from pathlib import Path
from typing import Any
import json

import torch
from tqdm import tqdm
import yaml

from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina
from openbabel import openbabel

from model.diffusion import LigandDiffusion
from model.flow_matching import LigandFlowMatching
from model.common import AtomCountPrior
from model.egnn import EGNN
from config.config import InferenceConfig
from datamodules import CrossDockedDataModule

from reconstruct import build_target


def bond_type_from_int(value: int) -> Chem.BondType:
    mapping = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    if int(value) not in mapping:
        raise ValueError(f"unknown bond type: {value}")
    return mapping[int(value)]


def build_reference_mol(reference: dict[str, Any]) -> Chem.Mol:
    atom_types = reference["ligand_element"].detach().cpu().long().tolist()
    pos = reference["ligand_pos"].detach().cpu().float()
    bond_index = reference["ligand_bond_index"].detach().cpu().long()
    bond_type = reference["ligand_bond_type"].detach().cpu().long().tolist()

    rw_mol = Chem.RWMol()
    for z in atom_types:
        rw_mol.AddAtom(Chem.Atom(int(z)))

    seen = set()
    for edge_idx in range(bond_index.shape[1]):
        i = int(bond_index[0, edge_idx])
        j = int(bond_index[1, edge_idx])
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        rw_mol.AddBond(a, b, bond_type_from_int(bond_type[edge_idx]))

    mol = rw_mol.GetMol()
    conf = Chem.Conformer(len(atom_types))
    for i, xyz in enumerate(pos.tolist()):
        conf.SetAtomPosition(i, (float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.KekulizeException:
        Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    return mol


def is_usable_mol(mol: Chem.Mol | None) -> bool:
    if mol is None:
        return False
    try:
        test_mol = Chem.Mol(mol)
        Chem.SanitizeMol(test_mol)
        if len(Chem.GetMolFrags(test_mol)) != 1:
            return False
        return True
    except Exception:
        return False


def protein_to_pdbqt_string(protein: dict[str, Any]) -> str:
    protein_pos = protein["protein_pos"].detach().cpu().float()
    protein_element = protein["protein_element"].detach().cpu().long()

    if "protein_batch" in protein:
        mask = protein["protein_batch"].detach().cpu().long() == 0
        protein_pos = protein_pos[mask]
        protein_element = protein_element[mask]

    ob_rec = openbabel.OBMol()
    ob_rec.BeginModify()
    for xyz, z in zip(protein_pos.tolist(), protein_element.tolist()):
        atom = ob_rec.NewAtom()
        atom.SetAtomicNum(int(z))
        atom.SetVector(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    ob_rec.EndModify()

    conv = openbabel.OBConversion()
    conv.SetOutFormat("pdbqt")
    conv.AddOption("r", openbabel.OBConversion.OUTOPTIONS)
    return conv.WriteString(ob_rec)


def mol_to_pdbqt_string(mol: Chem.Mol) -> str:
    work_mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(work_mol)
    work_mol = Chem.AddHs(work_mol, addCoords=True)

    if work_mol.GetNumConformers() == 0:
        status = AllChem.EmbedMolecule(work_mol, randomSeed=0)
        if status != 0:
            raise ValueError("rdkit embedding failed")

    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("mol", "pdbqt")
    ob_mol = openbabel.OBMol()
    ok = conv.ReadString(ob_mol, Chem.MolToMolBlock(work_mol))
    if not ok:
        raise ValueError("openbabel failed to read ligand mol block")
    return conv.WriteString(ob_mol)


def compute_box_from_reference(reference: dict[str, Any]) -> tuple[list[float], list[float]]:
    ligand_pos = reference["ligand_pos"].detach().cpu().float()
    center = ligand_pos.mean(dim=0).tolist()
    span = (ligand_pos.max(dim=0).values - ligand_pos.min(dim=0).values).tolist()

    # Vina search space guidance recommends including clear padding around the
    # bound ligand coordinates. +12 A total span gives ~6 A on each side.
    box_size = [max(22.0, float(v) + 12.0) for v in span]
    return center, box_size


def ensure_reference_cache(
    target_dir: Path,
    protein: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    dock_meta_path = target_dir / "docking.json"
    if dock_meta_path.exists():
        cached = json.loads(dock_meta_path.read_text(encoding="utf-8"))
        if cached.get("box_formula") == "ref_span_plus_12_min22":
            return cached

    receptor_path = target_dir / "receptor.pdbqt"
    reference_sdf_path = target_dir / "reference.sdf"
    reference_pdbqt_path = target_dir / "reference.pdbqt"

    receptor_path.write_text(protein_to_pdbqt_string(protein), encoding="utf-8")

    ref_mol = build_reference_mol(reference)
    Chem.MolToMolFile(ref_mol, str(reference_sdf_path))
    reference_pdbqt_path.write_text(mol_to_pdbqt_string(ref_mol), encoding="utf-8")

    center, box_size = compute_box_from_reference(reference)

    v = Vina(sf_name="vina")
    v.set_receptor(str(receptor_path))
    v.compute_vina_maps(center=center, box_size=box_size)
    v.set_ligand_from_file(str(reference_pdbqt_path))
    reference_vina_score = float(v.score()[0])

    payload = {
        "center": center,
        "box_size": box_size,
        "box_formula": "ref_span_plus_12_min22",
        "receptor_pdbqt": receptor_path.name,
        "reference_sdf": reference_sdf_path.name,
        "reference_pdbqt": reference_pdbqt_path.name,
        "reference_vina_score": reference_vina_score,
    }
    dock_meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def finalize_generated_samples(
    target_dir: Path,
    raw_target: dict[str, Any],
    atom_type_decoder: dict[int, Any],
) -> dict[str, Any]:
    built = build_target(raw_target, atom_type_decoder)

    lig_dir = target_dir / "ligands"
    lig_dir.mkdir(parents=True, exist_ok=True)

    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    for result in built["built_samples"]:
        if not result.ok:
            invalid_rows.append(
                {
                    "sample_idx": int(result.sample_idx),
                    "error": str(result.error),
                }
            )
            continue

        if not is_usable_mol(result.mol):
            invalid_rows.append(
                {
                    "sample_idx": int(result.sample_idx),
                    "error": "invalid_rdkit_mol",
                }
            )
            continue

        mol = Chem.Mol(result.mol)
        Chem.SanitizeMol(mol)

        sdf_path = lig_dir / f"{int(result.sample_idx):04d}.sdf"
        pdbqt_path = lig_dir / f"{int(result.sample_idx):04d}.pdbqt"

        try:
            Chem.MolToMolFile(mol, str(sdf_path))
            pdbqt_path.write_text(mol_to_pdbqt_string(mol), encoding="utf-8")
        except Exception as exc:
            if sdf_path.exists():
                sdf_path.unlink()
            if pdbqt_path.exists():
                pdbqt_path.unlink()
            invalid_rows.append(
                {
                    "sample_idx": int(result.sample_idx),
                    "error": f"pdbqt_export_failed: {exc}",
                }
            )
            continue

        valid_rows.append(
            {
                "sample_idx": int(result.sample_idx),
                "sdf": sdf_path.name,
                "pdbqt": pdbqt_path.name,
            }
        )

    payload = {
        "target_idx": int(raw_target["target_idx"]),
        "n_samples": len(raw_target["samples"]),
        "n_valid": len(valid_rows),
        "valid_fraction": (len(valid_rows) / len(raw_target["samples"])) if raw_target["samples"] else 0.0,
        "valid_samples": valid_rows,
        "invalid_samples": invalid_rows,
    }
    (target_dir / "reconstruction.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    with open("config/inference/base_config.yaml", "r") as f:
        cfg = InferenceConfig(**yaml.safe_load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples_per_target = getattr(cfg, "samples_per_target", 100)
    save_trajectory = getattr(cfg, "save_trajectory", True)

    dm = CrossDockedDataModule(
        lmdb_path=cfg.lmdb_path,
        split_pt_path=cfg.split_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
    )

    denoiser = EGNN(
        num_layers=cfg.num_layers,
        hidden_dim=cfg.hidden_dim,
        edge_feat_dim=cfg.edge_feat_dim,
        num_r_gaussian=cfg.num_r_gaussian,
        message_passing_mode=cfg.message_passing_mode,
        k=cfg.k,
        cutoff_mode=cfg.cutoff_mode,
        update_x=True,
        norm=cfg.norm,
    ).to(device)

    model = LigandDiffusion(
        denoiser=denoiser,
        hidden_dim=cfg.hidden_dim,
        num_types=cfg.num_types,
        steps=cfg.steps,
    ).to(device)

    ckpt = torch.load(cfg.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["diffusion"], strict=True)
    model.eval()

    if "prior" in ckpt:
        prior = AtomCountPrior.from_state_dict(ckpt["prior"])
    else:
        print("fitting atom count prior...")
        prior = AtomCountPrior.fit(dm.ds_train, n_bins=10)

    out_dir = Path("inference") / Path(cfg.ckpt).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    protein_keys = [
        "protein_pos",
        "protein_batch",
        "protein_element",
        "protein_atom_to_aa_type",
        "protein_is_backbone",
    ]

    ref_keys = [
        "ligand_pos",
        "ligand_element",
        "ligand_bond_index",
        "ligand_bond_type",
        "affinity",
    ]

    amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp else torch.float32

    if cfg.num_types > len(dm.ligand_elements):
        raise ValueError(
            f"cfg.num_types={cfg.num_types} exceeds available ligand elements "
            f"({len(dm.ligand_elements)}): {tuple(int(z) for z in dm.ligand_elements)}"
        )
    atom_type_decoder: dict[int, Any] = {
        i: int(z)
        for i, z in enumerate(dm.ligand_elements[:cfg.num_types])
    }

    all_recon = []

    for target_idx, batch in enumerate(tqdm(dm.test_dataloader(), desc="target")):
        target_dir = out_dir / f"{target_idx:06d}"
        target_dir.mkdir(parents=True, exist_ok=True)

        protein_cpu = {k: batch[k].cpu() for k in protein_keys}
        reference_cpu = {k: batch[k].cpu() for k in ref_keys if k in batch}

        raw_target = {
            "target_idx": target_idx,
            "protein": protein_cpu,
            "reference": reference_cpu,
            "samples": [],
        }

        ensure_reference_cache(target_dir, protein_cpu, reference_cpu)

        protein_gpu = {k: batch[k].to(device, non_blocking=True) for k in protein_keys}
        samples = []

        sample_batch_size = 50
        for sample_idx in tqdm(range(0, samples_per_target, sample_batch_size), leave=False):
            current_batch = min(sample_batch_size, samples_per_target - sample_idx)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp,
            ):
                if save_trajectory:
                    sample = model.sample(
                        protein_gpu,
                        prior,
                        return_trajectory=True,
                        trajectory_stride=1,
                        num_samples=current_batch,
                    )
                else:
                    sample = model.sample(
                        protein_gpu,
                        prior,
                        num_samples=current_batch,
                    )

            ligand_batch = sample["ligand_batch"].detach().cpu().long()
            ligand_pos = sample["ligand_pos"].detach().cpu()
            ligand_type = sample["ligand_type"].detach().cpu()
            protein_batch = sample["protein_batch"].detach().cpu().long()
            protein_pos = sample["protein_pos"].detach().cpu()
            traj = sample.get("trajectory")

            for local_idx in range(current_batch):
                lig_mask = ligand_batch == local_idx
                prot_mask = protein_batch == local_idx

                out_sample = {
                    "sample_idx": sample_idx + local_idx,
                    "ligand_pos": ligand_pos[lig_mask],
                    "ligand_type": ligand_type[lig_mask],
                    "ligand_batch": torch.zeros(int(lig_mask.sum().item()), dtype=torch.long),
                    "protein_pos": protein_pos[prot_mask],
                    "protein_batch": torch.zeros(int(prot_mask.sum().item()), dtype=torch.long),
                }
                if traj is not None:
                    out_sample["trajectory"] = [
                        {
                            "t": frame["t"],
                            "ligand_pos": frame["ligand_pos"][lig_mask].detach().cpu(),
                            "ligand_type": frame["ligand_type"][lig_mask].detach().cpu(),
                        }
                        for frame in traj
                    ]
                samples.append(out_sample)

        raw_target["samples"] = samples
        torch.save(raw_target, target_dir / "target.pt")

        recon = finalize_generated_samples(target_dir, raw_target, atom_type_decoder)
        all_recon.append(recon)

        print(
            f"target {target_idx}: "
            f"{recon['n_valid']}/{recon['n_samples']} valid "
            f"({recon['valid_fraction']:.3f})"
        )

    summary = {
        "n_targets": len(all_recon),
        "n_samples_total": sum(row["n_samples"] for row in all_recon),
        "n_valid_total": sum(row["n_valid"] for row in all_recon),
        "valid_fraction_mean": (
            sum(row["valid_fraction"] for row in all_recon) / len(all_recon)
            if all_recon else None
        ),
    }
    (out_dir / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
