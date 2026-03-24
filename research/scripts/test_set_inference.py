from __future__ import annotations

import contextlib
import json
import statistics
from pathlib import Path
from typing import Any
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import QED
from vina import Vina
import torch

from reconstruct_molecule import build_target, is_valid_mol, build_reference_mol
from common import compute_vina_box, mol_to_pdbqt_string, protein_to_pdbqt_string
from common import (
    build_atom_type_decoder,
    build_datamodule,
    build_diffusion_model,
    build_guidance_model,
    compute_normalized_sa,
    load_config,
    resolve_path,
)
from config.config import InferenceConfig
from model.common import AtomCountPrior


PROTEIN_KEYS = (
    "protein_pos",
    "protein_batch",
    "protein_element",
    "protein_atom_to_aa_type",
    "protein_is_backbone",
)
REFERENCE_KEYS = (
    "ligand_pos",
    "ligand_element",
    "ligand_bond_index",
    "ligand_bond_type",
    "ligand_formal_charge",
    "ligand_smiles",
    "affinity",
    "vina_score",
)


def list_target_dirs(inference_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in inference_dir.iterdir()
        if path.is_dir()
        and (path / "docking.json").exists()
        and (path / "reconstruction.json").exists()
    )


def load_mol_from_sdf(path: Path) -> Chem.Mol | None:

    return Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False)


def compute_qed(mol: Chem.Mol) -> float:
    return QED.qed(mol)


def compute_high_affinity_fraction(
    vina_scores: list[float],
    reference_vina_score: float | None,
) -> float | None:
    if not vina_scores or reference_vina_score is None:
        return None
    return sum(score <= reference_vina_score for score in vina_scores) / len(vina_scores)


def score_ligand_pose(vina: Any, ligand_pdbqt_path: Path) -> tuple[float, float]:
    vina.set_ligand_from_file(str(ligand_pdbqt_path))
    vina_score = float(vina.score()[0])
    vina_min = float(vina.optimize()[0])
    return vina_score, vina_min


def evaluate_target_dir(path: Path) -> dict[str, Any]:

    docking = json.loads((path / "docking.json").read_text(encoding="utf-8"))
    recon = json.loads((path / "reconstruction.json").read_text(encoding="utf-8"))

    box_size = [float(value) for value in docking["box_size"]]
    if docking.get("box_formula") != "ref_span_plus_12_min22":
        # Legacy inference folders used span+8 A (+4 A each side), which can clip
        # generated ligands near the edge. Upgrade to span+12 A semantics.
        box_size = [max(22.0, value + 4.0) for value in box_size]

    vina = Vina(sf_name="vina")
    vina.set_receptor(str(path / docking["receptor_pdbqt"]))
    vina.compute_vina_maps(center=docking["center"], box_size=box_size)

    vina_scores: list[float] = []
    vina_min_scores: list[float] = []
    qed_scores: list[float] = []
    sa_scores: list[float] = []
    lig_dir = path / "ligands"

    for row in recon["valid_samples"]:
        mol = load_mol_from_sdf(lig_dir / row["sdf"])
        if mol is None:
            continue
        vina_score, vina_min = score_ligand_pose(vina, lig_dir / row["pdbqt"])
        vina_scores.append(vina_score)
        vina_min_scores.append(vina_min)
        qed_scores.append(compute_qed(mol))
        sa_scores.append(compute_normalized_sa(mol))

    return {
        "target_idx": recon["target_idx"],
        "n_samples": recon["n_samples"],
        "n_valid": recon["n_valid"],
        "valid_fraction": recon["valid_fraction"],
        "vina_scores": vina_scores,
        "vina_min_scores": vina_min_scores,
        "vina_min_mean": statistics.mean(vina_min_scores) if vina_min_scores else None,
        "vina_min_median": statistics.median(vina_min_scores) if vina_min_scores else None,
        "high_affinity_fraction": compute_high_affinity_fraction(
            vina_scores,
            docking.get("reference_vina_score"),
        ),
        "qed_scores": qed_scores,
        "sa_scores": sa_scores,
        "invalid_samples": recon["invalid_samples"],
    }


def summarize_results(target_results: list[dict[str, Any]]) -> dict[str, Any]:
    all_vina = [score for row in target_results for score in row["vina_scores"]]
    all_vina_min = [score for row in target_results for score in row["vina_min_scores"]]
    all_qed = [score for row in target_results for score in row["qed_scores"]]
    all_sa = [score for row in target_results for score in row["sa_scores"]]
    high_affinity = [
        row["high_affinity_fraction"]
        for row in target_results
        if row["high_affinity_fraction"] is not None
    ]
    valid_fracs = [row["valid_fraction"] for row in target_results]

    return {
        "n_targets": len(target_results),
        "n_samples_total": sum(row["n_samples"] for row in target_results),
        "n_valid_total": sum(row["n_valid"] for row in target_results),
        "valid_fraction_mean": statistics.mean(valid_fracs) if valid_fracs else None,
        "valid_fraction_median": statistics.median(valid_fracs) if valid_fracs else None,
        "vina_mean": statistics.mean(all_vina) if all_vina else None,
        "vina_median": statistics.median(all_vina) if all_vina else None,
        "vina_min_mean": statistics.mean(all_vina_min) if all_vina_min else None,
        "vina_min_median": statistics.median(all_vina_min) if all_vina_min else None,
        "qed_mean": statistics.mean(all_qed) if all_qed else None,
        "qed_median": statistics.median(all_qed) if all_qed else None,
        "sa_mean": statistics.mean(all_sa) if all_sa else None,
        "sa_median": statistics.median(all_sa) if all_sa else None,
        "high_affinity_mean": statistics.mean(high_affinity) if high_affinity else None,
        "high_affinity_median": statistics.median(high_affinity) if high_affinity else None,
    }


def save_results(payload: dict[str, Any], out_path: Path) -> None:
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_inference_dir(inference_dir: Path) -> dict[str, Any]:
    target_results = [
        evaluate_target_dir(path)
        for path in tqdm(list_target_dirs(inference_dir), desc="eval")
    ]
    return {
        "inference_dir": str(inference_dir),
        "targets": target_results,
        "summary": summarize_results(target_results),
    }


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
    center, box_size = compute_vina_box(reference["ligand_pos"])

    Chem.MolToMolFile(ref_mol, str(reference_sdf_path))
    reference_pdbqt_path.write_text(mol_to_pdbqt_string(ref_mol), encoding="utf-8")

    vina = Vina(sf_name="vina")
    vina.set_receptor(str(receptor_path))
    vina.compute_vina_maps(center=center, box_size=box_size)
    vina.set_ligand_from_file(str(reference_pdbqt_path))

    payload = {
        "center": center,
        "box_size": box_size,
        "box_formula": "ref_span_plus_12_min22",
        "receptor_pdbqt": receptor_path.name,
        "reference_sdf": reference_sdf_path.name,
        "reference_pdbqt": reference_pdbqt_path.name,
        "reference_vina_score": float(vina.score()[0]),
    }
    dock_meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def split_generated_batch(
    sample: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]],
    *,
    sample_offset: int,
    batch_size: int,
    use_explicit_graph: bool,
) -> list[dict[str, Any]]:

    ligand_batch = sample["ligand_batch"].detach().cpu().long()
    ligand_pos = sample["ligand_pos"].detach().cpu()
    ligand_type = sample["ligand_type"].detach().cpu()
    protein_batch = sample["protein_batch"].detach().cpu().long()
    protein_pos = sample["protein_pos"].detach().cpu()
    ligand_charge = sample.get("ligand_charge")
    ligand_bond_index = sample.get("ligand_bond_index")
    ligand_bond_type = sample.get("ligand_bond_type")
    trajectory = sample.get("trajectory")

    if isinstance(ligand_charge, torch.Tensor):
        ligand_charge = ligand_charge.detach().cpu()
    if isinstance(ligand_bond_index, torch.Tensor):
        ligand_bond_index = ligand_bond_index.detach().cpu().long()
    if isinstance(ligand_bond_type, torch.Tensor):
        ligand_bond_type = ligand_bond_type.detach().cpu().long()

    rows = []
    for local_idx in range(batch_size):
        lig_mask = ligand_batch == local_idx
        prot_mask = protein_batch == local_idx
        row = {
            "sample_idx": sample_offset + local_idx,
            "ligand_pos": ligand_pos[lig_mask],
            "ligand_type": ligand_type[lig_mask],
            "ligand_batch": torch.zeros(int(lig_mask.sum().item()), dtype=torch.long),
            "protein_pos": protein_pos[prot_mask],
            "protein_batch": torch.zeros(int(prot_mask.sum().item()), dtype=torch.long),
        }

        if use_explicit_graph and ligand_charge is not None:
            row["ligand_charge"] = ligand_charge[lig_mask]
        if use_explicit_graph and ligand_bond_index is not None and ligand_bond_type is not None:
            global_lig_idx = torch.where(lig_mask)[0]
            global_to_local = torch.full((ligand_batch.shape[0],), -1, dtype=torch.long)
            global_to_local[global_lig_idx] = torch.arange(global_lig_idx.shape[0], dtype=torch.long)
            bond_mask = (
                lig_mask[ligand_bond_index[0]] & lig_mask[ligand_bond_index[1]]
                if ligand_bond_index.numel() > 0
                else torch.zeros((0,), dtype=torch.bool)
            )
            sample_bond_index = ligand_bond_index[:, bond_mask]
            if sample_bond_index.numel() > 0:
                sample_bond_index = global_to_local[sample_bond_index]
            row["ligand_bond_index"] = sample_bond_index
            row["ligand_bond_type"] = ligand_bond_type[bond_mask]

        if trajectory is not None:
            row["trajectory"] = [
                {
                    "t": frame["t"],
                    "ligand_pos": frame["ligand_pos"][lig_mask].detach().cpu(),
                    "ligand_type": frame["ligand_type"][lig_mask].detach().cpu(),
                }
                for frame in trajectory
            ]

        rows.append(row)

    return rows


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
            invalid_rows.append({"sample_idx": int(result.sample_idx), "error": str(result.error)})
            continue
        if not is_valid_mol(result.mol):
            invalid_rows.append({"sample_idx": int(result.sample_idx), "error": "invalid_rdkit_mol"})
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
        "valid_fraction": len(valid_rows) / len(raw_target["samples"]) if raw_target["samples"] else 0.0,
        "valid_samples": valid_rows,
        "invalid_samples": invalid_rows,
    }
    (target_dir / "reconstruction.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:

    cfg = load_config(InferenceConfig, "config/inference/flow_matching.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = build_datamodule(cfg, batch_size=1, return_text_fields=True, drop_last=False)

    ckpt = torch.load(resolve_path(cfg.ckpt), map_location="cpu")
    model = build_diffusion_model(cfg, device)
    load_result = model.load_state_dict(ckpt["diffusion"], strict=False)
    model.eval()

    use_prior = any(key.startswith("count_head.") for key in load_result.missing_keys)
    use_explicit_graph = not any(
        key.startswith(("bond_head.", "charge_head.")) for key in load_result.missing_keys
    )
    if use_prior and "prior" in ckpt:
        prior = AtomCountPrior.from_state_dict(ckpt["prior"])
    elif use_prior:
        print("fitting atom count prior...")
        prior = AtomCountPrior.fit(dm.ds_train, n_bins=10)
    else:
        prior = None

    guidance_model = None
    if cfg.guidance_ckpt:
        guidance_ckpt = torch.load(resolve_path(cfg.guidance_ckpt), map_location="cpu")
        guidance_model = build_guidance_model(cfg, device)
        guidance_model.load_state_dict(
            guidance_ckpt["guidance"] if "guidance" in guidance_ckpt else guidance_ckpt,
            strict=True,
        )
        guidance_model.eval()

    out_dir = Path("inference") / Path(cfg.ckpt).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp else torch.float32
    atom_type_decoder = build_atom_type_decoder(dm.ligand_elements, cfg.num_types)
    all_recon = []
    all_eval = []

    for target_idx, batch in enumerate(tqdm(dm.test_dataloader(), desc="target")):
        if int(batch["ligand_counts"].numel()) != 1:
            raise ValueError(
                "test_set_inference requires batch_size=1 so each target "
                "contains a single protein-ligand complex"
            )

        target_dir = out_dir / f"{target_idx:06d}"
        target_dir.mkdir(parents=True, exist_ok=True)

        protein_cpu = {key: batch[key].cpu() for key in PROTEIN_KEYS}
        reference_cpu = {
            key: batch[key].cpu() if torch.is_tensor(batch[key]) else batch[key][0]
            for key in REFERENCE_KEYS
            if key in batch
        }
        raw_target = {
            "target_idx": target_idx,
            "protein": protein_cpu,
            "reference": reference_cpu,
            "samples": [],
        }

        ensure_reference_cache(target_dir, protein_cpu, reference_cpu)
        protein_gpu = {key: batch[key].to(device, non_blocking=True) for key in PROTEIN_KEYS}

        for sample_idx in tqdm(
            range(0, cfg.samples_per_target, cfg.sample_batch_size),
            leave=False,
        ):
            current_batch = min(cfg.sample_batch_size, cfg.samples_per_target - sample_idx)
            inference_ctx = (
                contextlib.nullcontext()
                if guidance_model is not None and cfg.guidance_scale > 0
                else torch.inference_mode()
            )
            with inference_ctx, torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                sample = model.sample(
                    protein_gpu,
                    prior,
                    return_trajectory=cfg.save_trajectory,
                    num_samples=current_batch,
                    guidance_model=guidance_model,
                    guidance_lower_is_better=cfg.guidance_lower_is_better,
                    guidance_scale=cfg.guidance_scale,
                    guidance_clip=cfg.guidance_clip,
                )
            raw_target["samples"].extend(
                split_generated_batch(
                    sample,
                    sample_offset=sample_idx,
                    batch_size=current_batch,
                    use_explicit_graph=use_explicit_graph,
                )
            )

        torch.save(raw_target, target_dir / "target.pt")
        recon = finalize_generated_samples(target_dir, raw_target, atom_type_decoder)
        all_recon.append(recon)
        all_eval.append(evaluate_target_dir(target_dir))
        print(
            f"target {target_idx}: {recon['n_valid']}/{recon['n_samples']} valid "
            f"({recon['valid_fraction']:.3f})"
        )

    reconstruction_summary = {
        "n_targets": len(all_recon),
        "n_samples_total": sum(row["n_samples"] for row in all_recon),
        "n_valid_total": sum(row["n_valid"] for row in all_recon),
        "valid_fraction_mean": (
            sum(row["valid_fraction"] for row in all_recon) / len(all_recon) if all_recon else None
        ),
    }
    save_results(reconstruction_summary, out_dir / "reconstruction_summary.json")
    print(json.dumps(reconstruction_summary, indent=2))

    eval_payload = {
        "inference_dir": str(out_dir),
        "targets": all_eval,
        "summary": summarize_results(all_eval),
    }
    save_results(eval_payload, out_dir / "eval.json")
    print(json.dumps(eval_payload["summary"], indent=2))


if __name__ == "__main__":
    main()
