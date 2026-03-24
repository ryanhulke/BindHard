from __future__ import annotations

import gzip
import math
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar
import json
import torch
import yaml
from model.egnn import EGNN
from model.flow_matching import LigandFlowMatching
from model.sampling_guidance import PocketAffinityGuidance


RESEARCH_ROOT = Path(__file__).resolve().parent
ConfigT = TypeVar("ConfigT")


def resolve_path(path_like: str | Path, *, allow_missing: bool = False) -> Path:
    path = Path(path_like)
    if path.exists():
        return path
    rooted = RESEARCH_ROOT / path
    if rooted.exists() or (allow_missing and rooted.parent.exists()):
        return rooted
    return path


def load_config(config_type: type[ConfigT], path_like: str | Path) -> ConfigT:
    config_path = resolve_path(path_like)
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return config_type(**payload)


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_denoiser(cfg: Any, device: torch.device) -> EGNN:

    return EGNN(
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


def build_diffusion_model(cfg: Any, device: torch.device) -> LigandFlowMatching:

    return LigandFlowMatching(
        denoiser=build_denoiser(cfg, device),
        num_types=cfg.num_types,
        steps=cfg.steps,
        type_loss_scale=getattr(cfg, "type_loss_scale", 4.0),
        bond_loss_scale=getattr(cfg, "bond_loss_scale", 8.0),
        charge_loss_scale=getattr(cfg, "charge_loss_scale", 1000.0),
        count_loss_scale=getattr(cfg, "count_loss_scale", 0.5),
        protein_noise_std=getattr(cfg, "protein_noise_std", 0.1),
        hidden_dim=cfg.hidden_dim,
        max_ligand_atoms=cfg.max_ligand_atoms,
    ).to(device)


def build_guidance_model(cfg: Any, device: torch.device) -> PocketAffinityGuidance:
    return PocketAffinityGuidance(
        denoiser=build_denoiser(cfg, device),
        num_types=cfg.num_types,
        hidden_dim=cfg.hidden_dim,
    ).to(device)


def build_optimizer(model: torch.nn.Module, cfg: Any, device: torch.device) -> torch.optim.Optimizer:
    optimizer_name = str(getattr(cfg, "optimizer", "adamw")).lower()
    optimizer_kwargs: dict[str, Any] = {
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    if optimizer_name != "adamw":
        raise ValueError(f"unsupported optimizer: {optimizer_name!r}")
    return torch.optim.AdamW(model.parameters(), **optimizer_kwargs)


def amp_settings(device: torch.device, precision: str) -> tuple[bool, torch.dtype]:
    amp = precision.lower()
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(amp, torch.float32)
    return device.type == "cuda" and amp in {"bf16", "fp16"}, dtype


def build_atom_type_decoder(ligand_elements: tuple[int, ...], num_types: int) -> dict[int, int]:
    if num_types > len(ligand_elements):
        raise ValueError(
            f"num_types={num_types} exceeds available ligand elements "
            f"({len(ligand_elements)}): {ligand_elements}"
        )
    return {index: atomic_num for index, atomic_num in enumerate(ligand_elements[:num_types])}


def protein_to_pdbqt_string(protein: dict[str, Any]) -> str:
    from openbabel import openbabel

    protein_pos = protein["protein_pos"].detach().cpu().float()
    protein_element = protein["protein_element"].detach().cpu().long()

    if "protein_batch" in protein:
        mask = protein["protein_batch"].detach().cpu().long() == 0
        protein_pos = protein_pos[mask]
        protein_element = protein_element[mask]

    receptor = openbabel.OBMol()
    receptor.BeginModify()
    for xyz, atomic_num in zip(protein_pos.tolist(), protein_element.tolist()):
        atom = receptor.NewAtom()
        atom.SetAtomicNum(int(atomic_num))
        atom.SetVector(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    receptor.EndModify()

    conversion = openbabel.OBConversion()
    conversion.SetOutFormat("pdbqt")
    conversion.AddOption("r", openbabel.OBConversion.OUTOPTIONS)
    return conversion.WriteString(receptor)


def mol_to_pdbqt_string(mol) -> str:
    from openbabel import openbabel
    from rdkit import Chem
    from rdkit.Chem import AllChem

    work_mol = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    Chem.RemoveStereochemistry(work_mol)

    if work_mol.GetNumConformers() == 0 and AllChem.EmbedMolecule(work_mol, randomSeed=0) != 0:
        raise ValueError("rdkit embedding failed")

    conversion = openbabel.OBConversion()
    conversion.SetInAndOutFormats("mol", "pdbqt")
    ob_mol = openbabel.OBMol()
    if not conversion.ReadString(ob_mol, Chem.MolToMolBlock(work_mol)):
        raise ValueError("openbabel failed to read ligand mol block")
    return conversion.WriteString(ob_mol)


def compute_vina_box(ligand_pos: torch.Tensor) -> tuple[list[float], list[float]]:
    coords = ligand_pos.detach().cpu().float()
    center = coords.mean(dim=0).tolist()
    span = (coords.max(dim=0).values - coords.min(dim=0).values).tolist()
    # Vina search space guidance recommends including clear padding around the
    # bound ligand coordinates. +12 A total span gives ~6 A on each side.
    return center, [max(22.0, float(value) + 12.0) for value in span]


class VinaPoseScorer:
    def __init__(
        self,
        protein: dict[str, Any],
        box_ligand_pos: torch.Tensor,
        work_dir: str | Path,
    ):
        from vina import Vina

        self.work_path = Path(work_dir)
        self.work_path.mkdir(parents=True, exist_ok=True)
        self.receptor_path = self.work_path / "receptor.pdbqt"
        self.ligand_path = self.work_path / "ligand.pdbqt"
        self.receptor_path.write_text(protein_to_pdbqt_string(protein), encoding="utf-8")

        center, box_size = compute_vina_box(box_ligand_pos)
        self.vina = Vina(sf_name="vina")
        self.vina.set_receptor(str(self.receptor_path))
        self.vina.compute_vina_maps(center=center, box_size=box_size)

    def score_mol(self, mol) -> float:
        self.ligand_path.write_text(mol_to_pdbqt_string(mol), encoding="utf-8")
        self.vina.set_ligand_from_file(str(self.ligand_path))
        return float(self.vina.score()[0])


@lru_cache(maxsize=1)
def load_sa_fragment_scores() -> dict[int, float]:
    for relative_path in ("data/fpscores.pkl", "data/fpscores.pkl.gz"):
        scores_path = resolve_path(relative_path, allow_missing=True)
        if not scores_path.exists():
            continue

        scores: dict[int, float] = {}
        open_fn = gzip.open if scores_path.suffix == ".gz" else scores_path.open
        with open_fn(scores_path, "rb") as f:
            for entry in pickle.load(f):
                for bit_id in entry[1:]:
                    scores[int(bit_id)] = float(entry[0])
        return scores

    raise FileNotFoundError(
        "synthetic accessibility scoring requires data/fpscores.pkl "
        "or data/fpscores.pkl.gz"
    )


def compute_normalized_sa(mol) -> float:
    """Return normalized SA in [0, 1], where higher means easier synthesis."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

    fpscores = load_sa_fragment_scores()
    sparse_fp = rdFingerprintGenerator.GetMorganGenerator(radius=2).GetSparseCountFingerprint(mol)
    nonzero = sparse_fp.GetNonzeroElements()

    feature_count = int(sum(nonzero.values()))
    if feature_count <= 0:
        return float("nan")

    fragment_score = 0.0
    for bit_id, count in nonzero.items():
        fragment_score += fpscores.get(int(bit_id), -4.0) * int(count)
    fragment_score /= float(feature_count)

    n_atoms = mol.GetNumAtoms()
    n_chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ring_info = mol.GetRingInfo()
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    n_macrocycles = sum(1 for ring in ring_info.AtomRings() if len(ring) > 8)

    size_penalty = n_atoms**1.005 - n_atoms
    stereo_penalty = math.log10(n_chiral_centers + 1)
    spiro_penalty = math.log10(n_spiro + 1)
    bridge_penalty = math.log10(n_bridgeheads + 1)
    macrocycle_penalty = math.log10(2.0) if n_macrocycles > 0 else 0.0

    complexity_score = (
        fragment_score
        - size_penalty
        - stereo_penalty
        - spiro_penalty
        - bridge_penalty
        - macrocycle_penalty
    )

    score_correction = 0.0
    num_bits = len(nonzero)
    if n_atoms > num_bits and num_bits > 0:
        score_correction = math.log(float(n_atoms) / float(num_bits)) * 0.5

    sa_raw = complexity_score + score_correction
    sa_raw = 11.0 - (sa_raw - (-4.0) + 1.0) / (2.5 - (-4.0)) * 9.0

    if sa_raw > 8.0:
        sa_raw = 8.0 + math.log(sa_raw - 8.0)

    sa_raw = min(10.0, max(1.0, sa_raw))
    sa = (10.0 - sa_raw) / 9.0
    return float(min(1.0, max(0.0, sa)))


def compute_vina_sa_reward(
    scorer: VinaPoseScorer,
    ligand_pos: torch.Tensor,
    ligand_type: torch.Tensor,
    atom_type_decoder: dict[int, Any],
    *,
    vina_clip_low: float,
    vina_clip_high: float,
    vina_offset: float,
    vina_divisor: float,
    sa_shift: float,
    sa_scale: float,
    ligand_bond_index: torch.Tensor | None = None,
    ligand_bond_type: torch.Tensor | None = None,
    ligand_charge: torch.Tensor | None = None,
) -> tuple[float | None, dict[str, float] | None]:
    from reconstruct_molecule import infer_mol_from_geometry

    if vina_divisor == 0:
        raise ValueError("vina_divisor must be non-zero")
    if sa_scale == 0:
        raise ValueError("sa_scale must be non-zero")

    mol = infer_mol_from_geometry(
        ligand_pos=ligand_pos,
        ligand_type=ligand_type,
        atom_type_decoder=atom_type_decoder,
        ligand_bond_index=ligand_bond_index,
        ligand_bond_type=ligand_bond_type,
        ligand_charge=ligand_charge,
    )
    if mol is None:
        return None, None

    vina_score = scorer.score_mol(mol)
    sa = compute_normalized_sa(mol)

    vina_score_clip = float(max(vina_clip_low, min(vina_clip_high, vina_score)))
    reward = (-vina_score_clip + vina_offset) / vina_divisor + (sa - sa_shift) / sa_scale
    if not math.isfinite(reward):
        return None, None

    return float(reward), {
        "vina_score": float(vina_score),
        "vina_score_clip": vina_score_clip,
        "sa": float(sa),
        "reward": float(reward),
    }


def score_bound_pose(
    protein: dict[str, Any],
    mol,
    ligand_pos: torch.Tensor,
    work_dir: str | Path,
) -> float:
    return VinaPoseScorer(
        protein=protein,
        box_ligand_pos=ligand_pos,
        work_dir=work_dir,
    ).score_mol(mol)


VINA_SCORE_PROTOCOL = "crossdocked_vina_score_v1"
DEFAULT_GUIDANCE_TARGET = "vina_score"


def vina_score_metadata() -> dict[str, Any]:
    return {
        "kind": "meta",
        "protocol": VINA_SCORE_PROTOCOL,
        "target": DEFAULT_GUIDANCE_TARGET,
        "lower_is_better": True,
        "score_mode": "vina_score_only",
        "box_formula": "ref_span_plus_12_min22",
        "center_mode": "protein_mean",
    }


def validate_metadata(
    metadata: dict[str, Any],
    *,
    expected_protocol: str,
    expected_target: str | None,
) -> None:
    protocol = metadata.get("protocol")
    if protocol != expected_protocol:
        raise ValueError(
            f"label sidecar protocol mismatch: expected {expected_protocol!r}, got {protocol!r}"
        )
    if expected_target is not None:
        target = metadata.get("target")
        if target != expected_target:
            raise ValueError(
                f"label sidecar target mismatch: expected {expected_target!r}, got {target!r}"
            )


def read_label_sidecar(path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    label_path = Path(path)
    if not label_path.exists():
        raise FileNotFoundError(f"label sidecar not found: {label_path}")

    metadata: dict[str, Any] | None = None
    records: dict[str, dict[str, Any]] = {}
    with label_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            kind = row.get("kind", "sample")
            if kind == "meta":
                metadata = row
                continue

            sample_key = row.get("sample_key")
            if sample_key is None:
                raise ValueError(
                    f"label sidecar row {line_no} is missing 'sample_key': {label_path}"
                )
            records[str(sample_key)] = row

    if metadata is None:
        raise ValueError(f"label sidecar is missing metadata row: {label_path}")
    return metadata, records


def load_scalar_labels(
    path: str | Path,
    *,
    target_name: str,
    split: str | None = None,
    expected_protocol: str = VINA_SCORE_PROTOCOL,
) -> dict[str, float]:
    metadata, records = read_label_sidecar(path)
    validate_metadata(
        metadata,
        expected_protocol=expected_protocol,
        expected_target=target_name,
    )

    out: dict[str, float] = {}
    for sample_key, row in records.items():
        if split is not None and row.get("split") != split:
            continue
        if row.get("status") != "ok":
            continue
        value = row.get(target_name)
        if value is None:
            continue
        out[sample_key] = float(value)
    return out
