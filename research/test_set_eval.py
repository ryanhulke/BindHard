from pathlib import Path
from typing import Any
import json
import math
import pickle
import statistics

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors
from vina import Vina


def list_target_dirs(inference_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in inference_dir.iterdir()
        if path.is_dir()
        and (path / "docking.json").exists()
        and (path / "reconstruction.json").exists()
    )


def load_mol_from_sdf(path: Path) -> Chem.Mol | None:
    mol = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False)
    if mol is None:
        return None
    return mol


def compute_qed(mol: Chem.Mol) -> float:
    return QED.qed(mol)


def compute_sa(mol: Chem.Mol) -> float:
    data = pickle.load(Path("data/fpscores.pkl").open("rb"))
    fpscores: dict[int, float] = {}
    for entry in data:
        for bit_id in entry[1:]:
            fpscores[bit_id] = float(entry[0])

    sa_mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    sparse_fp = sa_mfpgen.GetSparseCountFingerprint(mol)
    nze = sparse_fp.GetNonzeroElements()

    score1 = 0.0
    nf = 0
    for bit_id, count in nze.items():
        nf += count
        score1 += fpscores.get(bit_id, -4.0) * count

    if nf == 0:
        return float("nan")

    score1 /= nf

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
    macrocycle_penalty = math.log10(2) if n_macrocycles > 0 else 0.0

    score2 = (
        -size_penalty
        - stereo_penalty
        - spiro_penalty
        - bridge_penalty
        - macrocycle_penalty
    )

    score3 = 0.0
    num_bits = len(nze)
    if n_atoms > num_bits and num_bits > 0:
        score3 = math.log(float(n_atoms) / num_bits) * 0.5

    sa_score = score1 + score2 + score3

    raw_min = -4.0
    raw_max = 2.5
    sa_score = 11.0 - (sa_score - raw_min + 1.0) / (raw_max - raw_min) * 9.0

    if sa_score > 8.0:
        sa_score = 8.0 + math.log(sa_score - 8.0)

    if sa_score > 10.0:
        return 10.0
    if sa_score < 1.0:
        return 1.0
    return sa_score


def compute_high_affinity_fraction(
    vina_scores: list[float],
    reference_vina_score: float | None,
) -> float | None:
    if not vina_scores or reference_vina_score is None:
        return None
    return sum(score <= reference_vina_score for score in vina_scores) / len(vina_scores)


def evaluate_target_dir(path: Path) -> dict[str, Any]:
    docking = json.loads((path / "docking.json").read_text(encoding="utf-8"))
    recon = json.loads((path / "reconstruction.json").read_text(encoding="utf-8"))

    box_size = [float(v) for v in docking["box_size"]]
    if docking.get("box_formula") != "ref_span_plus_12_min22":
        # Legacy inference folders used span+8 A (+4 A each side), which can clip
        # generated ligands near the edge. Upgrade to span+12 A semantics.
        box_size = [max(22.0, v + 4.0) for v in box_size]

    v = Vina(sf_name="vina")
    v.set_receptor(str(path / docking["receptor_pdbqt"]))
    v.compute_vina_maps(center=docking["center"], box_size=box_size)

    lig_dir = path / "ligands"

    vina_scores: list[float] = []
    qed_scores: list[float] = []
    sa_scores: list[float] = []

    for row in recon["valid_samples"]:
        pdbqt_path = lig_dir / row["pdbqt"]
        sdf_path = lig_dir / row["sdf"]

        mol = load_mol_from_sdf(sdf_path)
        if mol is None:
            continue

        v.set_ligand_from_file(str(pdbqt_path))
        vina_scores.append(float(v.score()[0]))
        qed_scores.append(compute_qed(mol))
        sa_scores.append(compute_sa(mol))

    return {
        "target_idx": recon["target_idx"],
        "n_samples": recon["n_samples"],
        "n_valid": recon["n_valid"],
        "valid_fraction": recon["valid_fraction"],
        "vina_scores": vina_scores,
        "vina_min": min(vina_scores) if vina_scores else None,
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
    all_qed = [score for row in target_results for score in row["qed_scores"]]
    all_sa = [score for row in target_results for score in row["sa_scores"]]

    vina_mins = [
        row["vina_min"]
        for row in target_results
        if row["vina_min"] is not None
    ]

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
        "vina_min_mean": statistics.mean(vina_mins) if vina_mins else None,
        "vina_min_median": statistics.median(vina_mins) if vina_mins else None,
        "qed_mean": statistics.mean(all_qed) if all_qed else None,
        "qed_median": statistics.median(all_qed) if all_qed else None,
        "sa_mean": statistics.mean(all_sa) if all_sa else None,
        "sa_median": statistics.median(all_sa) if all_sa else None,
        "high_affinity_mean": statistics.mean(high_affinity) if high_affinity else None,
        "high_affinity_median": statistics.median(high_affinity) if high_affinity else None,
    }


def save_results(payload: dict[str, Any], out_path: Path) -> None:
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    inference_dir = Path("inference") / "gat_egnn_diffusion_last"
    out_path = inference_dir / "eval.json"

    target_dirs = list_target_dirs(inference_dir)
    target_results = []

    for path in tqdm(target_dirs, desc="eval"):
        target_results.append(evaluate_target_dir(path))

    payload = {
        "inference_dir": str(inference_dir),
        "targets": target_results,
        "summary": summarize_results(target_results),
    }

    save_results(payload, out_path)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()