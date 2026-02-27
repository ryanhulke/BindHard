from pathlib import Path
from typing import Any
import json
import statistics
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import rdMolDescriptors
# also need the calculation for SA; reference TargetDiff repo. They use a custom calculation

from vina import vina
import subprocess # for docking

from build_molecule import (
    build_target,
    list_target_files,
    load_target_file,
    valid_fraction,
    valid_results,
)


def run_vina_for_mol(
    protein: dict[str, Any],
    mol: Any,
) -> float:
    raise NotImplementedError


def compute_qed(mol: Any) -> float:
    raise NotImplementedError


def compute_sa(mol: Any) -> float:
    raise NotImplementedError


def compute_high_affinity_fraction(
    vina_scores: list[float],
    reference: dict[str, Any],
) -> float | None:
    raise NotImplementedError


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def evaluate_target_file(
    path: Path,
    atom_type_decoder: dict[int, Any],
) -> dict[str, Any]:
    target = load_target_file(path)
    built = build_target(target, atom_type_decoder)

    good = valid_results(built["built_samples"])

    vina_scores = [
        run_vina_for_mol(built["protein"], result.mol)
        for result in good
    ]
    qed_scores = [compute_qed(result.mol) for result in good]
    sa_scores = [compute_sa(result.mol) for result in good]

    return {
        "target_idx": built["target_idx"],
        "n_samples": len(built["built_samples"]),
        "n_valid": len(good),
        "valid_fraction": valid_fraction(built["built_samples"]),
        "vina_scores": vina_scores,
        "vina_min": min(vina_scores) if vina_scores else None,
        "high_affinity_fraction": compute_high_affinity_fraction(vina_scores, built["reference"]),
        "qed_scores": qed_scores,
        "sa_scores": sa_scores,
        "invalid_samples": [
            {
                "sample_idx": result.sample_idx,
                "error": result.error,
            }
            for result in built["built_samples"]
            if not result.ok
        ],
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
        "valid_fraction_mean": mean_or_none(valid_fracs),
        "valid_fraction_median": median_or_none(valid_fracs),
        "vina_mean": mean_or_none(all_vina),
        "vina_median": median_or_none(all_vina),
        "vina_min_mean": mean_or_none(vina_mins),
        "vina_min_median": median_or_none(vina_mins),
        "qed_mean": mean_or_none(all_qed),
        "qed_median": median_or_none(all_qed),
        "sa_mean": mean_or_none(all_sa),
        "sa_median": median_or_none(all_sa),
        "high_affinity_mean": mean_or_none(high_affinity),
        "high_affinity_median": median_or_none(high_affinity),
    }


def save_results(payload: dict[str, Any], out_path: Path) -> None:
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    inference_dir = Path("inference") / "best"
    out_path = inference_dir / "eval.json"

    atom_type_decoder: dict[int, Any] = {}

    target_files = list_target_files(inference_dir)
    target_results = []

    for path in tqdm(target_files, desc="eval"):
        target_results.append(evaluate_target_file(path, atom_type_decoder))

    payload = {
        "inference_dir": str(inference_dir),
        "targets": target_results,
        "summary": summarize_results(target_results),
    }

    save_results(payload, out_path)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()