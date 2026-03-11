from pathlib import Path
import math
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import compute_sa
from research.reconstruct import build_target, load_target_file, valid_results
import pytest



def test_compute_sa_with_inference_molecule(monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)

    target_path = REPO_ROOT / "inference" / "graphAttn_flow_matching_best" / "000000" / "target.pt"

    target = load_target_file(target_path)
    built = build_target(target, atom_type_decoder={})
    good = valid_results(built["built_samples"])
    assert good, "no valid molecules could be reconstructed from inference output"

    sa = compute_sa(good[0].mol)
    assert math.isfinite(sa), f"SA score must be finite, got {sa}"
    assert 1.0 <= sa <= 10.0, f"SA score must be in [1, 10], got {sa}"
