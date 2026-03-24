import math
import pytest
from rdkit import Chem

from common import compute_normalized_sa, load_sa_fragment_scores, resolve_path


def test_load_sa_fragment_scores_reads_gzip_bundle() -> None:
    load_sa_fragment_scores.cache_clear()

    assert not resolve_path("data/fpscores.pkl", allow_missing=True).exists()
    assert resolve_path("data/fpscores.pkl.gz", allow_missing=True).exists()

    scores = load_sa_fragment_scores()

    assert scores


def test_compute_sa_returns_normalized_score() -> None:
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None

    sa = compute_normalized_sa(mol)

    assert math.isfinite(sa)
    assert 0.0 <= sa <= 1.0
    assert sa == pytest.approx(compute_normalized_sa(mol))


def test_compute_sa_keeps_higher_is_better_direction() -> None:
    easy = Chem.MolFromSmiles("CCO")
    harder = Chem.MolFromSmiles("C1CC2CCC3CC(C2)CC1C3")
    assert easy is not None
    assert harder is not None

    assert compute_normalized_sa(easy) > compute_normalized_sa(harder)
