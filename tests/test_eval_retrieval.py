"""CMC/mAP math pinned on tiny contrived rankings."""
import numpy as np
import pytest

from eval.retrieval import evaluate_retrieval


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_perfect_retrieval():
    q = np.stack([unit([1, 0]), unit([0, 1])])
    g = np.stack([unit([1, 0.05]), unit([0.05, 1]), unit([1, 1])])
    res = evaluate_retrieval(q, ["a", "b"], ["c1", "c2"],
                             g, ["a", "b", "z"], ["c9", "c9", "c9"])
    assert res.rank1 == 1.0
    assert res.mean_ap == 1.0


def test_rank_boundary_exact():
    # Query "a": correct gallery item ranks exactly 2nd ("b" outranks it).
    q = np.stack([unit([1, 0])])
    g = np.stack([unit([1, 0.01]), unit([0.9, 0.3]), unit([0, 1])])
    res = evaluate_retrieval(q, ["a"], ["c1"],
                             g, ["b", "a", "c"], ["c9", "c9", "c9"])
    assert res.rank(1) == 0.0
    assert res.rank(2) == 1.0, "hit at rank 2 must count at k=2, not k=3"
    assert res.mean_ap == pytest.approx(0.5)


def test_same_camera_same_id_excluded():
    # The only matching gallery item shares the query's camera; with every
    # query skipped there is nothing to score and the harness must say so
    # loudly rather than return vacuous numbers.
    q = np.stack([unit([1, 0])])
    g = np.stack([unit([1, 0]), unit([0, 1])])
    with pytest.raises(ValueError, match="no query"):
        evaluate_retrieval(q, ["a"], ["c1"], g, ["a", "b"], ["c1", "c2"])


def test_mixed_skip_accounting():
    q = np.stack([unit([1, 0]), unit([0, 1])])
    g = np.stack([unit([1, 0]), unit([0, 1])])
    res = evaluate_retrieval(q, ["a", "b"], ["c1", "c5"],
                             g, ["a", "b"], ["c1", "c2"])
    # query a: its only positive is same-camera -> skipped
    # query b: positive at c2 -> scored, rank 1
    assert res.n_queries_skipped == 1
    assert res.n_queries_scored == 1
    assert res.rank1 == 1.0


def test_map_multiple_positives():
    # Positives at ranks 1 and 3: AP = (1/1 + 2/3) / 2.
    q = np.stack([unit([1, 0])])
    g = np.stack([unit([1, 0.0]), unit([0.9, 0.4]), unit([0.9, 0.43]), unit([0, 1])])
    res = evaluate_retrieval(q, ["a"], ["c1"],
                             g, ["a", "b", "a", "c"], ["c2", "c9", "c3", "c9"])
    assert res.mean_ap == pytest.approx((1.0 + 2 / 3) / 2)


def test_errors_on_empty():
    with pytest.raises(ValueError):
        evaluate_retrieval(np.zeros((0, 2)), [], [], np.zeros((1, 2)), ["a"], ["c"])
