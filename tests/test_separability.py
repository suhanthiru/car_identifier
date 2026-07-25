"""Separability protocols: does the fair/adversarial split actually expose
the mining bias it exists to expose?

These use synthetic embeddings with a KNOWN structure, so the expected
answers are derivable rather than observed — the point is to pin the
protocol's behaviour, not to measure any real embedder.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from eval.separability import (
    evaluate_separability, positive_pairs, separability_auc,
    topk_bucket_negatives, uniform_bucket_negatives,
)


@dataclass(frozen=True)
class _Rec:
    vehicle_id: int
    camera_id: str
    color: str = "silver"
    body_type: str = "sedan"


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


# ------------------------------------------------------------------ auc

def test_auc_perfect_separation():
    assert separability_auc([0.9, 0.8, 0.95], [0.1, 0.2, 0.3]) == 1.0


def test_auc_fully_inverted():
    assert separability_auc([0.1, 0.2], [0.8, 0.9]) == 0.0


def test_auc_all_ties_is_exactly_half():
    # Every value identical -> no information -> 0.5, not a sort artefact.
    assert separability_auc([0.5] * 4, [0.5] * 6) == pytest.approx(0.5)


def test_auc_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one positive"):
        separability_auc([], [0.5])


# ------------------------------------------------- protocol construction

def test_positive_pairs_split_by_camera():
    records = [_Rec(1, "c001"), _Rec(1, "c002"), _Rec(1, "c001")]
    emb = _normalize(np.random.default_rng(0).normal(size=(3, 8)))

    cross = positive_pairs(records, emb, cross_camera=True)
    same = positive_pairs(records, emb, cross_camera=False)

    # indices 0-1 and 1-2 cross cameras; 0-2 shares c001.
    assert {tuple(sorted((p.a_index, p.b_index))) for p in cross} == {(0, 1), (1, 2)}
    assert {tuple(sorted((p.a_index, p.b_index))) for p in same} == {(0, 2)}
    assert all(p.same_vehicle for p in cross + same)


def test_negatives_never_pair_a_vehicle_with_itself():
    records = [_Rec(1, "c001"), _Rec(1, "c002"), _Rec(2, "c001"), _Rec(2, "c002")]
    emb = _normalize(np.random.default_rng(1).normal(size=(4, 8)))
    ids = [r.vehicle_id for r in records]

    for pairs in (uniform_bucket_negatives(records, emb),
                  topk_bucket_negatives(records, emb)):
        assert pairs
        assert all(not p.same_vehicle for p in pairs)
        assert all(ids[p.a_index] != ids[p.b_index] for p in pairs)


# ------------------------------------------------------- the actual bias

def _known_good_fixture():
    """A deliberately GOOD embedder: each vehicle sits on its own tight
    cluster, so same-vehicle pairs are far more similar than different-
    vehicle pairs.

    Two planted twin PAIRS sit almost on top of each other — the confusable
    tail. The proportions matter: 2 twin pairs out of 12 vehicles is ~3% of
    all cross-identity pairs, so uniform sampling mostly misses them (fair
    AUC stays high, correctly reporting a good embedder) while top-k mining
    fills almost its entire quota from them (adversarial AUC collapses).
    That asymmetry IS the bias this module exists to surface.
    """
    rng = np.random.default_rng(7)
    dim, n_vehicles, per_vehicle = 16, 12, 4
    centers = rng.normal(size=(n_vehicles, dim))
    for twin_of, twin in ((0, 1), (2, 3)):
        centers[twin] = centers[twin_of] + 0.01 * rng.normal(size=dim)

    records, rows = [], []
    for vid in range(n_vehicles):
        for k in range(per_vehicle):
            records.append(_Rec(vid, f"c{k % 2:03d}"))
            rows.append(centers[vid] + 0.02 * rng.normal(size=dim))
    return records, _normalize(np.asarray(rows))


def test_fair_protocol_sees_a_good_embedder_as_good():
    records, emb = _known_good_fixture()
    report, _ = evaluate_separability(records, emb, per_bucket=40)
    # Tight clusters + uniformly sampled negatives => near-perfect.
    assert report.fair_auc > 0.95


def test_adversarial_protocol_understates_the_same_embedder():
    """The regression this whole module exists for: scoring the SAME
    embeddings against top-k mined negatives drags the AUC down, because
    the planted twin dominates the negative set. Same data, same model,
    much worse number."""
    records, emb = _known_good_fixture()
    report, _ = evaluate_separability(records, emb, per_bucket=40)

    assert report.adversarial_auc < report.fair_auc
    assert report.difficulty_gap > 0.2
    # And the ceiling stays high, confirming nothing upstream is broken.
    assert report.same_camera_auc is not None and report.same_camera_auc > 0.9


def test_evaluate_requires_cross_camera_positives():
    records = [_Rec(1, "c001"), _Rec(1, "c001"), _Rec(2, "c001")]
    emb = _normalize(np.random.default_rng(3).normal(size=(3, 8)))
    with pytest.raises(ValueError, match="no cross-camera"):
        evaluate_separability(records, emb)
