"""Hard-negative mining, reliability, ablation, and cross-camera validation
tests on contrived data with known structure."""
from pathlib import Path

import numpy as np
import pytest

from calibration.isotonic import fit
from datasets.cityflow import CityFlowScenario, TrackSpan
from datasets.veri776 import VeriImage
from eval.ablation import run_ablation
from eval.cross_camera import (
    compare_fusion_on_real_transitions, fit_windows, validate_transit_veto,
)
from eval.hard_negatives import MinedPair, hardest_pairs, mine_pairs
from eval.reliability import compute_reliability
from reasoning.corroboration import UPDATE_THRESHOLD


def img(vid, cam, color="red", body="sedan"):
    return VeriImage(path=Path(f"{vid}_{cam}.jpg"), vehicle_id=vid,
                     camera_id=cam, timestamp=0, color=color, body_type=body)


def embeddings_for(structure):
    """structure: list of base vectors; jitter per image, normalized."""
    rng = np.random.default_rng(0)
    out = []
    for base in structure:
        v = np.asarray(base, dtype=np.float32) + rng.normal(0, 0.02, len(base))
        out.append(v / np.linalg.norm(v))
    return np.stack(out).astype(np.float32)


# ------------------------------------------------------------------- mining

def test_mining_buckets_and_labels():
    images = [
        img("1", "c1"), img("1", "c2"),               # red sedan, id 1
        img("2", "c1"), img("2", "c3"),               # red sedan, id 2 (lookalike)
        img("3", "c1", color="blue", body="bus"),     # easy negative
    ]
    base = [[1, 0, 0], [1, 0.1, 0], [0.95, 0.2, 0], [0.9, 0.25, 0], [0, 0, 1]]
    emb = embeddings_for(base)
    pairs = mine_pairs(images, emb, hard_per_bucket=10, random_negatives=20)
    hard = [p for p in pairs if p.hard_negative]
    assert hard, "same-color same-body cross-id pairs must be mined"
    assert all(p.bucket == "red/sedan" for p in hard)
    assert all(not p.same_vehicle for p in hard)
    positives = [p for p in pairs if p.same_vehicle]
    assert positives
    top = hardest_pairs(pairs, top=3)
    assert all(a.similarity >= b.similarity for a, b in zip(top, top[1:]))


# -------------------------------------------------------------- reliability

def test_reliability_perfectly_calibrated_synthetic():
    rng = np.random.default_rng(1)
    pairs = []
    # Construct pairs where P(same | sim) is exactly sim (clipped): the
    # isotonic fit should then be near-identity and ECE small.
    for _ in range(4000):
        sim = float(rng.uniform(0, 1))
        pairs.append(MinedPair(similarity=sim, same_vehicle=bool(rng.random() < sim),
                               hard_negative=False, a_index=0, b_index=1, bucket=""))
    model = fit(pairs)
    report = compute_reliability(pairs, model, n_bins=10)
    assert report.ece < 0.06
    assert sum(b.count for b in report.bins) == 4000


def test_reliability_requires_pairs():
    model = fit([MinedPair(0.1 * i, i % 2 == 0, False, 0, 1, "") for i in range(11)])
    with pytest.raises(ValueError):
        compute_reliability([], model)


# ----------------------------------------------------------------- ablation

def test_ablation_cascade_beats_raw_on_lookalikes():
    """Contrived world where raw thresholding false-alerts on a look-alike:
    the cascade must convert those FPs into reviews/vetoes."""
    gallery = [img("1", "c9"), img("2", "c9"), img("3", "c9", color="blue")]
    queries = [img("1", "c1"), img("2", "c1"),
               img("4", "c1"),                       # unflagged lookalike of 1/2
               img("5", "c1", color="blue")]         # blue car resembling red "1"
    g_emb = embeddings_for([[1, 0, 0], [0.90, 0.44, 0], [0, 1, 0]])
    q_emb = embeddings_for([[1, 0.03, 0], [0.90, 0.43, 0],
                            [0.985, 0.17, 0],         # between ids 1 and 2
                            [1, 0.05, 0]])            # blue query, red top match

    metrics, cases = run_ablation(queries, q_emb, gallery, g_emb,
                                  threshold=0.6, margin=0.03)
    raw = next(m for m in metrics if m.name == "raw")
    cascade = next(m for m in metrics if m.name == "cascade")
    assert raw.false_positives >= 1, "raw policy must false-alert here"
    assert cascade.false_positives < raw.false_positives
    assert cascade.precision > raw.precision
    assert cascade.reviews >= 1, "ambiguous lookalike goes to review"
    reasons = " | ".join(c.reason for c in cases)
    assert "refused to pick" in reasons


def test_ablation_attr_veto_fires():
    gallery = [img("1", "c9", color="red")]
    queries = [img("9", "c1", color="blue")]  # high sim but wrong color
    emb = embeddings_for([[1, 0]])
    metrics, cases = run_ablation(queries, emb, gallery, emb.copy(),
                                  threshold=0.5)
    cascade = next(m for m in metrics if m.name == "cascade")
    assert cascade.alerts == 0
    assert "contradiction" in cases[0].reason


def test_ablation_extra_attrs_hook():
    gallery = [img("1", "c9")]
    queries = [img("2", "c1")]
    emb = embeddings_for([[1, 0]])
    calls = []

    def geom(kind, i):
        calls.append((kind, i))
        return {"geom_bucket": "long" if kind == "gallery" else "short"}

    metrics, cases = run_ablation(queries, emb, gallery, emb.copy(),
                                  threshold=0.5, extra_attrs=geom)
    cascade = next(m for m in metrics if m.name == "cascade")
    assert cascade.alerts == 0, "3D geometry contradiction must veto"
    assert ("query", 0) in calls and ("gallery", 0) in calls


# ------------------------------------------------------------- cross-camera

def scenario():
    spans = []
    # 12 vehicles crossing c1 -> c2 with elapsed 20..31s, and some noise.
    for vid in range(12):
        spans.append(TrackSpan("S01", "c001", vid, 0.0, 10.0 + vid % 3))
        spans.append(TrackSpan("S01", "c002", vid, 30.0 + vid, 40.0 + vid))
    return CityFlowScenario(name="S01", cameras=("c001", "c002"),
                            homographies={}, spans=tuple(spans))


def test_transit_veto_validation():
    val = validate_transit_veto(scenario(), train_frac=0.5, speedup=5.0)
    assert val.n_real > 0
    assert val.false_veto_rate <= 0.2, "real transitions must rarely be vetoed"
    assert val.catch_rate == 1.0, "5x-faster-than-any-observed must always veto"


def test_fit_windows_ignores_overlap_negative_gaps():
    from datasets.cityflow import Transition
    w = fit_windows([Transition("S", 1, "a", "b", -2.0),
                     Transition("S", 2, "a", "b", 10.0)])
    assert w.windows[("a", "b")] == (10.0, 10.0)
    assert not w.is_impossible("x", "y", 0.1), "unknown pairs never veto"


def test_fusion_comparison_noisy_or_overshoots_capped_does_not():
    cmp = compare_fusion_on_real_transitions(scenario())
    assert cmp.n_vehicles == 12
    assert cmp.noisy_or_overshoot_rate > 0.9
    assert cmp.capped_overshoot_rate == 0.0
    assert max(cmp.capped_belief) < UPDATE_THRESHOLD
    assert cmp.noisy_or_belief[-1] > 0.99
