"""Calibration tests with a cheap fake embedder (structure-preserving:
same vehicle -> near-identical vectors, look-alikes -> close, others -> far).
The real-OSNet path is exercised by calibration/run.py, not unit tests."""
import numpy as np
import pytest

from calibration.dataset import PairConfig, build_pairs
from calibration.isotonic import (
    build_report, choose_threshold, fit, load_model, make_reid_prob_fn,
    pr_sweep, save,
)
from sim.emitter import build_default_world


@pytest.fixture(scope="module")
def world():
    return build_default_world()


@pytest.fixture(scope="module")
def pairs(world):
    by_pixels: dict[bytes, np.ndarray] = {}
    group_axis: dict[str, int] = {}

    def fake_embed(crop):
        # Deterministic per crop; crops of one vehicle cluster because the
        # sprite pixels barely differ -> use coarse pixel stats as the vector.
        small = crop[::16, ::16].astype(np.float32).mean(axis=2).ravel()[:24]
        v = small + 1.0
        return (v / np.linalg.norm(v)).astype(np.float32)

    _ = group_axis, by_pixels
    return build_pairs(world, fake_embed, PairConfig(crops_per_vehicle=2,
                                                     negatives_per_vehicle=2))


def test_pairs_have_structure(pairs):
    positives = [p for p in pairs if p.same_vehicle]
    hard = [p for p in pairs if p.hard_negative]
    easy = [p for p in pairs if not p.same_vehicle and not p.hard_negative]
    assert positives and hard and easy
    mean = lambda ps: float(np.mean([p.similarity for p in ps]))
    assert mean(positives) > mean(easy)
    assert mean(hard) > mean(easy), "lookalike negatives must be the hard ones"


def test_isotonic_fit_monotone_and_bounded(pairs):
    model = fit(pairs)
    probs = [model.predict(s) for s in np.linspace(0, 1, 50)]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert all(b >= a - 1e-9 for a, b in zip(probs, probs[1:])), "must be monotone"
    assert len(model.version) == 10


def test_version_tracks_content(pairs):
    assert fit(pairs).version == fit(pairs).version
    perturbed = list(pairs[1:])
    assert fit(perturbed).version != fit(pairs).version


def test_sweep_recall_decreases_with_threshold(pairs):
    sweep = pr_sweep(pairs)
    recalls = [p.recall for p in sweep]
    assert all(b <= a + 1e-9 for a, b in zip(recalls, recalls[1:]))


def test_chosen_threshold_meets_target_or_best_f1(pairs):
    sweep = pr_sweep(pairs)
    t = choose_threshold(sweep, target_precision=0.9)
    point = min((p for p in sweep if p.threshold >= t), key=lambda p: p.threshold)
    best_f1 = max(sweep, key=lambda p: p.f1)
    assert point.precision >= 0.9 or t == best_f1.threshold


def test_report_save_load_roundtrip(pairs, tmp_path):
    report = build_report(pairs)
    path = tmp_path / "cal.json"
    save(report, path)
    loaded = load_model(path)
    assert loaded.version == report.model.version
    assert loaded.predict(0.9) == pytest.approx(report.model.predict(0.9))
    assert "simulator" in loaded.note

    prob_fn, label = make_reid_prob_fn(loaded)
    assert label == f"isotonic-{loaded.version}"
    assert prob_fn(0.9) == pytest.approx(loaded.predict(0.9))


def test_fit_rejects_tiny_datasets(pairs):
    with pytest.raises(ValueError):
        fit(pairs[:3])
