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


def test_threshold_above_all_similarities_is_never_chosen(pairs):
    """A threshold that accepts nothing scores 0/0 precision, reported as 1.0.

    Regression: that vacuous point used to win — on VeRi-776 it selected
    0.960 when the highest observed similarity was 0.957, so the alert policy
    provably never alerted, and RESULTS.md published it as 100% precision on
    both arms of the ablation. A policy that never fires is not a policy that
    never errs.
    """
    sweep = pr_sweep(pairs)
    highest = max(p.similarity for p in pairs)
    t = choose_threshold(sweep, target_precision=0.95)
    assert t <= highest, f"chose {t}, above every observed similarity {highest}"
    chosen = min((p for p in sweep if abs(p.threshold - t) < 1e-9),
                 key=lambda p: p.threshold)
    assert not chosen.degenerate
    assert chosen.n_predicted > 0


def test_sweep_marks_degenerate_points(pairs):
    sweep = pr_sweep(pairs, thresholds=np.array([0.0, 1.5]))
    accepts_all, accepts_none = sweep
    assert accepts_all.n_predicted == len(pairs)
    assert not accepts_all.degenerate
    assert accepts_none.n_predicted == 0
    assert accepts_none.degenerate
    # The 1.0 is the 0/0 placeholder the bug relied on; keep it visible so
    # nothing starts treating it as a measurement again.
    assert accepts_none.precision == 1.0


def test_sweep_with_no_live_points_refuses_to_choose():
    """Better to raise than to invent an operating point from an empty sweep."""
    empty = pr_sweep([], thresholds=np.array([0.5, 0.9]))
    with pytest.raises(ValueError, match="zero pairs"):
        choose_threshold(empty)


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


def test_split_by_identity_shares_no_vehicle_between_halves():
    """The fit half and the scored half must not share a vehicle.

    A pair-level split is not enough. Two crops of one car carry the same
    paint, the same camera white balance and the same plate, so a fit that saw
    one of them has effectively seen the identity it is later scored on, and
    the "held-out" number stays optimistic. Splitting on the vehicle is the
    honest version.
    """
    from calibration.isotonic import split_by_identity

    class P:
        def __init__(self, a, b):
            self.a, self.b = a, b

    pairs = [P(a, b) for a in range(12) for b in range(12) if a <= b]
    fit_pairs, eval_pairs = split_by_identity(
        pairs, lambda p: (p.a, p.b), holdout_frac=0.3, seed=3)
    assert fit_pairs and eval_pairs, "split produced an empty half"
    fit_ids = {v for p in fit_pairs for v in (p.a, p.b)}
    eval_ids = {v for p in eval_pairs for v in (p.a, p.b)}
    assert not (fit_ids & eval_ids), (
        f"vehicles in both halves: {sorted(fit_ids & eval_ids)}")


def test_report_records_whether_its_numbers_are_held_out(pairs):
    """A calibration artifact must say whether it was scored out of sample.

    The threshold, the PR sweep and the hard-negative FPR used to be computed
    on the very pairs the isotonic regression was fitted to. Isotonic
    regression minimises error against those labels by construction, so an
    in-sample ECE near zero is a property of the method rather than evidence
    the mapping generalises. The flag exists so that can never again be
    reported as a generalisation result by accident.
    """
    from calibration.isotonic import build_report

    in_sample = build_report(pairs)
    assert in_sample.held_out is False

    cut = len(pairs) // 2
    split = build_report(pairs[:cut], eval_pairs=pairs[cut:])
    assert split.held_out is True
    assert split.n_fit_pairs == cut
    assert split.n_eval_pairs == len(pairs) - cut
