"""Isotonic calibration of ReID similarity -> P(same vehicle), versioned.

Isotonic regression fits a monotone step function from cosine similarity to
empirical same-vehicle probability. Monotonicity is the right constraint:
"more similar" should never mean "less likely the same" — but the *shape*
is learned from data, so a similarity of 0.95 maps to whatever fraction of
0.95-similar pairs were actually the same vehicle, look-alikes included.

Every fitted model carries a content-derived version string, and every
decision that uses it names that version in its fact list. Artifacts are
plain JSON (breakpoints + metadata), so serving needs numpy interp only.

HONESTY NOTE: calibrated on synthetic sprite pairs, this measures the
simulator's confusability, not real-world accuracy. The PR sweep below
answers "at what similarity does the SIMULATOR stop producing false
matches", nothing more.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from calibration.dataset import SimilarityPair

HONESTY_NOTE = (
    "Calibrated on synthetic sprite pairs: this measures the simulator's "
    "appearance distribution, not real-world ReID accuracy.")


@dataclass(frozen=True)
class CalibrationModel:
    version: str
    x: tuple[float, ...]        # similarity breakpoints (ascending)
    y: tuple[float, ...]        # calibrated P(same) at each breakpoint
    n_pairs: int
    n_hard_negatives: int
    note: str = HONESTY_NOTE

    def predict(self, similarity: float) -> float:
        return float(np.interp(similarity, self.x, self.y))


@dataclass(frozen=True)
class SweepPoint:
    threshold: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class CalibrationReport:
    model: CalibrationModel
    sweep: tuple[SweepPoint, ...]
    chosen_threshold: float
    target_precision: float
    hard_negative_fpr_at_threshold: float = field(default=0.0)


def _version_of(pairs: list[SimilarityPair]) -> str:
    # Versioned by pair content only (similarity, label, hardness), so any
    # pair-like object works — synthetic SimilarityPairs and real MinedPairs
    # alike — and the version changes iff the calibration data changes.
    h = hashlib.sha1()
    for p in sorted(pairs, key=lambda p: (p.similarity, p.same_vehicle, p.hard_negative)):
        h.update(f"{p.similarity:.6f}|{p.same_vehicle}|{p.hard_negative}".encode())
    return h.hexdigest()[:10]


def fit(pairs: list[SimilarityPair]) -> CalibrationModel:
    if len(pairs) < 10:
        raise ValueError("not enough pairs to calibrate")
    from sklearn.isotonic import IsotonicRegression

    sims = np.array([p.similarity for p in pairs])
    labels = np.array([1.0 if p.same_vehicle else 0.0 for p in pairs])
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(sims, labels)
    # Dense grid keeps the JSON artifact simple and the runtime sklearn-free.
    grid = np.linspace(float(sims.min()), float(sims.max()), 101)
    return CalibrationModel(
        version=_version_of(pairs),
        x=tuple(float(v) for v in grid),
        y=tuple(float(v) for v in iso.predict(grid)),
        n_pairs=len(pairs),
        n_hard_negatives=sum(p.hard_negative for p in pairs),
    )


def pr_sweep(pairs: list[SimilarityPair], thresholds: np.ndarray | None = None
             ) -> tuple[SweepPoint, ...]:
    """Precision/recall of 'similarity >= t means same vehicle' over t."""
    if thresholds is None:
        thresholds = np.linspace(0.5, 1.0, 51)
    points = []
    for t in thresholds:
        tp = sum(1 for p in pairs if p.same_vehicle and p.similarity >= t)
        fp = sum(1 for p in pairs if not p.same_vehicle and p.similarity >= t)
        fn = sum(1 for p in pairs if p.same_vehicle and p.similarity < t)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        points.append(SweepPoint(float(t), precision, recall, f1))
    return tuple(points)


def choose_threshold(sweep: tuple[SweepPoint, ...], target_precision: float = 0.95
                     ) -> float:
    """Lowest threshold meeting target precision (max recall subject to it).

    On sprite data with true look-alikes, no threshold may reach the target
    — hard negatives are near-duplicates by construction. Falling back to
    the best-F1 point is deliberate: it demonstrates that a similarity
    threshold cannot separate look-alikes, which is the project's thesis.
    """
    meeting = [p for p in sweep if p.precision >= target_precision]
    if meeting:
        return min(meeting, key=lambda p: p.threshold).threshold
    return max(sweep, key=lambda p: p.f1).threshold


def build_report(pairs: list[SimilarityPair], target_precision: float = 0.95
                 ) -> CalibrationReport:
    model = fit(pairs)
    sweep = pr_sweep(pairs)
    threshold = choose_threshold(sweep, target_precision)
    hard = [p for p in pairs if p.hard_negative]
    hard_fpr = (sum(1 for p in hard if p.similarity >= threshold) / len(hard)
                if hard else 0.0)
    return CalibrationReport(
        model=model, sweep=sweep, chosen_threshold=threshold,
        target_precision=target_precision,
        hard_negative_fpr_at_threshold=hard_fpr)


# ------------------------------------------------------------- persistence

def save(report: CalibrationReport, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": report.model.version,
        "note": report.model.note,
        "x": list(report.model.x),
        "y": list(report.model.y),
        "n_pairs": report.model.n_pairs,
        "n_hard_negatives": report.model.n_hard_negatives,
        "chosen_threshold": report.chosen_threshold,
        "target_precision": report.target_precision,
        "hard_negative_fpr_at_threshold": report.hard_negative_fpr_at_threshold,
        "sweep": [[s.threshold, s.precision, s.recall, s.f1] for s in report.sweep],
    }
    p.write_text(json.dumps(payload, indent=1))


def load_model(path: str | Path) -> CalibrationModel:
    data = json.loads(Path(path).read_text())
    return CalibrationModel(
        version=str(data["version"]),
        x=tuple(data["x"]), y=tuple(data["y"]),
        n_pairs=int(data["n_pairs"]),
        n_hard_negatives=int(data["n_hard_negatives"]),
        note=str(data.get("note", HONESTY_NOTE)))


def make_reid_prob_fn(model: CalibrationModel) -> tuple[Callable[[float], float], str]:
    """(prob_fn, label) ready for reasoning.cascade.CascadeConfig."""
    return model.predict, f"isotonic-{model.version}"
