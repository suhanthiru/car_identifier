"""Reliability of the calibrated P(same): binned accuracy + ECE.

A calibrated model's predicted probability should equal the empirical
frequency: among all pairs it scored ~0.7, about 70% should truly be the
same vehicle. The reliability diagram plots exactly that; ECE (expected
calibration error) is the bin-weighted gap. Computation is separated from
plotting so the numbers are unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibration.isotonic import CalibrationModel
from eval.hard_negatives import MinedPair


@dataclass(frozen=True)
class ReliabilityBin:
    lo: float
    hi: float
    mean_predicted: float
    empirical_accuracy: float
    count: int


@dataclass(frozen=True)
class ReliabilityReport:
    bins: tuple[ReliabilityBin, ...]
    ece: float
    n_pairs: int


def compute_reliability(
    pairs: list[MinedPair], model: CalibrationModel, n_bins: int = 10
) -> ReliabilityReport:
    if not pairs:
        raise ValueError("no pairs to evaluate")
    preds = np.array([model.predict(p.similarity) for p in pairs])
    labels = np.array([p.same_vehicle for p in pairs], dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    ece = 0.0
    for lo, hi in zip(edges, edges[1:]):
        mask = (preds >= lo) & (preds < hi if hi < 1.0 else preds <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(preds[mask].mean())
        acc = float(labels[mask].mean())
        ece += (count / len(pairs)) * abs(mean_pred - acc)
        bins.append(ReliabilityBin(float(lo), float(hi), mean_pred, acc, count))
    return ReliabilityReport(bins=tuple(bins), ece=float(ece), n_pairs=len(pairs))
