"""Separate isotonic calibration for the render-and-compare signal.

Mirrors calibration/isotonic.py but keeps its own artifact and version label
(`rendercmp-<version>`). The render-match raw score has a different
distribution from ReID cosine similarity, so it MUST be calibrated on its own
render-vs-crop pairs and never share the ReID isotonic model. Fit on
cross-view pairs (query at view A vs the target model rendered toward A).

HONESTY: calibrated on synthetic/rendered pairs this measures the renderer +
simulator, not real-world identification. On cargen's stub prior the render is
a procedural sedan and the signal is meaningless — which is why verify_match
gates on model maturity and the RESULTS row is PENDING without a real backend.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HONESTY_NOTE = (
    "Render-and-compare calibration on synthetic/rendered pairs: measures the "
    "renderer + simulator, not real-world identification.")


@dataclass(frozen=True)
class RenderMatchModel:
    version: str
    x: tuple[float, ...]        # raw render-match score breakpoints
    y: tuple[float, ...]        # calibrated P(same)
    n_pairs: int
    note: str = HONESTY_NOTE

    def predict(self, raw: float) -> float:
        return float(np.interp(raw, self.x, self.y))


def _version_of(raws: np.ndarray, labels: np.ndarray) -> str:
    h = hashlib.sha1()
    for r, y in sorted(zip(raws.tolist(), labels.tolist())):
        h.update(f"{r:.6f}|{int(y)}".encode())
    return h.hexdigest()[:10]


def fit(raw_scores: list[float], same_vehicle: list[bool]) -> RenderMatchModel:
    if len(raw_scores) < 10:
        raise ValueError("not enough render-match pairs to calibrate")
    from sklearn.isotonic import IsotonicRegression

    raws = np.asarray(raw_scores, dtype=float)
    labels = np.asarray([1.0 if s else 0.0 for s in same_vehicle], dtype=float)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(raws, labels)
    grid = np.linspace(float(raws.min()), float(raws.max()), 101)
    return RenderMatchModel(
        version=_version_of(raws, labels),
        x=tuple(float(v) for v in grid),
        y=tuple(float(v) for v in iso.predict(grid)),
        n_pairs=len(raw_scores),
    )


def save(model: RenderMatchModel, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "version": model.version, "note": model.note,
        "x": list(model.x), "y": list(model.y), "n_pairs": model.n_pairs,
    }, indent=1))


def load_model(path: str | Path) -> RenderMatchModel:
    data = json.loads(Path(path).read_text())
    return RenderMatchModel(
        version=str(data["version"]), x=tuple(data["x"]), y=tuple(data["y"]),
        n_pairs=int(data["n_pairs"]), note=str(data.get("note", HONESTY_NOTE)))


def labeled_version(model: RenderMatchModel) -> str:
    return f"rendercmp-{model.version}"
