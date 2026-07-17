"""THE MONEY ABLATION: raw ReID alerting vs cascade + vetoes.

Task framing: each query sighting proposes an alert against its best gallery
match. A policy decides alert / refuse. Ground truth says whether the top
match is actually the same vehicle. We compare:

  raw      — alert whenever top-1 similarity >= threshold. What a
             single-score system does.
  cascade  — same threshold, plus the symbolic layer:
               * hard-attribute veto: top-1 whose labeled color or body
                 type contradicts the query's cannot alert;
               * ambiguity refusal: if a *different* vehicle's gallery
                 entry scores within `margin` of the top-1, the system
                 refuses to pick between look-alikes and routes to review
                 instead of alerting.

Metrics per policy: precision, recall, F1 over alerts, plus the review rate
(cascade only — refusals are not free, and we report their cost honestly).
An optional `extra_attrs` hook merges additional per-image attributes (the
car3d 3D-geometry channel) into the contradiction check, giving the
with/without-3D ablation row on the same code path.

Honest note: the attribute channel here uses the dataset's own labels, i.e.
a perfect attribute classifier. The measured delta is therefore an UPPER
BOUND on what a real attribute head would buy; the RESULTS.md table says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from datasets.veri776 import VeriImage

# ("query"|"gallery", image index) -> extra attributes to merge (3D geometry).
AttrFn = Callable[[str, int], Mapping[str, str]]


@dataclass(frozen=True)
class PolicyMetrics:
    name: str
    threshold: float
    alerts: int
    true_positives: int
    false_positives: int
    missed: int             # queries with a true match available but no alert
    reviews: int            # refusals routed to a human (cascade only)

    @property
    def precision(self) -> float:
        return self.true_positives / self.alerts if self.alerts else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.missed
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def row(self) -> dict:
        return {
            "policy": self.name, "threshold": round(self.threshold, 3),
            "precision": round(self.precision, 4), "recall": round(self.recall, 4),
            "f1": round(self.f1, 4), "alerts": self.alerts,
            "false_positives": self.false_positives, "review_rate": self.reviews,
        }


@dataclass(frozen=True)
class AblationCase:
    """One query's outcome under a policy — kept for the failure gallery."""

    query_index: int
    top_index: int
    similarity: float
    correct: bool
    action: str             # "alert" | "review" | "none"
    reason: str             # plain English, shown in the failure gallery


def _attrs_contradict(a: Mapping[str, str], b: Mapping[str, str]) -> str:
    for key in sorted(set(a) & set(b)):
        if a[key] and b[key] and a[key] != b[key]:
            return f"{key} contradiction: query {a[key]} vs match {b[key]}"
    return ""


def run_ablation(
    query_images: Sequence[VeriImage],
    query_emb: np.ndarray,
    gallery_images: Sequence[VeriImage],
    gallery_emb: np.ndarray,
    threshold: float,
    margin: float = 0.05,
    extra_attrs: AttrFn | None = None,
) -> tuple[list[PolicyMetrics], list[AblationCase]]:
    """Score both policies on identical rankings; return metrics + cases."""
    sims = query_emb @ gallery_emb.T
    g_ids = np.asarray([im.vehicle_id for im in gallery_images])
    g_cams = np.asarray([im.camera_id for im in gallery_images])

    def attrs_of(images, i, kind: str) -> dict[str, str]:
        im = images[i]
        base = {"color": im.color, "body_type": im.body_type}
        if extra_attrs is not None:
            base.update(extra_attrs(kind, i))
        return base

    counters = {name: dict(alerts=0, tp=0, fp=0, missed=0, reviews=0)
                for name in ("raw", "cascade")}
    cases: list[AblationCase] = []

    for qi, q in enumerate(query_images):
        valid = ~((g_ids == q.vehicle_id) & (g_cams == q.camera_id))
        if not valid.any():
            continue
        row = np.where(valid, sims[qi], -np.inf)
        top = int(np.argmax(row))
        top_sim = float(row[top])
        correct = g_ids[top] == q.vehicle_id
        has_true_match = bool((g_ids[valid] == q.vehicle_id).any())

        # raw policy
        c = counters["raw"]
        if top_sim >= threshold:
            c["alerts"] += 1
            c["tp" if correct else "fp"] += int(1)
        elif has_true_match:
            c["missed"] += 1

        # cascade policy
        c = counters["cascade"]
        action, reason = "none", f"similarity {top_sim:.2f} below threshold"
        if top_sim >= threshold:
            contradiction = _attrs_contradict(
                attrs_of(query_images, qi, "query"),
                attrs_of(gallery_images, top, "gallery"))
            rival_rows = np.where(valid & (g_ids != g_ids[top]), sims[qi], -np.inf)
            rival_sim = float(np.max(rival_rows))
            if contradiction:
                action, reason = "none", f"hard-attribute veto: {contradiction}"
            elif top_sim - rival_sim < margin:
                action = "review"
                reason = (f"ambiguous look-alikes: best {top_sim:.2f} vs rival "
                          f"{rival_sim:.2f} (margin < {margin}); refused to pick")
            else:
                action, reason = "alert", f"similarity {top_sim:.2f} clear of rivals"
        if action == "alert":
            c["alerts"] += 1
            c["tp" if correct else "fp"] += int(1)
        else:
            if action == "review":
                c["reviews"] += 1
            if has_true_match:
                c["missed"] += 1
        cases.append(AblationCase(
            query_index=qi, top_index=top, similarity=top_sim,
            correct=bool(correct), action=action, reason=reason))

    metrics = [
        PolicyMetrics(name=name, threshold=threshold, alerts=c["alerts"],
                      true_positives=c["tp"], false_positives=c["fp"],
                      missed=c["missed"], reviews=c["reviews"])
        for name, c in counters.items()
    ]
    return metrics, cases
