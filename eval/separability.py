"""Separability: can the embedder tell these vehicles apart at all?

Distinct from `eval/hard_negatives.py`, and the distinction is the whole
point of this module. That module mines the **top-k most similar**
cross-identity pairs per (color, body) bucket. That is exactly right for
*fitting a calibration curve* — the isotonic map has to see the hard tail
or it will be overconfident — but it is the wrong sample for *judging an
embedder*, because it selects worst-case negatives while positives are
drawn uniformly. Scoring one against the other reliably produces an AUC
below 0.5 no matter how good the embedder is, which then reads as "the
embedder ranks different cars above same cars" when the data only
supports "the single most confusable look-alike pair beats the average
same-car pair".

So this module reports BOTH, side by side, plus a ceiling:

- **fair**        — positives restricted to CROSS-camera pairs (the actual
                    operational task: re-identify across viewpoints), negatives
                    sampled UNIFORMLY within the same appearance bucket.
- **adversarial** — the top-k mining above, relabelled honestly.
- **same-camera** — positives from the SAME camera. Near-duplicate frames, so
                    this is a sanity ceiling: if it is not high, something is
                    broken upstream (crops, preprocessing, the model load).

The gap between fair and adversarial is the look-alike difficulty measure
— the quantity the "identical cars" question is actually about.

Records are duck-typed: any object with `vehicle_id`, `camera_id`,
`color` and `body_type` works (`datasets.veri776.VeriImage` and
`scripts.calibrate_cityflow._CropRecord` both already do, which is the
same convention `mine_pairs` relies on).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval.hard_negatives import MinedPair


@dataclass(frozen=True)
class SeparabilityReport:
    fair_auc: float
    adversarial_auc: float
    same_camera_auc: float | None      # None when no same-camera positives exist
    n_cross_camera_positives: int
    n_same_camera_positives: int
    n_uniform_negatives: int
    n_topk_negatives: int

    @property
    def difficulty_gap(self) -> float:
        """How much harder the worst-case look-alikes are than average
        negatives. Large gap = the embedder works but a confusable tail
        exists; near-zero gap = nothing special about the tail."""
        return self.fair_auc - self.adversarial_auc

    def summary(self) -> dict:
        return {
            "fair_auc": round(self.fair_auc, 4),
            "adversarial_auc": round(self.adversarial_auc, 4),
            "same_camera_auc": (round(self.same_camera_auc, 4)
                                if self.same_camera_auc is not None else None),
            "difficulty_gap": round(self.difficulty_gap, 4),
            "n_cross_camera_positives": self.n_cross_camera_positives,
            "n_same_camera_positives": self.n_same_camera_positives,
            "n_uniform_negatives": self.n_uniform_negatives,
            "n_topk_negatives": self.n_topk_negatives,
        }


def separability_auc(positive_sims, negative_sims) -> float:
    """P(a random positive scores above a random negative), via the
    Mann-Whitney U statistic (equivalent to ROC-AUC, threshold-free).

    0.5 = no separation, 1.0 = perfect, <0.5 = inverted on THIS sample.
    Ties get average ranks, so an all-identical input returns exactly 0.5
    rather than an artefact of sort order.
    """
    pos = np.asarray(positive_sims, dtype=float)
    neg = np.asarray(negative_sims, dtype=float)
    if pos.size == 0 or neg.size == 0:
        raise ValueError(
            f"need at least one positive and one negative "
            f"(got {pos.size} / {neg.size})")

    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="mergesort")
    ordered = combined[order]
    ranks = np.empty(combined.size, dtype=float)
    i = 0
    while i < ordered.size:
        j = i
        while j + 1 < ordered.size and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # 1-based average rank
        i = j + 1

    u = ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def _bucket_of(record) -> str:
    return f"{record.color}/{record.body_type}"


def _pair(embeddings, i: int, j: int, *, same_vehicle: bool,
          hard_negative: bool, bucket: str) -> MinedPair:
    return MinedPair(
        similarity=float(embeddings[i] @ embeddings[j]),
        same_vehicle=same_vehicle, hard_negative=hard_negative,
        a_index=int(i), b_index=int(j), bucket=bucket)


def positive_pairs(
    records, embeddings: np.ndarray, *, cross_camera: bool,
    per_id: int = 6, seed: int = 11,
) -> list[MinedPair]:
    """Same-vehicle pairs, split by whether the two crops come from
    DIFFERENT cameras (the real task) or the SAME camera (near-duplicate
    frames — a sanity ceiling, not a re-identification)."""
    rng = np.random.default_rng(seed)
    ids = np.asarray([r.vehicle_id for r in records])
    cams = np.asarray([r.camera_id for r in records])

    out: list[MinedPair] = []
    for vid in np.unique(ids):
        idx = np.nonzero(ids == vid)[0]
        if idx.size < 2:
            continue
        candidates = [
            (int(a), int(b))
            for pos_a, a in enumerate(idx)
            for b in idx[pos_a + 1:]
            if (cams[a] != cams[b]) == cross_camera
        ]
        if not candidates:
            continue
        if len(candidates) > per_id:
            picked = rng.choice(len(candidates), size=per_id, replace=False)
            candidates = [candidates[int(p)] for p in picked]
        for a, b in candidates:
            out.append(_pair(embeddings, a, b, same_vehicle=True,
                             hard_negative=False, bucket=_bucket_of(records[a])))
    return out


def uniform_bucket_negatives(
    records, embeddings: np.ndarray, per_bucket: int = 40, seed: int = 13,
) -> list[MinedPair]:
    """Different-vehicle pairs sampled UNIFORMLY inside each appearance
    bucket — representative look-alikes, not the worst case. This is the
    negative set a fair separability judgement needs."""
    rng = np.random.default_rng(seed)
    ids = np.asarray([r.vehicle_id for r in records])
    buckets: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        if r.color and r.body_type:
            buckets.setdefault(_bucket_of(r), []).append(i)

    out: list[MinedPair] = []
    for bucket, members in sorted(buckets.items()):
        idx = np.asarray(members)
        if idx.size < 2:
            continue
        cand_a, cand_b = np.triu_indices(idx.size, k=1)
        cross = ids[idx[cand_a]] != ids[idx[cand_b]]
        cand_a, cand_b = cand_a[cross], cand_b[cross]
        if cand_a.size == 0:
            continue
        take = min(per_bucket, cand_a.size)
        picked = rng.choice(cand_a.size, size=take, replace=False)
        for p in picked:
            out.append(_pair(embeddings, idx[cand_a[p]], idx[cand_b[p]],
                             same_vehicle=False, hard_negative=False,
                             bucket=bucket))
    return out


def topk_bucket_negatives(
    records, embeddings: np.ndarray, per_bucket: int = 40,
) -> list[MinedPair]:
    """The adversarial set: the highest-similarity different-vehicle pairs
    per bucket. Mirrors `eval/hard_negatives.py:_hard_negative_pairs` (kept
    here so a separability run is self-contained and its selection bias is
    visible at the call site rather than buried in a calibration helper)."""
    ids = np.asarray([r.vehicle_id for r in records])
    buckets: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        if r.color and r.body_type:
            buckets.setdefault(_bucket_of(r), []).append(i)

    out: list[MinedPair] = []
    for bucket, members in sorted(buckets.items()):
        idx = np.asarray(members)
        if idx.size < 2:
            continue
        sims = embeddings[idx] @ embeddings[idx].T
        cross = ids[idx][:, None] != ids[idx][None, :]
        upper = np.triu(np.ones_like(sims, dtype=bool), k=1)
        cand_a, cand_b = np.nonzero(cross & upper)
        if cand_a.size == 0:
            continue
        order = np.argsort(-sims[cand_a, cand_b])[:per_bucket]
        for k in order:
            out.append(_pair(embeddings, idx[cand_a[k]], idx[cand_b[k]],
                             same_vehicle=False, hard_negative=True,
                             bucket=bucket))
    return out


def natural_population_pairs(
    records, embeddings: np.ndarray, per_id: int = 6,
    negatives_per_positive: int = 3, seed: int = 17,
) -> list[MinedPair]:
    """Pairs at the base rate a deployed target actually faces.

    The calibration sample from `eval/hard_negatives.py:mine_pairs` takes the
    top-k MOST SIMILAR cross-identity pairs per bucket. That is the right
    sample for measuring the confusable tail, and the wrong one for fitting
    P(same): it over-represents look-alikes so heavily that within it, higher
    similarity genuinely predicts "different vehicle". Isotonic regression can
    only emit a non-decreasing curve, so fed that sample it collapses to a
    CONSTANT — which is exactly what the serving curves did (flat 0.499 across
    the whole operating range), making the appearance signal contribute the
    same amount to every candidate and discriminate nothing.

    Here negatives are drawn uniformly from every OTHER vehicle, so
    look-alikes appear at their true frequency rather than a mined one.
    Confusable pairs are still present — they are simply no longer the
    majority of the sample. Deleting them outright would swing the curve to
    the opposite error (over-confident on precisely the pairs that matter).
    """
    rng = np.random.default_rng(seed)
    ids = np.asarray([r.vehicle_id for r in records])
    pairs = positive_pairs(records, embeddings, cross_camera=True,
                           per_id=per_id, seed=seed)

    n = len(records)
    for _ in range(len(pairs) * negatives_per_positive):
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if ids[i] == ids[j]:
            continue
        pairs.append(_pair(embeddings, i, j, same_vehicle=False,
                           hard_negative=False, bucket=_bucket_of(records[i])))
    return pairs


def evaluate_separability(
    records, embeddings: np.ndarray, per_id: int = 6, per_bucket: int = 40,
    seed: int = 11,
) -> tuple[SeparabilityReport, dict[str, list[MinedPair]]]:
    """Fair vs adversarial separability, plus the same-camera ceiling.

    Returns the report and the underlying pair sets (so callers can plot
    the distributions without re-mining).
    """
    cross_pos = positive_pairs(records, embeddings, cross_camera=True,
                               per_id=per_id, seed=seed)
    same_pos = positive_pairs(records, embeddings, cross_camera=False,
                              per_id=per_id, seed=seed)
    uniform_neg = uniform_bucket_negatives(records, embeddings, per_bucket, seed + 2)
    topk_neg = topk_bucket_negatives(records, embeddings, per_bucket)

    if not cross_pos:
        raise ValueError(
            "no cross-camera same-vehicle pairs found — separability cannot be "
            "judged on this set (every vehicle appears on only one camera)")

    cross_sims = [p.similarity for p in cross_pos]
    uniform_sims = [p.similarity for p in uniform_neg]
    topk_sims = [p.similarity for p in topk_neg]

    return SeparabilityReport(
        fair_auc=separability_auc(cross_sims, uniform_sims),
        adversarial_auc=separability_auc(cross_sims, topk_sims),
        same_camera_auc=(separability_auc([p.similarity for p in same_pos],
                                          uniform_sims) if same_pos else None),
        n_cross_camera_positives=len(cross_pos),
        n_same_camera_positives=len(same_pos),
        n_uniform_negatives=len(uniform_neg),
        n_topk_negatives=len(topk_neg),
    ), {
        "cross_camera_positives": cross_pos,
        "same_camera_positives": same_pos,
        "uniform_negatives": uniform_neg,
        "topk_negatives": topk_neg,
    }
