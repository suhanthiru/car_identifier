"""Hard-negative mining on real data: same-color, same-body confusables.

Random negative pairs are easy — a red sedan vs a white bus teaches nothing.
The negatives that actually break a similarity threshold are different
vehicles sharing color AND body type. We bucket gallery images by
(color, body_type) from the dataset's own labels, and mine the highest-
similarity cross-identity pairs inside each bucket. Those pairs:

- are the calibration set's negatives (so the isotonic map sees the truth
  about how similar different-vehicle pairs really get);
- populate the confusable-pair gallery in RESULTS.md;
- drive the look-alike false-positive measurements in the ablation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datasets.veri776 import VeriImage


@dataclass(frozen=True)
class MinedPair:
    similarity: float
    same_vehicle: bool
    hard_negative: bool
    a_index: int            # indices into the image/embedding arrays
    b_index: int
    bucket: str             # "red/sedan" etc.; "" for random fillers


def mine_pairs(
    images: list[VeriImage],
    embeddings: np.ndarray,
    hard_per_bucket: int = 40,
    positives_per_id: int = 4,
    random_negatives: int = 400,
    seed: int = 7,
) -> list[MinedPair]:
    """Positives + bucketed hard negatives + a few random fillers."""
    rng = np.random.default_rng(seed)
    ids = np.asarray([im.vehicle_id for im in images])
    pairs: list[MinedPair] = []

    pairs.extend(_positive_pairs(images, embeddings, ids, positives_per_id, rng))
    pairs.extend(_hard_negative_pairs(images, embeddings, ids, hard_per_bucket))

    # Random cross-bucket fillers keep the calibration's low-similarity tail
    # populated; without them the isotonic fit has no anchor near zero.
    n = len(images)
    for _ in range(random_negatives):
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if ids[i] == ids[j]:
            continue
        pairs.append(MinedPair(
            similarity=float(embeddings[i] @ embeddings[j]),
            same_vehicle=False, hard_negative=False,
            a_index=i, b_index=j, bucket=""))
    return pairs


def _positive_pairs(images, embeddings, ids, per_id, rng) -> list[MinedPair]:
    out: list[MinedPair] = []
    for vid in np.unique(ids):
        idx = np.nonzero(ids == vid)[0]
        if len(idx) < 2:
            continue
        count = min(per_id, len(idx) * (len(idx) - 1) // 2)
        seen = set()
        while len(seen) < count:
            i, j = rng.choice(idx, size=2, replace=False)
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            out.append(MinedPair(
                similarity=float(embeddings[i] @ embeddings[j]),
                same_vehicle=True, hard_negative=False,
                a_index=int(i), b_index=int(j), bucket=""))
    return out


def _hard_negative_pairs(images, embeddings, ids, per_bucket) -> list[MinedPair]:
    """Top cross-identity similarities within each (color, body) bucket."""
    buckets: dict[str, list[int]] = {}
    for i, im in enumerate(images):
        if im.color and im.body_type:
            buckets.setdefault(f"{im.color}/{im.body_type}", []).append(i)

    out: list[MinedPair] = []
    for bucket, members in sorted(buckets.items()):
        if len(members) < 2:
            continue
        idx = np.asarray(members)
        sims = embeddings[idx] @ embeddings[idx].T
        bucket_ids = ids[idx]
        cross = bucket_ids[:, None] != bucket_ids[None, :]
        upper = np.triu(np.ones_like(sims, dtype=bool), k=1)
        cand_i, cand_j = np.nonzero(cross & upper)
        if cand_i.size == 0:
            continue
        order = np.argsort(-sims[cand_i, cand_j])[:per_bucket]
        for k in order:
            i, j = int(idx[cand_i[k]]), int(idx[cand_j[k]])
            out.append(MinedPair(
                similarity=float(sims[cand_i[k], cand_j[k]]),
                same_vehicle=False, hard_negative=True,
                a_index=i, b_index=j, bucket=bucket))
    return out


def hardest_pairs(pairs: list[MinedPair], top: int = 12) -> list[MinedPair]:
    """The confusable gallery: highest-similarity TRUE negatives."""
    hard = [p for p in pairs if p.hard_negative]
    return sorted(hard, key=lambda p: -p.similarity)[:top]
