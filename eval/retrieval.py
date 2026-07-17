"""Standard ReID retrieval metrics: CMC (Rank-k) and mAP.

Protocol follows the VeRi/Market convention: for each query, gallery images
of the SAME identity seen by the SAME camera are excluded (a same-camera
near-duplicate frame is not a re-identification), then:

- CMC@k: fraction of queries whose first correct gallery hit appears within
  the top-k ranked results;
- AP: area under the precision curve at each correct hit, averaged over
  queries with at least one valid positive -> mAP.

Pure numpy on precomputed embeddings, so the same code scores VeRi-776,
VehicleID, and any synthetic fixture identically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalResult:
    cmc: np.ndarray          # (max_rank,) cumulative match curve
    mean_ap: float
    n_queries_scored: int
    n_queries_skipped: int   # queries with no valid positives in the gallery

    @property
    def rank1(self) -> float:
        return float(self.cmc[0])

    def rank(self, k: int) -> float:
        return float(self.cmc[k - 1])

    def summary(self) -> dict:
        return {
            "rank1": round(self.rank(1), 4),
            "rank5": round(self.rank(5), 4),
            "rank10": round(self.rank(10), 4),
            "mAP": round(self.mean_ap, 4),
            "queries": self.n_queries_scored,
            "skipped": self.n_queries_skipped,
        }


def evaluate_retrieval(
    query_emb: np.ndarray,
    query_ids: list[str],
    query_cams: list[str],
    gallery_emb: np.ndarray,
    gallery_ids: list[str],
    gallery_cams: list[str],
    max_rank: int = 50,
) -> RetrievalResult:
    """Score retrieval with the same-camera exclusion protocol."""
    if len(query_ids) != query_emb.shape[0] or len(gallery_ids) != gallery_emb.shape[0]:
        raise ValueError("embedding/id count mismatch")
    if query_emb.shape[0] == 0 or gallery_emb.shape[0] == 0:
        raise ValueError("empty query or gallery")

    sims = query_emb @ gallery_emb.T          # embeddings are L2-normalized
    g_ids = np.asarray(gallery_ids)
    g_cams = np.asarray(gallery_cams)
    max_rank = min(max_rank, gallery_emb.shape[0])

    cmc_accum = np.zeros(max_rank)
    aps: list[float] = []
    skipped = 0

    for i, (qid, qcam) in enumerate(zip(query_ids, query_cams)):
        order = np.argsort(-sims[i])
        keep = ~((g_ids[order] == qid) & (g_cams[order] == qcam))
        matches = (g_ids[order][keep] == qid)
        if not matches.any():
            skipped += 1
            continue
        ranked = matches[:max_rank]
        first_hit = int(np.argmax(matches))
        if first_hit < max_rank:
            cmc_accum[first_hit:] += 1
        elif ranked.any():  # pragma: no cover — defensive; argmax covers it
            cmc_accum[int(np.argmax(ranked)):] += 1
        # Average precision over the full ranking.
        hit_positions = np.nonzero(matches)[0]
        precisions = (np.arange(len(hit_positions)) + 1) / (hit_positions + 1)
        aps.append(float(precisions.mean()))

    scored = len(query_ids) - skipped
    if scored == 0:
        raise ValueError("no query had a valid gallery positive")
    return RetrievalResult(
        cmc=cmc_accum / scored,
        mean_ap=float(np.mean(aps)),
        n_queries_scored=scored,
        n_queries_skipped=skipped,
    )
