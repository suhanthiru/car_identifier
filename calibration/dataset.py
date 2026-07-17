"""Similarity-pair dataset for ReID calibration.

Builds labeled (cosine similarity, same-vehicle?) pairs from the synthetic
world, with the negatives that actually matter: HARD negatives are pairs of
look-alike siblings — same make/model/color/body, different vehicle. Those
are the pairs a raw similarity threshold gets wrong, and exactly what the
isotonic map needs to see to stay honest about ambiguity.

HONESTY NOTE (repeated in every calibration artifact): these pairs come
from rendered sprites. The resulting calibration measures the *simulator's*
appearance distribution, not real-world ReID accuracy. Numbers derived
from it say nothing about performance on real footage.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from sim.emitter import SimWorld

EmbedFn = Callable[[object], np.ndarray]  # crop (BGR ndarray) -> unit vector


@dataclass(frozen=True)
class SimilarityPair:
    similarity: float
    same_vehicle: bool
    hard_negative: bool     # different vehicle, identical class attributes
    vehicle_a: str
    vehicle_b: str


@dataclass(frozen=True)
class PairConfig:
    crops_per_vehicle: int = 3
    negatives_per_vehicle: int = 3
    seed: int = 101


def _sample_embeddings(
    world: SimWorld, embed_fn: EmbedFn, cfg: PairConfig
) -> dict[str, list[np.ndarray]]:
    from sim.render import render_vehicle_crop

    rng = random.Random(cfg.seed)
    cams = list(world.graph.camera_ids())
    out: dict[str, list[np.ndarray]] = {}
    for vehicle in world.fleet:
        embs = []
        for _ in range(cfg.crops_per_vehicle):
            cam = rng.choice(cams)
            t = rng.uniform(0, 3600)
            embs.append(embed_fn(render_vehicle_crop(vehicle, cam, t)))
        out[vehicle.vehicle_id] = embs
    return out


def build_pairs(
    world: SimWorld, embed_fn: EmbedFn, config: PairConfig | None = None
) -> list[SimilarityPair]:
    cfg = config or PairConfig()
    rng = random.Random(cfg.seed + 1)
    embeddings = _sample_embeddings(world, embed_fn, cfg)
    by_id = {v.vehicle_id: v for v in world.fleet}
    pairs: list[SimilarityPair] = []

    ids = sorted(embeddings)
    for vid in ids:
        embs = embeddings[vid]
        # Positives: every within-vehicle crop pair.
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                pairs.append(SimilarityPair(
                    float(np.dot(embs[i], embs[j])), True, False, vid, vid))
        # Negatives: sampled cross-vehicle pairs. Look-alike siblings are
        # always included (the hard cases); the rest are random fillers.
        me = by_id[vid]
        siblings = [o for o in ids if o != vid and by_id[o].lookalike_group
                    and by_id[o].lookalike_group == me.lookalike_group]
        others = [o for o in ids if o != vid and o not in siblings]
        chosen = siblings + rng.sample(others, k=min(cfg.negatives_per_vehicle, len(others)))
        for other in chosen:
            a = rng.choice(embs)
            b = rng.choice(embeddings[other])
            hard = by_id[other].class_attrs == me.class_attrs
            pairs.append(SimilarityPair(
                float(np.dot(a, b)), False, hard, vid, other))
    return pairs
