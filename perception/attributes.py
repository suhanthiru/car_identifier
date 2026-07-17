"""Class-attribute extraction, with per-field provenance.

- color: a REAL heuristic — dominant body color from crop pixels, snapped
  to the nearest named color. Simple image processing, honestly imperfect
  (camera color casts cause misreads, which is realistic and useful).
- make/model/body_type: SIMULATED. The sprites carry no make/model signal a
  real classifier could learn, so these come from ground truth with an
  injected confusion rate (mistaking a Camry for an Altima, etc.).
- instance attributes (damage/stickers/racks): SIMULATED, passed through
  from ground truth with a miss probability. No real detector backs these.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from sim.fleet import CLASS_VOCAB
from sim.model import VehicleIdentity
from sim.render import COLOR_BGR


@dataclass(frozen=True)
class AttributeNoiseConfig:
    class_confusion_prob: float = 0.08   # make/model swapped for a same-body peer
    color_from_pixels: bool = True
    instance_attr_miss_prob: float = 0.25
    seed: int = 31


def estimate_color(crop_bgr: np.ndarray) -> str:
    """Nearest named color of the central body region. Real heuristic."""
    h, w = crop_bgr.shape[:2]
    region = crop_bgr[int(h * 0.35): int(h * 0.60), int(w * 0.25): int(w * 0.75)]
    mean = region.reshape(-1, 3).mean(axis=0)
    names = list(COLOR_BGR)
    dists = [float(np.linalg.norm(mean - np.array(COLOR_BGR[n]))) for n in names]
    return names[int(np.argmin(dists))]


def perceive_class_attrs(
    vehicle: VehicleIdentity,
    crop_bgr: np.ndarray | None,
    event_id: str,
    config: AttributeNoiseConfig,
) -> dict[str, str]:
    """Class attrs as the pipeline would report them (possibly wrong)."""
    rng = random.Random(f"{config.seed}|attrs|{event_id}")
    make, model, body = vehicle.make, vehicle.model, vehicle.body_type
    if rng.random() < config.class_confusion_prob:
        peers = [(mk, md, bd) for mk, md, bd in CLASS_VOCAB
                 if bd == body and (mk, md) != (make, model)]
        if peers:
            make, model, body = rng.choice(peers)
    if config.color_from_pixels and crop_bgr is not None:
        color = estimate_color(crop_bgr)
    else:
        color = vehicle.color
    return {"make": make, "model": model, "body_type": body, "color": color}


def perceive_instance_attrs(
    vehicle: VehicleIdentity, event_id: str, config: AttributeNoiseConfig
) -> dict[str, str]:
    """Instance attrs with misses. Simulator-labeled ground truth."""
    rng = random.Random(f"{config.seed}|inst|{event_id}")
    return {
        k: v for k, v in vehicle.instance_attrs.items()
        if rng.random() > config.instance_attr_miss_prob
    }
