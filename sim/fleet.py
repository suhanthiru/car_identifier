"""Fleet generation, including deliberately confusable look-alike clusters.

The whole point of the reasoning layer is handling vehicles that a
similarity score alone cannot separate. So the fleet generator plants
clusters of vehicles sharing every class attribute (make/model/color/body),
differing only by plate and — sometimes — instance attributes. At least one
pair per cluster has *no* distinguishing instance attributes at all: on
appearance alone they are the same vehicle, and only plate reads or
physical-plausibility reasoning can tell them apart.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from sim.model import VehicleIdentity

# Fabricated plate alphabet: skip letters commonly confused by OCR (I, O, Q)
# so injected OCR noise is something we control, not an artifact.
PLATE_LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"
PLATE_DIGITS = "0123456789"

# (make, model, body_type) vocabulary — common, deliberately generic.
CLASS_VOCAB = [
    ("Toyota", "Camry", "sedan"),
    ("Honda", "Civic", "sedan"),
    ("Ford", "F-150", "pickup"),
    ("Chevrolet", "Equinox", "suv"),
    ("Nissan", "Altima", "sedan"),
    ("Ram", "1500", "pickup"),
    ("Hyundai", "Tucson", "suv"),
    ("Subaru", "Outback", "wagon"),
]

COLORS = ["silver", "black", "white", "gray", "red", "blue"]

# Instance-attribute vocabulary. These are simulator-labeled ground truth;
# no real damage/sticker detector exists in this project.
INSTANCE_ATTRS = {
    "damage": ["dented rear bumper", "cracked windshield", "scraped left door"],
    "sticker": ["parking permit rear window", "oval bumper sticker", "university decal"],
    "accessory": ["roof rack", "tow hitch", "bull bar"],
}


@dataclass(frozen=True)
class FleetConfig:
    """Knobs for fleet generation."""

    # Each tuple: (cluster size, how many members get a distinguishing mark).
    # marked < size guarantees at least one unmarked confusable pair.
    lookalike_clusters: tuple[tuple[int, int], ...] = ((4, 2), (3, 1), (2, 0))
    background_vehicles: int = 8
    seed: int = 7


def _plate(rng: random.Random, used: set[str]) -> str:
    while True:
        p = (
            "".join(rng.choice(PLATE_LETTERS) for _ in range(3))
            + "-"
            + "".join(rng.choice(PLATE_DIGITS) for _ in range(4))
        )
        if p not in used:
            used.add(p)
            return p


def _instance_attrs(rng: random.Random, count: int) -> dict[str, str]:
    kinds = rng.sample(sorted(INSTANCE_ATTRS), k=min(count, len(INSTANCE_ATTRS)))
    return {k: rng.choice(INSTANCE_ATTRS[k]) for k in kinds}


def generate_fleet(config: FleetConfig | None = None) -> tuple[VehicleIdentity, ...]:
    """Build the full vehicle population: look-alike clusters + background."""
    cfg = config or FleetConfig()
    rng = random.Random(cfg.seed)
    used_plates: set[str] = set()
    vehicles: list[VehicleIdentity] = []

    class_choices = rng.sample(CLASS_VOCAB, k=len(CLASS_VOCAB))

    for idx, (size, marked) in enumerate(cfg.lookalike_clusters):
        if marked >= size:
            raise ValueError("each cluster needs at least one unmarked member")
        make, model, body = class_choices[idx]
        color = rng.choice(COLORS)
        group = f"cluster-{idx + 1}"
        for member in range(size):
            has_mark = member < marked  # first `marked` members get marks
            vehicles.append(
                VehicleIdentity(
                    vehicle_id=f"veh-{group}-{member + 1}",
                    plate=_plate(rng, used_plates),
                    make=make,
                    model=model,
                    body_type=body,
                    color=color,
                    instance_attrs=_instance_attrs(rng, 1) if has_mark else {},
                    lookalike_group=group,
                )
            )

    remaining_classes = class_choices[len(cfg.lookalike_clusters):]
    for i in range(cfg.background_vehicles):
        make, model, body = remaining_classes[i % len(remaining_classes)]
        vehicles.append(
            VehicleIdentity(
                vehicle_id=f"veh-bg-{i + 1}",
                plate=_plate(rng, used_plates),
                make=make,
                model=model,
                body_type=body,
                color=rng.choice(COLORS),
                instance_attrs=_instance_attrs(rng, rng.choice([0, 0, 1])),
            )
        )
    return tuple(vehicles)


def lookalike_groups(fleet: tuple[VehicleIdentity, ...]) -> dict[str, list[VehicleIdentity]]:
    groups: dict[str, list[VehicleIdentity]] = {}
    for v in fleet:
        if v.lookalike_group:
            groups.setdefault(v.lookalike_group, []).append(v)
    return groups
