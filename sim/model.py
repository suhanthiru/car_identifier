"""Core datatypes for the synthetic world.

Everything in this package is SIMULATED. Cameras, vehicles, plates, and GPS
coordinates are fabricated. The `truth` fields on events are the simulator's
hidden ground-truth channel — downstream reasoning code must never read them
except through the evaluation harness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class CameraSpec:
    """A fixed roadside camera (simulated node)."""

    camera_id: str
    name: str
    lat: float
    lon: float
    # Compass heading the camera faces, degrees. Cosmetic for the map display.
    heading_deg: float = 0.0


@dataclass(frozen=True)
class TransitEdge:
    """Directed road link between two cameras with a physical transit window.

    A vehicle seen at `src` can only plausibly appear at `dst` between
    `min_s` and `max_s` seconds later. The reasoning layer uses these
    windows as hard physical vetoes, so the simulator must honor them.
    """

    src: str
    dst: str
    distance_m: float
    min_s: float
    typical_s: float
    max_s: float

    def contains(self, dt_s: float) -> bool:
        return self.min_s <= dt_s <= self.max_s


@dataclass(frozen=True)
class VehicleIdentity:
    """Ground-truth identity of a simulated vehicle.

    `class_attrs` are things a real classifier could plausibly output
    (make/model/color/body type). `instance_attrs` are distinguishing marks
    (damage, stickers, roof racks) — in this demo they are simulator-labeled
    ground truth, "detected" trivially; no real model backs them.
    """

    vehicle_id: str
    plate: str
    make: str
    model: str
    body_type: str
    color: str
    instance_attrs: Mapping[str, str] = field(default_factory=dict)
    # Non-empty when this vehicle was generated as part of a deliberately
    # confusable look-alike cluster (same class attrs as its siblings).
    lookalike_group: str = ""

    @property
    def class_attrs(self) -> dict[str, str]:
        return {
            "make": self.make,
            "model": self.model,
            "body_type": self.body_type,
            "color": self.color,
        }


@dataclass(frozen=True)
class RouteStop:
    """One camera passage on a vehicle's route, at an absolute sim time."""

    camera_id: str
    arrival_s: float


@dataclass(frozen=True)
class Route:
    """A realized route: the ordered camera passages for one vehicle."""

    vehicle_id: str
    stops: tuple[RouteStop, ...]

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(s.camera_id for s in self.stops)


@dataclass(frozen=True)
class SightingEvent:
    """A raw sighting emitted by a (simulated) edge camera node.

    `truth` is the simulator's hidden channel. Perception/reasoning code
    receives observations derived from it (with injected noise), never the
    truth object itself — see perception/observe.py.
    """

    event_id: str
    camera_id: str
    timestamp_s: float
    lat: float
    lon: float
    truth: VehicleIdentity
