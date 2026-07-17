"""Hand-built observations/profiles for reasoning-layer tests."""
from __future__ import annotations

import numpy as np

from perception.types import SOURCE_HEURISTIC, SOURCE_SIM, Observation, PlateRead
from reasoning.profile import LastSeen, TargetProfile

CAMRY = {"make": "Toyota", "model": "Camry", "body_type": "sedan", "color": "silver"}


def unit_vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


def make_obs(
    event_id: str = "evt-1",
    camera_id: str = "cam-ctr",
    t: float = 1000.0,
    plate: str | None = None,
    plate_conf: float = 0.9,
    class_attrs: dict | None = None,
    instance_attrs: dict | None = None,
    embedding: np.ndarray | None = None,
) -> Observation:
    return Observation(
        event_id=event_id,
        camera_id=camera_id,
        timestamp_s=t,
        lat=0.0,
        lon=0.0,
        embedding=embedding if embedding is not None else unit_vec(1),
        plate=PlateRead(plate, plate_conf, SOURCE_SIM) if plate else None,
        class_attrs=class_attrs if class_attrs is not None else dict(CAMRY),
        class_attrs_source=SOURCE_HEURISTIC,
        instance_attrs=instance_attrs or {},
        detection_source="sim-fallback",
        eval_truth_id="veh-test",
    )


def make_profile(
    target_id: str = "tgt-1",
    plate: str = "ABC-1234",
    class_attrs: dict | None = None,
    instance_attrs: dict | None = None,
    gallery: tuple[np.ndarray, ...] = (),
    last_seen: LastSeen | None = None,
) -> TargetProfile:
    return TargetProfile(
        target_id=target_id,
        label="test target",
        plate=plate,
        class_attrs=class_attrs if class_attrs is not None else dict(CAMRY),
        instance_attrs=instance_attrs or {},
        gallery=gallery,
        last_seen=last_seen,
    )
