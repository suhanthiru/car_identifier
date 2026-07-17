"""Target profiles: what the system believes about a flagged vehicle.

Profiles are immutable — every update returns a new profile and keeps the
old one reachable as a snapshot, so any automated update can be rolled
back when an operator rejects it (Phase 4 gates *when* updates may happen;
this module only defines *what* they are).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

# A profile keeps at most this many embeddings; oldest are dropped first.
# Bounded so one target cannot accumulate an unbounded gallery.
MAX_GALLERY = 12


@dataclass(frozen=True)
class LastSeen:
    camera_id: str
    timestamp_s: float
    event_id: str


@dataclass(frozen=True)
class TargetProfile:
    target_id: str
    label: str = ""                      # operator-facing name for the flag
    plate: str = ""                      # empty = plate unknown
    class_attrs: Mapping[str, str] = field(default_factory=dict)
    instance_attrs: Mapping[str, str] = field(default_factory=dict)
    # Appearance gallery: tuple of L2-normalized embeddings from sightings
    # that cleared the update gate. Tuple (not list) to keep it immutable.
    gallery: tuple[np.ndarray, ...] = ()
    last_seen: LastSeen | None = None
    # Monotonically increasing on every accepted update; pairs with the
    # profile_updates audit table server-side.
    version: int = 0

    def with_sighting(
        self,
        embedding: np.ndarray,
        last_seen: LastSeen,
        observed_instance_attrs: Mapping[str, str] | None = None,
        learned_plate: str = "",
    ) -> "TargetProfile":
        """New profile incorporating a vetted sighting (gate is elsewhere)."""
        gallery = (*self.gallery, embedding)[-MAX_GALLERY:]
        merged_attrs = dict(self.instance_attrs)
        if observed_instance_attrs:
            merged_attrs.update(observed_instance_attrs)
        return replace(
            self,
            gallery=gallery,
            last_seen=last_seen,
            instance_attrs=merged_attrs,
            plate=self.plate or learned_plate,
            version=self.version + 1,
        )

    def moved_since(self, other: "TargetProfile") -> bool:
        return self.last_seen != other.last_seen


def profile_from_flag(
    target_id: str,
    label: str,
    plate: str = "",
    class_attrs: Mapping[str, str] | None = None,
    instance_attrs: Mapping[str, str] | None = None,
) -> TargetProfile:
    """A fresh profile from an operator's flag_target request.

    The operator may know as little as "silver Camry, partial plate" —
    empty fields are legitimate and the cascade treats them as absent
    evidence, not as wildcards that match anything strongly.
    """
    return TargetProfile(
        target_id=target_id,
        label=label,
        plate=plate.upper().strip(),
        class_attrs=dict(class_attrs or {}),
        instance_attrs=dict(instance_attrs or {}),
    )
