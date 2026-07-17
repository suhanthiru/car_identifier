"""Track lifecycle: TENTATIVE -> CONFIRMED -> COASTING -> LOST.

State semantics:
- TENTATIVE: something matched a flagged target on appearance-grade
  evidence; not yet trusted. Promoted by plate-grade evidence at once, or
  by repeated consistent sightings.
- CONFIRMED: actively tracked with recent corroboration.
- COASTING: no sighting inside the expected window; position is dead
  reckoning only, displayed with decreasing confidence.
- LOST: coasted too long; the track needs re-acquisition (which restarts
  it as TENTATIVE unless plate-grade evidence confirms immediately).

All transitions are pure functions returning new Track values (no
mutation), so the tracker's history is trivially auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

TENTATIVE = "tentative"
CONFIRMED = "confirmed"
COASTING = "coasting"
LOST = "lost"

# Appearance-grade sightings needed to promote TENTATIVE -> CONFIRMED.
PROMOTE_AFTER_SIGHTINGS = 2
# Seconds without any association before CONFIRMED starts coasting.
COAST_AFTER_S = 240.0
# Seconds of total silence before a coasting track is declared lost.
LOST_AFTER_S = 600.0


@dataclass(frozen=True)
class Track:
    target_id: str
    state: str = TENTATIVE
    consecutive_sightings: int = 0
    last_association_s: float = 0.0
    entered_state_s: float = 0.0

    def _to(self, state: str, now_s: float) -> "Track":
        if state == self.state:
            return self
        return replace(self, state=state, entered_state_s=now_s)


def on_association(track: Track, now_s: float, plate_grade: bool) -> Track:
    """A sighting was associated with this target (verdict != rejected)."""
    t = replace(
        track,
        consecutive_sightings=track.consecutive_sightings + 1,
        last_association_s=now_s,
    )
    if plate_grade:
        return t._to(CONFIRMED, now_s)
    if t.state == TENTATIVE and t.consecutive_sightings >= PROMOTE_AFTER_SIGHTINGS:
        return t._to(CONFIRMED, now_s)
    if t.state in (COASTING, LOST):
        # Re-acquired on appearance evidence: back to tentative, counter reset
        # to 1 (this sighting) — it must earn confirmation again.
        return replace(t._to(TENTATIVE, now_s), consecutive_sightings=1)
    return t


def on_rejection(track: Track, now_s: float) -> Track:
    """A candidate sighting was vetoed; consecutive-evidence streak breaks."""
    return replace(track, consecutive_sightings=0)


def on_tick(track: Track, now_s: float) -> Track:
    """Advance time with no new sightings; handles coasting/lost decay."""
    silent_for = now_s - track.last_association_s
    if track.state == CONFIRMED and silent_for > COAST_AFTER_S:
        return track._to(COASTING, now_s)
    if track.state == COASTING and silent_for > LOST_AFTER_S:
        return track._to(LOST, now_s)
    if track.state == TENTATIVE and silent_for > LOST_AFTER_S:
        return track._to(LOST, now_s)
    return track
