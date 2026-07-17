"""Next-camera prediction from the adjacency graph.

Given where a target was last seen and how long ago, list each neighboring
camera with its transit window and a status:

- upcoming:     too early; the vehicle cannot have arrived yet
- expected-now: inside the transit window — watch this camera
- overdue:      past the window; the vehicle stopped, turned off, or was missed

The console uses this to highlight camera nodes; the tracker uses the
all-overdue condition as its cue that a track should start coasting.
"""
from __future__ import annotations

from dataclasses import dataclass

from sim.road_graph import RoadGraph

STATUS_UPCOMING = "upcoming"
STATUS_EXPECTED = "expected-now"
STATUS_OVERDUE = "overdue"


@dataclass(frozen=True)
class CameraPrediction:
    camera_id: str
    window_start_s: float   # absolute sim time the window opens
    window_end_s: float
    status: str


def predict_next_cameras(
    graph: RoadGraph,
    last_camera_id: str,
    last_seen_s: float,
    now_s: float,
) -> tuple[CameraPrediction, ...]:
    """Predictions for every camera adjacent to the last sighting."""
    preds = []
    for neighbor in sorted(graph.neighbors(last_camera_id)):
        window = graph.transit_window(last_camera_id, neighbor)
        start, end = last_seen_s + window[0], last_seen_s + window[1]
        if now_s < start:
            status = STATUS_UPCOMING
        elif now_s <= end:
            status = STATUS_EXPECTED
        else:
            status = STATUS_OVERDUE
        preds.append(CameraPrediction(neighbor, start, end, status))
    return tuple(preds)


def all_overdue(predictions: tuple[CameraPrediction, ...]) -> bool:
    """True when every plausible next camera's window has closed."""
    return bool(predictions) and all(p.status == STATUS_OVERDUE for p in predictions)
