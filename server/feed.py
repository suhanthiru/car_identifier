"""SIMULATED edge tier: per-camera processes that perceive and report.

Each camera in the road graph gets its own asyncio task standing in for an
edge node. The task takes that camera's slice of the simulated sighting
stream, runs the (real) perception glue locally — embedding, plate-read
channel, attributes — and POSTs compact observations to the central
server, crop attached as base64 PNG.

None of this is real infrastructure: there is no mesh, no camera hardware,
no remote host. The point of keeping the per-camera task structure is that
the partition of work (perceive at the edge, reason at the center) matches
the architecture the README describes.
"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass

import httpx

from perception.observe import PerceptionConfig, Perceptor
from perception.types import Observation
from sim.emitter import SimWorld, iter_sightings
from sim.model import SightingEvent


@dataclass(frozen=True)
class FeedConfig:
    base_url: str = "http://127.0.0.1:8000"
    time_scale: float = 8.0     # sim seconds per wall-clock second
    send_crops: bool = True


class FeedClock:
    """Replay clock that stops while the operator has the feed paused.

    Every camera runs as its own task off one shared timeline, so pausing
    cannot be a per-task sleep: the cameras would drift apart and the
    cross-camera transit times -- which the reasoning layer actually scores
    against -- would be wrong on resume. Instead all tasks read elapsed time
    from here, and paused wall-time is simply never counted. Freezing and
    resuming is then invisible to the replay: a car that took 8 seconds to
    cross still took 8 seconds, however long you stared at it.

    `state` is any object with a `feed_paused` attribute (the server's
    app.state), read live so the toggle takes effect within a tick.
    """

    TICK_S = 0.2               # cap on how long a sleeping task waits to notice

    def __init__(self, state: object | None = None, start: float | None = None):
        self._state = state
        self._start = start if start is not None else asyncio.get_running_loop().time()
        self._paused_total = 0.0
        self._paused_since: float | None = None

    @property
    def paused(self) -> bool:
        return bool(getattr(self._state, "feed_paused", False))

    def elapsed(self) -> float:
        """Wall seconds since start, excluding time spent paused."""
        now = asyncio.get_running_loop().time()
        frozen = self._paused_total
        if self._paused_since is not None:
            frozen += now - self._paused_since
        return now - self._start - frozen

    def _sync(self) -> None:
        if self.paused and self._paused_since is None:
            self._paused_since = asyncio.get_running_loop().time()
        elif not self.paused and self._paused_since is not None:
            self._paused_total += asyncio.get_running_loop().time() - self._paused_since
            self._paused_since = None

    async def wait_until(self, offset_s: float) -> None:
        """Sleep until `offset_s` of unpaused time has elapsed."""
        while True:
            self._sync()
            if not self.paused:
                remaining = offset_s - self.elapsed()
                if remaining <= 0:
                    return
                await asyncio.sleep(min(remaining, self.TICK_S))
            else:
                await asyncio.sleep(self.TICK_S)


def observation_payload(obs: Observation, send_crop: bool = True) -> dict:
    """Serialize an Observation into the report_sighting body."""
    crop_b64 = ""
    clip_b64: list[str] = []
    if send_crop:
        import cv2

        def _encode(img):
            ok, png = cv2.imencode(".png", img)
            return base64.b64encode(png.tobytes()).decode("ascii") if ok else ""

        if obs.crop is not None:
            crop_b64 = _encode(obs.crop)
        clip_b64 = [b for f in obs.clip_frames if (b := _encode(f))]
    return {
        "event_id": obs.event_id,
        "camera_id": obs.camera_id,
        "timestamp_s": obs.timestamp_s,
        "lat": obs.lat,
        "lon": obs.lon,
        "embedding": [float(x) for x in obs.embedding],
        "plate": (
            {"text": obs.plate.text, "confidence": obs.plate.confidence,
             "source": obs.plate.source}
            if obs.plate else None),
        "class_attrs": dict(obs.class_attrs),
        "class_attrs_source": obs.class_attrs_source,
        "instance_attrs": dict(obs.instance_attrs),
        "detection_source": obs.detection_source,
        "crop_png_b64": crop_b64,
        "clip_frames_b64": clip_b64,
        "eval_truth_id": obs.eval_truth_id,
    }


async def _edge_node(
    camera_id: str,
    events: list[SightingEvent],
    perceptor: Perceptor,
    client: httpx.AsyncClient,
    cfg: FeedConfig,
    t0: float,
    clock: FeedClock,
) -> int:
    """One simulated edge node: replay this camera's events in scaled time."""
    sent = 0
    for event in events:
        due = (event.timestamp_s - t0) / cfg.time_scale
        await clock.wait_until(due)
        # Perception is synchronous CPU work; keep the loop responsive.
        obs = await asyncio.to_thread(perceptor.process, event)
        if obs is None:
            continue  # simulated missed detection
        # Perception took real time; if pause landed during it, hold the
        # report rather than letting sightings trickle into a frozen
        # console. Returns immediately when running.
        await clock.wait_until(due)
        resp = await client.post(
            f"{cfg.base_url}/api/sightings",
            json=observation_payload(obs, cfg.send_crops),
            timeout=30.0)
        resp.raise_for_status()
        sent += 1
    return sent


async def run_feed(
    world: SimWorld,
    cfg: FeedConfig | None = None,
    perception: PerceptionConfig | None = None,
    pipeline_state: object | None = None,
) -> dict[str, int]:
    """Run every simulated edge node to completion; returns sent-counts."""
    cfg = cfg or FeedConfig()
    perceptor = Perceptor(world.graph, perception or PerceptionConfig())
    by_camera: dict[str, list[SightingEvent]] = {c: [] for c in world.graph.camera_ids()}
    for event in iter_sightings(world):
        by_camera[event.camera_id].append(event)
    t0 = min((evs[0].timestamp_s for evs in by_camera.values() if evs), default=0.0)

    async with httpx.AsyncClient() as client:
        clock = FeedClock(pipeline_state)
        results = await asyncio.gather(*(
            _edge_node(cam, events, perceptor, client, cfg, t0, clock)
            for cam, events in by_camera.items() if events))
    cameras = [cam for cam, events in by_camera.items() if events]
    return dict(zip(cameras, results))
