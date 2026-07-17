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


def observation_payload(obs: Observation, send_crop: bool = True) -> dict:
    """Serialize an Observation into the report_sighting body."""
    crop_b64 = ""
    if send_crop and obs.crop is not None:
        import cv2

        ok, png = cv2.imencode(".png", obs.crop)
        if ok:
            crop_b64 = base64.b64encode(png.tobytes()).decode("ascii")
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
        "eval_truth_id": obs.eval_truth_id,
    }


async def _edge_node(
    camera_id: str,
    events: list[SightingEvent],
    perceptor: Perceptor,
    client: httpx.AsyncClient,
    cfg: FeedConfig,
    t0: float,
    wall_start: float,
) -> int:
    """One simulated edge node: replay this camera's events in scaled time."""
    sent = 0
    loop = asyncio.get_running_loop()
    for event in events:
        due = wall_start + (event.timestamp_s - t0) / cfg.time_scale
        delay = due - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        # Perception is synchronous CPU work; keep the loop responsive.
        obs = await asyncio.to_thread(perceptor.process, event)
        if obs is None:
            continue  # simulated missed detection
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
) -> dict[str, int]:
    """Run every simulated edge node to completion; returns sent-counts."""
    cfg = cfg or FeedConfig()
    perceptor = Perceptor(world.graph, perception or PerceptionConfig())
    by_camera: dict[str, list[SightingEvent]] = {c: [] for c in world.graph.camera_ids()}
    for event in iter_sightings(world):
        by_camera[event.camera_id].append(event)
    t0 = min((evs[0].timestamp_s for evs in by_camera.values() if evs), default=0.0)

    async with httpx.AsyncClient() as client:
        wall_start = asyncio.get_running_loop().time()
        results = await asyncio.gather(*(
            _edge_node(cam, events, perceptor, client, cfg, t0, wall_start)
            for cam, events in by_camera.items() if events))
    cameras = [cam for cam, events in by_camera.items() if events]
    return dict(zip(cameras, results))
