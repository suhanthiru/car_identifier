"""Real-data edge tier for CityFlow: per-camera async replay of real
ground-truth vehicle passages against the live server.

Parallel to server/feed.py's run_feed/_edge_node, but every vehicle in the
loaded scenario gets replayed (not a pre-picked one) -- corroboration and
ambiguity only get interesting with other real cars in the mix, and
flagging a different car mid-replay picking up its own matching sightings
is what makes "follow any vehicle, arbitrarily" real. One representative
sighting per (vehicle, camera) passage: the frame at the midpoint of that
passage, replayed in real time-order.
"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path

import httpx

from datasets.cityflow import CityFlowScenario
from datasets.cityflow_video import VideoFrameSource, bbox_for, vehicle_frame_spans
from perception.real_observe import RealPerceptor
from server.feed import observation_payload


@dataclass(frozen=True)
class CityFlowFeedConfig:
    base_url: str = "http://127.0.0.1:8000"
    time_scale: float = 8.0
    send_crops: bool = True


@dataclass(frozen=True)
class _Passage:
    vehicle_id: int
    frame: int
    timestamp_s: float


def _passages_by_camera(
    scenario: CityFlowScenario, camera_dirs: dict[str, Path],
) -> dict[str, list[_Passage]]:
    spans_by_cam: dict[str, list] = {}
    for span in scenario.spans:
        spans_by_cam.setdefault(span.camera_id, []).append(span)

    out: dict[str, list[_Passage]] = {}
    for cam, spans in spans_by_cam.items():
        cam_dir = camera_dirs.get(cam)
        if cam_dir is None:
            continue
        frames = vehicle_frame_spans(cam_dir / "gt" / "gt.txt")
        passages = []
        for span in spans:
            fr = frames.get(span.vehicle_id)
            if fr is None:
                continue
            passages.append(_Passage(
                vehicle_id=span.vehicle_id, frame=(fr[0] + fr[1]) // 2,
                timestamp_s=(span.enter_s + span.exit_s) / 2))
        passages.sort(key=lambda p: p.timestamp_s)
        out[cam] = passages
    return out


def build_vehicle_index(
    scenario: CityFlowScenario, camera_dirs: dict[str, Path],
) -> list[dict]:
    """One entry per vehicle in the scenario: its earliest camera/time and a
    real thumbnail crop from that first appearance -- what lets an operator
    browse and click "any car, arbitrarily" instead of typing an id."""
    import cv2

    earliest: dict[int, object] = {}
    for span in scenario.spans:
        cur = earliest.get(span.vehicle_id)
        if cur is None or span.enter_s < cur.enter_s:
            earliest[span.vehicle_id] = span

    sources: dict[str, VideoFrameSource] = {}
    out: list[dict] = []
    for vid, span in sorted(earliest.items()):
        thumb_b64 = ""
        cam_dir = camera_dirs.get(span.camera_id)
        if cam_dir is not None:
            frames = vehicle_frame_spans(cam_dir / "gt" / "gt.txt")
            fr = frames.get(vid)
            if fr is not None:
                bbox = bbox_for(cam_dir / "gt" / "gt.txt", fr[0], vid)
                if bbox is not None:
                    src = sources.setdefault(
                        span.camera_id, VideoFrameSource(cam_dir / "vdo.avi"))
                    crop = src.crop(fr[0], bbox)
                    if crop is not None:
                        ok, png = cv2.imencode(".png", crop)
                        if ok:
                            thumb_b64 = base64.b64encode(png.tobytes()).decode("ascii")
        out.append({
            "vehicle_id": vid, "first_camera": span.camera_id,
            "first_time_s": span.enter_s, "thumbnail_b64": thumb_b64,
        })
    for src in sources.values():
        src.close()
    return out


async def _edge_node(
    camera_id: str, passages: list[_Passage], perceptor: RealPerceptor,
    client: httpx.AsyncClient, cfg: CityFlowFeedConfig, t0: float, wall_start: float,
) -> int:
    sent = 0
    loop = asyncio.get_running_loop()
    for p in passages:
        due = wall_start + (p.timestamp_s - t0) / cfg.time_scale
        delay = due - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        obs = await asyncio.to_thread(
            perceptor.process, camera_id, p.vehicle_id, p.frame, p.timestamp_s)
        if obs is None:
            continue  # no bbox at that frame, or an unreadable crop
        resp = await client.post(
            f"{cfg.base_url}/api/sightings",
            json=observation_payload(obs, cfg.send_crops), timeout=30.0)
        resp.raise_for_status()
        sent += 1
    return sent


async def run_cityflow_feed(
    scenario: CityFlowScenario,
    camera_dirs: dict[str, Path],
    camera_positions: dict[str, tuple[float, float]],
    pipeline_state: object,
    cfg: CityFlowFeedConfig | None = None,
) -> dict[str, int]:
    """Replay every vehicle in `scenario` across its real cameras."""
    cfg = cfg or CityFlowFeedConfig()
    perceptor = RealPerceptor(camera_dirs, camera_positions, pipeline_state)
    by_camera = _passages_by_camera(scenario, camera_dirs)
    all_ts = [p.timestamp_s for passages in by_camera.values() for p in passages]
    t0 = min(all_ts, default=0.0)

    try:
        async with httpx.AsyncClient() as client:
            wall_start = asyncio.get_running_loop().time()
            results = await asyncio.gather(*(
                _edge_node(cam, passages, perceptor, client, cfg, t0, wall_start)
                for cam, passages in by_camera.items() if passages))
    finally:
        perceptor.close()
    cameras = [cam for cam, passages in by_camera.items() if passages]
    return dict(zip(cameras, results))
