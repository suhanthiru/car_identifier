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
import json
from dataclasses import dataclass
from pathlib import Path

import httpx

from datasets.cityflow import CityFlowScenario
from datasets.cityflow_video import VideoFrameSource, bbox_for, vehicle_frame_spans
from perception.real_observe import RealPerceptor
from server.feed import FeedClock, observation_payload


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


async def _publish_clock(state, clock, t0: float, time_scale: float,
                         tick_s: float = 0.2) -> dict:
    """Keep `state.replay_clock_s` on the replay's actual position.

    Runs alongside the camera tasks and ends when they do — it is cancelled
    with the rest of the gather. FeedClock already excludes paused wall time,
    so this stops dead on pause and resumes exactly where it left off, which is
    what makes the pause button visibly do something.
    """
    while True:
        state.replay_clock_s = t0 + clock.elapsed() * time_scale
        await asyncio.sleep(tick_s)


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


THUMBNAIL_W = 160          # browse-grid tiles render around this wide


def _index_cache_path(scenario_name: str) -> Path:
    return Path("data/cache") / f"vehicle_index_{scenario_name.lower()}.json"


def _encode_thumbnail(crop) -> str:
    """Display-sized PNG for the browse grid. Never used for embeddings."""
    import cv2

    h, w = crop.shape[:2]
    if w > THUMBNAIL_W:
        crop = cv2.resize(crop, (THUMBNAIL_W, max(1, round(h * THUMBNAIL_W / w))),
                          interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", crop)
    return base64.b64encode(png.tobytes()).decode("ascii") if ok else ""


def build_vehicle_index(
    scenario: CityFlowScenario, camera_dirs: dict[str, Path],
    use_cache: bool = True,
) -> list[dict]:
    """One entry per vehicle in the scenario: its earliest camera/time and a
    real thumbnail crop from that first appearance -- what lets an operator
    browse and click "any car, arbitrarily" instead of typing an id.

    Cached to disk because building it means seeking and decoding three
    frames per vehicle out of 1080p video: ~16s for S01, during which the
    browse panel is empty. That is the first thing anyone sees, so it is
    worth not paying twice. The cache is keyed on the scenario's track
    count, which changes if the underlying data does.
    """
    import cv2

    cache = _index_cache_path(scenario.name)
    # v4: entries gained `journeys` (real positive-gap camera hops from the
    # ground truth). Bumped so an existing cache from before that change is
    # rebuilt rather than served without the field the browse panel filters on.
    key = f"v4:{scenario.name}:{len(scenario.spans)}:{len(camera_dirs)}"
    if use_cache and cache.is_file():
        try:
            blob = json.loads(cache.read_text())
            if blob.get("key") == key:
                return blob["vehicles"]
        except (json.JSONDecodeError, KeyError, OSError):
            pass  # corrupt or truncated cache: rebuild rather than fail

    earliest: dict[int, object] = {}
    # How many distinct cameras saw each vehicle. Kept as information only:
    # every annotated vehicle in this dataset is multi-camera, so it filters
    # nothing (see the journeys below for what actually separates them).
    cameras_per_vehicle: dict[int, set[str]] = {}
    passages_per_vehicle: dict[int, int] = {}
    # How watchable each vehicle is. Measured on S01: seconds-visible runs from
    # 7s to 154s (median 20) and the journey span from 5s to 76s (median 16),
    # so "how long can I actually watch this car, and does it go anywhere"
    # separates the list hard. Crop size deliberately is NOT part of this — at
    # passage midpoint every vehicle is at least 74px on its short side
    # (median 269), so it discriminates nothing here.
    visible_s: dict[int, float] = {}
    first_s: dict[int, float] = {}
    last_s: dict[int, float] = {}
    # Real cross-camera journeys, straight from the dataset's ground truth:
    # this vehicle left one camera and arrived at another LATER, with a
    # measurable gap in between. That gap is the whole problem the system
    # exists for — identity across an interval where nothing was observed.
    #
    # Only positive gaps count. In S01 82% of consecutive camera pairs overlap
    # in time (54 of 95 vehicles are in all five views at once), and a vehicle
    # that never left anyone's sight poses no re-identification question at
    # all. Deliberately computed from gt.txt alone: it is a fact about the
    # footage, published before this system existed, so filtering on it makes
    # no claim about what the cascade will conclude and cannot select for
    # outcomes we already know.
    journeys: dict[int, list[dict]] = {}
    spans_by_vehicle: dict[int, list] = {}
    for span in scenario.spans:
        spans_by_vehicle.setdefault(span.vehicle_id, []).append(span)
    for vid, spans in spans_by_vehicle.items():
        ordered = sorted(spans, key=lambda s: s.enter_s)
        hops = []
        for a, b in zip(ordered, ordered[1:]):
            gap = b.enter_s - a.exit_s
            if a.camera_id != b.camera_id and gap > 0:
                hops.append({"from_camera": a.camera_id, "to_camera": b.camera_id,
                             "gap_s": round(gap, 1),
                             "at_s": round(a.exit_s, 1)})
        journeys[vid] = hops

    for span in scenario.spans:
        cur = earliest.get(span.vehicle_id)
        if cur is None or span.enter_s < cur.enter_s:
            earliest[span.vehicle_id] = span
        cameras_per_vehicle.setdefault(span.vehicle_id, set()).add(span.camera_id)
        passages_per_vehicle[span.vehicle_id] = \
            passages_per_vehicle.get(span.vehicle_id, 0) + 1
        visible_s[span.vehicle_id] = (visible_s.get(span.vehicle_id, 0.0)
                                      + max(0.0, span.exit_s - span.enter_s))
        first_s[span.vehicle_id] = min(first_s.get(span.vehicle_id, span.enter_s),
                                       span.enter_s)
        last_s[span.vehicle_id] = max(last_s.get(span.vehicle_id, span.exit_s),
                                      span.exit_s)

    sources: dict[str, VideoFrameSource] = {}
    out: list[dict] = []
    for vid, span in sorted(earliest.items()):
        thumb_b64 = ""
        gallery_b64: list[str] = []
        cam_dir = camera_dirs.get(span.camera_id)
        if cam_dir is not None:
            gt = cam_dir / "gt" / "gt.txt"
            fr = vehicle_frame_spans(gt).get(vid)
            if fr is not None:
                src = sources.setdefault(
                    span.camera_id, VideoFrameSource(cam_dir / "vdo.avi"))
                # First frame is the display thumbnail; first/mid/last
                # together are the flag's reference-gallery seeds. The
                # passage midpoint matters: it is the exact crop the feed
                # later reports as this camera's sighting, so a flagged
                # car's own passage can actually match itself instead of
                # depending on how far the car moved since frame one.
                for frame in {fr[0], (fr[0] + fr[1]) // 2, fr[1]}:
                    bbox = bbox_for(gt, frame, vid)
                    if bbox is None:
                        continue
                    crop = src.crop(frame, bbox)
                    if crop is None or not crop.size:
                        continue
                    ok, png = cv2.imencode(".png", crop)
                    if not ok:
                        continue
                    gallery_b64.append(
                        base64.b64encode(png.tobytes()).decode("ascii"))
                    if frame == fr[0]:
                        # The browse grid renders these ~150px wide, but the
                        # raw crops run to 500px+ and 95 of them at full size
                        # is a 30 MB response before anything is on screen.
                        # Downscale the DISPLAY copy only -- gallery_b64
                        # keeps full resolution, because those crops are what
                        # the flag seeds its appearance embeddings from.
                        thumb_b64 = _encode_thumbnail(crop)
        out.append({
            "vehicle_id": vid, "first_camera": span.camera_id,
            "first_time_s": span.enter_s, "thumbnail_b64": thumb_b64,
            "gallery_b64": gallery_b64,
            "n_cameras": len(cameras_per_vehicle.get(vid, ())),
            "n_passages": passages_per_vehicle.get(vid, 0),
            "cameras": sorted(cameras_per_vehicle.get(vid, ())),
            "visible_s": round(visible_s.get(vid, 0.0), 1),
            "span_s": round(last_s.get(vid, 0.0) - first_s.get(vid, 0.0), 1),
            "last_time_s": round(last_s.get(vid, 0.0), 2),
            "journeys": journeys.get(vid, []),
        })
    for src in sources.values():
        src.close()

    if use_cache:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"key": key, "vehicles": out}))
        except OSError as exc:      # a read-only checkout must still work
            print(f"vehicle index: could not cache ({exc}); rebuilding next start")
    return out


async def _edge_node(
    camera_id: str, passages: list[_Passage], perceptor: RealPerceptor,
    client: httpx.AsyncClient, cfg: CityFlowFeedConfig, t0: float, clock: FeedClock,
) -> int:
    sent = 0
    for p in passages:
        due = (p.timestamp_s - t0) / cfg.time_scale
        await clock.wait_until(due)
        obs = await asyncio.to_thread(
            perceptor.process, camera_id, p.vehicle_id, p.frame, p.timestamp_s)
        if obs is None:
            continue  # no bbox at that frame, or an unreadable crop
        # Perception took real time; if the operator hit pause during it,
        # hold the report rather than letting a handful of sightings land
        # after the console visibly froze. Returns immediately when running.
        await clock.wait_until(due)
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
    embedder=None,
) -> dict[str, int]:
    """Replay every vehicle in `scenario` across its real cameras.

    `embedder` defaults to RealPerceptor's own (OSNet). Pass a
    FastReidEmbedder to run the vehicle-finetuned backbone instead — it
    retrieves far better (see RESULTS.md) but is much heavier per crop, so
    the default stays OSNet for a console that has to keep up with a replay.
    Whichever is used must match the calibration artifact the server loaded.
    """
    cfg = cfg or CityFlowFeedConfig()
    perceptor = RealPerceptor(camera_dirs, camera_positions, pipeline_state,
                              embedder=embedder)
    by_camera = _passages_by_camera(scenario, camera_dirs)
    all_ts = [p.timestamp_s for passages in by_camera.values() for p in passages]
    t0 = min(all_ts, default=0.0)

    try:
        async with httpx.AsyncClient() as client:
            clock = FeedClock(pipeline_state)
            # Publish the replay position for the console to display. The
            # server's `sim_now` is max(observed timestamp), which is the right
            # clock for reasoning but a terrible one to look at: with five
            # independent camera tasks it sits at whatever the LEADING camera
            # last reported and cannot move until that same camera reports
            # again, so it freezes for seconds at a time while the other four
            # are still delivering sightings. That reads as "stalled" or
            # "paused" when the replay is running perfectly.
            publisher = asyncio.create_task(
                _publish_clock(pipeline_state, clock, t0, cfg.time_scale))
            try:
                results = await asyncio.gather(*(
                    _edge_node(cam, passages, perceptor, client, cfg, t0, clock)
                    for cam, passages in by_camera.items() if passages))
            finally:
                # Runs forever by design; it ends when the cameras do.
                publisher.cancel()
    finally:
        perceptor.close()
    cameras = [cam for cam, passages in by_camera.items() if passages]
    return dict(zip(cameras, results))
