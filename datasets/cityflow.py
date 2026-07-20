"""CityFlow / AI City MTMC loader.

What the cross-camera validation needs from the release:
- per-camera ground-truth tracks (`gt/gt.txt`, MOT format:
  frame,id,left,top,width,height,...) with a shared vehicle-id space
  per scenario;
- camera calibration (`calibration.txt`: a 3x3 homography image->GPS,
  formatted as `Homography matrix: r1;r2;r3` in most releases);
- frame rate (10 fps for the published scenarios unless a cfg says otherwise).

From those we derive REAL cross-camera transitions: for each vehicle,
consecutive (camera, exit_time) -> (camera, entry_time) hops with their
elapsed seconds — the ground truth against which the transit-time veto and
the corroboration fusion are validated. Presence-gated; see DATASETS.md.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datasets.config import cityflow_root
from sim.model import CameraSpec, TransitEdge
from sim.road_graph import RoadGraph, haversine_m

DEFAULT_FPS = 10.0
# Real-graph windows get this much slack beyond the widest observed hop —
# one measurement is not "the slowest this could ever be."
REAL_WINDOW_BUFFER_S = 5.0

# AIC22's own ReadMe.txt is explicit: "we do not have access to the exact
# GPS location of each camera, the GPS location for the approximate center
# of each scenario is provided" (cam_loc/<subset>.png captions). S03/S04/S05
# share one composite map ("S0345.png") and therefore one center point.
# These are the ONLY real georeferenced facts this release publishes.
SCENARIO_CENTER_GPS: dict[str, tuple[float, float]] = {
    "S01": (42.525678, -90.723601),
    "S02": (42.491916, -90.723723),
    "S03": (42.498780, -90.686393),
    "S04": (42.498780, -90.686393),
    "S05": (42.498780, -90.686393),
    "S06": (42.492448, -90.723343),
}
# Individual camera positions inside a scenario are clustered within this
# radius of the real center — plausible for an intersection-cluster camera
# network, and small enough not to imply surveyed precision we don't have.
CAMERA_CLUSTER_RADIUS_M = 120.0
_METERS_PER_DEG_LAT = 111_320.0


@dataclass(frozen=True)
class TrackSpan:
    """One vehicle's contiguous presence at one camera."""

    scenario: str
    camera_id: str
    vehicle_id: int
    enter_s: float
    exit_s: float


@dataclass(frozen=True)
class Transition:
    """A real cross-camera hop by one ground-truth vehicle."""

    scenario: str
    vehicle_id: int
    from_camera: str
    to_camera: str
    elapsed_s: float    # to.enter - from.exit; can be negative (overlap)


@dataclass(frozen=True)
class CityFlowScenario:
    name: str
    cameras: tuple[str, ...]
    homographies: dict[str, np.ndarray]      # camera -> 3x3 image->GPS
    spans: tuple[TrackSpan, ...]

    def transitions(self, min_gap_s: float = -5.0) -> tuple[Transition, ...]:
        """Consecutive camera hops per vehicle, time-ordered.

        Small negative gaps are kept (fields of view overlap in reality);
        `min_gap_s` only filters annotation glitches.
        """
        by_vehicle: dict[int, list[TrackSpan]] = {}
        for span in self.spans:
            by_vehicle.setdefault(span.vehicle_id, []).append(span)
        out: list[Transition] = []
        for vid, spans in sorted(by_vehicle.items()):
            spans.sort(key=lambda s: s.enter_s)
            for a, b in zip(spans, spans[1:]):
                if a.camera_id == b.camera_id:
                    continue
                gap = b.enter_s - a.exit_s
                if gap >= min_gap_s:
                    out.append(Transition(
                        scenario=self.name, vehicle_id=vid,
                        from_camera=a.camera_id, to_camera=b.camera_id,
                        elapsed_s=round(gap, 2)))
        return tuple(out)

    def center_gps(self) -> tuple[float, float] | None:
        """The one real georeferenced fact AIC22 publishes for this scenario."""
        return SCENARIO_CENTER_GPS.get(self.name)

    def _camera_local_layout(self) -> dict[str, tuple[float, float]]:
        """Relative camera layout from each homography's image-center mapping.

        NOT real GPS — AIC22's own ReadMe says per-camera GPS isn't
        published, only an approximate scenario-level center (see
        `SCENARIO_CENTER_GPS`). The homography maps into some local
        calibration frame of unknown scale/orientation, not WGS84 degrees;
        the raw output is unusable as literal lat/lon (values far outside
        valid ranges). It still carries real, if uncalibrated-to-Earth,
        relative-position information between cameras in the same scenario,
        which `camera_gps()` turns into an honest approximate placement.
        """
        out = {}
        for cam, H in self.homographies.items():
            pt = H @ np.array([960.0, 540.0, 1.0])
            if abs(pt[2]) > 1e-12:
                out[cam] = (float(pt[0] / pt[2]), float(pt[1] / pt[2]))
        return out

    def camera_gps(self) -> dict[str, tuple[float, float]]:
        """Approximate per-camera GPS: real scenario center + relative layout.

        Not surveyed positions. Recenters the homography's relative local
        layout on this scenario's real, dataset-documented center point and
        scales it to fit within `CAMERA_CLUSTER_RADIUS_M` of that point —
        preserving real relative geometry (which camera is where relative
        to the others) while never claiming precision the dataset doesn't
        publish. Empty if the scenario has no real center on record or no
        calibrated cameras.
        """
        center = self.center_gps()
        local = self._camera_local_layout()
        if center is None or not local:
            return {}
        cams = sorted(local)
        xs = np.array([local[c][0] for c in cams])
        ys = np.array([local[c][1] for c in cams])
        xs, ys = xs - xs.mean(), ys - ys.mean()
        radius = float(np.hypot(xs, ys).max())
        scale = (CAMERA_CLUSTER_RADIUS_M / radius) if radius > 1e-9 else 0.0
        lat0, lon0 = center
        lon_scale = _METERS_PER_DEG_LAT * math.cos(math.radians(lat0))
        out = {}
        for cam, x, y in zip(cams, xs * scale, ys * scale):
            out[cam] = (
                lat0 + float(y) / _METERS_PER_DEG_LAT,
                lon0 + float(x) / lon_scale,
            )
        return out

    def to_road_graph(self, road_factor: float = 1.0) -> RoadGraph:
        """A REAL, reasoning-capable RoadGraph built from this scenario.

        Camera positions come from `camera_gps()` — clustered around this
        scenario's real, dataset-documented center point (the only GPS
        AIC22 actually publishes), using the calibration homography only
        for real *relative* layout, never claiming surveyed per-camera
        precision. Transit windows come from ACTUALLY OBSERVED ground-truth
        vehicle hops (`transitions()`), unlike the synthetic world's
        `sim.road_graph.make_edge`, which derives windows from a guessed
        speed envelope. A camera pair with only one observed hop gets a
        wide window padded around it (one measurement is not "the slowest
        this could ever be"); a pair with several gets [min, median, max]
        as the real population implies. Camera pairs with zero observed
        transitions get no edge at all — `min_transit_s` returns None
        between them rather than a fabricated guess.

        This is what makes the operator console's map "automatically look
        real" honest: the graph is built from a real place and real
        observed travel times, with camera micro-positions honestly
        approximated rather than fabricated wholesale like the synthetic
        world's fictional Gridville.
        """
        gps = self.camera_gps()
        if not gps:
            raise ValueError(
                f"scenario {self.name} has no usable camera calibration and/or "
                f"no documented real center; cannot place cameras honestly")
        cameras = tuple(
            CameraSpec(camera_id=cam, name=cam, lat=lat, lon=lon)
            for cam, (lat, lon) in sorted(gps.items())
        )

        by_pair: dict[tuple[str, str], list[float]] = {}
        for t in self.transitions():
            if t.elapsed_s < 0 or t.from_camera not in gps or t.to_camera not in gps:
                continue  # negative gaps are fov overlap, not travel time
            by_pair.setdefault((t.from_camera, t.to_camera), []).append(t.elapsed_s)

        edges = []
        for (src, dst), raw_samples in sorted(by_pair.items()):
            samples = sorted(raw_samples)
            lo, hi = samples[0], samples[-1]
            typical = samples[len(samples) // 2]
            lat1, lon1 = gps[src]
            lat2, lon2 = gps[dst]
            distance_m = haversine_m(lat1, lon1, lat2, lon2) * road_factor
            min_s = max(0.1, lo)
            max_s = max(hi, min_s) + REAL_WINDOW_BUFFER_S
            typical_s = min(max(typical, min_s), max_s)
            edges.append(TransitEdge(
                src=src, dst=dst, distance_m=round(distance_m, 1),
                min_s=round(min_s, 1), typical_s=round(typical_s, 1),
                max_s=round(max_s, 1)))

        return RoadGraph(cameras=cameras, edges=tuple(edges))


class CityFlow:
    def __init__(self, root: Path | None = None):
        self.root = root or cityflow_root()
        if not self.exists(self.root):
            raise FileNotFoundError(
                f"CityFlow not found at {self.root}. The AI City Challenge data "
                f"requires a signed request — see DATASETS.md.")

    @staticmethod
    def exists(root: Path | None = None) -> bool:
        root = root or cityflow_root()
        return any(root.glob("*/S*/c*/gt/gt.txt"))

    def scenario_names(self) -> list[str]:
        return sorted({p.parent.name for split in self.root.iterdir() if split.is_dir()
                       for p in split.glob("S*/c*") if (p / "gt" / "gt.txt").exists()
                       for p in [p.parent / p.name]} |
                      {p.name for split in self.root.iterdir() if split.is_dir()
                       for p in split.glob("S*") if any(p.glob("c*/gt/gt.txt"))})

    def load_scenario(self, name: str, fps: float = DEFAULT_FPS) -> CityFlowScenario:
        scen_dir = next((d for split in sorted(self.root.iterdir()) if split.is_dir()
                         for d in [split / name] if d.is_dir()), None)
        if scen_dir is None:
            raise FileNotFoundError(f"scenario {name} not under {self.root}")
        # AIC22 cameras in one scenario start at different wall-clock times;
        # the offsets (seconds) are essential — cross-camera transit times are
        # meaningless without them. They live in the scenario/camera dir or a
        # cam_timing file; missing => 0 offset (single-clock fallback).
        offsets = _load_timing_offsets(scen_dir, self.root, name)
        cameras, spans, homographies = [], [], {}
        for cam_dir in sorted(scen_dir.glob("c*")):
            gt = cam_dir / "gt" / "gt.txt"
            if not gt.exists():
                continue
            cameras.append(cam_dir.name)
            cam_fps = _read_seqinfo_fps(cam_dir, fps)
            offset = offsets.get(cam_dir.name, 0.0)
            spans.extend(_spans_from_gt(gt, name, cam_dir.name, cam_fps, offset))
            calib = cam_dir / "calibration.txt"
            if calib.exists():
                H = parse_homography(calib.read_text())
                if H is not None:
                    homographies[cam_dir.name] = H
        return CityFlowScenario(
            name=name, cameras=tuple(cameras),
            homographies=homographies, spans=tuple(spans))


def parse_homography(text: str) -> np.ndarray | None:
    """Parse `Homography matrix: a b c;d e f;g h i` (variants tolerated)."""
    m = re.search(r"[Hh]omography[^:]*:\s*([-\d.eE+\s;,]+)", text)
    if not m:
        return None
    rows = [r for r in re.split(r";", m.group(1).strip()) if r.strip()]
    if len(rows) != 3:
        return None
    try:
        H = np.array([[float(v) for v in re.split(r"[,\s]+", r.strip()) if v]
                      for r in rows], dtype=np.float64)
    except ValueError:
        return None
    return H if H.shape == (3, 3) else None


def _spans_from_gt(
    gt_path: Path, scenario: str, camera_id: str, fps: float, offset_s: float = 0.0
) -> list[TrackSpan]:
    """Collapse per-frame MOT rows into one presence span per vehicle.

    CityFlow GT tracks don't leave and re-enter the same camera within a
    scenario in any way that matters for transit stats, so min/max frame per
    id is sufficient — and robust to occasional dropped frames. `offset_s`
    shifts this camera onto the shared scenario clock.
    """
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for line in gt_path.read_text().splitlines():
        parts = line.replace(";", ",").split(",")
        if len(parts) < 6:
            continue
        try:
            frame, vid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        first[vid] = min(first.get(vid, frame), frame)
        last[vid] = max(last.get(vid, frame), frame)
    return [
        TrackSpan(scenario=scenario, camera_id=camera_id, vehicle_id=vid,
                  enter_s=round(first[vid] / fps + offset_s, 2),
                  exit_s=round(last[vid] / fps + offset_s, 2))
        for vid in sorted(first)
    ]


def _read_seqinfo_fps(cam_dir: Path, default: float) -> float:
    """Per-camera fps from an MOT-style seqinfo.ini, if the release ships one."""
    seqinfo = cam_dir / "seqinfo.ini"
    if not seqinfo.exists():
        return default
    m = re.search(r"frameRate\s*=\s*([\d.]+)", seqinfo.read_text())
    return float(m.group(1)) if m else default


def _load_timing_offsets(scen_dir: Path, root: Path, name: str) -> dict[str, float]:
    """Camera start offsets (seconds) onto the shared scenario clock.

    AIC22 ships these as `cam_timing/<scenario>.txt` with lines
    `<camera> <offset_seconds> [fps]`. Some releases place a per-scenario
    file directly in the scenario dir. Absent => empty (single-clock), which
    the loader treats as zero offset for every camera.
    """
    candidates = [
        root / "cam_timing" / f"{name}.txt",
        scen_dir / "cam_timing.txt",
        scen_dir / f"{name}.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        offsets: dict[str, float] = {}
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            cam = parts[0] if parts[0].startswith("c") else f"c{int(parts[0]):03d}"
            try:
                offsets[cam] = float(parts[1])
            except ValueError:
                continue
        if offsets:
            return offsets
    return {}
