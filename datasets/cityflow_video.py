"""Real crop extraction from CityFlow video (`vdo.avi` per camera).

Two small, focused pieces used by the K live console's real feed:
- `bbox_for`: re-parses one camera's `gt.txt` for a specific frame's box.
  A separate, cheap lookup rather than extending `datasets/cityflow.py`'s
  well-tested `TrackSpan`/`transitions()` path, which only ever needed
  enter/exit seconds, not per-frame boxes.
- `VideoFrameSource`: wraps one `cv2.VideoCapture` per camera, opened once
  per feed run and reused for every crop -- not reopened per sighting,
  which would be prohibitively slow over a multi-minute replay.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def discover_camera_dirs(root: Path, scenario_name: str) -> dict[str, Path]:
    """camera_id -> its directory (has gt/gt.txt + vdo.avi).

    Mirrors `CityFlow.load_scenario()`'s directory discovery, duplicated
    here rather than exposed on `CityFlowScenario` to avoid changing that
    dataclass's stable, already-tested shape.
    """
    for split in sorted(p for p in root.iterdir() if p.is_dir()):
        scen_dir = split / scenario_name
        if not scen_dir.is_dir():
            continue
        return {
            cam_dir.name: cam_dir
            for cam_dir in sorted(scen_dir.glob("c*"))
            if (cam_dir / "gt" / "gt.txt").exists()
        }
    return {}


def vehicle_frame_spans(gt_path: Path) -> dict[int, tuple[int, int]]:
    """vehicle_id -> (first_frame, last_frame), straight from raw gt.txt.

    Independent of any seconds/offset conversion -- CityFlowFeed pairs this
    with datasets/cityflow.py's TrackSpan (already offset-adjusted to the
    scenario's shared clock) for the real timestamp of the same passage,
    rather than trying to invert TrackSpan's seconds back into a frame
    number, which would need the per-camera fps/offset all over again.
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
    return {vid: (first[vid], last[vid]) for vid in first}


def bbox_for(gt_path: Path, frame: int, vehicle_id: int) -> tuple[int, int, int, int] | None:
    """(left, top, width, height) for one vehicle at one frame, or None."""
    for line in gt_path.read_text().splitlines():
        parts = line.replace(";", ",").split(",")
        if len(parts) < 6:
            continue
        try:
            f, vid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if f != frame or vid != vehicle_id:
            continue
        try:
            return tuple(int(round(float(x))) for x in parts[2:6])  # type: ignore[return-value]
        except ValueError:
            return None
    return None


class VideoFrameSource:
    """One reused `cv2.VideoCapture` handle for a single camera's video."""

    def __init__(self, video_path: Path):
        self._path = video_path
        self._cap = None
        self._pos: int | None = None      # last decoded frame, for frame_at

    def _capture(self):
        import cv2

        if self._cap is None:
            cap = cv2.VideoCapture(str(self._path))
            if not cap.isOpened():
                raise FileNotFoundError(f"cannot open video {self._path}")
            self._cap = cap
        return self._cap

    def frame_at(self, frame: int) -> np.ndarray | None:
        """Whole frame `frame`, decoding forward when that is cheaper.

        `cap.set(CAP_PROP_POS_FRAMES)` on an inter-frame-coded AVI forces a
        seek to the preceding keyframe and re-decodes from there, which is
        expensive enough that a naive frame server stutters. Live playback
        almost always asks for a frame slightly AHEAD of the last one, so read
        forward instead; fall back to seeking only for a jump backwards or a
        skip long enough that reading would cost more.
        """
        import cv2

        cap = self._capture()
        current = getattr(self, "_pos", None)
        ahead = current is not None and 0 <= frame - current <= self.SEEK_AHEAD_MAX
        if not ahead:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame))
            current = frame - 1
        img = None
        while current is None or current < frame:
            ok, img = cap.read()
            if not ok or img is None:
                self._pos = None
                return None
            current = frame if current is None else current + 1
        self._pos = frame
        return img

    # Reading this many frames forward beats paying for a keyframe seek.
    SEEK_AHEAD_MAX = 60

    def crop(self, frame: int, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        import cv2

        cap = self._capture()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        # Keep frame_at's forward-read fast path honest: this just moved the
        # decoder, and a stale position would make it read from the wrong place.
        self._pos = frame if (ok and img is not None) else None
        if not ok or img is None:
            return None
        left, top, w, h = bbox
        left, top = max(0, left), max(0, top)
        crop = img[top:top + h, left:left + w]
        return crop if crop.size else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._pos = None
