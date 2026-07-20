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

    def _capture(self):
        import cv2

        if self._cap is None:
            cap = cv2.VideoCapture(str(self._path))
            if not cap.isOpened():
                raise FileNotFoundError(f"cannot open video {self._path}")
            self._cap = cap
        return self._cap

    def crop(self, frame: int, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        import cv2

        cap = self._capture()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
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
