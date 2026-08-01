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

import threading
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
    """One reused `cv2.VideoCapture` handle for a single camera's video.

    Every method that touches the handle takes `self._lock`, INCLUDING
    `close()`. Releasing a `cv2.VideoCapture` while another thread is inside
    `set()` or `read()` on it is an access violation — the process dies with
    no Python exception and no traceback.

    That is not hypothetical. A replay restart tears down the feed's
    RealPerceptor and closes its sources, while `asyncio.to_thread` workers
    from the outgoing run can still be inside `crop()`. Caught with the fault
    handler armed after repeated resets:

        Windows fatal exception: access violation
        Current thread (most recent call first):
          File "datasets/cityflow_video.py", line 176 in crop
          File "perception/real_observe.py", line 124 in process
          ... concurrent.futures worker

    The frame endpoint already guarded its own copies with an external
    per-camera lock for exactly this reason; the feed's copies had nothing.
    Owning the lock here means a caller cannot forget it.
    """

    def __init__(self, video_path: Path):
        self._path = video_path
        self._cap = None
        self._pos: int | None = None      # last decoded frame, for frame_at
        self._last: np.ndarray | None = None   # that frame's pixels, for re-asks
        self._n_frames: int | None = None      # container frame count, 0 = unknown
        # Reentrant: frame_at() calls frame_count(), which also takes it.
        self._lock = threading.RLock()

    def _capture(self):
        import cv2

        if self._cap is None:
            cap = cv2.VideoCapture(str(self._path))
            if not cap.isOpened():
                raise FileNotFoundError(f"cannot open video {self._path}")
            self._cap = cap
        return self._cap

    def frame_count(self) -> int:
        """Frames in the container, or 0 if it does not say.

        Read from the header once, not by decoding. Used to reject an
        out-of-range request before it reaches the decoder: `cap.set()` to a
        frame past the end makes FFmpeg scan for a keyframe that is not there,
        which costs far more than a normal read and then fails anyway. A live
        view polling twice a second past the end of its clip pays that on
        every tick -- with 25 cameras in S04 that is what made the console
        buffer and lock up as clips ran out.
        """
        import cv2

        with self._lock:
            if self._n_frames is None:
                try:
                    self._n_frames = max(0, int(
                        self._capture().get(cv2.CAP_PROP_FRAME_COUNT)))
                except Exception:        # noqa: BLE001 — treat as "unknown"
                    self._n_frames = 0
            return self._n_frames

    def frame_at(self, frame: int) -> np.ndarray | None:
        """Whole frame `frame`, decoding forward when that is cheaper.

        `cap.set(CAP_PROP_POS_FRAMES)` on an inter-frame-coded AVI forces a
        seek to the preceding keyframe and re-decodes from there, which is
        expensive enough that a naive frame server stutters. Live playback
        almost always asks for a frame slightly AHEAD of the last one, so read
        forward instead; fall back to seeking only for a jump backwards or a
        skip long enough that reading would cost more.

        Re-asking for the frame just returned is answered from `self._last`.
        That case used to fall through the forward-read loop without executing
        it once and return None, which the frame endpoint reports as a 404 and
        the console renders as "this camera's footage has ended" -- about a
        camera that is running perfectly. Every poll while the replay is
        PAUSED asks for the same clock value, so a paused console declared all
        five cameras dead; so did a second browser tab polling in step with
        the first.
        """
        import cv2

        with self._lock:
            cap = self._capture()
            if frame == self._pos and self._last is not None:
                return self._last
            # Cheap rejection before the decoder is touched at all.
            n = self.frame_count()
            if frame < 0 or (n and frame >= n):
                return None
            return self._decode_to(cap, frame, cv2)

    def _decode_to(self, cap, frame: int, cv2):
        """Forward-read or seek to `frame`. Caller holds the lock."""
        current = self._pos
        # Strictly ahead: frame == current with no cached pixels means the
        # decoder has already consumed it, so that must seek, not read.
        ahead = current is not None and 0 < frame - current <= self.SEEK_AHEAD_MAX
        if not ahead:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame))
            current = frame - 1
        img = None
        while current < frame:
            ok, img = cap.read()
            if not ok or img is None:
                self._pos = None
                self._last = None
                return None
            current += 1
        self._pos = frame
        self._last = img
        return img

    # Reading this many frames forward beats paying for a keyframe seek.
    SEEK_AHEAD_MAX = 60

    def crop(self, frame: int, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        import cv2

        with self._lock:
            return self._crop_locked(frame, bbox, cv2)

    def _crop_locked(self, frame, bbox, cv2) -> np.ndarray | None:
        cap = self._capture()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        # Keep frame_at's forward-read fast path honest: this just moved the
        # decoder, and a stale position would make it read from the wrong place.
        # The cached full frame is dropped rather than kept -- crop() is the
        # ingest path, one call per sighting per camera, and holding a 1080p
        # frame per source for a re-ask that never comes is pure footprint.
        self._pos = frame if (ok and img is not None) else None
        self._last = None
        if not ok or img is None:
            return None
        left, top, w, h = bbox
        left, top = max(0, left), max(0, top)
        crop = img[top:top + h, left:left + w]
        return crop if crop.size else None

    def close(self) -> None:
        # Under the lock: this is the release that crashed the process when it
        # landed while a worker was mid-read. Waiting for the in-flight decode
        # costs milliseconds; not waiting costs the whole server.
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._pos = None
            self._last = None
