"""Detection + dedup: YOLO with ByteTrack, and best-crop selection.

Off-the-shelf glue. One honest wrinkle, stated loudly:

Pretrained COCO YOLO does not reliably recognize this simulator's cartoon
sprites as vehicles (verified experimentally — it mostly sees nothing).
So the wrapper always *runs* the real detector, and when the detector
comes up empty it falls back to the simulator's ground-truth box. Every
Detection records which path produced it (`source="yolo"` vs
`source="sim-fallback"`), and the demo UI surfaces that. On real footage
(the optional real-clip mode) the same code runs with no fallback available.
"""
from __future__ import annotations

import numpy as np

from perception.types import Detection

# COCO class ids that count as vehicles.
VEHICLE_CLASS_IDS = {2, 5, 7}  # car, bus, truck
DEFAULT_CONF = 0.25


class VehicleDetector:
    """YOLO + ByteTrack over a short frame burst; picks the best crop."""

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = DEFAULT_CONF):
        self._model_name = model_name
        self._conf = conf
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self._model_name)
        return self._model

    def best_detection(
        self,
        frames: list[np.ndarray],
        fallback_boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> Detection | None:
        """Track vehicles across `frames`, return the single best crop.

        Best = highest (confidence x box area) among vehicle-class boxes of
        the most-seen track. If YOLO finds nothing and `fallback_boxes`
        (simulator ground truth, one per frame) is provided, fall back to
        the largest ground-truth box, honestly labeled.
        """
        model = self._load()
        candidates: list[tuple[int, float, tuple[int, int, int, int], int]] = []
        for idx, frame in enumerate(frames):
            results = model.track(
                frame, conf=self._conf, persist=idx > 0,
                tracker="bytetrack.yaml", verbose=False,
            )[0]
            if results.boxes is None:
                continue
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASS_IDS:
                    continue
                track_id = int(box.id[0]) if box.id is not None else -1
                xyxy = tuple(int(v) for v in box.xyxy[0].tolist())
                candidates.append((idx, float(box.conf[0]), xyxy, track_id))

        if candidates:
            return self._pick_best(frames, candidates)

        if fallback_boxes:
            areas = [
                ((x2 - x1) * (y2 - y1), i, (x1, y1, x2, y2))
                for i, (x1, y1, x2, y2) in enumerate(fallback_boxes)
            ]
            _, idx, box = max(areas)
            return Detection(
                crop=_crop(frames[idx], box),
                bbox_xyxy=box,
                confidence=0.0,  # the detector did not actually fire
                source="sim-fallback",
            )
        return None

    def _pick_best(self, frames, candidates) -> Detection:
        # Prefer the track seen most often (ByteTrack dedup), then the
        # highest confidence x area crop within it.
        counts: dict[int, int] = {}
        for _, _, _, tid in candidates:
            counts[tid] = counts.get(tid, 0) + 1
        best_track = max(counts, key=counts.get)
        in_track = [c for c in candidates if c[3] == best_track]

        def score(c):
            _, conf, (x1, y1, x2, y2), _ = c
            return conf * max(1, (x2 - x1) * (y2 - y1))

        idx, conf, box, _ = max(in_track, key=score)
        return Detection(
            crop=_crop(frames[idx], box), bbox_xyxy=box, confidence=conf, source="yolo"
        )


def _crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate box {box} for frame {w}x{h}")
    return frame[y1:y2, x1:x2].copy()
