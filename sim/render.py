"""Procedural vehicle sprite renderer.

Generates synthetic camera crops/frames for the perception pipeline. These
are deliberately cartoon-simple side-view sprites, not photorealistic
renders — and the docs say so. What matters for this project is the
*confusability structure*: two vehicles with identical class attributes
render as near-identical sprites (only instance attributes and per-sighting
jitter differ), so appearance embeddings genuinely cannot separate
look-alikes. That property is what gives the reasoning layer real work.

Determinism: sprite content is seeded by (vehicle_id, camera_id, timestamp),
so re-running a scenario reproduces identical pixels.
"""
from __future__ import annotations

import random

import numpy as np

from sim.model import CameraSpec, VehicleIdentity

CROP_W, CROP_H = 256, 160
FRAME_W, FRAME_H = 640, 384

# BGR body colors. Slightly desaturated so lighting jitter stays plausible.
COLOR_BGR: dict[str, tuple[int, int, int]] = {
    "silver": (200, 200, 205),
    "black": (35, 35, 35),
    "white": (245, 245, 245),
    "gray": (120, 120, 125),
    "red": (40, 40, 190),
    "blue": (170, 90, 30),
}

# Per-camera color cast (BGR offsets). This makes appearance error
# *correlated across sightings at the same camera* — the exact effect the
# corroboration layer's independence-trap handling has to survive.
CAMERA_TINT: dict[str, tuple[int, int, int]] = {}


def _camera_tint(camera_id: str) -> np.ndarray:
    if camera_id not in CAMERA_TINT:
        rng = random.Random(f"tint:{camera_id}")
        CAMERA_TINT[camera_id] = (
            rng.randint(-18, 18), rng.randint(-18, 18), rng.randint(-18, 18)
        )
    return np.array(CAMERA_TINT[camera_id], dtype=np.int16)


def _body_geometry(body_type: str) -> tuple[list[tuple[float, float]], float]:
    """Cabin polygon (fractions of body box) + body height fraction."""
    cabins = {
        "sedan": [(0.22, 0.0), (0.34, -0.42), (0.66, -0.42), (0.80, 0.0)],
        "suv": [(0.12, 0.0), (0.18, -0.52), (0.82, -0.52), (0.90, 0.0)],
        "pickup": [(0.10, 0.0), (0.16, -0.50), (0.46, -0.50), (0.50, 0.0)],
        "wagon": [(0.16, 0.0), (0.26, -0.46), (0.88, -0.46), (0.92, 0.0)],
    }
    heights = {"sedan": 0.30, "suv": 0.36, "pickup": 0.34, "wagon": 0.32}
    return cabins.get(body_type, cabins["sedan"]), heights.get(body_type, 0.30)


def render_vehicle_crop(
    vehicle: VehicleIdentity,
    camera_id: str,
    timestamp_s: float,
    size: tuple[int, int] = (CROP_W, CROP_H),
) -> np.ndarray:
    """Render one BGR crop of `vehicle` as seen at `camera_id`."""
    import cv2

    w, h = size
    rng = random.Random(f"{vehicle.vehicle_id}|{camera_id}|{timestamp_s:.1f}")
    img = np.full((h, w, 3), 90, dtype=np.uint8)  # asphalt backdrop
    img += np.uint8(rng.randint(0, 12))

    # Body box with per-sighting scale/position jitter.
    scale = rng.uniform(0.82, 0.98)
    bw, bh_frac = int(w * 0.82 * scale), _body_geometry(vehicle.body_type)[1]
    bh = int(h * bh_frac * 2.0 * scale)
    x0 = (w - bw) // 2 + rng.randint(-8, 8)
    y1 = int(h * 0.82) + rng.randint(-4, 4)  # bottom of body
    y0 = y1 - bh

    body = np.array(COLOR_BGR[vehicle.color], dtype=np.int16)
    body = np.clip(body + rng.randint(-10, 10), 0, 255)
    body_c = tuple(int(v) for v in body)

    cv2.rectangle(img, (x0, y0), (x0 + bw, y1), body_c, -1)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y1), tuple(int(v * 0.6) for v in body), 2)

    # Cabin + windows.
    cabin, _ = _body_geometry(vehicle.body_type)
    cabin_h = int(bh * 1.05)
    pts = np.array(
        [(x0 + int(fx * bw), y0 + int(fy * cabin_h)) for fx, fy in cabin], dtype=np.int32
    )
    cv2.fillPoly(img, [pts], body_c)
    cv2.polylines(img, [pts], True, tuple(int(v * 0.6) for v in body), 2)
    win = pts.copy()
    win[:, 1] = np.clip(win[:, 1] + 4, 0, h - 1)
    shrink = (win.mean(axis=0) * 0.18 + win * 0.82).astype(np.int32)
    cv2.fillPoly(img, [shrink], (70, 55, 45))

    # Wheels.
    wheel_r = max(6, bh // 3)
    for fx in (0.20, 0.80):
        cx, cy = x0 + int(fx * bw), y1
        cv2.circle(img, (cx, cy), wheel_r, (25, 25, 25), -1)
        cv2.circle(img, (cx, cy), wheel_r // 2, (160, 160, 160), -1)

    # Plate: white tag near the rear. Text is rendered but tiny; the
    # perception layer's plate reads are simulated separately (see
    # perception/observe.py) — we do not pretend this text drives OCR.
    pw, ph = max(34, bw // 6), max(10, bh // 5)
    px, py = x0 + bw - pw - 6, y1 - ph - 2
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (235, 235, 235), -1)
    cv2.putText(img, vehicle.plate, (px + 1, py + ph - 2),
                cv2.FONT_HERSHEY_PLAIN, 0.55, (30, 30, 30), 1, cv2.LINE_AA)

    _draw_instance_attrs(cv2, img, vehicle, rng, x0, y0, bw, bh, y1)

    # Camera color cast + global brightness jitter + sensor noise.
    tinted = img.astype(np.int16) + _camera_tint(camera_id) + rng.randint(-14, 14)
    img = np.clip(tinted, 0, 255).astype(np.uint8)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 3.5, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _draw_instance_attrs(cv2, img, vehicle, rng, x0, y0, bw, bh, y1) -> None:
    """Visualize simulator-labeled marks. No real detector backs these."""
    attrs = vehicle.instance_attrs
    if "accessory" in attrs:  # roof rack / hitch / bull bar -> roof bars
        ry = y0 - max(3, bh // 8)
        cv2.line(img, (x0 + bw // 4, ry), (x0 + 3 * bw // 4, ry), (40, 40, 40), 3)
    if "sticker" in attrs:  # bright square on the rear
        sx, sy = x0 + bw - 18, y0 + bh // 3
        cv2.rectangle(img, (sx, sy), (sx + 10, sy + 10), (60, 200, 240), -1)
    if "damage" in attrs:  # dull blotch on the body side
        cx, cy = x0 + bw // 3, (y0 + y1) // 2
        cv2.ellipse(img, (cx, cy), (14, 8), 20, 0, 360, (70, 70, 75), -1)


def render_frame(
    vehicle: VehicleIdentity,
    camera: CameraSpec,
    timestamp_s: float,
) -> np.ndarray:
    """Full simulated camera frame: road scene with the vehicle composited in.

    Used when exercising the detector stage; single-vehicle frames keep the
    demo honest about what the sim can support (no dense traffic scenes).
    """
    import cv2

    rng = random.Random(f"frame|{vehicle.vehicle_id}|{camera.camera_id}|{timestamp_s:.1f}")
    frame = np.full((FRAME_H, FRAME_W, 3), 105, dtype=np.uint8)
    # Road band + dashes.
    cv2.rectangle(frame, (0, FRAME_H // 3), (FRAME_W, FRAME_H), (80, 80, 82), -1)
    for x in range(0, FRAME_W, 60):
        cv2.line(frame, (x, 2 * FRAME_H // 3), (x + 28, 2 * FRAME_H // 3), (200, 200, 200), 3)
    # Sky-ish top band.
    cv2.rectangle(frame, (0, 0), (FRAME_W, FRAME_H // 3), (150, 130, 115), -1)

    crop = render_vehicle_crop(vehicle, camera.camera_id, timestamp_s)
    ch, cw = crop.shape[:2]
    x = rng.randint(20, FRAME_W - cw - 20)
    y = rng.randint(FRAME_H // 2 - 20, FRAME_H - ch - 10)
    frame[y:y + ch, x:x + cw] = crop
    return frame
