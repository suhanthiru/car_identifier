"""Plate reading.

Two backends, chosen by the pipeline config:

- SimulatedPlateReader: the backend the synthetic demo uses. It does NOT do
  OCR. It takes the ground-truth plate and applies a controlled error model
  (missed reads, per-character substitutions with confusion pairs). Every
  read is labeled source="sim". This is deliberate: the sprite plates are a
  few pixels tall, and pretending a real OCR engine reads them would be
  dressing a stub up as real.

- FastPlateOcrReader: a thin adapter over the open `fast-plate-ocr` engine
  for the optional real-clip mode. Labeled source="model".

The error model matters more than realism: plate misreads are what force
the identity cascade to fall through to weaker evidence tiers.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from perception.types import SOURCE_MODEL, SOURCE_SIM, PlateRead

# Visually-confusable character pairs (what real ALPR tends to swap).
CONFUSIONS = {
    "8": "B", "B": "8", "5": "S", "S": "5", "2": "Z", "Z": "2",
    "0": "D", "D": "0", "1": "7", "7": "1", "E": "F", "F": "E",
}


@dataclass(frozen=True)
class PlateNoiseConfig:
    """Controlled error model for simulated plate reads."""

    read_prob: float = 0.80          # chance the plate is read at all
    char_error_prob: float = 0.06    # per-character substitution chance
    # Confidence bands: correct reads score high, corrupted reads lower,
    # with overlap — so confidence alone cannot separate them.
    clean_conf: tuple[float, float] = (0.82, 0.99)
    noisy_conf: tuple[float, float] = (0.55, 0.90)


class SimulatedPlateReader:
    """Ground-truth plate + controlled noise. source="sim" on every read."""

    def __init__(self, config: PlateNoiseConfig | None = None, seed: int = 23):
        self._cfg = config or PlateNoiseConfig()
        self._seed = seed

    def read(self, true_plate: str, event_id: str) -> PlateRead | None:
        """Deterministic per event: same event always reads the same."""
        rng = random.Random(f"{self._seed}|plate|{event_id}")
        if rng.random() > self._cfg.read_prob:
            return None
        chars = list(true_plate)
        corrupted = False
        for i, ch in enumerate(chars):
            if ch != "-" and rng.random() < self._cfg.char_error_prob:
                chars[i] = CONFUSIONS.get(ch, rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ"))
                corrupted = True
        lo, hi = self._cfg.noisy_conf if corrupted else self._cfg.clean_conf
        return PlateRead(text="".join(chars), confidence=rng.uniform(lo, hi), source=SOURCE_SIM)


class FastPlateOcrReader:
    """Real OCR adapter (fast-plate-ocr) for the optional real-clip mode."""

    def __init__(self, model_name: str = "european-plates-mobile-vit-v2-model"):
        self._model_name = model_name
        self._engine = None

    def _load(self):
        if self._engine is None:
            try:
                from fast_plate_ocr import ONNXPlateRecognizer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "fast-plate-ocr is not installed; real-clip plate reading "
                    "is unavailable. The synthetic demo uses SimulatedPlateReader."
                ) from exc
            self._engine = ONNXPlateRecognizer(self._model_name)
        return self._engine

    def read(self, plate_crop_bgr: np.ndarray) -> PlateRead | None:
        import cv2

        engine = self._load()
        gray = cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2GRAY)
        texts = engine.run(gray)
        if not texts or not texts[0].strip("_"):
            return None
        # fast-plate-ocr pads with underscores; it does not expose a
        # per-read confidence, so we report a fixed mid confidence.
        return PlateRead(text=texts[0].strip("_"), confidence=0.7, source=SOURCE_MODEL)
