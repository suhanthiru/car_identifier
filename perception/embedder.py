"""Appearance embedding via OSNet (torchreid), used as the ReID signal.

Off-the-shelf glue, not a contribution of this project. Notes for honesty:

- Default weights are OSNet ImageNet-pretrained (auto-downloaded by
  torchreid). The VeRi-776 vehicle-ReID checkpoint must be fetched manually
  from the torchreid model zoo; pass its path as `weights_path` to use it.
- On this project's cartoon sprites the embedding measures sprite
  similarity, not real vehicle identity. Look-alike vehicles are *supposed*
  to collide in embedding space here — that is the scenario the reasoning
  layer exists to handle.
"""
from __future__ import annotations

import numpy as np

# Input size (height, width). OSNet is fully convolutional + global pooling,
# so a wide vehicle aspect works fine.
INPUT_H, INPUT_W = 128, 256


class ReidEmbedder:
    """Lazy-loading OSNet embedder producing L2-normalized vectors."""

    def __init__(self, arch: str = "osnet_x0_25", weights_path: str | None = None):
        self._arch = arch
        self._weights_path = weights_path
        self._model = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import warnings

        import torch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torchreid

        model = torchreid.models.build_model(
            name=self._arch, num_classes=1, pretrained=self._weights_path is None
        )
        if self._weights_path:
            torchreid.reid.utils.load_pretrained_weights(model, self._weights_path)
        model.eval()
        self._model = model
        self._torch = torch

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        return self.embed_batch([crop_bgr])[0]

    def embed_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Embed BGR crops -> (n, d) float32, rows L2-normalized."""
        import cv2

        self._load()
        torch = self._torch
        batch = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (INPUT_W, INPUT_H)).astype(np.float32) / 255.0
            # ImageNet normalization, matching torchreid's training transforms.
            rgb = (rgb - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
            batch.append(rgb.transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(batch).astype(np.float32))
        with torch.no_grad():
            feats = self._model(x).cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.maximum(norms, 1e-12)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two already-normalized embeddings."""
    return float(np.dot(a, b))


def max_similarity(query: np.ndarray, gallery: list[np.ndarray]) -> float:
    """Max cosine similarity of `query` against a target's embedding set.

    Targets keep a *set* of embeddings (one per confirmed sighting) and we
    match on the max, so one bad crop cannot poison the whole profile.
    """
    if not gallery:
        return 0.0
    return max(cosine_similarity(query, g) for g in gallery)
