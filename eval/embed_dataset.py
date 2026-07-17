"""Batch-embed real dataset crops with the OSNet embedder, with caching.

Embedding 50k VeRi images on CPU takes a while; the cache keys on
(dataset name, split, arch, image count) and lands in data/eval_cache/.
Cache files are derived artifacts — safe to delete, never committed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from perception.embedder import ReidEmbedder

CACHE_DIR = Path("data/eval_cache")
BATCH = 32


def _cache_key(tag: str, paths: list[Path], arch: str) -> Path:
    h = hashlib.sha1(f"{tag}|{arch}|{len(paths)}".encode())
    for p in paths[:50]:
        h.update(p.name.encode())
    return CACHE_DIR / f"{tag}-{h.hexdigest()[:12]}.npz"


def embed_images(
    paths: list[Path],
    tag: str,
    embedder: ReidEmbedder | None = None,
    arch: str = "osnet_x0_25",
    progress: bool = True,
) -> np.ndarray:
    """(n, d) L2-normalized embeddings for image files, cached on disk."""
    import cv2

    cache = _cache_key(tag, paths, arch)
    if cache.exists():
        with np.load(cache) as data:
            if data["n"] == len(paths):
                return data["embeddings"]
    embedder = embedder or ReidEmbedder(arch=arch)
    out: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    for i, path in enumerate(paths):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"unreadable image: {path}")
        batch.append(img)
        if len(batch) == BATCH or i == len(paths) - 1:
            out.append(embedder.embed_batch(batch))
            batch = []
            if progress and (len(out) % 20 == 0):
                done = sum(a.shape[0] for a in out)
                print(f"  embedded {done}/{len(paths)}", flush=True)
    emb = np.concatenate(out).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, embeddings=emb, n=len(paths))
    return emb
