"""VehicleID (PKU) loader.

Layout: `image/<name>.jpg` + `train_test_split/*.txt` with lines
"<image_stem> <vehicle_id>". Standard protocol: for each test list size
(800/1600/2400 identities), one random image per identity forms the gallery
and the rest are queries; we use a seeded split for reproducibility.

Presence-gated like every real dataset here (see DATASETS.md).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from datasets.config import vehicleid_root


@dataclass(frozen=True)
class VehicleIdImage:
    path: Path
    vehicle_id: str


@dataclass(frozen=True)
class VehicleIdSplit:
    query: tuple[VehicleIdImage, ...]
    gallery: tuple[VehicleIdImage, ...]


class VehicleID:
    def __init__(self, root: Path | None = None):
        self.root = root or vehicleid_root()
        if not self.exists(self.root):
            raise FileNotFoundError(
                f"VehicleID not found at {self.root}. Manual request required — "
                f"see DATASETS.md.")

    @staticmethod
    def exists(root: Path | None = None) -> bool:
        root = root or vehicleid_root()
        return (root / "image").is_dir() and (root / "train_test_split").is_dir()

    def _read_list(self, name: str) -> list[VehicleIdImage]:
        path = self.root / "train_test_split" / name
        if not path.exists():
            return []
        images = []
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            stem, vid = parts
            images.append(VehicleIdImage(self.root / "image" / f"{stem}.jpg", vid))
        return images

    def test_split(self, size: int = 800, seed: int = 0) -> VehicleIdSplit:
        """Standard protocol: 1 gallery image per identity, rest query."""
        images = self._read_list(f"test_list_{size}.txt")
        if not images:
            raise FileNotFoundError(
                f"test_list_{size}.txt missing under {self.root}/train_test_split")
        by_id: dict[str, list[VehicleIdImage]] = {}
        for img in images:
            by_id.setdefault(img.vehicle_id, []).append(img)
        rng = random.Random(seed)
        gallery, query = [], []
        for vid in sorted(by_id):
            group = sorted(by_id[vid], key=lambda i: i.path.name)
            pick = rng.randrange(len(group))
            gallery.append(group[pick])
            query.extend(g for k, g in enumerate(group) if k != pick)
        return VehicleIdSplit(query=tuple(query), gallery=tuple(gallery))
