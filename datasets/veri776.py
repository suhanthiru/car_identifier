"""VeRi-776 loader.

Filename convention: `0002_c002_00030600_0.jpg` ->
vehicle id 0002, camera c002, timestamp 00030600, sequence 0.

`train_label.xml` / `test_label.xml` carry per-image color/type ids, which is
what makes VeRi the right first benchmark here: color+type buckets are how we
mine REAL confusable look-alikes (same color, same body type, different
vehicle) instead of easy random negatives.

Presence-gated: `Veri776.exists()` must be True before anything runs, and the
loader never invents fields the release on disk doesn't have.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from datasets.config import veri_root

_NAME_RE = re.compile(r"^(\d+)_c(\d+)_(\d+)_(\d+)\.jpg$")

# Official VeRi color/type id -> name maps (from the dataset's list files).
COLOR_NAMES = {
    1: "yellow", 2: "orange", 3: "green", 4: "gray", 5: "red",
    6: "blue", 7: "white", 8: "golden", 9: "brown", 10: "black",
}
TYPE_NAMES = {
    1: "sedan", 2: "suv", 3: "van", 4: "hatchback", 5: "mpv",
    6: "pickup", 7: "bus", 8: "truck", 9: "estate",
}


@dataclass(frozen=True)
class VeriImage:
    path: Path
    vehicle_id: str
    camera_id: str
    timestamp: int
    color: str = ""     # "" when the release lacks a label for this image
    body_type: str = ""


@dataclass(frozen=True)
class Veri776:
    root: Path
    train: tuple[VeriImage, ...]
    query: tuple[VeriImage, ...]
    gallery: tuple[VeriImage, ...]

    @staticmethod
    def exists(root: Path | None = None) -> bool:
        root = root or veri_root()
        return all(
            (root / p).exists()
            for p in ("image_query", "image_test", "name_query.txt", "name_test.txt")
        )

    @staticmethod
    def load(root: Path | None = None) -> "Veri776":
        root = root or veri_root()
        if not Veri776.exists(root):
            raise FileNotFoundError(
                f"VeRi-776 not found at {root}. It requires a manual research-use "
                f"request — see DATASETS.md for instructions and expected layout.")
        labels = {}
        for xml_name in ("train_label.xml", "test_label.xml"):
            labels.update(_parse_label_xml(root / xml_name))
        return Veri776(
            root=root,
            train=_load_split(root, "image_train", "name_train.txt", labels),
            query=_load_split(root, "image_query", "name_query.txt", labels),
            gallery=_load_split(root, "image_test", "name_test.txt", labels),
        )

    def identities(self) -> set[str]:
        return {img.vehicle_id for img in (*self.query, *self.gallery)}


def parse_name(name: str) -> tuple[str, str, int] | None:
    """filename -> (vehicle_id, camera_id, timestamp), or None if malformed."""
    m = _NAME_RE.match(name.strip())
    if not m:
        return None
    return m.group(1), f"c{m.group(2)}", int(m.group(3))


def _parse_label_xml(path: Path) -> dict[str, tuple[str, str]]:
    """imageName -> (color, type). Tolerant of the file being absent."""
    if not path.exists():
        return {}
    # VeRi XMLs declare encoding="gb2312", but expat (ET.parse's underlying
    # parser) doesn't support multi-byte encodings and raises ValueError on
    # them — decode with Python's own gb2312 codec first, then hand ET
    # already-decoded text (the values themselves are pure ASCII digits).
    root = ET.fromstring(path.read_bytes().decode("gb2312"))
    out: dict[str, tuple[str, str]] = {}
    for item in root.iter("Item"):
        name = item.get("imageName", "")
        color = COLOR_NAMES.get(int(item.get("colorID", 0) or 0), "")
        vtype = TYPE_NAMES.get(int(item.get("typeID", 0) or 0), "")
        if name:
            out[name] = (color, vtype)
    return out


def _load_split(
    root: Path, image_dir: str, list_name: str,
    labels: dict[str, tuple[str, str]],
) -> tuple[VeriImage, ...]:
    list_path = root / list_name
    if not list_path.exists():
        return ()
    images = []
    for line in list_path.read_text().splitlines():
        name = line.strip()
        parsed = parse_name(name)
        if parsed is None:
            continue  # tolerate stray lines; malformed names carry no labels
        vehicle_id, camera_id, ts = parsed
        color, body = labels.get(name, ("", ""))
        images.append(VeriImage(
            path=root / image_dir / name,
            vehicle_id=vehicle_id, camera_id=camera_id, timestamp=ts,
            color=color, body_type=body,
        ))
    return tuple(images)
