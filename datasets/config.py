"""Dataset root paths — env-overridable, presence-gated.

Real datasets require manual request/download (see DATASETS.md). Nothing in
the eval harness runs against a dataset whose root fails its presence check,
and nothing ever substitutes synthetic numbers for a missing dataset.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_BASE = Path("data/datasets")


def veri_root() -> Path:
    return Path(os.environ.get("EYES_VERI_ROOT", DEFAULT_BASE / "VeRi"))


def vehicleid_root() -> Path:
    return Path(os.environ.get("EYES_VEHICLEID_ROOT", DEFAULT_BASE / "VehicleID"))


def cityflow_root() -> Path:
    return Path(os.environ.get("EYES_CITYFLOW_ROOT", DEFAULT_BASE / "CityFlow"))
