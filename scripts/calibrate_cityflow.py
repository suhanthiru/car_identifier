"""Per-deployment ReID calibration for a CityFlow scenario.

    python scripts/calibrate_cityflow.py [--scenario S01] [--embedder osnet|fastreid]

The project's rule is that a similarity->P(same) map is only valid for the
deployment it was fitted on: the synthetic artifact maps real same-car
similarities (~0.8) to p~0, and the VeRi-fitted curve answers a different
dataset's question. This script fits the CityFlow answer from the
scenario's own ground truth: it crops every (vehicle, camera) passage at
two points, embeds them with the same backbone the live console uses, and
saves an isotonic artifact that scripts/run_cityflow_console.py auto-loads.

The pair sample is deliberately BASE-RATE, not hard-mined. An earlier
version fitted on eval/hard_negatives.py:mine_pairs, whose top-k selection
made look-alikes the bulk of the sample; within it higher similarity
predicted "different vehicle", and since isotonic regression can only emit
a non-decreasing curve it collapsed to a constant (~0.5 across the whole
operating range). Appearance then contributed an identical amount to every
candidate and discriminated nothing. Hard mining still drives the
separability reporting in RESULTS.md — it just must not define P(same).

Ground-truth identity is used HERE (fitting an offline calibration, the
sanctioned use) and never on the serving path.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from calibration.isotonic import build_report, save
from datasets.cityflow import CityFlow
from datasets.cityflow_video import (
    VideoFrameSource, bbox_for, discover_camera_dirs, vehicle_frame_spans,
)
from datasets.config import cityflow_root
from eval.separability import natural_population_pairs
from perception.attributes import estimate_color
from perception.embedder import ReidEmbedder
from perception.fastreid_backbone import FastReidEmbedder

ARTIFACT_DIR = Path("calibration/artifacts")


@dataclass(frozen=True)
class _CropRecord:
    """The minimal image record mine_pairs needs. body_type is constant, so
    hard-negative buckets become per-estimated-color: different vehicles the
    color heuristic cannot tell apart — exactly the look-alikes the live
    console has to score."""
    vehicle_id: int
    camera_id: str
    color: str
    body_type: str = "vehicle"


def artifact_path(scenario: str, embedder: str = "osnet") -> Path:
    suffix = "" if embedder == "osnet" else f"_{embedder}"
    return ARTIFACT_DIR / f"cityflow_{scenario.lower()}{suffix}.json"


def collect_crops(scenario, camera_dirs) -> tuple[list[_CropRecord], list[np.ndarray]]:
    records: list[_CropRecord] = []
    crops: list[np.ndarray] = []
    for cam, cam_dir in sorted(camera_dirs.items()):
        gt = cam_dir / "gt" / "gt.txt"
        spans = vehicle_frame_spans(gt)
        source = VideoFrameSource(cam_dir / "vdo.avi")
        try:
            for vid, (first, last) in sorted(spans.items()):
                # Two sample points per passage give same-camera positives.
                for frac in (1 / 3, 2 / 3):
                    frame = first + int((last - first) * frac)
                    bbox = bbox_for(gt, frame, vid)
                    if bbox is None:
                        continue
                    crop = source.crop(frame, bbox)
                    if crop is None or crop.size == 0:
                        continue
                    records.append(_CropRecord(
                        vehicle_id=vid, camera_id=cam,
                        color=estimate_color(crop)))
                    crops.append(crop)
        finally:
            source.close()
    return records, crops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="S01")
    parser.add_argument("--embedder", choices=["osnet", "fastreid"], default="osnet")
    args = parser.parse_args()

    root = cityflow_root()
    if not CityFlow.exists(root):
        raise SystemExit(f"CityFlow not found at {root}; see DATASETS.md.")
    scenario = CityFlow(root).load_scenario(args.scenario)
    camera_dirs = discover_camera_dirs(root, args.scenario)

    records, crops = collect_crops(scenario, camera_dirs)
    print(f"{args.scenario}: {len(crops)} real crops across "
          f"{len({r.vehicle_id for r in records})} vehicles, "
          f"embedder={args.embedder}")
    embedder = (FastReidEmbedder() if args.embedder == "fastreid"
                else ReidEmbedder())
    embeddings = embedder.embed_batch(crops)

    # Base-rate sample, NOT the top-k mined one. mine_pairs is still the
    # right tool for reporting the confusable tail (see eval/separability.py
    # and the RESULTS separability table), but fitting P(same) on it made
    # look-alikes the majority of the sample, inverted the similarity/identity
    # relationship, and collapsed the isotonic fit to a constant — an
    # appearance signal that contributed the same value to every candidate.
    pairs = natural_population_pairs(records, embeddings)
    n_pos = sum(p.same_vehicle for p in pairs)
    print(f"{len(pairs)} base-rate pairs ({n_pos} cross-camera positives, "
          f"{len(pairs) - n_pos} negatives sampled uniformly over all "
          f"other vehicles)")
    report = build_report(pairs, note=(
        f"Calibrated on CityFlow {args.scenario} ground-truth crops (real "
        f"footage, this deployment's own cameras) with {args.embedder} "
        f"embeddings. Cross-camera positives against negatives drawn at their "
        f"natural base rate, so look-alikes appear as often as they really do "
        f"— fitting on top-k mined negatives instead collapses the isotonic "
        f"curve to a constant. Not transferable to other scenarios, datasets "
        f"or embedders."))
    out = artifact_path(args.scenario, args.embedder)
    save(report, out)
    print(f"saved {out} (version {report.model.version}, "
          f"chosen_threshold {report.chosen_threshold:.2f})")


if __name__ == "__main__":
    main()
