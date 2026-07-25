"""Field-standard ReID retrieval metrics (Rank-1/5/10, mAP) on CityFlow.

    python scripts/eval_cityflow_retrieval.py [--scenario S01] [--embedder osnet|fastreid]

The project already reports VeRi-776 retrieval numbers in RESULTS.md
(30.7% Rank-1 / 7.6% mAP on the ImageNet-pretrained OSNet default). This
script produces the equivalent numbers on CityFlow's real multi-camera
footage, which is what actually settles the question the look-alike
investigation ran into: is the embedder weak, or is this data genuinely
hard?

It deliberately reuses `eval/retrieval.py:evaluate_retrieval` rather than
scoring by hand. That function already implements the standard protocol --
for each query, gallery images of the SAME identity from the SAME camera
are excluded, so a near-duplicate frame from the same viewpoint cannot be
counted as a re-identification. Getting that exclusion wrong is the
classic way to publish a flattering ReID number, so the protocol is
shared with the VeRi path rather than reimplemented here.

Every crop comes from CityFlow ground-truth boxes via
`scripts/calibrate_cityflow.py:collect_crops`; identity labels are used
only to score the ranking afterwards, never fed to the embedder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from calibrate_cityflow import collect_crops
from datasets.cityflow import CityFlow
from datasets.cityflow_video import discover_camera_dirs
from datasets.config import cityflow_root
from eval.plots import plot_cmc
from eval.retrieval import evaluate_retrieval
from perception.embedder import ReidEmbedder
from perception.fastreid_backbone import FastReidEmbedder


def split_query_gallery(records, embeddings: np.ndarray, seed: int = 5):
    """One query per (vehicle, camera) passage; everything else is gallery.

    Splitting per-passage rather than randomly matters: it guarantees each
    query's positives live on OTHER cameras (the same-camera ones are
    excluded by the protocol anyway), so the task being scored really is
    cross-camera re-identification.
    """
    rng = np.random.default_rng(seed)
    by_passage: dict[tuple[int, str], list[int]] = {}
    for i, r in enumerate(records):
        by_passage.setdefault((r.vehicle_id, r.camera_id), []).append(i)

    query_idx: list[int] = []
    for key in sorted(by_passage):
        members = by_passage[key]
        query_idx.append(int(rng.choice(members)))
    query_set = set(query_idx)
    gallery_idx = [i for i in range(len(records)) if i not in query_set]
    return query_idx, gallery_idx


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
    n_vehicles = len({r.vehicle_id for r in records})
    n_cameras = len({r.camera_id for r in records})
    print(f"{args.scenario}: {len(crops)} real crops, {n_vehicles} vehicles, "
          f"{n_cameras} cameras, embedder={args.embedder}")

    embedder = FastReidEmbedder() if args.embedder == "fastreid" else ReidEmbedder()
    embeddings = embedder.embed_batch(crops)

    query_idx, gallery_idx = split_query_gallery(records, embeddings)
    print(f"{len(query_idx)} queries (one per vehicle-camera passage), "
          f"{len(gallery_idx)} gallery crops")

    result = evaluate_retrieval(
        embeddings[query_idx],
        [str(records[i].vehicle_id) for i in query_idx],
        [records[i].camera_id for i in query_idx],
        embeddings[gallery_idx],
        [str(records[i].vehicle_id) for i in gallery_idx],
        [records[i].camera_id for i in gallery_idx],
    )
    summary = result.summary()
    print(f"\nCityFlow {args.scenario} cross-camera retrieval "
          f"({args.embedder}), same-camera-same-id excluded:")
    print(f"  Rank-1  : {summary['rank1'] * 100:.1f}%")
    print(f"  Rank-5  : {summary['rank5'] * 100:.1f}%")
    print(f"  Rank-10 : {summary['rank10'] * 100:.1f}%")
    print(f"  mAP     : {summary['mAP'] * 100:.1f}%")
    print(f"  scored {summary['queries']} queries "
          f"({summary['skipped']} skipped: no cross-camera positive in gallery)")

    tag = args.scenario.lower()
    if args.embedder != "osnet":
        tag = f"{tag}_{args.embedder}"
    cmc_path = plot_cmc(
        result.cmc,
        f"CityFlow {args.scenario} CMC ({args.embedder}, "
        f"Rank-1 {summary['rank1'] * 100:.1f}% / mAP {summary['mAP'] * 100:.1f}%)",
        f"cityflow_{tag}_cmc.png")
    out = Path(f"data/retrieval_{tag}.json")
    out.write_text(json.dumps({
        "scenario": args.scenario, "embedder": args.embedder,
        "n_crops": len(crops), "n_vehicles": n_vehicles, "n_cameras": n_cameras,
        **summary,
    }, indent=1))
    print(f"\nplot: {cmc_path}\nsummary: {out}")


if __name__ == "__main__":
    main()
