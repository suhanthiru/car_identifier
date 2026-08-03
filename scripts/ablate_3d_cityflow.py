"""THE 3D ABLATION: does reconstructed geometry help identification?

    python scripts/ablate_3d_cityflow.py [--scenario S01] [--vehicles 40]
                                         [--embedder osnet|fastreid]

The cascade can consult 3D-derived proportions (`geom3d:body_profile`,
`geom3d:length_class` from `car3d/geometry.py`) as attributes alongside colour.
The claim is that these help where 2D appearance fails hardest: the same car
seen front-on at one camera and side-on at another changes completely in pixels
and not at all in shape. Nothing had ever tested that claim.

This runs `eval/ablation.py:run_ablation` twice over IDENTICAL rankings — once
with `extra_attrs=None`, once with the geometry channel wired in — so the only
difference between the two rows is the 3D evidence.

WHY CITYFLOW AND NOT VERI-776
    Reconstruction costs ~98 s per crop on an RTX 3080 Ti. VeRi's evaluation
    split is 13,257 images, i.e. ~350 GPU-hours, so a subsample is mandatory
    either way. CityFlow is the better subsample: it is real multi-camera
    footage where the viewpoint genuinely changes between cameras, which is
    precisely the condition the 3D channel is supposed to address. VeRi's
    query/gallery pairs are mostly the same few viewpoints.

WHAT IS SUBSAMPLED, AND HOW
    `--vehicles N` keeps the first N ground-truth vehicle ids in sorted order
    and every crop belonging to them. Sorted-and-deterministic rather than
    random: the subsample is reproducible and nobody can quietly re-roll it
    until the delta looks good. All of a kept vehicle's passages are kept, so
    cross-camera positives survive the cut.

HONEST NOTE
    Unlike the VeRi ablation's colour/body-type channel — which uses dataset
    labels, i.e. a perfect classifier — this channel is MEASURED, from real
    reconstructions of real crops. The delta here is therefore not an upper
    bound; it is what this pipeline actually produces. That also means a null
    result is a real result: if 3D does not help, that is the finding.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from calibrate_cityflow import collect_crops
from eval_cityflow_retrieval import split_query_gallery

from calibration.isotonic import build_report
from car3d.compat import InsufficientDetail
from car3d.geometry import signature_to_attrs
from car3d.profile_model import Target3DModel, build_pipeline
from datasets.cityflow import CityFlow
from datasets.cityflow_video import discover_camera_dirs
from datasets.config import cityflow_root
from eval.ablation import run_ablation
from eval.hard_negatives import mine_pairs

CACHE = Path("data/ablate3d_signatures.json")


@dataclass(frozen=True)
class _Img:
    """The shape run_ablation expects, over CityFlow records.

    CityFlow publishes no body-type label, so that field is empty and
    `_attrs_contradict` skips it — only colour and the 3D keys can veto here.
    """

    vehicle_id: str
    camera_id: str
    color: str
    body_type: str = ""


def reconstruct_attrs(crops, records, cache_path: Path, limit_px: int = 64):
    """{crop index -> geometry attrs}, cached on disk between runs.

    Cached because this is the expensive half by three orders of magnitude: a
    crash four hours in should not cost four hours. Keyed by vehicle/camera/
    index so a re-run with a different subsample reuses what it can.
    """
    # Cache validity across the 2026-08-01 reconstruction changes: checked, and
    # unaffected. LightGlue, the landmark radius and the registration gate only
    # apply from the SECOND view onward, and this loop fuses exactly one crop
    # per model, so registration never runs. prune_opacity and densify_reach do
    # apply to the bootstrap, so they were measured directly on 30 real S01
    # crops under both settings: 0.05/8 gave 7/30 usable at median observed
    # fraction 0.095, and 0.12/5 gave 7/30 at 0.094 — a delta of zero crops.
    # The cached signatures therefore still describe the current pipeline.
    cached = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    pipeline = build_pipeline()
    out: dict[int, dict[str, str]] = {}
    todo = [i for i in range(len(crops))]
    t0 = time.time()
    done = 0
    for i in todo:
        rec = records[i]
        key = f"{rec.vehicle_id}|{rec.camera_id}|{i}"
        if key in cached:
            out[i] = cached[key]
            continue
        crop = crops[i]
        if min(crop.shape[:2]) < limit_px:
            cached[key] = {}
            out[i] = {}
            continue
        model = Target3DModel(f"ablate3d-{key.replace('|', '-')}",
                              Path("data/ablate3d_models"), pipeline=pipeline)
        try:
            outcome = model.fuse_confirmed_crop(
                crop, key, reason="3D ablation", export=False)
            attrs = signature_to_attrs(outcome.geometry)
        except InsufficientDetail:
            attrs = {}
        except Exception as exc:                      # noqa: BLE001
            print(f"  reconstruction failed for {key}: {type(exc).__name__}: {exc}")
            attrs = {}
        cached[key] = attrs
        out[i] = attrs
        done += 1
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached, indent=1))
        if done % 5 == 0:
            rate = (time.time() - t0) / done
            left = (len(todo) - len(out)) * rate
            print(f"  reconstructed {done} new ({len(out)}/{len(todo)} total), "
                  f"{rate:.0f}s each, ~{left/60:.0f} min left")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="S01")
    ap.add_argument("--vehicles", type=int, default=40,
                    help="how many ground-truth vehicles to keep (default 40; "
                         "cost is ~98s per crop, ~4 crops per vehicle)")
    ap.add_argument("--embedder", choices=["osnet", "fastreid"], default="osnet")
    ap.add_argument("--out", default="data/ablation3d_s01.json")
    args = ap.parse_args()

    root = cityflow_root()
    scen = CityFlow(root).load_scenario(args.scenario)
    cams = discover_camera_dirs(root, args.scenario)
    records, crops = collect_crops(scen, cams)
    print(f"{args.scenario}: {len(crops)} crops, "
          f"{len({r.vehicle_id for r in records})} vehicles")

    keep_ids = sorted({r.vehicle_id for r in records})[:args.vehicles]
    idx = [i for i, r in enumerate(records) if r.vehicle_id in keep_ids]
    records = [records[i] for i in idx]
    crops = [crops[i] for i in idx]
    print(f"subsample: {len(crops)} crops over {len(keep_ids)} vehicles "
          f"(deterministic: first {args.vehicles} ids in sorted order)")

    if args.embedder == "fastreid":
        from perception.fastreid_backbone import FastReidEmbedder
        embedder = FastReidEmbedder()
    else:
        from perception.embedder import ReidEmbedder
        embedder = ReidEmbedder()
    emb = np.vstack([embedder.embed_batch(crops[i:i + 32])
                     for i in range(0, len(crops), 32)])

    print(f"\nreconstructing (~98s each; cached in {CACHE})")
    attrs_by_index = reconstruct_attrs(crops, records, CACHE)
    with_geom = sum(1 for a in attrs_by_index.values() if a)
    print(f"usable geometry on {with_geom}/{len(crops)} crops "
          f"({100*with_geom/max(len(crops),1):.0f}%) — the rest were too small "
          f"or their signature was not trustworthy")

    q_idx, g_idx = split_query_gallery(records, emb)
    q_imgs = [_Img(records[i].vehicle_id, records[i].camera_id, records[i].color)
              for i in q_idx]
    g_imgs = [_Img(records[i].vehicle_id, records[i].camera_id, records[i].color)
              for i in g_idx]
    q_emb, g_emb = emb[q_idx], emb[g_idx]

    pairs = mine_pairs(
        [_Img(records[i].vehicle_id, records[i].camera_id, records[i].color)
         for i in g_idx], g_emb)
    threshold = build_report(pairs).chosen_threshold
    print(f"threshold {threshold:.3f} (fitted on this subsample's gallery pairs)")

    def geom_attrs(kind: str, i: int) -> dict:
        src = q_idx if kind == "query" else g_idx
        return attrs_by_index.get(src[i], {})

    rows = {}
    for label, hook in (("2D only (colour)", None), ("+ 3D geometry", geom_attrs)):
        metrics, cases = run_ablation(q_imgs, q_emb, g_imgs, g_emb,
                                      threshold=threshold, extra_attrs=hook)
        cas = next(m for m in metrics if m.name == "cascade")
        rows[label] = cas.row()
        vetoes = sum(1 for c in cases if "veto" in c.reason)
        rows[label]["attribute_vetoes"] = vetoes
        print(f"{label:18} precision={cas.precision:.1%} recall={cas.recall:.1%} "
              f"alerts={cas.alerts} fp={cas.false_positives} "
              f"reviews={cas.reviews} vetoes={vetoes}")

    summary = {
        "scenario": args.scenario, "embedder": args.embedder,
        "vehicles_kept": len(keep_ids), "crops": len(crops),
        "crops_with_geometry": with_geom, "threshold": threshold,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
