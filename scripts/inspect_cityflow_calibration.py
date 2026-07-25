"""Diagnose the CityFlow S01 per-deployment ReID calibration: is the flat,
conservative curve a genuine property of this data, or an artifact of the
small (795-crop) sample?

    python scripts/inspect_cityflow_calibration.py [--scenario S01]

Answers two separate questions:
1. RAW SEPARABILITY (independent of any fit or sample size): do same-
   vehicle and different-vehicle-same-color-bucket similarities actually
   separate on this data? Reported as an AUC-like statistic (fraction of
   (positive, hard-negative) pairs where the positive scores higher) --
   0.5 = no separation at all, 1.0 = perfect.
2. SAMPLE-SIZE STABILITY: refit the isotonic curve on bootstrap subsamples
   (50%/75%/100% of the mined pairs) and compare the resulting curve and
   chosen threshold. A curve that swings a lot under subsampling means
   "not enough data yet"; a curve that barely moves means the flat region
   is real, not noise.

Writes eval/figures/cityflow_{scenario}_similarity_dist.png (raw
separability), reuses eval/plots.plot_reliability/plot_pr_sweep for the
already-fitted curve, and data/cityflow_{scenario}_pairs.json (the raw
mined pairs, for any follow-up analysis without re-embedding).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from calibration.isotonic import build_report
from calibrate_cityflow import collect_crops
from datasets.cityflow import CityFlow
from datasets.cityflow_video import discover_camera_dirs
from datasets.config import cityflow_root
from eval.hard_negatives import mine_pairs
from eval.plots import (
    plot_pr_sweep, plot_reliability, plot_similarity_distributions,
)
from eval.reliability import compute_reliability
from eval.separability import evaluate_separability, separability_auc
from perception.embedder import ReidEmbedder
from perception.fastreid_backbone import FastReidEmbedder


def bootstrap_stability(pairs, n_boot: int = 8, seed: int = 0) -> list[dict]:
    """Refit on random subsamples at decreasing fractions; report how much
    the fitted curve (sampled at a fixed grid) and chosen threshold move."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.5, 0.95, 10)
    full_report = build_report(pairs)
    full_curve = np.array([full_report.model.predict(g) for g in grid])

    out = []
    for frac in (1.0, 0.75, 0.5):
        thresholds, curve_devs = [], []
        for _ in range(n_boot if frac < 1.0 else 1):
            if frac < 1.0:
                idx = rng.choice(len(pairs), size=int(len(pairs) * frac), replace=False)
                sample = [pairs[i] for i in idx]
            else:
                sample = pairs
            try:
                report = build_report(sample)
            except ValueError:
                continue
            curve = np.array([report.model.predict(g) for g in grid])
            curve_devs.append(float(np.mean(np.abs(curve - full_curve))))
            thresholds.append(report.chosen_threshold)
        out.append({
            "fraction": frac, "n_pairs": int(len(pairs) * frac),
            "mean_curve_deviation_from_full_fit": (
                round(float(np.mean(curve_devs)), 4) if curve_devs else None),
            "chosen_threshold_range": (
                [round(min(thresholds), 3), round(max(thresholds), 3)]
                if thresholds else None),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="S01")
    parser.add_argument("--embedder", choices=["osnet", "fastreid"], default="osnet",
                        help="osnet = current default (ImageNet-pretrained, "
                             "never vehicle-finetuned); fastreid = VeRi-776 "
                             "vehicle-ReID-finetuned checkpoint")
    parser.add_argument("--per-bucket", type=int, default=40,
                        help="negatives sampled per appearance bucket in the "
                             "fair/adversarial separability protocols")
    args = parser.parse_args()

    root = cityflow_root()
    if not CityFlow.exists(root):
        raise SystemExit(f"CityFlow not found at {root}; see DATASETS.md.")
    scenario = CityFlow(root).load_scenario(args.scenario)
    camera_dirs = discover_camera_dirs(root, args.scenario)

    records, crops = collect_crops(scenario, camera_dirs)
    print(f"{args.scenario}: {len(crops)} real crops, "
          f"{len({r.vehicle_id for r in records})} vehicles, "
          f"embedder={args.embedder}")
    embedder = FastReidEmbedder() if args.embedder == "fastreid" else ReidEmbedder()
    embeddings = embedder.embed_batch(crops)

    pairs = mine_pairs(records, embeddings)
    positives = [p for p in pairs if p.same_vehicle]
    hard_negs = [p for p in pairs if p.hard_negative]
    random_negs = [p for p in pairs if not p.same_vehicle and not p.hard_negative]
    print(f"mined {len(pairs)} pairs: {len(positives)} positive, "
          f"{len(hard_negs)} hard negative, {len(random_negs)} random negative")

    pos_sims = [p.similarity for p in positives]
    hard_sims = [p.similarity for p in hard_negs]
    rand_sims = [p.similarity for p in random_negs]
    print(f"\npositive similarity:      mean={np.mean(pos_sims):.3f} "
          f"median={np.median(pos_sims):.3f} std={np.std(pos_sims):.3f}")
    print(f"hard-negative similarity: mean={np.mean(hard_sims):.3f} "
          f"median={np.median(hard_sims):.3f} std={np.std(hard_sims):.3f}")
    print(f"random-negative similarity: mean={np.mean(rand_sims):.3f} "
          f"median={np.median(rand_sims):.3f} std={np.std(rand_sims):.3f}")

    auc_hard = separability_auc(pos_sims, hard_sims)
    auc_random = separability_auc(pos_sims, rand_sims)
    print(f"\nseparability of the CALIBRATION sample "
          f"(AUC, 0.5=no separation, 1.0=perfect):")
    print(f"  positives vs hard negatives (same color/body bucket): {auc_hard:.3f}")
    print(f"  positives vs random negatives (any bucket):           {auc_random:.3f}")
    print(f"  NOTE: these use mine_pairs' calibration sample -- top-k mined")
    print(f"  negatives vs uniformly-sampled positives. Right for FITTING the")
    print(f"  curve, biased for JUDGING the embedder. See the fair split below.")

    # The honest embedder judgement: cross-camera positives (the real task)
    # against uniformly-sampled same-bucket negatives, with the top-k number
    # reported beside it so the selection bias is visible rather than implied.
    sep, sep_pairs = evaluate_separability(records, embeddings,
                                           per_bucket=args.per_bucket)
    sep_summary = sep.summary()
    print(f"\nseparability, FAIR vs ADVERSARIAL protocols "
          f"(eval/separability.py):")
    print(f"  fair        (cross-camera positives vs uniform same-bucket "
          f"negatives): {sep_summary['fair_auc']:.3f}")
    print(f"  adversarial (cross-camera positives vs top-k mined "
          f"negatives):   {sep_summary['adversarial_auc']:.3f}")
    if sep_summary["same_camera_auc"] is not None:
        print(f"  same-camera ceiling (near-duplicate frames, sanity check):     "
              f"{sep_summary['same_camera_auc']:.3f}")
    print(f"  look-alike difficulty gap (fair - adversarial):                "
          f"{sep_summary['difficulty_gap']:.3f}")
    print(f"  [{sep_summary['n_cross_camera_positives']} cross-camera positives, "
          f"{sep_summary['n_same_camera_positives']} same-camera, "
          f"{sep_summary['n_uniform_negatives']} uniform negatives, "
          f"{sep_summary['n_topk_negatives']} top-k negatives]")

    print(f"\nbootstrap stability (refit on subsamples, compare to full fit):")
    stability = bootstrap_stability(pairs)
    for s in stability:
        print(f"  {int(s['fraction']*100)}% of pairs (~{s['n_pairs']}): "
              f"mean curve deviation={s['mean_curve_deviation_from_full_fit']}, "
              f"chosen_threshold range={s['chosen_threshold_range']}")

    report = build_report(pairs, note=(
        f"Diagnostic refit for inspection, same data as "
        f"scripts/calibrate_cityflow.py --scenario {args.scenario}."))
    rel = compute_reliability(pairs, report.model)
    print(f"\nreliability: ECE={rel.ece:.4f} over {rel.n_pairs} pairs, "
          f"chosen_threshold={report.chosen_threshold:.3f}, "
          f"hard_negative_fpr_at_threshold={report.hard_negative_fpr_at_threshold:.3f}")

    tag = args.scenario.lower()
    if args.embedder != "osnet":
        tag = f"{tag}_{args.embedder}"
    dist_path = plot_similarity_distributions(
        pos_sims, hard_sims, rand_sims, f"cityflow_{tag}_similarity_dist.png",
        f"CityFlow {args.scenario}: raw similarity separability "
        f"(AUC vs hard negatives = {auc_hard:.2f})")
    # Same three-way plot under the fair protocol, so the two pictures can be
    # put side by side: identical embeddings, different negative selection.
    fair_dist_path = plot_similarity_distributions(
        [p.similarity for p in sep_pairs["cross_camera_positives"]],
        [p.similarity for p in sep_pairs["topk_negatives"]],
        [p.similarity for p in sep_pairs["uniform_negatives"]],
        f"cityflow_{tag}_fair_dist.png",
        f"CityFlow {args.scenario}: cross-camera positives "
        f"(fair AUC {sep_summary['fair_auc']:.2f} / adversarial {sep_summary['adversarial_auc']:.2f})")
    rel_path = plot_reliability(rel.bins, rel.ece, f"cityflow_{tag}_reliability.png")
    sweep_path = plot_pr_sweep(report.sweep, report.chosen_threshold,
                               f"cityflow_{tag}_sweep.png")
    print(f"\nplots written: {dist_path}, {fair_dist_path}, {rel_path}, {sweep_path}")

    summary_path = Path(f"data/separability_{tag}.json")
    summary_path.write_text(json.dumps({
        "scenario": args.scenario, "embedder": args.embedder,
        "n_crops": len(crops), "n_vehicles": len({r.vehicle_id for r in records}),
        "calibration_sample": {
            "auc_vs_hard_negatives": round(auc_hard, 4),
            "auc_vs_random_negatives": round(auc_random, 4),
        },
        "fair_protocol": sep_summary,
        "reliability_ece": round(rel.ece, 4),
        "chosen_threshold": round(report.chosen_threshold, 4),
    }, indent=1))
    print(f"separability summary: {summary_path}")

    pairs_out = Path(f"data/cityflow_{tag}_pairs.json")
    pairs_out.write_text(json.dumps([
        {"similarity": p.similarity, "same_vehicle": p.same_vehicle,
         "hard_negative": p.hard_negative, "bucket": p.bucket}
        for p in pairs], indent=1))
    print(f"raw pairs: {pairs_out}")


if __name__ == "__main__":
    main()
