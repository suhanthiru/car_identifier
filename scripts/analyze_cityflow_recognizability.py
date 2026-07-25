"""Offline recognizability analysis (no server, no HTTP replay): for every
CityFlow vehicle with >=2 real camera passages, evaluate whether flagging
it from its FIRST passage lets the cascade react to its OWN SUBSEQUENT
passages -- in isolation, one target at a time, no other flagged profiles
competing for rank_candidates.best.

This is the correction to scripts/sweep_cityflow.py's batch run: flagging
all 95 vehicles simultaneously meant only ONE target could ever "win" the
top slot per real sighting, so that sweep mostly measured 95-way
look-alike auction noise, not per-vehicle recognizability. Evaluating one
target at a time answers the actually relevant question for picking a
demo car: "if an operator flags THIS car, how much of its own real
cross-camera story does the system pick up."

Simulates a diligent operator: whenever a subsequent passage crosses the
reaction threshold (and isn't vetoed), the profile is folded in via
TargetProfile.with_sighting -- the same effect as accepting the review --
so later passages see a growing gallery and an up-to-date last_seen for
the transit check, exactly like a real session where the operator accepts
each review as it appears.

    python scripts/analyze_cityflow_recognizability.py [--scenario S01]

Ground truth only groups a vehicle's own real passages and scores the
result afterward -- it never enters the cascade itself. Color, plate OCR,
and embeddings all come from RealPerceptor exactly as they would in
server/real_feed.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from datasets.cityflow import CityFlow
from datasets.cityflow_video import discover_camera_dirs, vehicle_frame_spans
from datasets.config import cityflow_root
from perception.embedder import ReidEmbedder
from perception.fastreid_backbone import FastReidEmbedder
from perception.plates import FastPlateOcrReader
from perception.real_observe import RealPerceptor
from reasoning.cascade import (
    VERDICT_CANDIDATE, VERDICT_CONFIRMED, VERDICT_LIKELY, VERDICT_REJECTED,
    VERDICT_UNDECIDED, CascadeConfig, evaluate,
)

# Verdicts that would actually surface to an operator as a proposed match.
# REJECTED is deliberately excluded: on a true passage it is a miss, and on
# an impostor it is the system correctly saying no -- lumping it in with
# "reacted" hides both.
PROPOSING_VERDICTS = (VERDICT_LIKELY, VERDICT_CONFIRMED, VERDICT_CANDIDATE)
from reasoning.profile import LastSeen, profile_from_flag


def build_cascade_config(scenario_name: str, embedder_tag: str) -> CascadeConfig:
    suffix = "" if embedder_tag == "osnet" else f"_{embedder_tag}"
    artifact = Path(f"calibration/artifacts/cityflow_{scenario_name.lower()}{suffix}.json")
    if not artifact.is_file():
        print("no per-deployment calibration found; using uncalibrated-linear")
        return CascadeConfig()
    from calibration.isotonic import load_model, make_reid_prob_fn

    prob_fn, label = make_reid_prob_fn(load_model(artifact))
    print(f"using calibration {label} ({artifact})")
    return CascadeConfig(reid_prob_fn=prob_fn, reid_calibration_label=label)


def _aggregate_query_embedding(embedder, obs) -> np.ndarray:
    """Mean-pool the passage's multiple real clip frames into one query
    embedding (renormalized) -- standard multi-shot query aggregation,
    reduces single-frame noise (blur/pose/partial occlusion) instead of
    betting everything on the one midpoint crop."""
    frames = list(obs.clip_frames) or [obs.crop]
    if len(frames) == 1:
        return obs.embedding
    embs = embedder.embed_batch(frames)
    mean = embs.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else obs.embedding


def analyze_vehicle(vid, spans, camera_dirs, graph, perceptor, cascade_config,
                    embedder, multi_crop: bool, obs_cache=None, seed_cache=None):
    spans = sorted(spans, key=lambda s: s.enter_s)
    seed = spans[0]
    seed_frames = vehicle_frame_spans(camera_dirs[seed.camera_id] / "gt" / "gt.txt")
    fr = seed_frames.get(vid)
    if fr is None:
        return None
    obs0 = perceptor.process(seed.camera_id, vid, (fr[0] + fr[1]) // 2,
                             (seed.enter_s + seed.exit_s) / 2)
    if obs0 is None:
        return None
    # Multi-shot GALLERY: keep each of the seed passage's real clip frames as
    # its own gallery entry (not averaged) so max_similarity can pick
    # whichever crop best matches a later sighting -- mirrors the real
    # flag-seeding path (server/api.py seeds first/mid/last crops too).
    seed_gallery = (tuple(embedder.embed_batch(list(obs0.clip_frames)))
                    if multi_crop and obs0.clip_frames else (obs0.embedding,))
    profile = dataclasses.replace(
        profile_from_flag(f"v{vid}", f"vehicle {vid}", "", obs0.class_attrs, {}),
        gallery=seed_gallery,
        # A flag IS a sighting: the operator pointed at this vehicle on this
        # camera at this time. Leaving last_seen unset (the old behaviour)
        # silently disabled the whole space-time tier for every real-data
        # target, and deadlocked it: last_seen is only written on
        # association, association needs the score bar, and the score bar is
        # what space-time was supposed to help clear.
        last_seen=LastSeen(seed.camera_id, obs0.timestamp_s, obs0.event_id))
    if seed_cache is not None:
        # The pristine seeded profile, kept for the impostor pass so the
        # false-positive measurement is independent of how much the profile
        # happened to evolve during the recall pass.
        seed_cache[vid] = (profile, obs0.class_attrs.get("color", ""))

    evaluated, reacted, proposed = 0, 0, 0
    cameras_reacted: set[str] = set()
    trace: list[dict] = []
    for span in spans[1:]:
        frames = vehicle_frame_spans(camera_dirs[span.camera_id] / "gt" / "gt.txt")
        fr = frames.get(vid)
        if fr is None:
            continue
        obs = perceptor.process(span.camera_id, vid, (fr[0] + fr[1]) // 2,
                                (span.enter_s + span.exit_s) / 2)
        if obs is None:
            continue
        evaluated += 1
        query_obs = obs
        if multi_crop:
            query_emb = _aggregate_query_embedding(embedder, obs)
            query_obs = dataclasses.replace(obs, embedding=query_emb)
        if obs_cache is not None:
            # Drop the pixel payload: the impostor pass only re-runs
            # evaluate(), which needs the embedding/attrs/plate, not images.
            obs_cache[(vid, span.camera_id)] = dataclasses.replace(
                query_obs, crop=None, clip_frames=())
        decision = evaluate(query_obs, profile, graph, cascade_config)
        trace.append({
            "camera": span.camera_id, "t": round(span.enter_s, 1),
            "verdict": decision.verdict, "score": round(decision.score, 3),
            "distinctiveness": round(decision.distinctiveness, 3),
            "refused_to_individuate": decision.refused_to_individuate,
        })
        if decision.verdict in PROPOSING_VERDICTS:
            proposed += 1
        if decision.verdict != VERDICT_UNDECIDED:
            reacted += 1
            cameras_reacted.add(span.camera_id)
            if decision.verdict != VERDICT_REJECTED:
                learned_plate = ""
                if obs.plate is not None and "_" not in obs.plate.text:
                    learned_plate = obs.plate.text
                profile = profile.with_sighting(
                    query_obs.embedding, LastSeen(span.camera_id, span.enter_s, obs.event_id),
                    learned_plate=learned_plate)

    return {
        "vehicle_id": vid,
        "total_ground_truth_passages": len(spans),
        "seed_camera": seed.camera_id,
        "evaluated_subsequent_passages": evaluated,
        "reacted_subsequent_passages": reacted,
        "proposed_subsequent_passages": proposed,
        "distinct_cameras_reacted": len(cameras_reacted),
        "distinct_cameras_including_seed": len(cameras_reacted | {seed.camera_id}),
        "final_gallery_size": len(profile.gallery),
        "final_plate": profile.plate,
        "trace": trace,
    }


def impostor_pass(seed_cache, obs_cache, graph, cascade_config,
                  per_vehicle: int = 12, seed: int = 17) -> dict:
    """False-positive rate: does a target's profile propose matches against
    OTHER vehicles' passages?

    Without this axis the recall number is uninterpretable -- a 0% reaction
    rate scores identically to a perfect system and to one that has simply
    stopped working, and any change that makes the system react to
    everything would read as progress.

    Impostors are sampled with a deliberate bias toward the SAME estimated
    color as the target, because that is the hard case; sampling uniformly
    would mostly draw obviously-different vehicles and flatter the result.
    Same-color and different-color rates are reported separately so the
    bias cannot hide inside a single averaged number.
    """
    rng = np.random.default_rng(seed)
    colors = {vid: color for vid, (_, color) in seed_cache.items()}

    totals = {"same_color": [0, 0], "diff_color": [0, 0]}   # [evaluations, proposals]
    worst: list[dict] = []
    for vid, (profile, color) in sorted(seed_cache.items()):
        others = [k for k in obs_cache if k[0] != vid]
        same = [k for k in others if colors.get(k[0]) == color]
        diff = [k for k in others if colors.get(k[0]) != color]
        # Two thirds of the budget on same-color impostors, but never more
        # than exist; the remainder from the easy pool as a control.
        n_same = min(len(same), (per_vehicle * 2) // 3)
        n_diff = min(len(diff), per_vehicle - n_same)
        picked = (
            [same[int(i)] for i in rng.choice(len(same), n_same, replace=False)]
            + [diff[int(i)] for i in rng.choice(len(diff), n_diff, replace=False)]
        )
        for key in picked:
            bucket = "same_color" if colors.get(key[0]) == color else "diff_color"
            decision = evaluate(obs_cache[key], profile, graph, cascade_config)
            totals[bucket][0] += 1
            if decision.verdict in PROPOSING_VERDICTS:
                totals[bucket][1] += 1
                worst.append({
                    "target_vehicle": vid, "impostor_vehicle": key[0],
                    "impostor_camera": key[1], "verdict": decision.verdict,
                    "score": round(decision.score, 3), "bucket": bucket,
                })

    evaluations = totals["same_color"][0] + totals["diff_color"][0]
    proposals = totals["same_color"][1] + totals["diff_color"][1]
    def _rate(pair):
        return round(pair[1] / pair[0], 4) if pair[0] else None
    return {
        "impostor_evaluations": evaluations,
        "impostor_proposals": proposals,
        "false_positive_rate": round(proposals / evaluations, 4) if evaluations else None,
        "same_color": {"evaluations": totals["same_color"][0],
                       "proposals": totals["same_color"][1],
                       "rate": _rate(totals["same_color"])},
        "diff_color": {"evaluations": totals["diff_color"][0],
                       "proposals": totals["diff_color"][1],
                       "rate": _rate(totals["diff_color"])},
        "examples": sorted(worst, key=lambda w: -w["score"])[:15],
    }


def summarize(results: list[dict]) -> None:
    single_camera = [r for r in results if r["total_ground_truth_passages"] < 2]
    multi = [r for r in results if r["total_ground_truth_passages"] >= 2]
    scored = [r for r in multi if r["evaluated_subsequent_passages"] > 0]
    print(f"\n{len(results)} vehicles total: {len(single_camera)} only ever "
          f"appear on one camera (cannot demo cross-camera corroboration), "
          f"{len(multi)} cross multiple cameras")
    if not scored:
        print("no multi-camera vehicle had an evaluable subsequent passage")
        return

    never_reacted = [r for r in scored if r["reacted_subsequent_passages"] == 0]
    print(f"of {len(scored)} multi-camera vehicles with a usable trace, "
          f"{len(scored) - len(never_reacted)} reacted to at least one "
          f"subsequent passage of their own "
          f"({100 * (len(scored) - len(never_reacted)) / len(scored):.0f}%)")

    reacting = [r for r in scored if r["reacted_subsequent_passages"] > 0]
    if not reacting:
        print("none reacted -- nothing to rank")
        return
    scores = [r["distinct_cameras_including_seed"] for r in reacting]
    q75 = float(np.percentile(scores, 75))
    print(f"\ndistinct-cameras-including-seed distribution among reacting "
          f"vehicles: min={min(scores)} median={np.median(scores):.1f} "
          f"p75={q75:.1f} max={max(scores)}")

    ranked = sorted(reacting, key=lambda r: (
        r["distinct_cameras_including_seed"], r["reacted_subsequent_passages"]),
        reverse=True)
    upper_quartile = [r for r in ranked
                      if r["distinct_cameras_including_seed"] >= q75]
    print(f"\n{len(upper_quartile)} vehicles at/above the 75th percentile "
          f"-- the demo-candidate pool (deliberately not the single best "
          f"case):")
    for r in upper_quartile[:20]:
        print(f"  vehicle {r['vehicle_id']:>3}: seed={r['seed_camera']} "
              f"+{r['reacted_subsequent_passages']}/{r['evaluated_subsequent_passages']} "
              f"subsequent reactions across "
              f"{r['distinct_cameras_including_seed']} distinct cameras, "
              f"gallery grew to {r['final_gallery_size']}, "
              f"plate={r['final_plate'] or 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="S01")
    parser.add_argument("--embedder", choices=["osnet", "fastreid"], default="osnet")
    parser.add_argument("--multi-crop", action="store_true",
                        help="mean-pool query frames + multi-entry gallery "
                             "from each passage's real clip frames, instead "
                             "of a single midpoint crop")
    parser.add_argument("--impostors-per-vehicle", type=int, default=12,
                        help="impostor passages evaluated per target for the "
                             "false-positive rate (2/3 same-color, the hard case)")
    args = parser.parse_args()

    root = cityflow_root()
    if not CityFlow.exists(root):
        raise SystemExit(f"CityFlow not found at {root}; see DATASETS.md.")
    scenario = CityFlow(root).load_scenario(args.scenario)
    camera_dirs = discover_camera_dirs(root, args.scenario)
    graph = scenario.to_road_graph()
    cascade_config = build_cascade_config(args.scenario, args.embedder)

    by_vehicle: dict[int, list] = {}
    for span in scenario.spans:
        by_vehicle.setdefault(span.vehicle_id, []).append(span)
    print(f"{len(by_vehicle)} vehicles in {args.scenario}, embedder={args.embedder}, "
          f"multi_crop={args.multi_crop}")

    embedder = FastReidEmbedder() if args.embedder == "fastreid" else ReidEmbedder()
    plate_reader = FastPlateOcrReader()
    perceptor = RealPerceptor(camera_dirs, scenario.camera_gps(),
                              pipeline_state=SimpleNamespace(enable_plate_ocr=True),
                              embedder=embedder, plate_reader=plate_reader)

    t0 = time.monotonic()
    results = []
    obs_cache: dict[tuple[int, str], object] = {}
    seed_cache: dict[int, tuple] = {}
    try:
        for vid, spans in sorted(by_vehicle.items()):
            r = analyze_vehicle(vid, spans, camera_dirs, graph, perceptor,
                               cascade_config, embedder, args.multi_crop,
                               obs_cache=obs_cache, seed_cache=seed_cache)
            if r is not None:
                results.append(r)
            else:
                results.append({
                    "vehicle_id": vid, "total_ground_truth_passages": len(spans),
                    "seed_camera": None, "evaluated_subsequent_passages": 0,
                    "reacted_subsequent_passages": 0,
                    "proposed_subsequent_passages": 0, "distinct_cameras_reacted": 0,
                    "distinct_cameras_including_seed": 0, "final_gallery_size": 0,
                    "final_plate": "", "trace": [],
                })
    finally:
        perceptor.close()
    print(f"analyzed {len(results)} vehicles in {time.monotonic() - t0:.0f}s")

    # Precision axis. Cheap: re-runs evaluate() over cached observations, no
    # video decoding or re-embedding.
    impostors = impostor_pass(seed_cache, obs_cache, graph, cascade_config,
                              per_vehicle=args.impostors_per_vehicle)

    summarize(results)

    # Recall and false-positive rate must be read together: either alone is
    # trivially optimizable in the wrong direction.
    evaluated = sum(r["evaluated_subsequent_passages"] for r in results)
    proposed = sum(r["proposed_subsequent_passages"] for r in results)
    print(f"\n--- two-sided operational metric ---")
    print(f"RECALL   : {proposed}/{evaluated} genuine cross-camera passages "
          f"produced a proposed match "
          f"({100 * proposed / evaluated:.1f}%)" if evaluated else
          "RECALL   : no evaluable passages")
    fpr = impostors["false_positive_rate"]
    print(f"PRECISION: {impostors['impostor_proposals']}/"
          f"{impostors['impostor_evaluations']} impostor passages falsely "
          f"proposed ({100 * fpr:.1f}% FPR)" if fpr is not None else
          "PRECISION: no impostor evaluations")
    for bucket, label in (("same_color", "same estimated color (hard)"),
                          ("diff_color", "different color (control)")):
        b = impostors[bucket]
        if b["rate"] is not None:
            print(f"    {label:<28}: {b['proposals']}/{b['evaluations']} "
                  f"({100 * b['rate']:.1f}%)")
    if impostors["examples"]:
        print(f"  worst false proposals:")
        for e in impostors["examples"][:5]:
            print(f"    target v{e['target_vehicle']} <- impostor "
                  f"v{e['impostor_vehicle']} @{e['impostor_camera']} "
                  f"score {e['score']} ({e['verdict']}, {e['bucket']})")

    suffix = "" if args.embedder == "osnet" else f"_{args.embedder}"
    if args.multi_crop:
        suffix += "_multicrop"
    out = Path(f"data/recognizability_{args.scenario.lower()}{suffix}.json")
    out.write_text(json.dumps({
        "scenario": args.scenario, "embedder": args.embedder,
        "multi_crop": args.multi_crop,
        "recall": {"evaluated_passages": evaluated, "proposed": proposed,
                   "rate": round(proposed / evaluated, 4) if evaluated else None},
        "impostors": impostors,
        "per_vehicle": results,
    }, indent=2))
    print(f"\nfull per-vehicle traces: {out}")


if __name__ == "__main__":
    main()
