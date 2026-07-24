"""Full-sweep audit: flag EVERY vehicle in a CityFlow scenario and measure
how the live reasoning pipeline actually reacts to each one's real
sightings -- an honest ranked report, not a cherry-picked demo pick.

    python scripts/sweep_cityflow.py [--scenario S01] [--port 8013]

Distinct from scripts/run_cityflow_console.py (a live, real-time,
single-operator console): this flags all vehicles at once and compresses
the whole scenario into one fast batch pass (large --time-scale). That is
a HARDER, more honest condition than "one operator flags one car in
isolation" -- with every vehicle simultaneously tracked, look-alikes can
draw ambiguous candidate-set reviews against each other, same as a real
multi-target console session would. Cars with no valid ground-truth crop
at their first appearance get flagged label-only (no seeded evidence,
matching what the browse UI would actually do for a thumbnail-less tile)
-- their zero-reaction rate is reported honestly, not filtered out.

Ground-truth vehicle identity is used to score results after the fact,
exactly like eval/ -- never fed into the reasoning path itself.

Writes data/sweep_{scenario}.json and prints a ranked summary, including
the vehicles sitting at the upper quartile of "the system reacted to it"
strength -- the intended pool for the eventual demo pick, deliberately
NOT the single best case.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sqlite3
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import numpy as np
import uvicorn

from datasets.cityflow import CityFlow
from datasets.cityflow_video import discover_camera_dirs
from datasets.config import cityflow_root
from server.real_feed import CityFlowFeedConfig, build_vehicle_index, run_cityflow_feed

DB_PATH_TMPL = "data/eyes-sweep-{scenario}.sqlite"
CROPS_DIR_TMPL = "data/crops-sweep-{scenario}"


def reset(db_path: Path, crops_dir: Path) -> None:
    if db_path.exists():
        db_path.unlink()
    if crops_dir.exists():
        shutil.rmtree(crops_dir, ignore_errors=True)


def start_server(port: int, db_path: Path, crops_dir: Path, root: Path,
                 scenario_name: str, graph, calibration_path: str):
    from server.api import create_app

    app = create_app(
        graph=graph, db_url=f"sqlite:///{db_path.as_posix()}",
        crops_dir=str(crops_dir), world_source="real",
        calibration_path=calibration_path, enable_plate_ocr=True,
        cityflow_root=str(root), cityflow_scenario_name=scenario_name)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, app.state


def wait_for_server(base: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/api/cameras", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.3)
    raise RuntimeError("server did not come up in time")


def flag_all_vehicles(base: str, vehicle_index: list[dict]) -> dict[int, str]:
    """POST /api/targets for every vehicle, exactly like a tile click would
    (thumbnail as reference_crop_b64, remaining passage crops as the
    gallery). Vehicles with no recoverable crop get a label-only flag --
    the same degraded case the browse UI hits for a thumbnail-less tile."""
    target_by_vehicle: dict[int, str] = {}
    with httpx.Client(timeout=60.0) as client:
        for v in vehicle_index:
            body = {"label": f"vehicle {v['vehicle_id']} (real, "
                             f"first seen {v['first_camera']})"}
            if v["thumbnail_b64"]:
                body["reference_crop_b64"] = v["thumbnail_b64"]
                body["reference_gallery_b64"] = v["gallery_b64"]
            resp = client.post(f"{base}/api/targets", json=body)
            resp.raise_for_status()
            target_by_vehicle[v["vehicle_id"]] = resp.json()["target_id"]
    return target_by_vehicle


def score_results(db_path: Path, base: str,
                  target_by_vehicle: dict[int, str],
                  vehicle_index: dict[int, dict]) -> list[dict]:
    live_targets = httpx.get(f"{base}/api/targets", timeout=30.0).json()

    db = sqlite3.connect(db_path)
    event_camera = dict(db.execute(
        "select event_id, camera_id from sightings").fetchall())
    reactions: dict[str, set[str]] = {}   # target_id -> distinct cameras
    review_counts: dict[str, int] = {}
    association_counts: dict[str, int] = {}
    for target_id, event_id, kind in db.execute(
            "select target_id, event_id, 'review' from reviews "
            "union all "
            "select target_id, event_id, 'assoc' from corroboration_chains"):
        cam = event_camera.get(event_id)
        if cam:
            reactions.setdefault(target_id, set()).add(cam)
        if kind == "review":
            review_counts[target_id] = review_counts.get(target_id, 0) + 1
        else:
            association_counts[target_id] = association_counts.get(target_id, 0) + 1
    truth_counts = dict(db.execute(
        "select truth_id, count(*) from sightings group by truth_id").fetchall())
    db.close()

    out = []
    for vid, target_id in target_by_vehicle.items():
        live = live_targets.get(target_id, {})
        cams = sorted(reactions.get(target_id, set()))
        out.append({
            "vehicle_id": vid, "target_id": target_id,
            "had_seed_photo": bool(vehicle_index[vid]["thumbnail_b64"]),
            "ground_truth_sightings": truth_counts.get(str(vid), 0),
            "distinct_cameras_reacted": len(cams),
            "cameras_reacted": cams,
            "review_events": review_counts.get(target_id, 0),
            "association_events": association_counts.get(target_id, 0),
            "final_state": live.get("state", "unknown"),
            "final_belief": round(live.get("belief", 0.0), 3),
        })
    return out


def summarize(results: list[dict]) -> None:
    n = len(results)
    reacted = [r for r in results if r["distinct_cameras_reacted"] > 0]
    confirmed = [r for r in results if r["final_state"] == "confirmed"]
    print(f"\n{n} vehicles flagged, {len(reacted)} produced at least one "
          f"system reaction ({100 * len(reacted) / n:.0f}%), "
          f"{len(confirmed)} reached CONFIRMED ({100 * len(confirmed) / n:.0f}%)")

    no_seed = [r for r in results if not r["had_seed_photo"]]
    if no_seed:
        print(f"{len(no_seed)} vehicles had no recoverable reference crop "
              f"(label-only flag, degraded case): "
              f"{sum(1 for r in no_seed if r['distinct_cameras_reacted'] > 0)} "
              f"of those still reacted")

    ranked = sorted(reacted, key=lambda r: (
        r["distinct_cameras_reacted"], r["association_events"] > 0), reverse=True)
    if not ranked:
        print("no vehicle produced any system reaction -- nothing to rank")
        return
    scores = [r["distinct_cameras_reacted"] for r in ranked]
    q75 = float(np.percentile(scores, 75))
    print(f"\ndistinct-cameras-reacted distribution: min={min(scores)} "
          f"median={np.median(scores):.1f} p75={q75:.1f} max={max(scores)}")

    upper_quartile = [r for r in ranked if r["distinct_cameras_reacted"] >= q75]
    print(f"\n{len(upper_quartile)} vehicles at/above the 75th percentile "
          f"(the honest demo-candidate pool, NOT the single best case):")
    for r in upper_quartile[:15]:
        print(f"  vehicle {r['vehicle_id']:>3} (target {r['target_id']}): "
              f"{r['distinct_cameras_reacted']} cameras reacted "
              f"{r['cameras_reacted']}, {r['review_events']} review(s), "
              f"{r['association_events']} auto-corroboration(s), "
              f"final={r['final_state']} (belief {r['final_belief']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="S01")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--time-scale", type=float, default=300.0,
                        help="compress the whole scenario for a fast batch pass")
    args = parser.parse_args()

    root = cityflow_root()
    if not CityFlow.exists(root):
        raise SystemExit(f"CityFlow not found at {root}; see DATASETS.md.")
    cf = CityFlow(root)
    scenario = cf.load_scenario(args.scenario)
    camera_dirs = discover_camera_dirs(root, args.scenario)
    graph = scenario.to_road_graph()

    db_path = Path(DB_PATH_TMPL.format(scenario=args.scenario.lower()))
    crops_dir = Path(CROPS_DIR_TMPL.format(scenario=args.scenario.lower()))
    reset(db_path, crops_dir)

    calibration = Path(f"calibration/artifacts/cityflow_{args.scenario.lower()}.json")
    calibration_path = str(calibration) if calibration.is_file() else ""
    print(f"reid calibration: {calibration_path or 'uncalibrated-linear fallback'}")

    base = f"http://127.0.0.1:{args.port}"
    server, app_state = start_server(
        args.port, db_path, crops_dir, root, args.scenario, graph, calibration_path)
    wait_for_server(base)
    print(f"server up at {base}")

    print("building vehicle index (real crops from ground truth)...")
    vehicle_list = build_vehicle_index(scenario, camera_dirs)
    vehicle_index = {v["vehicle_id"]: v for v in vehicle_list}
    print(f"{len(vehicle_list)} vehicles in {args.scenario}, "
          f"{sum(1 for v in vehicle_list if v['thumbnail_b64'])} with a "
          f"recoverable reference crop")

    print("flagging every vehicle (this is the slow part -- "
          "real embeddings, sequential)...")
    t0 = time.monotonic()
    target_by_vehicle = flag_all_vehicles(base, vehicle_list)
    print(f"flagged {len(target_by_vehicle)} vehicles in "
          f"{time.monotonic() - t0:.0f}s")

    print(f"replaying the full scenario at {args.time_scale}x "
          f"(no crops saved -- this pass is for scoring only)...")
    t0 = time.monotonic()
    counts = asyncio.run(run_cityflow_feed(
        scenario, camera_dirs, scenario.camera_gps(), app_state,
        CityFlowFeedConfig(base_url=base, time_scale=args.time_scale,
                           send_crops=False)))
    print(f"replay done in {time.monotonic() - t0:.0f}s: "
          f"{sum(counts.values())} sightings across {len(counts)} cameras")

    results = score_results(db_path, base, target_by_vehicle, vehicle_index)
    summarize(results)

    out_path = Path(f"data/sweep_{args.scenario.lower()}.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nfull per-vehicle results: {out_path}")

    server.should_exit = True


if __name__ == "__main__":
    main()
