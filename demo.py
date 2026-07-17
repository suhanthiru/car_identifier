"""One-command end-to-end demo. Everything on screen is synthetic.

    python demo.py [--time-scale 8] [--port 8010] [--no-browser] [--keep-db]

Spins up the central server, flags two demo targets from the look-alike
cluster (one by plate, one by appearance only), then replays the simulated
world through the per-camera edge tasks. Open the printed URL: you should
see contacts moving, the plate target confirming automatically, and the
appearance-only target stacking up review-queue items that wait for you.
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import threading
import time
import webbrowser
from pathlib import Path

import httpx
import uvicorn

from server.feed import FeedConfig, run_feed
from sim.emitter import build_default_world

DB_PATH = Path("data/eyes.sqlite")
CROPS_DIR = Path("data/crops")


def reset_demo_data() -> Path:
    """Wipe the previous run's artifacts (synthetic only). If another
    process holds the DB open, fall back to a fresh timestamped file."""
    db_path = DB_PATH
    if DB_PATH.exists():
        try:
            DB_PATH.unlink()
        except PermissionError:
            db_path = DB_PATH.with_name(f"eyes-{int(time.time())}.sqlite")
            print(f"note: {DB_PATH} is in use elsewhere; using {db_path}")
    if CROPS_DIR.exists():
        shutil.rmtree(CROPS_DIR, ignore_errors=True)
    return db_path


def start_server(port: int, db_path: Path) -> uvicorn.Server:
    from server.api import create_app

    app = create_app(db_url=f"sqlite:///{db_path.as_posix()}")
    config = uvicorn.Config(app, host="127.0.0.1", port=port,
                            log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server


def wait_for_server(base: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/api/cameras", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.3)
    raise RuntimeError("server did not come up in time")


def flag_demo_targets(base: str) -> None:
    world = build_default_world()
    cluster = [v for v in world.fleet if v.lookalike_group == "cluster-1"]
    by_plate, by_marks = cluster[0], cluster[1]
    with httpx.Client() as c:
        c.post(f"{base}/api/targets", json={
            "label": f"{by_plate.color} {by_plate.make} {by_plate.model} — case 12",
            "plate": by_plate.plate,
            "class_attrs": by_plate.class_attrs,
        }).raise_for_status()
        c.post(f"{base}/api/targets", json={
            "label": (f"{by_marks.color} {by_marks.model} w/ "
                      f"{next(iter(by_marks.instance_attrs.values()), 'no marks')}"),
            "plate": "",
            "class_attrs": by_marks.class_attrs,
            "instance_attrs": dict(by_marks.instance_attrs),
        }).raise_for_status()
    print(f"flagged 2 targets from the look-alike cluster "
          f"({len(cluster)} visually identical vehicles are on the road)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-scale", type=float, default=8.0,
                        help="sim seconds per wall second (default 8)")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--keep-db", action="store_true",
                        help="keep the previous run's database")
    args = parser.parse_args()

    db_path = reset_demo_data() if not args.keep_db else DB_PATH

    base = f"http://127.0.0.1:{args.port}"
    server = start_server(args.port, db_path)
    wait_for_server(base)
    print(f"server up at {base}")
    flag_demo_targets(base)
    if not args.no_browser:
        webbrowser.open(base)
    print(f"replaying the synthetic world at {args.time_scale}x...")

    world = build_default_world()
    counts = asyncio.run(run_feed(world, FeedConfig(
        base_url=base, time_scale=args.time_scale)))
    print(f"feed complete: {sum(counts.values())} sightings across "
          f"{len(counts)} cameras")
    print("console stays live for review-queue work — ctrl+c to quit")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.should_exit = True
        print("bye")


if __name__ == "__main__":
    main()
