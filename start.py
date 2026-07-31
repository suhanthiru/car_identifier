"""One command to run everything. Start here.

    python start.py

On a fresh checkout this installs what it needs first (CUDA torch if the
machine has an NVIDIA GPU, the requirements, the cargen 3D bridge if it can
find it) and then runs. See setup_env.py for the details, or run
`python start.py --check` to see what it would do without doing it.

Once the environment is there, it looks at what is actually available on
this machine and runs the best demo it can, rather than failing on a
missing prerequisite:

- CityFlow present  -> real-data console (real footage, real cameras)
- CityFlow absent   -> synthetic world (always works, no downloads)
- cargen present    -> 3D reconstruction panel enabled
- cargen absent     -> everything else still runs

Nothing here needs a GPU. Override any of it with the flags below; run
`python start.py --help` to see them.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Only stdlib and setup_env (also stdlib-only) at module scope: on a fresh
# checkout the third-party imports below would be the first thing to fail,
# and a traceback is a worse answer than an offer to install them.
import setup_env

BANNER = "=" * 66


def _detect_cityflow() -> tuple[bool, str]:
    """(available, human-readable reason)."""
    try:
        from datasets.cityflow import CityFlow
        from datasets.config import cityflow_root
    except Exception as exc:                      # noqa: BLE001 - report, never crash
        return False, f"loader unavailable ({exc})"
    root = cityflow_root()
    if not CityFlow.exists(root):
        return False, f"not found at {root}"
    return True, str(root)


def _detect_cargen() -> tuple[bool, str]:
    """(available, what the 3D panel will actually produce).

    `import trimesh` used to stand in for "real backend available", which was
    wrong in the direction that matters: trimesh is a mesh library, not a
    reconstructor, so the banner claimed real geometry while both generative
    backends were quietly falling back to stubs. Probe the two backends cargen
    would really build — cheap, since each fails on a missing import or a
    missing checkout rather than by loading weights.
    """
    try:
        import cargen  # noqa: F401
    except Exception as exc:                      # noqa: BLE001
        return False, f"not installed ({type(exc).__name__})"

    from cargen import backends

    def _probe(build) -> bool:
        try:
            build()
            return True
        except Exception:                         # noqa: BLE001 — absent == stub
            return False

    prior = _probe(backends.build_prior_generator)
    seg = _probe(backends.build_segmenter)
    if prior and seg:
        return True, "installed, real prior + segmenter"
    if prior or seg:
        got = "prior" if prior else "segmenter"
        return True, f"installed, real {got} only (the other falls back to a stub)"
    return True, ("installed, STUB geometry only — procedural sedan, not a "
                  "reconstruction (provenance and fusion are real)")


def _device_note() -> str:
    """torch is installed as a CPU-only build in some environments, so
    report what will actually be used rather than what is hoped for."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return "torch unavailable"
    if torch.cuda.is_available():
        return f"GPU - {torch.cuda.get_device_name(0)}"
    return f"CPU (torch {torch.__version__}; a CUDA build would use the GPU)"


def _print_plan(mode: str, cityflow: tuple[bool, str], cargen: tuple[bool, str],
                port: int, time_scale: float) -> None:
    print(BANNER)
    print("  EYES EVERYWHERE")
    print(BANNER)
    print(f"  mode        : {mode}")
    print(f"  CityFlow    : {'yes - ' if cityflow[0] else 'no  - '}{cityflow[1]}")
    print(f"  3D (cargen) : {'yes - ' if cargen[0] else 'no  - '}{cargen[1]}")
    print(f"  compute     : {_device_note()}")
    print(f"  speed       : {time_scale}x real time")
    print(f"  console     : http://127.0.0.1:{port}")
    print(f"  inspector   : http://127.0.0.1:{port}/inspector.html")
    print(BANNER)


def _wait_for_server(base: str, timeout_s: float = 60.0) -> None:
    import httpx

    deadline = time.monotonic() + timeout_s
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/api/cameras", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError as exc:
            last = exc
            time.sleep(0.3)
    raise RuntimeError(f"server did not come up within {timeout_s:.0f}s ({last})")


def _instance_suffix() -> str:
    """Per-instance suffix for the runtime data paths, from --instance.

    The database, crops and 3D directories were fixed names, so a second
    console on a second port silently shared all three with the first. Three
    launched together raced each other's `_reset` + `create_all` and two died
    on `sqlite3.OperationalError: table targets already exists` — a raw
    traceback, with nothing to suggest the real cause was another instance.
    Empty by default, so the normal single-console path is unchanged.
    """
    raw = os.environ.get("EYES_INSTANCE", "").strip()
    safe = "".join(c for c in raw if c.isalnum() or c in "-_")
    return f"-{safe}" if safe else ""


def _reset(db_path: Path, crops_dir: Path, targets3d_dir: Path | None = None) -> Path:
    """Clear the previous run so each launch starts clean. Falls back to a
    timestamped DB if another process still holds the file open.

    The 3D models must go too. Target ids restart at `tgt-001` every launch,
    so a surviving `data/targets3d/tgt-001` is silently adopted by whatever
    vehicle happens to be flagged first next time — the new run fuses into the
    old run's asset and reports its splat count, which is how a session with a
    real SF3D backend produced a 20k-splat cloud built by a previous session's
    stub. Same-name reuse across runs is contamination, not caching.
    """
    if db_path.exists():
        try:
            db_path.unlink()
        except PermissionError:
            db_path = db_path.with_name(f"{db_path.stem}-{int(time.time())}.sqlite")
            print(f"  note: previous database still in use; using {db_path.name}")
    for stale in (crops_dir, targets3d_dir):
        if stale is not None and stale.exists():
            shutil.rmtree(stale, ignore_errors=True)
    return db_path


def run_synthetic(port: int, time_scale: float, open_browser: bool) -> None:
    from demo import flag_demo_targets
    from server.api import create_app
    from server.feed import FeedConfig, run_feed
    from sim.emitter import build_default_world

    inst = _instance_suffix()
    crops_dir = f"data/crops{inst}"
    db_path = _reset(Path(f"data/eyes{inst}.sqlite"), Path(crops_dir),
                     Path(f"data/targets3d{inst}"))
    app = create_app(db_url=f"sqlite:///{db_path.as_posix()}",
                     crops_dir=crops_dir,
                     targets3d_dir=f"data/targets3d{inst}")
    server = _serve(app, port)
    base = f"http://127.0.0.1:{port}"
    _wait_for_server(base)
    flag_demo_targets(base)
    if open_browser:
        webbrowser.open(base)
    print(f"\n  replaying the synthetic world at {time_scale}x - watch the console\n")

    _run_until_quit(server, app, lambda: run_feed(
        build_default_world(),
        FeedConfig(base_url=base, time_scale=time_scale),
        pipeline_state=app.state))


def run_cityflow(port: int, time_scale: float, open_browser: bool, scenario: str,
                 root: str) -> None:
    from datasets.cityflow import CityFlow
    from datasets.cityflow_video import discover_camera_dirs
    from server.api import create_app
    from server.real_feed import CityFlowFeedConfig, run_cityflow_feed

    scen = CityFlow(Path(root)).load_scenario(scenario)
    graph = scen.to_road_graph()
    camera_dirs = discover_camera_dirs(Path(root), scenario)
    print(f"  loaded {scenario}: {len(graph.cameras)} real cameras, "
          f"{len(scen.spans)} ground-truth tracks, "
          f"{len(graph.edges)} observed transit routes")

    # Cameras in a scenario start recording at different wall-clock times.
    # Without the offsets every cross-camera elapsed time is wrong in a way
    # that still looks reasonable, so say which case this run is in.
    from datasets.cityflow import timing_offsets_found
    if not timing_offsets_found(Path(root), scenario):
        print(f"  WARNING: no camera timing offsets for {scenario} "
              f"(expected {root}\\cam_timestamp\\{scenario}.txt).\n"
              f"           Cameras do not share a clock; transit times will be "
              f"wrong without them.")

    # Calibration is per-deployment AND per-embedder; if the artifact for this
    # scenario is missing the cascade falls back to an uncalibrated curve and
    # says so, rather than silently borrowing another deployment's numbers.
    calibration = Path(f"calibration/artifacts/cityflow_{scenario.lower()}.json")
    if not calibration.is_file():
        print(f"  note: no calibration for {scenario}; using the uncalibrated "
              f"fallback (run scripts/calibrate_cityflow.py to fit one)")

    # Separate 3D directory per mode, for the same reason crops are separate:
    # a synthetic sprite's model and a real vehicle's model must never share
    # a target id's directory.
    inst = _instance_suffix()
    crops_dir, models_dir = f"data/crops-cityflow{inst}", f"data/targets3d-cityflow{inst}"
    db_path = _reset(Path(f"data/eyes-cityflow{inst}.sqlite"), Path(crops_dir),
                     Path(models_dir))
    app = create_app(
        graph=graph, db_url=f"sqlite:///{db_path.as_posix()}",
        crops_dir=crops_dir, world_source="real",
        targets3d_dir=models_dir,
        calibration_path=str(calibration) if calibration.is_file() else "",
        cityflow_root=root, cityflow_scenario_name=scenario)
    server = _serve(app, port)
    base = f"http://127.0.0.1:{port}"
    _wait_for_server(base)
    if open_browser:
        webbrowser.open(base)
    print(f"\n  replaying {scenario} at {time_scale}x - browse the thumbnails "
          f"and click any car to track it\n")

    def make_feed():
        # Read the scenario off state each time rather than closing over the
        # one loaded at startup: a scenario switch replaces it between runs,
        # and a captured reference would keep replaying the old footage while
        # the map and the browse list showed the new scenario.
        active = app.state.cityflow_scenario
        return run_cityflow_feed(
            active, app.state.cityflow_camera_dirs, active.camera_gps(),
            app.state,
            CityFlowFeedConfig(base_url=base, time_scale=time_scale))

    _run_until_quit(server, app, make_feed)


async def _supervise_feed(app, make_feed, on_idle=None) -> dict:
    """Run the replay, and rerun it from t=0 whenever a reset is requested.

    POST /api/reset only sets a flag: the feed's camera tasks live in this
    loop's event loop while the request is served on a uvicorn worker thread,
    so tearing them down from the handler would cancel tasks mid-POST. Here we
    own them, and can cancel, wipe and restart in a defined order.

    Clearing is bundled with the rewind rather than offered separately. Rewound
    footage beside targets and beliefs earned from the previous pass would show
    a past frame next to present-tense conclusions — exactly what
    /api/feed_control refuses to do by offering pause and not rewind.
    """
    state = app.state
    announced = False
    while True:
        task = asyncio.create_task(make_feed())
        while not task.done():
            if getattr(state, "restart_requested", False):
                task.cancel()
                break
            await asyncio.sleep(0.1)
        try:
            counts = await task
        except asyncio.CancelledError:
            counts = {}
        # The feed finished on its own. Do NOT return: the console stays live
        # and RESTART is most likely to be pressed exactly here, once the
        # operator has watched the whole replay. Returning handed control to a
        # sleep loop that never looked at the flag, so the button did nothing
        # and the client sat out its 20s watchdog — "slow, or simply doesn't
        # work". Keep owning the loop and wait for the request instead.
        if not getattr(state, "restart_requested", False):
            if not announced:
                announced = True
                # The console needs to know too. A finished replay leaves the
                # clock parked at the end with paused=false, which reads as a
                # hang: nothing in the interface said "there is no more tape".
                state.replay_complete = True
                if on_idle is not None:
                    on_idle(counts)
            while not getattr(state, "restart_requested", False):
                await asyncio.sleep(0.1)
        announced = False
        state.replay_complete = False
        print("\n  reset requested - clearing this run and replaying from t=0\n")
        # Let any in-flight 3D fusion actually finish before the directories go.
        #
        # This used to call j.cancel() and move straight on, which is not what
        # the comment claimed and not what happened: cancel() returns False for
        # a future that has already started and does nothing to stop it. A
        # reconstruction takes ~98s on the GPU with SF3D, so reset_runtime then
        # deleted data/targets3d out from under a running exporter — dropping
        # files mid-write and, twice, taking the process down with no Python
        # traceback at all. Drop only the jobs that have not begun; wait for the
        # ones that have.
        jobs = list(getattr(state, "car3d_jobs", {}).values())
        if jobs:
            def _drain() -> None:
                for job in jobs:
                    if job.cancel():
                        continue          # never started; safe to drop
                    try:
                        job.result(timeout=180)
                    except Exception:     # noqa: BLE001 — already reported
                        pass

            print(f"  waiting for {len(jobs)} in-flight 3D fusion(s)...")
            await asyncio.get_running_loop().run_in_executor(None, _drain)
        # A scenario switch is a reset with a different scenario attached: new
        # cameras, new graph, new footage, so nothing from the old run could
        # carry across meaningfully. Swap before reset_runtime so the rebuilt
        # tracker is built against the new graph.
        pending = getattr(state, "pending_scenario", "")
        if pending:
            state.pending_scenario = ""
            print(f"  switching scenario -> {pending}")
            state.load_scenario(pending)
            # Build the browse index here, while the console is showing
            # "restarting", rather than leaving the first request after the
            # reload to pay for it. It decodes three frames per vehicle out of
            # 1080p video (~16s for 95 vehicles) and caches to disk, so this is
            # a one-off per scenario. In an executor because it is blocking
            # work and the server has to keep answering during it.
            from server.real_feed import build_vehicle_index

            print("  building the browse index for the new scenario...")
            await asyncio.get_running_loop().run_in_executor(
                None, build_vehicle_index,
                state.cityflow_scenario, state.cityflow_camera_dirs)
        state.reset_runtime()
        await state.manager.broadcast({
            "type": "reset_done", "run_generation": state.run_generation,
            "scenario": (state.cityflow_scenario.name
                         if state.cityflow_scenario is not None else "")})


def _serve(app, port: int) -> "uvicorn.Server":
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    return server


def _run_until_quit(server: "uvicorn.Server", app, make_feed) -> None:
    """Own the replay for the life of the process.

    The supervisor never returns now, because the console outlives the feed:
    after a replay ends the operator keeps working the queue, and may press
    RESTART at any point. This used to hand off to a plain sleep loop that
    ignored the restart flag, which made the button dead precisely when it was
    most likely to be used.
    """
    def announce(counts: dict) -> None:
        print(f"\n  replay complete: {sum(counts.values())} sightings across "
              f"{len(counts)} cameras.")
        print("  the console stays live - keep working the review queue.")
        print("  press RESTART in the console to replay from t=0; ctrl+c to quit.\n")

    try:
        asyncio.run(_supervise_feed(app, make_feed, on_idle=announce))
    except KeyboardInterrupt:
        server.should_exit = True
        print("  bye")


def _ensure_environment(*, force: bool, refuse: bool) -> bool:
    """Install missing dependencies before the demo needs them.

    Returns False when the caller should stop. Installing several GB without
    asking would be rude, so an interactive run confirms first; a
    non-interactive one (CI, a pipe) refuses and prints the command instead
    of silently blocking on a prompt nobody can answer.
    """
    if refuse or not (force or setup_env.needs_setup()):
        missing = setup_env.missing_core()
        if missing and refuse:
            print(f"missing: {', '.join(missing)}. Drop --no-setup, or:\n"
                  f"  {Path(sys.executable).name} -m pip install -r requirements.txt")
            return False
        return True

    print(BANNER)
    print("  EYES EVERYWHERE - first-run setup")
    print(BANNER)
    setup_env.print_report()
    print(BANNER)
    print("\n  Setting this up means a multi-GB download (torch and friends).")

    if not sys.stdin.isatty():
        print(f"  Not an interactive terminal, so not starting it unasked. Run:\n"
              f"    python setup_env.py\n")
        return False
    try:
        if input("  Install now? [Y/n] ").strip().lower() in ("n", "no"):
            print("  Skipped. `python setup_env.py` when you're ready.")
            return False
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return False

    if not setup_env.bootstrap():
        print("\n  Setup did not complete; see the pip output above.")
        return False

    # The freshly-installed packages are invisible to this process — it has
    # already resolved (and in torch's case cached) the old import state. Hand
    # off to a new interpreter rather than importing into a stale one.
    print("\n  Setup complete — restarting.\n")
    argv = [a for a in sys.argv[1:] if a not in ("--setup",)]
    raise SystemExit(subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                     *argv]).returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["auto", "synthetic", "cityflow"],
                        default="auto", help="default: auto-detect")
    parser.add_argument("--scenario", default="S01", help="CityFlow scenario")
    parser.add_argument("--time-scale", type=float, default=None,
                        help="seconds of footage per wall second "
                             "(default 8 synthetic, 4 CityFlow)")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--instance", default="",
                        help="name this instance so its database, crops and 3D "
                             "models do not collide with another console's "
                             "(required to run two at once on different ports)")
    parser.add_argument("--no-3d", action="store_true",
                        help="skip the 3D panel even if cargen is installed")
    parser.add_argument("--3d-identification", dest="three_d_identification",
                        action="store_true",
                        help="let 3D geometry vote on identity (OFF by default: "
                             "the measured ablation found it vetoes correct "
                             "matches — see RESULTS.md)")
    parser.add_argument("--check", action="store_true",
                        help="report the environment and exit, installing nothing")
    parser.add_argument("--setup", action="store_true",
                        help="run the install step even if nothing looks missing")
    parser.add_argument("--no-setup", action="store_true",
                        help="never install; fail instead if something is missing")
    args = parser.parse_args()

    if args.check:
        print(BANNER)
        print("  EYES EVERYWHERE - environment")
        print(BANNER)
        setup_env.print_report()
        cargen = _detect_cargen()
        print(f"  3D backend  : {cargen[1]}")
        cityflow = _detect_cityflow()
        print(f"  CityFlow    : {'yes - ' if cityflow[0] else 'no  - '}{cityflow[1]}")
        print(BANNER)
        return
    if not _ensure_environment(force=args.setup, refuse=args.no_setup):
        return

    cityflow = _detect_cityflow()
    cargen = _detect_cargen()

    mode = args.mode
    if mode == "auto":
        mode = "cityflow" if cityflow[0] else "synthetic"
    if mode == "cityflow" and not cityflow[0]:
        raise SystemExit(
            f"CityFlow requested but unavailable: {cityflow[1]}.\n"
            f"See DATASETS.md, or just run `python start.py --mode synthetic`.")

    # 3D is opt-out rather than opt-in here: the whole point of one command is
    # that you see everything the machine can actually do.
    enable_3d = cargen[0] and not args.no_3d
    if args.instance:
        os.environ["EYES_INSTANCE"] = args.instance
    os.environ["EYES_ENABLE_3D"] = "1" if enable_3d else "0"
    # Visual by default, evidential only on request — the ablation measured the
    # geometry channel removing correct matches and no incorrect ones.
    os.environ["EYES_ENABLE_3D_IDENTIFICATION"] = (
        "1" if (enable_3d and args.three_d_identification) else "0")
    if not enable_3d and cargen[0]:
        cargen = (True, "installed, disabled via --no-3d")
    elif enable_3d:
        cargen = (cargen[0], cargen[1] + (
            "; feeding identification (--3d-identification)"
            if args.three_d_identification else "; visual only"))

    time_scale = args.time_scale
    if time_scale is None:
        time_scale = 4.0 if mode == "cityflow" else 8.0

    _print_plan("real CityFlow footage" if mode == "cityflow"
                else "synthetic world (no datasets needed)",
                cityflow, cargen, args.port, time_scale)

    if mode == "cityflow":
        run_cityflow(args.port, time_scale, not args.no_browser,
                     args.scenario, cityflow[1])
    else:
        run_synthetic(args.port, time_scale, not args.no_browser)


if __name__ == "__main__":
    main()
