"""FastAPI app: ingest, target management, review queue, live console feed.

Endpoint map (brief-name -> route):
    flag_target             POST   /api/targets
    update_target_profile   PATCH  /api/targets/{target_id}   (operator, audited)
    report_sighting         POST   /api/sightings
    get_alerts              GET    /api/alerts
    camera_registry         GET    /api/cameras
    camera_adjacency        GET    /api/adjacency
    operator_confirm_match  POST   /api/reviews/{review_id}/resolve
    console stream          WS     /ws/console

Everything ingested here is synthetic — the "edge nodes" are local
processes replaying the simulator (see server/feed.py).
"""
from __future__ import annotations

import base64
import binascii
import dataclasses
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select

from audit.store import load_entries as audit_load
from audit.store import record as audit_record
from audit.store import verify as audit_verify
from perception.types import Observation, PlateRead
from reasoning.profile import profile_from_flag
from server import db as dbm
from server.schemas import (
    FeedControlRequest, FlagTargetRequest, InspectRequest, PipelineConfigRequest,
    ProfileEditRequest, ReviewResolution, SightingReport,
)
from server.ws import ConnectionManager
from sim.road_graph import RoadGraph, default_world
from tracking.tracker import FleetTracker

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _clip_urls(row) -> list[str]:
    """Ordered /api/crops URLs for a sighting row's saved clip frames (the
    frames were written as '{event_id}.f{i}.png'). Empty when the row is
    missing or was saved as a still only (clip_frame_count == 0)."""
    if row is None or not row.clip_frame_count:
        return []
    return [f"/api/crops/{row.event_id}.f{i}.png"
            for i in range(row.clip_frame_count)]


def create_app(
    graph: RoadGraph | None = None,
    db_url: str = dbm.DEFAULT_DB_URL,
    crops_dir: str = "data/crops",
    calibration_path: str = "calibration/artifacts/latest.json",
    enable_3d: bool | None = None,
    enable_3d_identification: bool | None = None,
    targets3d_dir: str = "data/targets3d",
    world_source: str = "synthetic",
    enable_plate_ocr: bool = True,
    cityflow_root: str | None = None,
    cityflow_scenario_name: str | None = None,
) -> FastAPI:
    """enable_3d: build/maintain a cargen 3D model per target, fusing crops
    only on gated (plate/operator-confirmed) updates. Off by default: the
    CPU reconstruction adds seconds per confirmed sighting. Env override:
    EYES_ENABLE_3D=1.

    world_source: "synthetic" (default, our fictional Gridville graph) or
    "real" (a graph built from an actual dataset's real camera GPS, e.g.
    CityFlowScenario.to_road_graph()). Purely descriptive — it changes
    nothing about how the graph is used server-side — but the console and
    inspector read it via GET /api/world_source to decide whether their map
    draws our own fabricated road network or a real basemap tile layer
    under the (real) camera positions. Never mix the two: a "real" graph
    with fabricated street-name labels drawn on top would misrepresent
    fiction as fact."""
    import os

    if enable_3d is None:
        enable_3d = os.environ.get("EYES_ENABLE_3D", "0") == "1"
    # Separate switch, default OFF, because the two uses of the 3D model have
    # different evidence behind them. Showing an operator what was reconstructed
    # is useful and honest. Letting that geometry vote on identity is not: the
    # measured ablation (scripts/ablate_3d_cityflow.py, 795 real CityFlow crops)
    # found it usable on only 14% of crops and, where usable, vetoing 4 correct
    # matches for 0 incorrect ones. An attribute channel that contradicts on
    # same-vehicle pairs as often as on different-vehicle pairs is worse than no
    # channel. Turn it on deliberately, per deployment, with numbers to justify
    # it — EYES_ENABLE_3D_IDENTIFICATION=1.
    if enable_3d_identification is None:
        enable_3d_identification = (
            os.environ.get("EYES_ENABLE_3D_IDENTIFICATION", "0") == "1")
    if world_source not in ("synthetic", "real"):
        raise ValueError(f"world_source must be 'synthetic' or 'real', got {world_source!r}")
    graph = graph or default_world()
    cascade_config = None
    if calibration_path and Path(calibration_path).is_file():
        # Versioned isotonic map: decisions cite `isotonic-<version>` in
        # their fact lists instead of the uncalibrated fallback.
        from calibration.isotonic import load_model, make_reid_prob_fn
        from reasoning.cascade import CascadeConfig

        prob_fn, label = make_reid_prob_fn(load_model(calibration_path))
        cascade_config = CascadeConfig(reid_prob_fn=prob_fn,
                                       reid_calibration_label=label)
    # Reconstruction takes tens of seconds and must not run inside the
    # sighting-ingest request. One worker, so fusions also serialize against
    # each other for the GPU. Created before the app so its shutdown can be
    # bound into the lifespan.
    car3d_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="car3d")

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        yield
        # Let queued reconstructions finish rather than abandoning them
        # half-written; the asset on disk is the audit trail.
        car3d_executor.shutdown(wait=True)

    app = FastAPI(title="Eyes Everywhere (synthetic demo)", lifespan=_lifespan)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request, exc: RequestValidationError):
        """Report a rejected body as 422, even when the body cannot be echoed.

        FastAPI's default handler includes the offending input in the error
        detail. When that input is exactly what made the request invalid —
        `Infinity`, `NaN`, a 5 MB string — serialising the explanation fails
        and the client gets a 500, which reads as "you broke the server"
        rather than "the server refused your input". The distinction matters:
        one is a crash to investigate, the other is validation working.
        """
        safe = []
        for err in exc.errors():
            safe.append({"loc": [str(p) for p in err.get("loc", ())],
                         "msg": str(err.get("msg", "invalid")),
                         "type": str(err.get("type", "value_error"))})
        return JSONResponse(status_code=422, content={"detail": safe})
    Path(crops_dir).mkdir(parents=True, exist_ok=True)
    if db_url.startswith("sqlite:///"):
        db_path = db_url.removeprefix("sqlite:///")
        if "/" in db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    engine = dbm.make_engine(db_url)
    with Session(engine) as session:
        dbm.store_graph(session, graph)

    state = app.state
    state.graph = graph
    state.engine = engine

    def _verify_against_model3d(target_id: str, obs: Observation):
        """Render-and-compare P(same) for the cascade's look-alike tiebreak.

        Analysis-by-synthesis: the target's own 3D model is rendered into the
        query crop's viewpoint and compared like-for-like, which is what makes
        this survive the cross-view case that 2D ReID is worst at.

        Abstains (None) for anything it cannot answer honestly — no crop, an
        immature model, or any failure at all. The cascade treats None as "no
        opinion" and leaves its ordering untouched; 3D corroborates, it never
        sinks a decision.

        Deliberately builds a Target3DModel with NO pipeline: verification only
        loads and renders an existing asset. Handing it the shared pipeline
        would make the first look-alike tie block the request thread on a
        multi-gigabyte model load it has no use for.
        """
        if obs.crop is None:
            return None
        try:
            from car3d.match import verify_match
            from car3d.profile_model import Target3DModel

            return verify_match(
                Target3DModel(target_id, state.targets3d_dir), obs.crop,
                state.embed_bgr, calibrator=state.render_calibrator,
                embed_batch_fn=state.embed_bgr_batch).score
        except Exception as exc:  # noqa: BLE001 — never sink a decision
            print(f"car3d: render verification unavailable for {target_id}: {exc}")
            return None

    if enable_3d and enable_3d_identification:
        # Feature D: the dependency-inverted hook reasoning/ declares but never
        # imports. Without this the verifier is built, tested and unreachable.
        from reasoning.cascade import CascadeConfig

        cascade_config = dataclasses.replace(
            cascade_config or CascadeConfig(),
            shortlist_verifier=_verify_against_model3d)

    state.tracker = FleetTracker(graph, cascade_config)
    state.manager = ConnectionManager()
    state.crops_dir = Path(crops_dir)
    state.sim_now = 0.0
    # Replay position for display, published by the feed. None until a
    # feed runs, so /api/stats can fall back to sim_now.
    state.replay_clock_s = None
    # Set by the feed supervisor when a replay reaches its end. A
    # parked clock with paused=false is otherwise indistinguishable
    # from a hang.
    state.replay_complete = False
    state.target_seq = itertools.count(1)
    state.enable_3d = enable_3d
    state.enable_3d_identification = enable_3d_identification
    state.targets3d_dir = Path(targets3d_dir)
    state.world_source = world_source
    # Real-clip mode only (perception/real_observe.py's RealPerceptor reads
    # this per-call): whether plate OCR runs at all. Runtime-flippable via
    # POST /api/pipeline_config so a single demo session can show both
    # states, not just a startup flag.
    state.enable_plate_ocr = enable_plate_ocr
    # Replay pause. The feeds read this live via server.feed.FeedClock, which
    # stops counting wall time while it is set -- so every camera task freezes
    # on one shared timeline and the cross-camera gaps the transit check
    # scores against survive the pause unchanged.
    state.feed_paused = False
    # Live tallies for the console's scale strip. Kept as plain counters rather
    # than derived from the DB per poll: the point is to show the replay moving
    # once a second, and a COUNT(*) over the audit tables every second to draw a
    # number is a silly amount of work for a display.
    state.counters = {
        "sightings": 0, "vehicles_seen": 0, "cross_camera_hops": 0,
        "reviews_raised": 0, "refusals": 0, "alerts": 0,
    }
    state._seen_vehicle_keys: set[str] = set()
    state._last_camera_by_vehicle: dict[str, str] = {}
    # Set by POST /api/reset; start.py's supervisor watches it, tears the feed
    # down, wipes runtime state and replays from t=0. Held on state because the
    # feed and the server live in different threads and this is the only thing
    # they need to agree on.
    state.restart_requested = False
    state.run_generation = 0
    state.footage_duration_s = 0.0
    # Shared lazy embedder for photo-seeded flags (see _flag_embedder below);
    # tests may inject a stub here to avoid the model load.
    state.flag_embedder = None

    # Real-clip mode only: the CityFlow scenario this server instance is
    # actively replaying (server/real_feed.py), if any. Set once at
    # startup by scripts/run_cityflow_console.py -- there is no live
    # scenario-switching, only vehicle selection within the active one.
    state.cityflow_root = Path(cityflow_root) if cityflow_root else None
    state.cityflow_scenario = None
    state.cityflow_camera_dirs = {}
    state.cityflow_vehicle_index = None  # lazily built + cached
    state.pending_scenario = ""          # set by POST /api/cityflow/scenario
    # Guards the browse-index build so only one ever runs; see
    # /api/cityflow/{scenario}/vehicles for why concurrent builds are fatal.
    state.index_lock = threading.Lock()
    state.index_building = False
    # Guards the per-camera video handles; see the frame endpoint for why
    # sharing one cv2.VideoCapture across threadpool workers is fatal.
    state.frame_lock = threading.Lock()
    state.frame_locks: dict[str, threading.Lock] = {}

    def load_scenario(name: str):
        """Point the server at a different scenario: graph, cameras, index.

        Called by the supervisor between runs, never mid-replay — the feed,
        the tracker and the map all have to agree on which cameras exist.
        """
        from datasets.cityflow import CityFlow
        from datasets.cityflow_video import discover_camera_dirs

        scen = CityFlow(state.cityflow_root).load_scenario(name)
        state.cityflow_scenario = scen
        state.cityflow_camera_dirs = discover_camera_dirs(state.cityflow_root, name)
        state.cityflow_vehicle_index = None      # rebuilt for the new scenario
        state.footage_duration_s = 0.0
        # Per-camera video handles belong to the old scenario's files, and must
        # be released under the SAME lock the frame endpoint decodes with.
        # Releasing a cv2.VideoCapture while a threadpool worker is inside
        # read() on it is a native crash, not an exception — the process just
        # disappears with no traceback, which is exactly what a scenario switch
        # did while frames were being served.
        with state.frame_lock:
            sources = getattr(state, "_frame_sources", None) or {}
            locks = getattr(state, "frame_locks", None) or {}
            for cam, src in sources.items():
                lock = locks.get(cam)
                if lock is not None:
                    with lock:
                        try:
                            src.close()
                        except Exception:        # noqa: BLE001
                            pass
                else:
                    try:
                        src.close()
                    except Exception:            # noqa: BLE001
                        pass
            state._frame_sources = {}
            state.frame_locks = {}
        state.graph = scen.to_road_graph()
        with Session(engine) as session:
            # store_graph merges but never deletes, so switching scenarios
            # would leave the previous one's cameras and edges in the registry.
            # The persisted topology must describe the scenario now running.
            from sqlalchemy import delete as sa_delete

            session.exec(sa_delete(dbm.AdjacencyRow))
            session.exec(sa_delete(dbm.CameraRow))
            session.commit()
            dbm.store_graph(session, state.graph)
        return scen

    state.load_scenario = load_scenario
    if cityflow_scenario_name and state.cityflow_root:
        load_scenario(cityflow_scenario_name)
    state.render_embedder = None       # lazy ReidEmbedder for render-and-compare
    _rc_path = Path("car3d/artifacts/render_compare.json")
    state.render_calibrator = None
    if _rc_path.is_file():
        from car3d.calibration import load_model as _load_rc
        state.render_calibrator = _load_rc(_rc_path)

    def _embed_bgr(bgr):
        if state.render_embedder is None:
            from perception.embedder import ReidEmbedder
            state.render_embedder = ReidEmbedder()
        return state.render_embedder.embed(bgr)

    def _embed_bgr_batch(images):
        if state.render_embedder is None:
            from perception.embedder import ReidEmbedder
            state.render_embedder = ReidEmbedder()
        return state.render_embedder.embed_batch(images)

    state.embed_bgr = _embed_bgr
    state.embed_bgr_batch = _embed_bgr_batch

    # --- 3D reconstruction resources -------------------------------------
    # The cargen Pipeline is built ONCE and shared. A real prior backend
    # (SF3D) loads ~4 GB of weights and holds most of an 8 GB GPU; building
    # one per fusion event — which is what constructing Target3DModel without
    # an injected pipeline does — reloaded all of it every single time.
    state.car3d_pipeline = None
    state.car3d_pipeline_lock = threading.Lock()
    state.car3d_executor = car3d_executor
    state.car3d_jobs: dict[str, object] = {}   # target_id -> most recent Future
    # Guards the worker's read-modify-write of a profile against the event
    # loop's own tracker mutations. Held for a dict swap, never across work.
    state.tracker_lock = threading.Lock()

    def _car3d_pipeline():
        if state.car3d_pipeline is None:
            with state.car3d_pipeline_lock:
                if state.car3d_pipeline is None:
                    from car3d.profile_model import build_pipeline

                    state.car3d_pipeline = build_pipeline()
        return state.car3d_pipeline

    state.car3d_pipeline_factory = _car3d_pipeline

    def _model_for(target_id: str):
        """A Target3DModel wired to the shared pipeline."""
        from car3d.profile_model import Target3DModel

        return Target3DModel(target_id, state.targets3d_dir,
                             pipeline=_car3d_pipeline())

    state.car3d_model_for = _model_for

    # Rehydrate flagged targets from a pre-existing DB so a server restart
    # neither collides on target ids nor forgets what was flagged. Live
    # track state (belief, lifecycle, gallery) is in-memory and resets —
    # the audit tables keep the history, the track re-earns confirmation.
    max_seq = 0
    with Session(engine) as session:
        for row in session.exec(select(dbm.TargetRow)).all():
            profile = profile_from_flag(
                row.target_id, row.label, row.plate,
                dbm.loads(row.class_attrs) or {},
                dbm.loads(row.instance_attrs) or {})
            try:
                state.tracker.flag_target(profile)
            except ValueError:
                pass
            suffix = row.target_id.rsplit("-", 1)[-1]
            if suffix.isdigit():
                max_seq = max(max_seq, int(suffix))
    state.target_seq = itertools.count(max_seq + 1)

    # ------------------------------------------------------------ 3d bridge

    def _run_fusion(target_id: str, event_id: str, reason: str,
                    timestamp_s: float) -> None:
        """Reconstruct one confirmed crop into a target's model. Worker thread.

        Owns its own Session: the request's session belongs to the request's
        thread and must not be handed across. Every failure degrades to a
        console note — 3D is corroborative and must never sink ingest.
        """
        import cv2

        from car3d.compat import InsufficientDetail
        from car3d.geometry import signature_to_attrs

        crop_path = state.crops_dir / f"{event_id}.png"
        crop = cv2.imread(str(crop_path)) if crop_path.exists() else None
        if crop is None:
            return
        try:
            # export=False: the .ply/.splat set is ~20 MB and the turntable is
            # six CPU renders. Both are regenerated on demand by the dossier
            # endpoints, which are opened far less often than sightings arrive.
            outcome = state.car3d_model_for(target_id).fuse_confirmed_crop(
                crop, event_id, reason=reason, timestamp=timestamp_s,
                export=False)
        except InsufficientDetail as exc:
            # Not a failure: the frame simply cannot support a reconstruction.
            print(f"car3d: skipped a crop for {target_id} — {exc}")
            return
        except Exception as exc:  # noqa: BLE001 — 3D must never sink ingest
            print(f"car3d: fusion failed for {target_id}: {exc}")
            return

        # The reconstruction still happened, is still exported, and is still
        # shown in the dossier — only its promotion to identity evidence is
        # gated. See create_app's enable_3d_identification.
        geom_attrs = (signature_to_attrs(outcome.geometry)
                      if state.enable_3d_identification else {})
        if not geom_attrs:
            return
        with state.tracker_lock:
            tracked = state.tracker.targets().get(target_id)
            if tracked is None:
                return
            profile = dataclasses.replace(
                tracked.profile,
                instance_attrs={**tracked.profile.instance_attrs, **geom_attrs},
                version=tracked.profile.version + 1)
            state.tracker.replace_profile(target_id, profile)
        with Session(state.engine) as session:
            session.add(dbm.ProfileUpdateRow(
                target_id=target_id, event_id=event_id,
                version=profile.version,
                reason="3D geometry attributes refreshed from the "
                       "fused model (gated fusion, reversible).",
                timestamp_s=timestamp_s))
            session.commit()

    def _fuse_3d_for_events(events, session: Session) -> None:
        """Queue gated (confirmed) sightings for fusion into per-target models.

        Runs only on profile_update events — the exact moments the profile
        gate opened — so cargen's pending-approval merge policy and this
        project's update gate stay one mechanism.

        Queued rather than executed: a real backend takes tens of seconds per
        crop, and this is called from the sighting-ingest path. `session` is
        accepted for signature compatibility with the other _persist helpers
        and deliberately unused — the worker opens its own.
        """
        if not state.enable_3d:
            return
        for ev in [e for e in events if e.kind == "profile_update" and e.event_id]:
            state.car3d_jobs[ev.target_id] = state.car3d_executor.submit(
                _run_fusion, ev.target_id, ev.event_id,
                str(ev.detail.get("reason", "gated update")), ev.timestamp_s)

    def _await_fusion(target_id: str, timeout: float = 300.0) -> None:
        """Block until this target's queued fusion settles.

        The dossier should show settled state rather than a half-built model,
        and it is operator-paced, so paying the wait here is right. One worker
        means FIFO: awaiting the newest job implies the earlier ones finished.
        """
        job = state.car3d_jobs.get(target_id)
        if job is None:
            return
        try:
            job.result(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 — 3D must never sink a request
            # _run_fusion reports everything it catches, but it cannot report
            # what it never entered: an import error or a signature change at
            # the top of the worker escapes past its try. Swallowing that
            # silently is how a completely dead 3D path looked like an empty
            # panel for weeks. Say it here instead.
            print(f"car3d: fusion worker for {target_id} died before reporting: "
                  f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------ helpers

    def _persist_events(session: Session, events, obs: Observation | None) -> None:
        for ev in events:
            session.add(dbm.AlertRow(
                kind=ev.kind, target_id=ev.target_id, event_id=ev.event_id,
                timestamp_s=ev.timestamp_s, detail=dbm.dumps(dict(ev.detail))))
            if ev.kind == "association":
                session.add(dbm.CorroborationRow(
                    target_id=ev.target_id, event_id=ev.event_id,
                    timestamp_s=ev.timestamp_s,
                    verdict=str(ev.detail.get("verdict", "")),
                    belief_after=float(ev.detail.get("belief", 0.0)),
                    facts=str(ev.detail.get("facts", ""))))
            elif ev.kind in ("review", "anomaly"):
                session.add(dbm.ReviewRow(
                    review_id=str(ev.detail["review_id"]),
                    target_id=ev.target_id, event_id=ev.event_id, kind=ev.kind,
                    score=float(ev.detail.get("score", 0.0)),
                    facts=str(ev.detail.get("facts", "")),
                    counterfactuals=dbm.dumps(list(ev.detail.get("counterfactuals", []))),
                    rivals=dbm.dumps(list(ev.detail.get("rivals", []))),
                    created_s=ev.timestamp_s))
            elif ev.kind == "profile_update":
                session.add(dbm.ProfileUpdateRow(
                    target_id=ev.target_id, event_id=ev.event_id,
                    version=int(ev.detail.get("version", 0)),
                    reason=str(ev.detail.get("reason", "")),
                    timestamp_s=ev.timestamp_s))
                if obs is not None:
                    row = session.get(dbm.TargetRow, ev.target_id)
                    # First associated sighting becomes the reference; a
                    # flag-photo placeholder ("{target_id}-ref.png") gives
                    # way to it so the dossier's targeting clip resolves.
                    if row and (not row.reference_crop
                                or row.reference_crop.endswith("-ref.png")):
                        row.reference_crop = f"{obs.event_id}.png"
                        session.add(row)

    def _sync_target_rows(session: Session) -> None:
        for target_id, tracked in state.tracker.targets().items():
            row = session.get(dbm.TargetRow, target_id)
            if row is None:
                continue
            row.state = tracked.track.state
            row.belief = tracked.corroboration.belief
            row.profile_version = tracked.profile.version
            row.plate = tracked.profile.plate
            row.instance_attrs = dbm.dumps(dict(tracked.profile.instance_attrs))
            session.add(row)

    async def _broadcast_events(events) -> None:
        for ev in events:
            await state.manager.broadcast({
                "type": ev.kind, "target_id": ev.target_id,
                "event_id": ev.event_id, "timestamp_s": ev.timestamp_s,
                "detail": dict(ev.detail)})

    def _observation_from_report(report: SightingReport) -> Observation:
        emb = np.asarray(report.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm <= 0:
            raise HTTPException(422, "embedding must be non-zero")
        plate = None
        if report.plate is not None:
            plate = PlateRead(report.plate.text.upper(), report.plate.confidence,
                              report.plate.source)
        return Observation(
            event_id=report.event_id, camera_id=report.camera_id,
            timestamp_s=report.timestamp_s, lat=report.lat, lon=report.lon,
            embedding=emb / norm, plate=plate,
            class_attrs=dict(report.class_attrs),
            class_attrs_source=report.class_attrs_source,
            instance_attrs=dict(report.instance_attrs),
            detection_source=report.detection_source,
            eval_truth_id=report.eval_truth_id)

    # ------------------------------------------------------------ targets

    def _flag_embedder():
        # Lazy: only flags that carry a reference photo pay the model load,
        # and one shared instance serves them all (ReidEmbedder locks its
        # own lazy load, same as the edge tier's shared embedder).
        if state.flag_embedder is None:
            from perception.embedder import ReidEmbedder

            state.flag_embedder = ReidEmbedder()
        return state.flag_embedder

    def _seed_profile_from_photo(profile, png_bytes: bytes,
                                 extra_pngs: list[bytes] = ()):
        """Honest evidence from the operator's own reference photos: the
        pixel-color heuristic (the same one sightings use) plus one
        appearance-gallery embedding per reference crop. ReID stays a
        capped tiebreaker — the photos let the cascade *consider* this
        target, they cannot confirm anything by themselves, and the
        distinctiveness floor still routes color+appearance matches to
        review as a candidate set."""
        import cv2

        from perception.attributes import estimate_color

        crop = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
        if crop is None or crop.size == 0:
            raise HTTPException(422, "reference_crop_b64 is not a decodable image")
        crops = [crop]
        for extra in extra_pngs:
            more = cv2.imdecode(np.frombuffer(extra, np.uint8), cv2.IMREAD_COLOR)
            if more is not None and more.size:   # extras are best-effort
                crops.append(more)
        class_attrs = dict(profile.class_attrs)
        class_attrs.setdefault("color", estimate_color(crop))
        embedder = _flag_embedder()
        return dataclasses.replace(
            profile, class_attrs=class_attrs,
            gallery=tuple(embedder.embed(c) for c in crops))

    @app.post("/api/targets", status_code=201)
    async def flag_target(req: FlagTargetRequest):
        target_id = f"tgt-{next(state.target_seq):03d}"
        profile = profile_from_flag(
            target_id, req.label, req.plate, req.class_attrs, req.instance_attrs)
        reference_crop = ""
        if req.reference_crop_b64:
            try:
                png = base64.b64decode(req.reference_crop_b64, validate=True)
                extras = [base64.b64decode(g, validate=True)
                          for g in req.reference_gallery_b64]
            except binascii.Error as exc:
                raise HTTPException(
                    422, "reference crop is not valid base64") from exc
            profile = _seed_profile_from_photo(profile, png, extras)
            reference_crop = f"{target_id}-ref.png"
            (state.crops_dir / reference_crop).write_bytes(png)
        state.tracker.flag_target(profile)
        with Session(engine) as session:
            session.add(dbm.TargetRow(
                target_id=target_id, label=req.label, plate=profile.plate,
                class_attrs=dbm.dumps(dict(profile.class_attrs)),
                instance_attrs=dbm.dumps(dict(profile.instance_attrs)),
                reference_crop=reference_crop,
                created_s=state.sim_now))
            audit_record(session, "operator", "flag_target",
                         {"target_id": target_id, "label": req.label,
                          "plate": profile.plate,
                          "photo_seeded": bool(req.reference_crop_b64)},
                         state.sim_now)
            session.commit()
        # Tell the console immediately. The targets list is painted from the
        # snapshot broadcast, which was only ever sent on sighting ingest — so
        # flagging while the feed was paused, or after the replay had finished,
        # ticked the tile green and left the panel saying "no targets flagged
        # yet" until a sighting happened to arrive. Flagging is the operator's
        # single most important action; it must be visible the moment it lands.
        await state.manager.broadcast({
            "type": "snapshot", "timestamp_s": state.sim_now,
            "targets": state.tracker.snapshot(state.sim_now)})
        return {"target_id": target_id}

    @app.get("/api/targets")
    def list_targets():
        return state.tracker.snapshot(state.sim_now)

    @app.get("/api/targets/{target_id}")
    def target_dossier(target_id: str):
        tracked = state.tracker.targets().get(target_id)
        if tracked is None:
            raise HTTPException(404, "unknown target")
        with Session(engine) as session:
            row = session.get(dbm.TargetRow, target_id)
            reference_crop = row.reference_crop if row else ""
            # "Targeting clip": the clip of the first sighting the tracker
            # associated to this target (the same event that set
            # reference_crop, named "{event_id}.png"). Empty until then.
            reference_clip: list[str] = []
            if reference_crop.endswith(".png"):
                ref_row = session.get(dbm.SightingRow, reference_crop[:-len(".png")])
                reference_clip = _clip_urls(ref_row)
            updates = [u.model_dump() for u in session.exec(
                select(dbm.ProfileUpdateRow)
                .where(dbm.ProfileUpdateRow.target_id == target_id)
                .order_by(dbm.ProfileUpdateRow.version)).all()]
            chain = [c.model_dump() for c in session.exec(
                select(dbm.CorroborationRow)
                .where(dbm.CorroborationRow.target_id == target_id)
                .order_by(dbm.CorroborationRow.timestamp_s)).all()]
            # Opening a target's full dossier is a data access worth logging;
            # per-thumbnail crop fetches are not (they are UI rendering).
            # Serialize the rows to dicts *before* this commit expires them.
            audit_record(session, "operator", "view_dossier",
                         {"target_id": target_id}, state.sim_now)
            session.commit()
        snap = state.tracker.snapshot(state.sim_now).get(target_id, {})
        return {
            "target_id": target_id,
            "label": tracked.profile.label,
            "plate": tracked.profile.plate,
            "class_attrs": dict(tracked.profile.class_attrs),
            "instance_attrs": dict(tracked.profile.instance_attrs),
            "gallery_size": len(tracked.profile.gallery),
            "reference_crop": reference_crop,
            "reference_clip": reference_clip,
            "live": snap,
            "profile_updates": updates,
            "corroboration_chain": chain,
        }

    @app.patch("/api/targets/{target_id}")
    def update_target_profile(target_id: str, req: ProfileEditRequest):
        """Operator-authority edit; still versioned + audited like any update."""
        tracked = state.tracker.targets().get(target_id)
        if tracked is None:
            raise HTTPException(404, "unknown target")
        profile = tracked.profile
        changes = {k: v for k, v in req.model_dump().items() if v is not None}
        if not changes:
            raise HTTPException(422, "no fields to update")
        profile = dataclasses.replace(
            profile,
            label=changes.get("label", profile.label),
            plate=changes.get("plate", profile.plate).upper()
            if "plate" in changes else profile.plate,
            class_attrs=changes.get("class_attrs", dict(profile.class_attrs)),
            instance_attrs=changes.get("instance_attrs", dict(profile.instance_attrs)),
            version=profile.version + 1)
        state.tracker.replace_profile(target_id, profile)
        with Session(engine) as session:
            session.add(dbm.ProfileUpdateRow(
                target_id=target_id, event_id="", version=profile.version,
                reason=f"Operator edited profile fields: {', '.join(sorted(changes))}.",
                timestamp_s=state.sim_now))
            _sync_target_rows(session)
            row = session.get(dbm.TargetRow, target_id)
            if row:
                row.label = profile.label
                row.class_attrs = dbm.dumps(dict(profile.class_attrs))
                session.add(row)
            audit_record(session, "operator", "update_target_profile",
                         {"target_id": target_id, "fields": sorted(changes),
                          "version": profile.version}, state.sim_now)
            session.commit()
        return {"target_id": target_id, "version": profile.version}

    @app.delete("/api/targets/{target_id}", status_code=204)
    def unflag_target(target_id: str):
        state.tracker.unflag_target(target_id)
        # Previously wrote nothing — an untraceable deletion. Now audited.
        with Session(engine) as session:
            audit_record(session, "operator", "unflag_target",
                         {"target_id": target_id}, state.sim_now)
            session.commit()

    # ---------------------------------------------------------- sightings

    @app.post("/api/sightings", status_code=202)
    async def report_sighting(report: SightingReport):
        obs = _observation_from_report(report)
        state.sim_now = max(state.sim_now, obs.timestamp_s)
        crop_name = ""
        if report.crop_png_b64:
            try:
                png = base64.b64decode(report.crop_png_b64, validate=True)
            except binascii.Error as exc:
                raise HTTPException(422, "crop_png_b64 is not valid base64") from exc
            crop_name = f"{obs.event_id}.png"
            (state.crops_dir / crop_name).write_bytes(png)

        # Short looping sighting clip, saved as sibling frames the review card
        # flips through; reuses the same /api/crops server as the still.
        clip_count = 0
        for i, frame_b64 in enumerate(report.clip_frames_b64):
            try:
                frame_png = base64.b64decode(frame_b64, validate=True)
            except binascii.Error as exc:
                raise HTTPException(
                    422, "clip_frames_b64 contains invalid base64") from exc
            (state.crops_dir / f"{obs.event_id}.f{i}.png").write_bytes(frame_png)
            clip_count += 1

        with state.tracker_lock:
            events = state.tracker.process_observation(obs)
            events += state.tracker.tick(state.sim_now)
        with Session(engine) as session:
            session.add(dbm.SightingRow(
                event_id=obs.event_id, camera_id=obs.camera_id,
                timestamp_s=obs.timestamp_s, lat=obs.lat, lon=obs.lon,
                plate_text=obs.plate.text if obs.plate else "",
                plate_conf=obs.plate.confidence if obs.plate else 0.0,
                plate_source=obs.plate.source if obs.plate else "",
                class_attrs=dbm.dumps(dict(obs.class_attrs)),
                instance_attrs=dbm.dumps(dict(obs.instance_attrs)),
                detection_source=obs.detection_source,
                crop_path=crop_name, clip_frame_count=clip_count,
                truth_id=obs.eval_truth_id))
            _persist_events(session, events, obs)
            _fuse_3d_for_events(events, session)
            _sync_target_rows(session)
            audit_record(session, f"camera:{obs.camera_id}", "report_sighting",
                         {"event_id": obs.event_id, "camera_id": obs.camera_id,
                          "outcomes": [e.kind for e in events]}, obs.timestamp_s)
            session.commit()
        _tally(obs, events)
        await state.manager.broadcast({
            "type": "contact", "event_id": obs.event_id,
            "camera_id": obs.camera_id, "timestamp_s": obs.timestamp_s,
            "lat": obs.lat, "lon": obs.lon,
            "class_attrs": dict(obs.class_attrs),
            "crop": f"/api/crops/{crop_name}" if crop_name else "",
            "detection_source": obs.detection_source})
        for hop in _hops_from(obs):
            # Lights the corresponding transit edge on the operator's map, so
            # the road graph shows the physics reasoning happening rather than
            # sitting there as decoration.
            await state.manager.broadcast(hop)
        await _broadcast_events(events)
        await state.manager.broadcast({
            "type": "snapshot", "timestamp_s": state.sim_now,
            "targets": state.tracker.snapshot(state.sim_now)})
        return {"events": [e.kind for e in events]}

    def _tally(obs, events) -> None:
        """Update the live scale counters from one ingested sighting."""
        c = state.counters
        c["sightings"] += 1
        # Ground-truth vehicle id when the dataset supplies one; otherwise the
        # event id, which at least counts distinct sightings honestly rather
        # than silently collapsing them.
        key = str(obs.eval_truth_id or obs.event_id)
        if key not in state._seen_vehicle_keys:
            state._seen_vehicle_keys.add(key)
            c["vehicles_seen"] = len(state._seen_vehicle_keys)
        for ev in events:
            if ev.kind == "review":
                c["reviews_raised"] += 1
            elif ev.kind == "alert":
                c["alerts"] += 1
            if ev.detail.get("refused_to_individuate"):
                c["refusals"] += 1

    def _hops_from(obs) -> list[dict]:
        """Cross-camera transitions for this sighting's ground-truth vehicle.

        Uses the dataset's own id, so this reports what the FOOTAGE did, not
        what the cascade concluded — it is a display of the data's structure,
        and the message says so via `source: "ground_truth"`. Drawing the
        system's own associations here instead would be a different claim and
        needs the association's previous camera, which the event does not
        currently carry.
        """
        vid = getattr(obs, "eval_truth_id", None)
        if vid is None:
            return []
        key = str(vid)
        previous = state._last_camera_by_vehicle.get(key)
        state._last_camera_by_vehicle[key] = obs.camera_id
        if not previous or previous == obs.camera_id:
            return []
        state.counters["cross_camera_hops"] += 1
        return [{"type": "hop", "from_camera": previous,
                 "to_camera": obs.camera_id, "timestamp_s": obs.timestamp_s,
                 "vehicle_id": key, "source": "ground_truth"}]

    # ------------------------------------------------------ reviews/alerts

    @app.get("/api/reviews")
    def list_reviews(status: str = "pending"):
        from reasoning.cascade import score_breakdown

        # Structured decision detail (signals, per-fact kind/check, real
        # score breakdown, candidate_ids) lives only on the tracker's live
        # in-memory PendingReview — ReviewRow only ever persisted flattened
        # text. Enrich pending rows from there; once resolved a review has
        # left that dict and the flattened DB fields are all that remain.
        pending_by_id = {p.review_id: p for p in state.tracker.pending_reviews()}
        with Session(engine) as session:
            rows = session.exec(
                select(dbm.ReviewRow).where(dbm.ReviewRow.status == status)
                .order_by(dbm.ReviewRow.created_s)).all()
            out = []
            for r in rows:
                sighting = session.get(dbm.SightingRow, r.event_id)
                target = session.get(dbm.TargetRow, r.target_id)
                live = pending_by_id.get(r.review_id)
                decision = live.decision if live else None
                out.append({
                    **r.model_dump(),
                    "rivals": dbm.loads(r.rivals),
                    "counterfactuals": dbm.loads(r.counterfactuals) or [],
                    "sighting_crop": (f"/api/crops/{sighting.crop_path}"
                                      if sighting and sighting.crop_path else ""),
                    "sighting_clip": _clip_urls(sighting),
                    "reference_crop": (f"/api/crops/{target.reference_crop}"
                                       if target and target.reference_crop else ""),
                    "target_label": target.label if target else "",
                    "structured_facts": (
                        [{"kind": f.kind, "text": f.text, "check": f.check}
                         for f in decision.facts] if decision else []),
                    "signals": (dataclasses.asdict(decision.signals)
                                if decision and decision.signals else None),
                    "distinctiveness": decision.distinctiveness if decision else None,
                    "candidate_ids": list(decision.candidate_ids) if decision else [],
                    "score_breakdown": (score_breakdown(decision.signals)
                                        if decision and decision.signals else {}),
                })
            return out

    @app.post("/api/reviews/{review_id}/resolve")
    async def operator_confirm_match(review_id: str, res: ReviewResolution):
        # Grab the sighting behind the review before resolution pops it, so
        # an accepted first match can set the target's reference crop/clip
        # (review-accepted targets otherwise never get a targeting clip).
        pending = next((r for r in state.tracker.pending_reviews()
                        if r.review_id == review_id), None)
        obs = pending.observation if pending else None
        try:
            with state.tracker_lock:
                events = state.tracker.resolve_review(
                    review_id, res.accept, state.sim_now)
        except KeyError:
            raise HTTPException(404, "unknown or already-resolved review")
        with Session(engine) as session:
            row = session.get(dbm.ReviewRow, review_id)
            if row:
                row.status = "accepted" if res.accept else "rejected"
                row.resolved_s = state.sim_now
                session.add(row)
            _persist_events(session, events, obs)
            _fuse_3d_for_events(events, session)
            _sync_target_rows(session)
            audit_record(session, "operator", "resolve_review",
                         {"review_id": review_id, "accepted": res.accept},
                         state.sim_now)
            session.commit()
        await _broadcast_events(events)
        await state.manager.broadcast({
            "type": "snapshot", "timestamp_s": state.sim_now,
            "targets": state.tracker.snapshot(state.sim_now)})
        return {"resolved": review_id, "accepted": res.accept}

    @app.get("/api/audit")
    def get_audit(limit: int = 100):
        """Recent audit entries + a live chain-integrity verdict."""
        with Session(engine) as session:
            entries = audit_load(session, limit=limit)
            result = audit_verify(session)
        return {
            "verified": result.ok,
            "length": result.length,
            "break_index": result.break_index,
            "reason": result.reason,
            "entries": [
                {"seq": e.seq, "timestamp_s": e.timestamp_s, "actor": e.actor,
                 "action": e.action, "payload_digest": e.payload_digest[:12],
                 "entry_hash": e.entry_hash[:12]}
                for e in entries],
        }

    @app.get("/api/alerts")
    def get_alerts(since_s: float = 0.0, target_id: str = "", limit: int = 200):
        with Session(engine) as session:
            q = select(dbm.AlertRow).where(dbm.AlertRow.timestamp_s >= since_s)
            if target_id:
                q = q.where(dbm.AlertRow.target_id == target_id)
            rows = session.exec(
                q.order_by(dbm.AlertRow.timestamp_s.desc()).limit(limit)).all()
        return [{**r.model_dump(), "detail": dbm.loads(r.detail)} for r in rows]

    # ---------------------------------------------------------- world info

    @app.get("/api/cameras")
    def camera_registry():
        return [dataclasses.asdict(c) for c in state.graph.cameras]

    @app.get("/api/adjacency")
    def camera_adjacency():
        return [dataclasses.asdict(e) for e in state.graph.edges]

    @app.get("/api/world_source")
    def world_source_info():
        """Tells the frontend whether the camera graph is our fictional
        Gridville world or built from a real dataset's real GPS — the map
        uses this to decide whether to draw its own road network or a real
        basemap tile layer under the camera pins. See create_app's
        world_source docstring for why these must never be mixed."""
        return {"source": state.world_source}

    @app.get("/api/pipeline_config")
    def pipeline_config_get():
        return {"plate_ocr": state.enable_plate_ocr}

    @app.post("/api/pipeline_config")
    def pipeline_config_set(req: PipelineConfigRequest):
        if req.plate_ocr is not None:
            state.enable_plate_ocr = req.plate_ocr
        return {"plate_ocr": state.enable_plate_ocr}

    @app.get("/api/feed_control")
    def feed_control_get():
        return {"paused": state.feed_paused}

    @app.post("/api/feed_control")
    def feed_control_set(req: FeedControlRequest):
        """Freeze or resume the replay.

        Only the CLOCK stops. Sightings already reported keep their state,
        the review queue stays workable while paused, and nothing is
        rewound -- the reasoning layer mutates belief and profiles as it
        goes, so running it backwards would show a past frame beside
        present-tense conclusions. Pause is honest; rewind would not be.
        """
        if req.paused is not None:
            state.feed_paused = req.paused
        return {"paused": state.feed_paused}

    def reset_runtime() -> None:
        """Wipe everything the last pass produced. Called by the supervisor
        between runs, never from a request handler — see POST /api/reset."""
        import shutil

        with state.tracker_lock:
            state.tracker = FleetTracker(state.graph, cascade_config)
        state.sim_now = 0.0
        # The next feed republishes this; leaving the old value would
        # show the previous run's position on a rewound clock.
        state.replay_clock_s = None
        state.replay_complete = False
        state.target_seq = itertools.count(1)
        state.counters = {k: 0 for k in state.counters}
        state._seen_vehicle_keys = set()
        state._last_camera_by_vehicle = {}
        state.car3d_jobs = {}
        state.feed_paused = False
        state.run_generation += 1
        # Everything the run produced. CameraRow/AdjacencyRow are deliberately
        # absent: they are the deployment's topology, not this pass's output,
        # and dropping them would leave the map with nothing to draw.
        #
        # Bulk DELETE rather than loading every row and deleting it one by one.
        # A finished S01 pass leaves ~400 sightings plus audit and decision
        # rows; round-tripping each through the ORM made the restart take
        # visible seconds, which is most of why it felt unreliable.
        from sqlalchemy import delete as sa_delete

        with Session(engine) as session:
            for model in (dbm.SightingRow, dbm.TargetRow, dbm.ProfileUpdateRow,
                          dbm.CorroborationRow, dbm.ReviewRow, dbm.AlertRow,
                          dbm.AuditRow):
                session.exec(sa_delete(model))
            session.commit()
        for directory in (state.crops_dir, state.targets3d_dir):
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir(parents=True, exist_ok=True)
        # Only clear the flag if nothing NEW asked for a restart while this one
        # was running. Clearing unconditionally swallowed the request: two
        # scenario switches 0.3s apart both answered 200 {"switching": true},
        # the first won, and the second silently never happened — the API
        # reporting success for work it had just discarded.
        if getattr(state, "pending_scenario", ""):
            state.restart_requested = True
        else:
            state.restart_requested = False

    state.reset_runtime = reset_runtime

    @app.get("/api/stats")
    def stats():
        """Clock + live tallies for the console's scale strip.

        `sim_now` is footage time, not wall time: it is whatever the newest
        ingested sighting carried, which is the only clock the reasoning layer
        actually uses. Reporting wall time here would drift from every
        timestamp shown elsewhere the moment anyone hits pause.
        """
        scen = state.cityflow_scenario
        duration = state.footage_duration_s
        if not duration and scen is not None:
            duration = max((sp.exit_s for sp in scen.spans), default=0.0)
            state.footage_duration_s = duration
        # Two clocks, deliberately. `sim_now` is max(observed timestamp) and is
        # what the reasoning layer uses; it is the honest answer to "what is the
        # latest evidence timestamp". It is a poor thing to display, because it
        # only moves when the leading camera reports and sits frozen for seconds
        # in between. `clock_s` is the replay's actual position and is what the
        # console shows, so a running replay looks like one.
        replay = getattr(state, "replay_clock_s", None)
        return {
            "sim_now": state.sim_now,
            "clock_s": state.sim_now if replay is None else min(replay, duration or replay),
            "footage_duration_s": duration,
            "paused": state.feed_paused,
            "replay_complete": bool(getattr(state, "replay_complete", False)),
            "world_source": state.world_source,
            "run_generation": state.run_generation,
            "scenario": scen.name if scen is not None else "",
            "counters": dict(state.counters),
            "scale": {
                "cameras": len(state.graph.cameras),
                "ground_truth_vehicles": (
                    len({sp.vehicle_id for sp in scen.spans}) if scen else 0),
                "ground_truth_tracks": len(scen.spans) if scen else 0,
                "transit_routes": len(state.graph.edges),
            },
        }

    @app.post("/api/cityflow/scenario")
    async def switch_scenario(req: dict):
        """Replay a different scenario without restarting the process.

        Implemented as a reset with a target scenario attached, because that is
        exactly what it is: a different scenario means different cameras, a
        different road graph and a different feed, so continuing the current
        run's targets and beliefs across the switch would be meaningless. The
        supervisor performs the swap between runs, where it owns the feed.
        """
        name = str(req.get("scenario") or "").strip()
        if state.cityflow_root is None:
            raise HTTPException(400, "not running a real-data scenario")
        from datasets.cityflow import CityFlow

        available = CityFlow(Path(state.cityflow_root)).scenario_names()
        if name not in available:
            raise HTTPException(
                404, f"unknown scenario {name!r}; have {available}")
        if state.cityflow_scenario is not None and name == state.cityflow_scenario.name:
            return {"switching": False, "scenario": name, "reason": "already active"}
        state.pending_scenario = name
        state.restart_requested = True
        await state.manager.broadcast({"type": "resetting", "scenario": name})
        return {"switching": True, "scenario": name}

    @app.post("/api/reset")
    async def reset_run():
        """Restart the replay from t=0 and clear everything it produced.

        Sets a flag rather than doing the work here: the feed runs in another
        thread's event loop (start.py), and tearing it down from a request
        handler would race the camera tasks mid-POST. The supervisor notices,
        cancels the feed, wipes runtime state, and replays from the top.

        Clearing is not optional and not a separate button. A rewound clock
        beside targets, reviews and beliefs accumulated from the previous pass
        would show past footage next to present-tense conclusions — the same
        reason `/api/feed_control` offers pause but never rewind.
        """
        state.restart_requested = True
        await state.manager.broadcast({"type": "resetting"})
        return {"restarting": True, "run_generation": state.run_generation}

    @app.get("/api/cityflow/camera/{camera_id}/frame.jpg")
    def cityflow_camera_frame(camera_id: str, t: float | None = None):
        """One JPEG of what this camera sees at footage time `t`.

        Decoding is sequential-with-seek-fallback (see
        cityflow_video.VideoFrameSource.frame_at): during normal playback the
        requested frame is a little ahead of the last one, which is a few cheap
        reads, and only a jump backwards or a long skip pays for a seek. Random
        seeking every request is what makes naive frame servers unusable on
        1080p AVI.
        """
        from fastapi.responses import Response

        cam_dir = (state.cityflow_camera_dirs or {}).get(camera_id)
        if cam_dir is None:
            raise HTTPException(404, f"no such camera: {camera_id!r}")
        import cv2

        from datasets.cityflow import DEFAULT_FPS
        from datasets.cityflow_video import VideoFrameSource

        sources = getattr(state, "_frame_sources", None)
        if sources is None:
            sources = state._frame_sources = {}
        # One lock per camera. This is a SYNC route, so FastAPI runs it in the
        # threadpool and several browsers — or one impatient one — can be
        # inside it at once. VideoFrameSource wraps a single cv2.VideoCapture
        # and mutates its decode position; driving one handle from several
        # threads is undefined behaviour in OpenCV and took the whole process
        # down under 40 concurrent frame requests, which is an ordinary load
        # for a console with five cameras and a couple of tabs open.
        with state.frame_lock:
            src = sources.get(camera_id)
            if src is None:
                src = sources[camera_id] = VideoFrameSource(cam_dir / "vdo.avi")
            lock = state.frame_locks.setdefault(camera_id, threading.Lock())
        at = state.sim_now if t is None else t
        # Bound the request before it reaches int(): t=1e308 overflowed the
        # conversion and 500'd.
        if not (at == at) or at in (float("inf"), float("-inf")):
            raise HTTPException(422, "t must be a finite number")
        at = min(max(0.0, at), 24 * 3600.0)
        with lock:
            img = src.frame_at(int(at * DEFAULT_FPS))
        if img is None:
            raise HTTPException(404, "no frame at that time")
        # Downscale before encoding. Source frames are 1080p and encode to
        # ~280 KB, which at a 4 fps refresh is both slower than the refresh
        # interval and pointless: the overlay renders far smaller than this.
        max_w = 960
        if img.shape[1] > max_w:
            scale = max_w / img.shape[1]
            img = cv2.resize(img, (max_w, int(img.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            raise HTTPException(500, "frame encode failed")
        return Response(content=jpg.tobytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/cityflow/{scenario}/timeline")
    def cityflow_timeline(scenario: str):
        """Per-camera passage lanes: when each vehicle was in each view.

        This is what makes the dataset's structure legible. In S01 a single
        vehicle is frequently present in all five cameras at once, which no
        other view in the console reveals and which is the reason the transit
        windows collapse to their floor.
        """
        scen = state.cityflow_scenario
        if scen is None or scenario != scen.name:
            raise HTTPException(404, f"scenario {scenario!r} is not active")
        lanes: dict[str, list[dict]] = {cam: [] for cam in scen.cameras}
        for sp in scen.spans:
            lanes.setdefault(sp.camera_id, []).append({
                "vehicle_id": sp.vehicle_id,
                "enter_s": round(sp.enter_s, 2), "exit_s": round(sp.exit_s, 2)})
        duration = max((sp.exit_s for sp in scen.spans), default=0.0)
        return {"scenario": scen.name, "duration_s": duration,
                "cameras": list(scen.cameras),
                "lanes": {c: sorted(v, key=lambda p: p["enter_s"])
                          for c, v in lanes.items()}}

    @app.get("/api/cityflow/scenarios")
    def cityflow_scenarios():
        """Every scenario in the dataset this server was pointed at, not
        just the one actively replaying -- discovery only; only the active
        scenario (below) can actually be browsed/flagged."""
        from datasets.cityflow import CityFlow

        if state.cityflow_root is None or not CityFlow.exists(state.cityflow_root):
            return []
        return CityFlow(state.cityflow_root).scenario_names()

    @app.get("/api/cityflow/{scenario}/vehicles")
    def cityflow_vehicles(scenario: str, min_cameras: int = 1):
        """`min_cameras=2` restricts the browse list to vehicles the ground
        truth actually saw at more than one camera.

        Flagging a single-camera vehicle cannot demonstrate re-identification:
        there is no second sighting for the cascade to associate, so the review
        queue stays empty and the run looks broken when it is working exactly
        as specified. Filtering server-side keeps the thumbnails out of the
        response rather than hiding them after they are sent."""
        if state.cityflow_scenario is None or scenario != state.cityflow_scenario.name:
            raise HTTPException(
                404, f"scenario {scenario!r} is not the active real-data scenario "
                     f"({state.cityflow_scenario.name if state.cityflow_scenario else 'none loaded'})")
        if state.cityflow_vehicle_index is None:
            # Never build inline. This endpoint is sync, so FastAPI runs it in
            # the threadpool, and a cold build decodes three frames per vehicle
            # out of 1080p video — tens of seconds holding a worker. A client
            # polling while it waited would start a SECOND build in a second
            # worker, and so on, until the pool and the CPU were both gone and
            # the whole API stopped answering. Kick off exactly one build in
            # the background and tell the caller to come back.
            import threading

            with state.index_lock:
                if not state.index_building:
                    state.index_building = True
                    scen, cams = state.cityflow_scenario, state.cityflow_camera_dirs

                    def _build():
                        from server.real_feed import build_vehicle_index
                        try:
                            index = build_vehicle_index(scen, cams)
                            # Only publish if the scenario has not changed
                            # under us; a switch mid-build must not install the
                            # previous scenario's vehicles.
                            if state.cityflow_scenario is scen:
                                state.cityflow_vehicle_index = index
                        except Exception as exc:      # noqa: BLE001
                            print(f"vehicle index build failed: {exc}")
                        finally:
                            state.index_building = False

                    threading.Thread(target=_build, daemon=True,
                                     name="vehicle-index").start()
            raise HTTPException(
                503, f"building the vehicle index for {scenario}; retry shortly")
        rows = state.cityflow_vehicle_index
        if min_cameras > 1:
            rows = [v for v in rows if v.get("n_cameras", 1) >= min_cameras]
        # Strip the reference gallery. It is 3 FULL-RESOLUTION crops per
        # vehicle and made this response 31 MB for S01's 95 vehicles — 94% of
        # the payload — to draw a grid of 90px thumbnails. The browser parsed
        # and held all of it on every load and every restart, which is most of
        # why the console felt slow to come back. The gallery is only needed
        # for the one vehicle actually being flagged, so it is fetched then.
        return [{k: val for k, val in v.items() if k != "gallery_b64"}
                for v in rows]

    @app.get("/api/cityflow/{scenario}/vehicles/{vehicle_id}/gallery")
    def cityflow_vehicle_gallery(scenario: str, vehicle_id: int):
        """Full-resolution passage crops for ONE vehicle, for seeding a flag.

        Separate from the browse list so the list stays small: these are the
        crops the appearance model embeds, so they must not be downscaled, and
        shipping 95 vehicles' worth to show thumbnails was pure waste.
        """
        scen = state.cityflow_scenario
        if scen is None or scenario != scen.name:
            raise HTTPException(404, f"scenario {scenario!r} is not active")
        if state.cityflow_vehicle_index is None:
            raise HTTPException(503, "vehicle index still building")
        for v in state.cityflow_vehicle_index:
            if v["vehicle_id"] == vehicle_id:
                return {"vehicle_id": vehicle_id,
                        "gallery_b64": v.get("gallery_b64") or []}
        raise HTTPException(404, f"no vehicle {vehicle_id} in {scenario}")

    # ---------------------------------------------------------- inspector

    @app.post("/api/inspect/evaluate")
    def inspect_evaluate(req: InspectRequest):
        """Reasoning sandbox: run the real cascade on hand-built inputs.

        No tracker, no DB write, no audit entry — this is a "what would the
        system conclude" tool, not a live sighting. It exists so a human can
        see the fact list, structured signals, distinctiveness score, and
        counterfactuals for ANY hypothetical scenario, not just ones that
        happen to occur during a sim run. ReID similarity is simulated
        directly (a slider, not a real image) via a pair of synthetic unit
        vectors constructed to have exactly the requested cosine similarity.
        """
        from reasoning.cascade import CascadeConfig, rank_candidates
        from reasoning.profile import LastSeen, TargetProfile

        known = set(state.graph.camera_ids())
        bad_cams = {req.sighting.camera_id} - known
        for t in req.targets:
            if t.last_seen_camera_id:
                bad_cams |= {t.last_seen_camera_id} - known
        if bad_cams:
            raise HTTPException(422, f"unknown camera id(s): {sorted(bad_cams)}")

        # A fixed basis vector for the sighting; each target's synthetic
        # gallery vector is placed so its cosine similarity to this one
        # equals exactly the requested reid_similarity (or omitted -> no
        # gallery -> ReID unavailable for that target, same as a
        # never-confirmed profile).
        dim = 8
        obs_emb = np.zeros(dim, dtype=np.float32)
        obs_emb[0] = 1.0
        plate = (PlateRead(req.sighting.plate_text.upper(), req.sighting.plate_confidence,
                           "sim") if req.sighting.plate_text else None)
        obs = Observation(
            event_id="sandbox", camera_id=req.sighting.camera_id,
            timestamp_s=req.sighting.timestamp_s, lat=0.0, lon=0.0,
            embedding=obs_emb, plate=plate,
            class_attrs=dict(req.sighting.class_attrs), class_attrs_source="sandbox",
            instance_attrs=dict(req.sighting.instance_attrs), detection_source="sandbox",
        )

        labels: dict[str, str] = {}
        profiles = []
        for t in req.targets:
            last_seen = (
                LastSeen(t.last_seen_camera_id, t.last_seen_timestamp_s, "sandbox-last-seen")
                if t.last_seen_camera_id and t.last_seen_timestamp_s is not None else None)
            gallery = ()
            if t.reid_similarity is not None:
                s = max(-1.0, min(1.0, t.reid_similarity))
                vec = np.zeros(dim, dtype=np.float32)
                vec[0], vec[1] = s, float(np.sqrt(max(0.0, 1.0 - s * s)))
                gallery = (vec,)
            profiles.append(TargetProfile(
                target_id=t.target_id, label=t.label, plate=t.plate.upper().strip(),
                class_attrs=dict(t.class_attrs), instance_attrs=dict(t.instance_attrs),
                gallery=gallery, last_seen=last_seen,
            ))
            labels[t.target_id] = t.label

        cfg = CascadeConfig(distinctiveness_floor=req.distinctiveness_floor) \
            if req.distinctiveness_floor is not None else CascadeConfig()
        ranked = rank_candidates(obs, profiles, state.graph, cfg)

        def decision_json(d):
            return {
                "target_id": d.target_id, "label": labels.get(d.target_id, d.target_id),
                "verdict": d.verdict, "score": round(d.score, 4),
                "deciding_tier": d.deciding_tier,
                "distinctiveness": round(d.distinctiveness, 4),
                "refused_to_individuate": d.refused_to_individuate,
                # Always present (possibly empty) so clients never branch on
                # its absence — only ranked.best carries the real candidate
                # set today, but every decision serializes the same shape.
                "candidate_ids": list(d.candidate_ids),
                "requires_review": d.requires_review, "anomaly": d.anomaly,
                "reid_similarity": round(d.reid_similarity, 4),
                "facts": [{"kind": f.kind, "text": f.text, "check": f.check}
                         for f in d.facts],
                "counterfactuals": [
                    {"signal": c.signal, "current_outcome": c.current_outcome,
                     "flipped_outcome": c.flipped_outcome, "boundary": c.boundary,
                     "text": c.text}
                    for c in d.counterfactuals],
                "signals": dataclasses.asdict(d.signals) if d.signals else None,
            }

        all_decisions = [
            decision_json(ranked.best if d.target_id == ranked.best.target_id else d)
            for d in ranked.all_decisions
        ]
        return {
            "best": decision_json(ranked.best),
            "margin": round(ranked.margin, 4),
            "all_decisions": all_decisions,
        }

    @app.get("/api/inspect/render")
    def inspect_render(camera_id: str, timestamp_s: float = 0.0, payload: str = "{}"):
        """A SYNTHETIC camera frame standing in for the sandbox's sighting or
        a target's last-confirmed view — the inspector's visual centerpiece.

        Nothing here is a real camera feed: it's the same procedural sprite
        renderer the sim uses, driven by whatever attrs are in the form right
        now, so the panel updates live as the operator edits them. Swapping
        this for real per-camera frames (once a real dataset is wired into a
        live inspector) is a natural extension, not a redesign — the frontend
        only cares that this URL returns a PNG for a given camera+attrs.
        """
        import json

        import cv2

        from sim.model import VehicleIdentity
        from sim.render import COLOR_BGR, render_frame

        if camera_id not in state.graph.camera_ids():
            raise HTTPException(422, f"unknown camera id: {camera_id}")
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            raise HTTPException(422, "payload must be valid JSON")
        if not isinstance(data, dict):
            raise HTTPException(422, "payload must be a JSON object")

        color = str(data.get("color") or "").lower()
        if color not in COLOR_BGR:
            color = "gray"  # renderer only knows a fixed palette; fall back honestly
        vehicle = VehicleIdentity(
            vehicle_id="sandbox",
            plate=str(data.get("plate") or "SANDBOX"),
            make=str(data.get("make") or "Unknown"),
            model=str(data.get("model") or "Vehicle"),
            body_type=str(data.get("body_type") or "sedan"),
            color=color,
            instance_attrs={str(k): str(v) for k, v in (data.get("instance_attrs") or {}).items()},
        )
        frame = render_frame(vehicle, state.graph.camera(camera_id), timestamp_s)
        ok, png = cv2.imencode(".png", frame)
        if not ok:
            raise HTTPException(500, "render failed")
        return Response(content=png.tobytes(), media_type="image/png")

    @app.get("/api/targets/{target_id}/model3d")
    def target_model3d(target_id: str):
        """3D-model status for the dossier. exists=false when 3D is disabled
        or nothing has been fused yet — the UI hides the section then."""
        from car3d.geometry import signature_from_cloud
        from car3d.profile_model import Target3DModel

        # An unknown target has no model state to report. Answering 200
        # {"exists": false} for an id that does not exist is the same answer as
        # "flagged, nothing fused yet", so a typo or a stale dossier link looked
        # like a live target with no reconstruction. /api/targets/{id} already
        # 404s; this now agrees with it.
        if target_id not in state.tracker.targets():
            raise HTTPException(404, "unknown target")

        # Settle any queued fusion first, so the dossier never reports a
        # target as having no model purely because the worker is mid-flight.
        _await_fusion(target_id)
        model = Target3DModel(target_id, state.targets3d_dir)
        if not model.exists():
            return {"exists": False, "enabled": state.enable_3d}
        asset = model.load()
        sig = signature_from_cloud(asset.cloud)
        # Derived artefacts are produced here rather than per fusion.
        model.ensure_turntable(provenance_overlay=True)
        return {
            "exists": True, "enabled": state.enable_3d,
            "observations": len(asset.observations),
            "n_splats": asset.cloud.n,
            "observed_fraction": sig.observed_fraction if sig else 0.0,
            "geometry": ({"body_profile": sig.body_profile,
                          "length_class": sig.length_class,
                          "lw_ratio": sig.lw_ratio, "hl_ratio": sig.hl_ratio,
                          "trustworthy": sig.trustworthy} if sig else None),
            "turntable": f"/api/targets/{target_id}/model3d/turntable_provenance.png",
            "exports": {
                "splat": f"/api/targets/{target_id}/model3d/model.splat",
                "ply": f"/api/targets/{target_id}/model3d/model.ply",
                "provenance_ply":
                    f"/api/targets/{target_id}/model3d/model_provenance.ply",
            },
        }

    @app.get("/api/targets/{target_id}/model3d/{name}")
    def target_model3d_file(target_id: str, name: str):
        from car3d.profile_model import Target3DModel

        # Same null-byte guard as /api/crops: resolve() raises rather than
        # returning a path, and an unrepresentable name is a 404, not a 500.
        try:
            base = (state.targets3d_dir / target_id / "exports").resolve()
            path = (base / name).resolve()
        except (ValueError, OSError):
            raise HTTPException(404, "no such model file") from None
        # Resolve traversal BEFORE generating anything, so a hostile `name`
        # cannot make us do work outside the export directory.
        if base not in path.parents:
            raise HTTPException(404, "no such model file")
        _await_fusion(target_id)
        model = Target3DModel(target_id, state.targets3d_dir)
        if name.endswith(".png"):
            model.ensure_turntable(provenance_overlay="provenance" in name)
        elif name.endswith((".splat", ".ply")):
            model.ensure_exports()
        if not path.is_file():
            raise HTTPException(404, "no such model file")
        # A model export (.splat/.ply) leaves the system — audit it. Turntable
        # PNGs are UI and not logged.
        if name.endswith((".splat", ".ply")):
            with Session(engine) as session:
                audit_record(session, "operator", "export_model3d",
                             {"target_id": target_id, "file": name}, state.sim_now)
                session.commit()
        media = "image/png" if name.endswith(".png") else "application/octet-stream"
        return FileResponse(path, media_type=media)

    @app.get("/api/crops/{name}")
    def get_crop(name: str):
        # resolve() raises ValueError on an embedded null byte rather than
        # returning a path, which surfaced as a 500 on `/api/crops/x%00.jpg`.
        # A name that cannot denote a file is simply not found.
        try:
            path = (state.crops_dir / name).resolve()
        except (ValueError, OSError):
            raise HTTPException(404, "no such crop") from None
        if state.crops_dir.resolve() not in path.parents or not path.is_file():
            raise HTTPException(404, "no such crop")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/snapshot")
    def get_snapshot():
        return {"timestamp_s": state.sim_now,
                "targets": state.tracker.snapshot(state.sim_now)}

    # -------------------------------------------------------------- ws/web

    @app.websocket("/ws/console")
    async def console_ws(ws: WebSocket):
        await state.manager.connect(ws)
        try:
            await ws.send_json({
                "type": "snapshot", "timestamp_s": state.sim_now,
                "targets": state.tracker.snapshot(state.sim_now)})
            while True:
                await ws.receive_text()  # console is read-mostly; ignore pings
        except WebSocketDisconnect:
            state.manager.disconnect(ws)

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="console")

        @app.middleware("http")
        async def _no_stale_console(request, call_next):
            # The console JS/CSS changes often between demo runs and browsers
            # otherwise serve day-old copies from disk cache; no-cache makes
            # them revalidate (304 when unchanged) instead of going stale.
            response = await call_next(request)
            if not request.url.path.startswith("/api"):
                response.headers.setdefault("Cache-Control", "no-cache")
            return response

    return app
