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
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select

from audit.store import load_entries as audit_load
from audit.store import record as audit_record
from audit.store import verify as audit_verify
from perception.types import Observation, PlateRead
from reasoning.profile import profile_from_flag
from server import db as dbm
from server.schemas import (
    FlagTargetRequest, InspectRequest, ProfileEditRequest, ReviewResolution,
    SightingReport,
)
from server.ws import ConnectionManager
from sim.road_graph import RoadGraph, default_world
from tracking.tracker import FleetTracker

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app(
    graph: RoadGraph | None = None,
    db_url: str = dbm.DEFAULT_DB_URL,
    crops_dir: str = "data/crops",
    calibration_path: str = "calibration/artifacts/latest.json",
    enable_3d: bool | None = None,
    targets3d_dir: str = "data/targets3d",
    world_source: str = "synthetic",
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
    app = FastAPI(title="Eyes Everywhere (synthetic demo)")
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
    state.tracker = FleetTracker(graph, cascade_config)
    state.manager = ConnectionManager()
    state.crops_dir = Path(crops_dir)
    state.sim_now = 0.0
    state.target_seq = itertools.count(1)
    state.enable_3d = enable_3d
    state.targets3d_dir = Path(targets3d_dir)
    state.world_source = world_source
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

    state.embed_bgr = _embed_bgr

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

    def _fuse_3d_for_events(events, session: Session) -> None:
        """Fuse gated (confirmed) sightings into per-target cargen models.

        Runs only on profile_update events — the exact moments the profile
        gate opened — so cargen's pending-approval merge policy and this
        project's update gate stay one mechanism. Failures degrade to a
        console note; 3D is corroborative, never load-bearing.
        """
        if not state.enable_3d:
            return
        import cv2

        from car3d.geometry import signature_to_attrs
        from car3d.profile_model import Target3DModel

        for ev in [e for e in events if e.kind == "profile_update" and e.event_id]:
            crop_path = state.crops_dir / f"{ev.event_id}.png"
            crop = cv2.imread(str(crop_path)) if crop_path.exists() else None
            if crop is None:
                continue
            try:
                model = Target3DModel(ev.target_id, state.targets3d_dir)
                outcome = model.fuse_confirmed_crop(
                    crop, ev.event_id,
                    reason=str(ev.detail.get("reason", "gated update")),
                    timestamp=ev.timestamp_s)
                model.turntable_png(provenance_overlay=True)
            except Exception as exc:  # noqa: BLE001 — 3D must never sink ingest
                print(f"car3d: fusion failed for {ev.target_id}: {exc}")
                continue
            geom_attrs = signature_to_attrs(outcome.geometry)
            if geom_attrs:
                tracked = state.tracker.targets().get(ev.target_id)
                if tracked is not None:
                    profile = dataclasses.replace(
                        tracked.profile,
                        instance_attrs={**tracked.profile.instance_attrs,
                                        **geom_attrs},
                        version=tracked.profile.version + 1)
                    state.tracker.replace_profile(ev.target_id, profile)
                    session.add(dbm.ProfileUpdateRow(
                        target_id=ev.target_id, event_id=ev.event_id,
                        version=profile.version,
                        reason="3D geometry attributes refreshed from the "
                               "fused model (gated fusion, reversible).",
                        timestamp_s=ev.timestamp_s))

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
                    if row and not row.reference_crop:
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

    @app.post("/api/targets", status_code=201)
    def flag_target(req: FlagTargetRequest):
        target_id = f"tgt-{next(state.target_seq):03d}"
        profile = profile_from_flag(
            target_id, req.label, req.plate, req.class_attrs, req.instance_attrs)
        state.tracker.flag_target(profile)
        with Session(engine) as session:
            session.add(dbm.TargetRow(
                target_id=target_id, label=req.label, plate=profile.plate,
                class_attrs=dbm.dumps(dict(profile.class_attrs)),
                instance_attrs=dbm.dumps(dict(profile.instance_attrs)),
                created_s=state.sim_now))
            audit_record(session, "operator", "flag_target",
                         {"target_id": target_id, "label": req.label,
                          "plate": profile.plate}, state.sim_now)
            session.commit()
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
                crop_path=crop_name, truth_id=obs.eval_truth_id))
            _persist_events(session, events, obs)
            _fuse_3d_for_events(events, session)
            _sync_target_rows(session)
            audit_record(session, f"camera:{obs.camera_id}", "report_sighting",
                         {"event_id": obs.event_id, "camera_id": obs.camera_id,
                          "outcomes": [e.kind for e in events]}, obs.timestamp_s)
            session.commit()
        await state.manager.broadcast({
            "type": "contact", "event_id": obs.event_id,
            "camera_id": obs.camera_id, "timestamp_s": obs.timestamp_s,
            "lat": obs.lat, "lon": obs.lon,
            "class_attrs": dict(obs.class_attrs),
            "crop": f"/api/crops/{crop_name}" if crop_name else "",
            "detection_source": obs.detection_source})
        await _broadcast_events(events)
        await state.manager.broadcast({
            "type": "snapshot", "timestamp_s": state.sim_now,
            "targets": state.tracker.snapshot(state.sim_now)})
        return {"events": [e.kind for e in events]}

    # ------------------------------------------------------ reviews/alerts

    @app.get("/api/reviews")
    def list_reviews(status: str = "pending"):
        with Session(engine) as session:
            rows = session.exec(
                select(dbm.ReviewRow).where(dbm.ReviewRow.status == status)
                .order_by(dbm.ReviewRow.created_s)).all()
            out = []
            for r in rows:
                sighting = session.get(dbm.SightingRow, r.event_id)
                target = session.get(dbm.TargetRow, r.target_id)
                out.append({
                    **r.model_dump(),
                    "rivals": dbm.loads(r.rivals),
                    "counterfactuals": dbm.loads(r.counterfactuals) or [],
                    "sighting_crop": (f"/api/crops/{sighting.crop_path}"
                                      if sighting and sighting.crop_path else ""),
                    "reference_crop": (f"/api/crops/{target.reference_crop}"
                                       if target and target.reference_crop else ""),
                    "target_label": target.label if target else "",
                })
            return out

    @app.post("/api/reviews/{review_id}/resolve")
    async def operator_confirm_match(review_id: str, res: ReviewResolution):
        try:
            events = state.tracker.resolve_review(review_id, res.accept, state.sim_now)
        except KeyError:
            raise HTTPException(404, "unknown or already-resolved review")
        with Session(engine) as session:
            row = session.get(dbm.ReviewRow, review_id)
            if row:
                row.status = "accepted" if res.accept else "rejected"
                row.resolved_s = state.sim_now
                session.add(row)
            _persist_events(session, events, None)
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
        return [dataclasses.asdict(c) for c in graph.cameras]

    @app.get("/api/adjacency")
    def camera_adjacency():
        return [dataclasses.asdict(e) for e in graph.edges]

    @app.get("/api/world_source")
    def world_source_info():
        """Tells the frontend whether the camera graph is our fictional
        Gridville world or built from a real dataset's real GPS — the map
        uses this to decide whether to draw its own road network or a real
        basemap tile layer under the camera pins. See create_app's
        world_source docstring for why these must never be mixed."""
        return {"source": state.world_source}

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

        known = set(graph.camera_ids())
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
        ranked = rank_candidates(obs, profiles, graph, cfg)

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

        if camera_id not in graph.camera_ids():
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
        frame = render_frame(vehicle, graph.camera(camera_id), timestamp_s)
        ok, png = cv2.imencode(".png", frame)
        if not ok:
            raise HTTPException(500, "render failed")
        return Response(content=png.tobytes(), media_type="image/png")

    @app.get("/api/targets/{target_id}/model3d")
    def target_model3d(target_id: str):
        """3D-model status for the dossier. exists=false when 3D is disabled
        or nothing has been fused yet — the UI hides the section then."""
        from car3d.profile_model import Target3DModel

        model = Target3DModel(target_id, state.targets3d_dir)
        if not model.exists():
            return {"exists": False, "enabled": state.enable_3d}
        asset = model.load()
        sig = model.geometry()
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
        base = (state.targets3d_dir / target_id / "exports").resolve()
        path = (base / name).resolve()
        if base not in path.parents or not path.is_file():
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
        path = (state.crops_dir / name).resolve()
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

    return app
