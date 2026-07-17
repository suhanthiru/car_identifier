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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select

from perception.types import Observation, PlateRead
from reasoning.profile import profile_from_flag
from server import db as dbm
from server.schemas import (
    FlagTargetRequest, ProfileEditRequest, ReviewResolution, SightingReport,
)
from server.ws import ConnectionManager
from sim.road_graph import RoadGraph, default_world
from tracking.tracker import FleetTracker

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app(
    graph: RoadGraph | None = None,
    db_url: str = dbm.DEFAULT_DB_URL,
    crops_dir: str = "data/crops",
) -> FastAPI:
    graph = graph or default_world()
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
    state.tracker = FleetTracker(graph)
    state.manager = ConnectionManager()
    state.crops_dir = Path(crops_dir)
    state.sim_now = 0.0
    state.target_seq = itertools.count(1)

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
            updates = session.exec(
                select(dbm.ProfileUpdateRow)
                .where(dbm.ProfileUpdateRow.target_id == target_id)
                .order_by(dbm.ProfileUpdateRow.version)).all()
            chain = session.exec(
                select(dbm.CorroborationRow)
                .where(dbm.CorroborationRow.target_id == target_id)
                .order_by(dbm.CorroborationRow.timestamp_s)).all()
        snap = state.tracker.snapshot(state.sim_now).get(target_id, {})
        return {
            "target_id": target_id,
            "label": tracked.profile.label,
            "plate": tracked.profile.plate,
            "class_attrs": dict(tracked.profile.class_attrs),
            "instance_attrs": dict(tracked.profile.instance_attrs),
            "gallery_size": len(tracked.profile.gallery),
            "reference_crop": (row.reference_crop if row else ""),
            "live": snap,
            "profile_updates": [u.model_dump() for u in updates],
            "corroboration_chain": [c.model_dump() for c in chain],
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
            session.commit()
        return {"target_id": target_id, "version": profile.version}

    @app.delete("/api/targets/{target_id}", status_code=204)
    def unflag_target(target_id: str):
        state.tracker.unflag_target(target_id)

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
            _sync_target_rows(session)
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
            _sync_target_rows(session)
            session.commit()
        await _broadcast_events(events)
        await state.manager.broadcast({
            "type": "snapshot", "timestamp_s": state.sim_now,
            "targets": state.tracker.snapshot(state.sim_now)})
        return {"resolved": review_id, "accepted": res.accept}

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
