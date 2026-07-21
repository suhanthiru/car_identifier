"""SQLite storage via SQLModel.

Tables mirror the brief: targets, cameras, camera_adjacency, sightings,
corroboration_chains, profile_updates, plus reviews. JSON-ish fields are
stored as serialized strings (SQLite-friendly, human-inspectable).

The sightings table keeps `truth_id` — the simulator's answer key. A real
deployment would not have this column; it exists so the calibration and
evaluation harnesses can score decisions, and nothing on the serving path
reads it.
"""
from __future__ import annotations

import json
from typing import Iterator

from sqlmodel import Field, Session, SQLModel, create_engine, select

DEFAULT_DB_URL = "sqlite:///data/eyes.sqlite"


class TargetRow(SQLModel, table=True):
    __tablename__ = "targets"
    target_id: str = Field(primary_key=True)
    label: str = ""
    plate: str = ""
    class_attrs: str = "{}"        # JSON
    instance_attrs: str = "{}"     # JSON
    state: str = "tentative"
    belief: float = 0.0
    profile_version: int = 0
    reference_crop: str = ""       # path of the crop shown in review UI
    created_s: float = 0.0


class CameraRow(SQLModel, table=True):
    __tablename__ = "cameras"
    camera_id: str = Field(primary_key=True)
    name: str = ""
    lat: float = 0.0
    lon: float = 0.0
    heading_deg: float = 0.0


class AdjacencyRow(SQLModel, table=True):
    __tablename__ = "camera_adjacency"
    id: int | None = Field(default=None, primary_key=True)
    src: str = Field(index=True)
    dst: str = ""
    distance_m: float = 0.0
    min_s: float = 0.0
    typical_s: float = 0.0
    max_s: float = 0.0


class SightingRow(SQLModel, table=True):
    __tablename__ = "sightings"
    event_id: str = Field(primary_key=True)
    camera_id: str = Field(index=True)
    timestamp_s: float = Field(index=True)
    lat: float = 0.0
    lon: float = 0.0
    plate_text: str = ""
    plate_conf: float = 0.0
    plate_source: str = ""
    class_attrs: str = "{}"
    instance_attrs: str = "{}"
    detection_source: str = ""
    crop_path: str = ""
    # Number of short-clip frames saved for this sighting as
    # "{event_id}.f{i}.png" in the crops dir (0 = still only). The console
    # derives their /api/crops URLs from event_id + this count.
    clip_frame_count: int = 0
    # Simulator ground truth — evaluation only, never read on the serving path.
    truth_id: str = ""


class CorroborationRow(SQLModel, table=True):
    __tablename__ = "corroboration_chains"
    id: int | None = Field(default=None, primary_key=True)
    target_id: str = Field(index=True)
    event_id: str = ""
    timestamp_s: float = 0.0
    verdict: str = ""
    belief_after: float = 0.0
    facts: str = ""


class ProfileUpdateRow(SQLModel, table=True):
    __tablename__ = "profile_updates"
    id: int | None = Field(default=None, primary_key=True)
    target_id: str = Field(index=True)
    event_id: str = ""
    version: int = 0
    reason: str = ""
    timestamp_s: float = 0.0


class ReviewRow(SQLModel, table=True):
    __tablename__ = "reviews"
    review_id: str = Field(primary_key=True)
    target_id: str = Field(index=True)
    event_id: str = ""
    kind: str = "review"           # review | anomaly
    status: str = "pending"        # pending | accepted | rejected
    score: float = 0.0
    facts: str = ""
    counterfactuals: str = "[]"    # JSON list of plain-English flip explanations
    rivals: str = "[]"             # JSON list of rival target ids
    created_s: float = 0.0
    resolved_s: float | None = None


class AlertRow(SQLModel, table=True):
    __tablename__ = "alerts"
    id: int | None = Field(default=None, primary_key=True)
    kind: str = ""
    target_id: str = Field(index=True)
    event_id: str = ""
    timestamp_s: float = 0.0
    detail: str = "{}"             # JSON


class AuditRow(SQLModel, table=True):
    __tablename__ = "audit_log"
    # Append-only, hash-chained (see audit/). id is storage order; seq is the
    # chain position. entry_hash commits to prev_hash + the entry contents.
    id: int | None = Field(default=None, primary_key=True)
    seq: int = Field(index=True)
    timestamp_s: float = 0.0
    actor: str = ""
    action: str = Field(default="", index=True)
    payload_digest: str = ""
    prev_hash: str = ""
    entry_hash: str = ""


def make_engine(db_url: str = DEFAULT_DB_URL, echo: bool = False):
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, echo=echo, connect_args=connect_args)
    SQLModel.metadata.create_all(engine)
    return engine


def get_session(engine) -> Iterator[Session]:
    with Session(engine) as session:
        yield session


def dumps(obj) -> str:
    return json.dumps(obj, default=str)


def loads(text: str):
    return json.loads(text) if text else None


def store_graph(session: Session, graph) -> None:
    """Idempotently persist the camera registry + adjacency."""
    for cam in graph.cameras:
        session.merge(CameraRow(
            camera_id=cam.camera_id, name=cam.name,
            lat=cam.lat, lon=cam.lon, heading_deg=cam.heading_deg))
    existing = set(session.exec(select(AdjacencyRow.src, AdjacencyRow.dst)).all())
    for e in graph.edges:
        if (e.src, e.dst) not in existing:
            session.add(AdjacencyRow(
                src=e.src, dst=e.dst, distance_m=e.distance_m,
                min_s=e.min_s, typical_s=e.typical_s, max_s=e.max_s))
    session.commit()
