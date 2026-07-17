"""SQLite-backed audit chain: appends AuditRow, verifies from the DB.

Append reads the current tail, chains onto it, and inserts — call it inside
the same transaction as the state change it records, so the audit entry and
the change commit together.
"""
from __future__ import annotations

from typing import Any, Mapping

from sqlmodel import Session, select

from audit.log import AuditEntry, ChainVerification, append_entry, verify_chain
from server.db import AuditRow


def _row_to_entry(row: AuditRow) -> AuditEntry:
    return AuditEntry(
        seq=row.seq, timestamp_s=row.timestamp_s, actor=row.actor,
        action=row.action, payload_digest=row.payload_digest,
        prev_hash=row.prev_hash, entry_hash=row.entry_hash,
    )


def _tail(session: Session) -> AuditEntry | None:
    row = session.exec(
        select(AuditRow).order_by(AuditRow.seq.desc()).limit(1)).first()
    return _row_to_entry(row) if row else None


def record(
    session: Session,
    actor: str,
    action: str,
    payload: Mapping[str, Any] | None,
    timestamp_s: float,
) -> AuditEntry:
    """Append one chained entry. Adds to the session (caller commits)."""
    entry = append_entry(_tail(session), actor, action, payload, timestamp_s)
    session.add(AuditRow(
        seq=entry.seq, timestamp_s=entry.timestamp_s, actor=entry.actor,
        action=entry.action, payload_digest=entry.payload_digest,
        prev_hash=entry.prev_hash, entry_hash=entry.entry_hash,
    ))
    return entry


def load_entries(session: Session, limit: int | None = None) -> list[AuditEntry]:
    q = select(AuditRow).order_by(AuditRow.seq)
    rows = session.exec(q).all()
    entries = [_row_to_entry(r) for r in rows]
    return entries[-limit:] if limit else entries


def verify(session: Session) -> ChainVerification:
    return verify_chain(load_entries(session))
