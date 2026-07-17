"""Hash-chain primitives — storage-agnostic and fully unit-testable.

Each entry commits to its predecessor:

    entry_hash = sha256(prev_hash + canonical_json(entry_without_hash))

`canonical_json` is deterministic (sorted keys, tight separators), so the same
logical entry always hashes identically. The genesis entry's `prev_hash` is a
fixed sentinel. `verify_chain` recomputes every link and reports the first
index where the stored hash disagrees — the tamper location.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

GENESIS_PREV_HASH = "0" * 64


@dataclass(frozen=True)
class AuditEntry:
    seq: int
    timestamp_s: float
    actor: str
    action: str
    payload_digest: str      # sha256 of the action's payload (not the payload itself)
    prev_hash: str
    entry_hash: str = ""     # filled by append_entry / entry_hash

    def _core(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp_s": self.timestamp_s,
            "actor": self.actor,
            "action": self.action,
            "payload_digest": self.payload_digest,
            "prev_hash": self.prev_hash,
        }


def canonical_json(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def payload_digest(payload: Mapping[str, Any] | None) -> str:
    return hashlib.sha256(canonical_json(payload or {}).encode()).hexdigest()


def entry_hash(entry: AuditEntry) -> str:
    return hashlib.sha256(
        (entry.prev_hash + canonical_json(entry._core())).encode()
    ).hexdigest()


def append_entry(
    prev: AuditEntry | None,
    actor: str,
    action: str,
    payload: Mapping[str, Any] | None,
    timestamp_s: float,
) -> AuditEntry:
    """Build the next chained entry after `prev` (None for the genesis entry)."""
    seq = 0 if prev is None else prev.seq + 1
    prev_hash = GENESIS_PREV_HASH if prev is None else prev.entry_hash
    entry = AuditEntry(
        seq=seq, timestamp_s=timestamp_s, actor=actor, action=action,
        payload_digest=payload_digest(payload), prev_hash=prev_hash,
    )
    return replace(entry, entry_hash=entry_hash(entry))


@dataclass(frozen=True)
class ChainVerification:
    ok: bool
    length: int
    break_index: int | None = None   # seq/list index of the first broken link
    reason: str = ""


def verify_chain(entries: Sequence[AuditEntry]) -> ChainVerification:
    """Walk the chain; localize the first tampered or mislinked entry."""
    prev_hash = GENESIS_PREV_HASH
    for i, e in enumerate(entries):
        if e.prev_hash != prev_hash:
            return ChainVerification(
                False, len(entries), i,
                f"entry {e.seq} prev_hash does not match the previous entry's hash "
                f"(chain broken — an earlier entry was altered, removed, or reordered)")
        if entry_hash(e) != e.entry_hash:
            return ChainVerification(
                False, len(entries), i,
                f"entry {e.seq} contents were altered after signing "
                f"(recomputed hash does not match the stored hash)")
        prev_hash = e.entry_hash
    return ChainVerification(True, len(entries))
