"""Audit-log tests: hash-chain integrity + tamper localization + DB store."""
from dataclasses import replace

import pytest
from sqlmodel import Session, select

from audit.log import (
    GENESIS_PREV_HASH, append_entry, entry_hash, payload_digest, verify_chain,
)
from audit.store import load_entries, record, verify
from server.db import AuditRow, make_engine


def build_chain(n: int):
    entries = []
    prev = None
    for i in range(n):
        prev = append_entry(prev, actor="operator", action=f"act-{i}",
                            payload={"i": i}, timestamp_s=float(i))
        entries.append(prev)
    return entries


def test_genesis_and_linkage():
    chain = build_chain(4)
    assert chain[0].prev_hash == GENESIS_PREV_HASH
    assert chain[0].seq == 0
    for a, b in zip(chain, chain[1:]):
        assert b.prev_hash == a.entry_hash
        assert b.seq == a.seq + 1


def test_verify_intact_chain():
    result = verify_chain(build_chain(10))
    assert result.ok and result.break_index is None and result.length == 10


def test_mutation_localized():
    chain = build_chain(8)
    # Tamper the payload digest of entry 5 without re-signing.
    chain[5] = replace(chain[5], payload_digest="deadbeef")
    result = verify_chain(chain)
    assert not result.ok
    assert result.break_index == 5, "verify must point at the altered entry"
    assert "altered" in result.reason


def test_reorder_detected():
    chain = build_chain(6)
    chain[2], chain[3] = chain[3], chain[2]  # swap → prev_hash mismatch
    result = verify_chain(chain)
    assert not result.ok
    assert result.break_index in (2, 3)


def test_deletion_detected():
    chain = build_chain(6)
    del chain[3]  # removing a link breaks the successor's prev_hash
    result = verify_chain(chain)
    assert not result.ok


def test_payload_digest_hides_payload():
    d = payload_digest({"plate": "ABC-1234", "secret": True})
    assert len(d) == 64
    # digest commits to content but is not reversible / not the payload
    assert "ABC-1234" not in d


def test_entry_hash_deterministic():
    a = append_entry(None, "op", "flag", {"x": 1}, 10.0)
    b = append_entry(None, "op", "flag", {"x": 1}, 10.0)
    assert a.entry_hash == b.entry_hash == entry_hash(a)


# ------------------------------------------------------------- DB store

@pytest.fixture()
def session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/audit.sqlite")
    with Session(engine) as s:
        yield s


def test_store_append_and_verify(session):
    for i in range(5):
        record(session, "operator", f"action-{i}", {"i": i}, float(i))
    session.commit()
    entries = load_entries(session)
    assert [e.seq for e in entries] == [0, 1, 2, 3, 4]
    assert verify(session).ok


def test_store_detects_db_tamper(session):
    for i in range(4):
        record(session, "operator", f"action-{i}", {"i": i}, float(i))
    session.commit()
    row = session.exec(select(AuditRow).where(AuditRow.seq == 2)).one()
    row.action = "tampered"          # edit a persisted row, don't re-sign
    session.add(row)
    session.commit()
    result = verify(session)
    assert not result.ok and result.break_index == 2


def test_store_limit(session):
    for i in range(10):
        record(session, "op", f"a{i}", {"i": i}, float(i))
    session.commit()
    assert [e.seq for e in load_entries(session, limit=3)] == [7, 8, 9]
