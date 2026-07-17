"""Tamper-evident audit log: an append-only, hash-chained record of every
state-changing action and query the server performs.

This is a STRUCTURAL accountability gate, not a policy one: it cannot decide
who is allowed to do what, but it makes after-the-fact tampering detectable —
any edit, deletion, or reordering of a past entry breaks the hash chain and
`verify_chain` localizes exactly where. What a deployment does with that
signal (who audits, how often, with what authority) is the policy layer, and
lives outside the code.
"""
from audit.log import (
    AuditEntry,
    ChainVerification,
    append_entry,
    entry_hash,
    payload_digest,
    verify_chain,
)

__all__ = [
    "AuditEntry",
    "ChainVerification",
    "append_entry",
    "entry_hash",
    "payload_digest",
    "verify_chain",
]
