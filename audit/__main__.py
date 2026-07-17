"""Audit CLI:  python -m audit verify [--db sqlite:///data/eyes.sqlite]

Walks the persisted audit chain and reports integrity. Exit code 0 = intact,
1 = tampered (with the broken entry's index/reason).
"""
from __future__ import annotations

import argparse
import sys

from sqlmodel import Session

from audit.store import load_entries, verify
from server.db import DEFAULT_DB_URL, make_engine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m audit")
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="verify the audit chain")
    v.add_argument("--db", default=DEFAULT_DB_URL)
    args = parser.parse_args(argv)

    engine = make_engine(args.db)
    with Session(engine) as session:
        entries = load_entries(session)
        result = verify(session)
    print(f"audit chain: {result.length} entries")
    if result.ok:
        print("OK — chain intact, no tampering detected.")
        return 0
    broken = entries[result.break_index] if result.break_index is not None else None
    where = f" at seq {broken.seq} (action '{broken.action}')" if broken else ""
    print(f"TAMPERED{where}: {result.reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
