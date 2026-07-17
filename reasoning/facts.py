"""Plain-English facts: the explainability backbone.

Every identity decision this system makes is justified by a list of Facts —
short, human-readable sentences with a machine-readable kind. The operator
console shows them verbatim next to each match proposal; nothing decides
silently. If a decision cannot articulate its facts, it does not happen.

Kinds:
- support: evidence for the match
- veto:    a hard disqualifier (any veto rejects the match outright)
- caution: evidence against, or a reason to distrust the support
- info:    neutral context worth showing the operator
"""
from __future__ import annotations

from dataclasses import dataclass

KIND_SUPPORT = "support"
KIND_VETO = "veto"
KIND_CAUTION = "caution"
KIND_INFO = "info"

VALID_KINDS = (KIND_SUPPORT, KIND_VETO, KIND_CAUTION, KIND_INFO)


@dataclass(frozen=True)
class Fact:
    kind: str
    text: str
    # Which check produced it, e.g. "plate", "transit", "attributes",
    # "corroboration", "reid". Lets the UI group and badge facts.
    check: str = ""

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(f"unknown fact kind: {self.kind!r}")
        if not self.text.strip():
            raise ValueError("fact text must be non-empty")


def support(text: str, check: str = "") -> Fact:
    return Fact(KIND_SUPPORT, text, check)


def veto(text: str, check: str = "") -> Fact:
    return Fact(KIND_VETO, text, check)


def caution(text: str, check: str = "") -> Fact:
    return Fact(KIND_CAUTION, text, check)


def info(text: str, check: str = "") -> Fact:
    return Fact(KIND_INFO, text, check)


def has_veto(facts: list[Fact]) -> bool:
    return any(f.kind == KIND_VETO for f in facts)


def render_facts(facts: list[Fact]) -> str:
    """One fact per line, prefixed by kind — what the console displays."""
    prefix = {KIND_SUPPORT: "[+]", KIND_VETO: "[X]", KIND_CAUTION: "[!]", KIND_INFO: "[i]"}
    return "\n".join(f"{prefix[f.kind]} {f.text}" for f in facts)
