"""Fit + sweep + save a calibration artifact from the default world.

    python -m calibration.run [output_path]

Uses the real OSNet embedder on rendered sprites. The saved JSON is the
versioned artifact the server loads at startup (see server/api.py). The
printed numbers describe the SIMULATOR only.
"""
from __future__ import annotations

import sys
from pathlib import Path

from calibration.dataset import PairConfig, build_pairs
from calibration.isotonic import build_report, save
from perception.embedder import ReidEmbedder
from sim.emitter import build_default_world

DEFAULT_OUT = Path("calibration/artifacts/latest.json")


def main(out_path: str | None = None) -> None:
    out = Path(out_path) if out_path else DEFAULT_OUT
    world = build_default_world()
    embedder = ReidEmbedder()
    print("building similarity pairs (real OSNet on rendered sprites)...")
    pairs = build_pairs(world, embedder.embed, PairConfig())
    positives = sum(p.same_vehicle for p in pairs)
    hard = sum(p.hard_negative for p in pairs)
    print(f"  {len(pairs)} pairs: {positives} positive, "
          f"{len(pairs) - positives} negative ({hard} hard look-alike negatives)")

    report = build_report(pairs)
    save(report, out)
    print(f"calibration version {report.model.version} -> {out}")
    print(f"  chosen threshold {report.chosen_threshold:.3f} "
          f"(target precision {report.target_precision:.2f})")
    print(f"  hard-negative false-positive rate at threshold: "
          f"{report.hard_negative_fpr_at_threshold:.2%}")
    print("  NOTE: these numbers measure the simulator, not the real world.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
