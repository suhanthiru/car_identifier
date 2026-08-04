"""Rank every CityFlow vehicle by how much of the system it exercises, and
freeze the best few per scenario as a committed showcase set.

    python scripts/rank_showcase.py                 # every scenario on disk
    python scripts/rank_showcase.py --scenario S01
    python scripts/rank_showcase.py --check         # fail if artifacts are stale

The console shows 95 vehicles for S01 and ~145 for S02 with nothing to say
which is worth watching. Most of them are not: they cross cameras with their
fields of view overlapping, so the system is never actually asked the question
it exists to answer. A handful carry the interesting cases. This picks those.

RANKED FROM GROUND TRUTH ONLY -- never from what the cascade concluded.
web/app.js already refuses to select browse vehicles on pipeline output, for
the reason that pre-running the pipeline and showing the cars we already knew
would work is selection on the outcome, and the demo would be answering a
question it had rigged. The same applies here, more so: this set is the first
thing a visitor sees. Every feature below comes from gt.txt and the road graph
built from observed hops, both of which are fixed properties of the footage. So
the set is deterministic, and `--check` enforces that.

data/recognizability_s01.json is a useful CHECK on these picks (it holds real
per-passage verdicts) but is deliberately not an input. Checked against it, the
S01 set behaves as each archetype predicts:

    v71 clean_hop         in_window=1 veto=0 -> candidate, then undecided
    v59 veto_and_support  in_window=3 veto=1 -> 1 rejected AND 3 candidate
    v33 never_unobserved  in_window=0 veto=0 -> undecided throughout
    v53 longest_watch     in_window=1 veto=0 -> candidate, then undecided
    v95 thinnest_evidence in_window=1 veto=0 -> candidate on its one passage

The pair is the exception and cannot be checked there: that harness evaluates
one target at a time, so a guard that only fires when two targets compete is
structurally unreachable in it.

WHY ARCHETYPES RATHER THAN A SINGLE SCORE. A scalar ranking returns six
variations of whichever feature dominates it. The point of this set is
coverage: one car that shows a clean unobserved hop, one that shows the system
refusing, one that shows why some cars never raise anything at all. Each
archetype below states its own selection rule and what it should make the
cascade do, so an operator can check the system against a stated prediction
instead of being told the answer afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.cityflow import CityFlow, CityFlowScenario
from datasets.config import cityflow_root

OUT_DIR = Path("data/showcase")

# Bumped when the feature set or the archetype rules change, so a stale
# committed artifact is detectable rather than silently served.
SCHEMA = "showcase-v1"


def _features(scenario: CityFlowScenario) -> dict[int, dict]:
    """Per-vehicle ground-truth facts. No video decode; ~0.15s for S01.

    The transit facts are computed the way the CASCADE will actually see them,
    which is not the way `transitions()` reports them, and the difference
    decides the whole prediction:

    * SEED-RELATIVE, not consecutive. An operator flags a car from one browse
      tile, and every later sighting is checked against that flagged moment.
      `last_seen` only advances when a match is accepted, and on plateless
      CityFlow most sightings never reach acceptance -- so in practice the
      comparison stays anchored at the seed passage.
    * MIDPOINT-to-MIDPOINT, not exit-to-enter. The live feed emits one
      timestamp per passage, its midpoint, so that is what the transit check
      differences.

    Using `transitions()`'s exit->enter gaps instead predicted the observed
    verdicts almost at chance. Midpoint seed-relative agrees with the real
    per-passage verdicts in data/recognizability_s01.json on 84 of 86
    evaluated S01 vehicles.

    (Worth knowing, and NOT compensated for here: the road-graph windows
    themselves are built by `to_road_graph()` from exit->enter gaps, while the
    veto queries them with midpoint deltas, which are systematically larger by
    half of each passage's duration. That inconsistency is in the system, not
    in this script; this script models what the system does, because the
    showcase's job is to predict what the operator will actually see.)
    """
    graph = scenario.to_road_graph()

    spans: dict[int, list] = {}
    for span in scenario.spans:
        spans.setdefault(span.vehicle_id, []).append(span)

    def mid(s) -> float:
        return (s.enter_s + s.exit_s) / 2.0

    hops: dict[int, list[dict]] = {}
    for vid, vspans in spans.items():
        vspans.sort(key=lambda s: s.enter_s)
        seed = vspans[0]
        for b in vspans[1:]:
            if b.camera_id == seed.camera_id:
                continue
            dt = round(mid(b) - mid(seed), 2)
            fastest = graph.min_transit_s(seed.camera_id, b.camera_id)
            window = graph.transit_window(seed.camera_id, b.camera_id)
            hops.setdefault(vid, []).append({
                "from_camera": seed.camera_id, "to_camera": b.camera_id,
                "dt_s": dt,
                # The transit veto as reasoning/signals.py applies it:
                # arriving before you could possibly have got there.
                "veto": bool(dt < 0 or (fastest is not None and dt < fastest)),
                # The one support signal a plateless target can earn, and
                # near-necessary for rising above undecided.
                "in_window": bool(window is not None
                                  and window[0] <= dt <= window[1]),
            })

    out: dict[int, dict] = {}
    for vid, vspans in sorted(spans.items()):
        vhops = hops.get(vid, [])
        # Raw consecutive gaps stay as DESCRIPTIVE facts for the copy -- "left
        # one camera and reappeared 9s later" is what a reader understands --
        # while the scoring above uses what the cascade evaluates.
        gaps = [round(b.enter_s - a.exit_s, 2)
                for a, b in zip(vspans, vspans[1:])
                if a.camera_id != b.camera_id]
        out[vid] = {
            "n_cameras": len({s.camera_id for s in vspans}),
            "n_passages": len(vspans),
            "cameras": sorted({s.camera_id for s in vspans}),
            "visible_s": round(sum(s.exit_s - s.enter_s for s in vspans), 1),
            "span_s": round(vspans[-1].exit_s - vspans[0].enter_s, 1),
            "first_time_s": round(vspans[0].enter_s, 2),
            "last_time_s": round(vspans[-1].exit_s, 2),
            "first_camera": vspans[0].camera_id,
            "n_hops": len(vhops),
            "positive_gap_hops": sum(1 for g in gaps if g > 0),
            "max_gap_s": round(max(gaps), 2) if gaps else 0.0,
            "max_overlap_s": round(min(gaps), 2) if gaps else 0.0,
            "in_window_hops": sum(1 for h in vhops if h["in_window"]),
            "veto_hops": sum(1 for h in vhops if h["veto"]),
            "hops": vhops,
        }
    return out


def _lookalike_pair(scenario: CityFlowScenario,
                    feats: dict[int, dict]) -> tuple[int, int] | None:
    """Two vehicles space-time cannot separate: co-present at one camera, and
    both going on to the same next camera.

    This is the honest ground-truth definition of the hard case. When two cars
    are at the same camera at the same moment and then both travel to the same
    next one, every space-time signal the system has fires IDENTICALLY for
    both -- the transit window cannot prefer one, so appearance is the only
    thing left, and appearance is exactly what look-alikes defeat. Flagging
    both is what exercises the multi-target ambiguity guard in
    tracking/tracker.py, which a one-target-at-a-time harness structurally
    cannot reach.

    Ranked by overlap duration: the longer they were genuinely side by side,
    the more sightings each gets to confuse the other.
    """
    by_cam: dict[str, list] = {}
    for span in scenario.spans:
        by_cam.setdefault(span.camera_id, []).append(span)

    onward: dict[int, set[str]] = {}
    for vid, f in feats.items():
        onward[vid] = {h["to_camera"] for h in f["hops"]}

    best, best_overlap = None, 0.0
    for cam, cam_spans in by_cam.items():
        cam_spans.sort(key=lambda s: s.enter_s)
        for i, a in enumerate(cam_spans):
            for b in cam_spans[i + 1:]:
                if b.enter_s >= a.exit_s:
                    break                      # sorted: nothing later overlaps
                if a.vehicle_id == b.vehicle_id:
                    continue
                if not (onward.get(a.vehicle_id, set())
                        & onward.get(b.vehicle_id, set())):
                    continue                   # no shared onward camera
                overlap = min(a.exit_s, b.exit_s) - max(a.enter_s, b.enter_s)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = (a.vehicle_id, b.vehicle_id)
    return best


# Each archetype: (key, headline, predicate, sort key -- higher is better).
# Order matters; earlier archetypes claim their vehicle first.
ARCHETYPES = [
    (
        "clean_hop",
        "Crosses cameras with a real gap",
        lambda f: f["veto_hops"] == 0 and f["in_window_hops"] >= 1,
        lambda f: f["max_gap_s"],
    ),
    (
        "veto_and_support",
        "One hop supported, another refused",
        # One veto and one support is already the interesting shape: the same
        # car with evidence pointing both ways. Requiring two supports read
        # better on S01 but emptied the archetype on S03, S04 and S05.
        lambda f: f["veto_hops"] >= 1 and f["in_window_hops"] >= 1,
        lambda f: (f["in_window_hops"], f["max_gap_s"]),
    ),
    (
        "never_unobserved",
        "Never once out of sight",
        lambda f: f["in_window_hops"] == 0 and f["n_cameras"] >= 3,
        lambda f: -f["max_overlap_s"],
    ),
    (
        "longest_watch",
        "The longest story in the scenario",
        lambda f: f["n_cameras"] >= 2,
        lambda f: f["visible_s"],
    ),
    (
        "thinnest_evidence",
        "The least the system gets to work with",
        lambda f: f["n_cameras"] >= 2,
        lambda f: -f["visible_s"],
    ),
]

WHY = {
    "clean_hop":
        "Left {first_camera} and reappeared {max_gap_s}s later at another "
        "camera, with nothing observing it in between. That gap is the whole "
        "problem the system exists to solve.",
    "veto_and_support":
        "{in_window_hops} of its hops land inside the window real vehicles "
        "were observed to take, and {veto_hops} arrived faster than is "
        "physically possible. One car, evidence pointing both ways.",
    "never_unobserved":
        "Its camera views overlap by up to {overlap_abs}s -- it is in two "
        "places' footage at once, so there is no unobserved gap to reason "
        "across.",
    "longest_watch":
        "Visible for {visible_s}s across {n_cameras} cameras, the most "
        "footage any vehicle here gets.",
    "thinnest_evidence":
        "Visible for only {visible_s}s across {n_cameras} cameras. Barely "
        "enough pixels to build a profile from.",
    "lookalike_pair":
        "Side by side with vehicle {partner} at the same camera, and both go "
        "on to the same next one. Every space-time signal fires identically "
        "for the two of them.",
}

EXPECT = {
    "clean_hop":
        "Expect a candidate verdict on the second camera: the transit window "
        "supports it, but colour alone cannot name one car.",
    "veto_and_support":
        "Expect the impossible hop to be rejected outright and the plausible "
        "ones to reach candidate. Watch the cascade panel disagree with "
        "itself across sightings -- that is the point.",
    "never_unobserved":
        "Expect undecided, repeatedly. With no gap to cross, the transit "
        "signal never fires and colour alone scores below the threshold. "
        "A car the system honestly has nothing to say about.",
    "longest_watch":
        "Expect the appearance gallery to grow across many sightings, and "
        "the score to stay put anyway -- more looks at the same silver "
        "sedan is not more distinctive.",
    "thinnest_evidence":
        "Expect a candidate anyway. A few seconds of footage and one "
        "plausible hop is enough for the system to narrow -- which is "
        "exactly why narrowing is not the same as identifying.",
    "lookalike_pair":
        "Flag BOTH. Expect the ambiguity guard to fire: when two targets "
        "score within 0.10 of each other the system refuses to pick and "
        "sends it to review instead.",
}


FEATURE_KEYS = (
    "n_cameras", "n_passages", "cameras", "visible_s", "span_s",
    "first_time_s", "last_time_s", "first_camera", "n_hops",
    "positive_gap_hops", "max_gap_s", "max_overlap_s", "in_window_hops",
    "veto_hops",
)


def rank(scenario: CityFlowScenario) -> dict:
    feats = _features(scenario)
    chosen: list[dict] = []
    taken: set[int] = set()

    # The pair claims its two vehicles FIRST. It is the only archetype that
    # needs a specific *combination*, and the only one that reaches the
    # multi-target ambiguity guard, so it cannot be assembled from leftovers.
    # Letting the single-vehicle archetypes go first silently destroyed it:
    # on S02 both members scored top of another archetype and the pair
    # dropped out of the set entirely; on S04 only one member survived, so
    # the "flag both" instruction pointed at a car that wasn't there.
    pair = _lookalike_pair(scenario, feats)
    if pair:
        for vid, partner in ((pair[0], pair[1]), (pair[1], pair[0])):
            f = feats[vid]
            taken.add(vid)
            chosen.append({
                "vehicle_id": vid,
                "archetype": "lookalike_pair",
                "headline": "Confusable with vehicle %d" % partner,
                "why": WHY["lookalike_pair"].format(partner=partner),
                "expect": EXPECT["lookalike_pair"],
                "partner_vehicle_id": partner,
                "features": {k: f[k] for k in FEATURE_KEYS},
            })

    for key, headline, predicate, sort_key in ARCHETYPES:
        pool = [(v, f) for v, f in feats.items()
                if v not in taken and predicate(f)]
        if not pool:
            continue
        vid, f = max(pool, key=lambda vf: sort_key(vf[1]))
        taken.add(vid)
        ctx = dict(f, overlap_abs=round(abs(f["max_overlap_s"]), 1))
        chosen.append({
            "vehicle_id": vid,
            "archetype": key,
            "headline": headline,
            "why": WHY[key].format(**ctx),
            "expect": EXPECT[key],
            "features": {k: f[k] for k in FEATURE_KEYS},
        })

    return {
        "schema": SCHEMA,
        "scenario": scenario.name,
        "generated_from": "ground truth (gt.txt + observed-hop road graph)",
        "n_vehicles_considered": len(feats),
        "vehicles": chosen,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="")
    ap.add_argument("--check", action="store_true",
                    help="verify committed artifacts match a fresh run")
    args = ap.parse_args()

    root = cityflow_root()
    if not CityFlow.exists(root):
        print(f"CityFlow not found at {root}. See DATASETS.md.")
        return 2

    ds = CityFlow(root)
    names = [args.scenario] if args.scenario else list(ds.scenario_names())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stale = []
    for name in names:
        report = rank(ds.load_scenario(name))
        path = OUT_DIR / f"{name}.json"
        text = json.dumps(report, indent=2) + "\n"
        if args.check:
            old = path.read_text(encoding="utf-8") if path.exists() else ""
            if old != text:
                stale.append(name)
            continue
        path.write_text(text, encoding="utf-8")
        picks = ", ".join(
            f"{v['vehicle_id']} ({v['archetype']})" for v in report["vehicles"])
        print(f"{name}: {report['n_vehicles_considered']} vehicles -> {picks}")
        print(f"  wrote {path}")

    if args.check:
        if stale:
            print(f"STALE: {', '.join(stale)} -- rerun scripts/rank_showcase.py")
            return 1
        print(f"showcase artifacts current for {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
