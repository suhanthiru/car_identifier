# Eyes Everywhere

A real-time, distributed vehicle-tracking and re-identification research demo whose headline deliverable is **honest, reproducible evaluation on public vehicle-ReID datasets** (VeRi-776, VehicleID, CityFlow). The system around the numbers — identity cascade, symbolic plausibility vetoes, capped-additive corroboration, a live operator console, and a per-target 3D model — exists to showcase and stress-test the method. Current results live in [RESULTS.md](RESULTS.md), regenerated end-to-end by one command.

> **Data envelope.** Real data means established public research datasets obtained under their research-use terms, and nothing else — no scraped feeds, no covert footage, no non-consented camera data. Until those datasets are downloaded (they require manual request forms; see [DATASETS.md](DATASETS.md)), every real-data section of RESULTS.md reads **PENDING**: the harness never substitutes synthetic numbers for missing real ones. The always-runnable demo uses a clearly-labeled synthetic world.

## Reproduce the results

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e "D:\comp_vison_projs\Car_Gen_and_Modeling_proj"   # cargen, for the 3D bridge
pytest -m "not slow"                 # ~150 tests, pure logic + fixtures
# obtain datasets per DATASETS.md, then:
python -m eval.run                   # regenerates RESULTS.md + figures
python demo.py                       # synthetic live-console demo (no datasets needed)
```

## What the evaluation measures

1. **Retrieval** — Rank-1/5/10, mAP, CMC curves on VeRi-776 and VehicleID with the standard same-camera exclusion protocol.
2. **Hard negatives mined from real data** — negative pairs are same-color/same-body different-vehicle confusables (bucketed on the dataset's own labels), because those are the pairs that actually break similarity thresholds. RESULTS.md shows a gallery of the hardest ones.
3. **Calibration** — isotonic similarity→P(same) fitted on the mined real pairs, with a reliability diagram (predicted vs empirical) and ECE to prove it, and alert thresholds derived from a precision/recall sweep. Every calibration artifact is content-versioned and decisions cite the version they used.
4. **The ablation** — precision/recall of alerting under (a) raw ReID score alone vs (b) the identity cascade with attribute vetoes and look-alike ambiguity refusal, on identical rankings. The attribute channel uses dataset labels (a perfect classifier), so the measured delta is an upper bound and is labeled as such.
5. **Cross-camera physics on CityFlow** — the transit-time veto validated against real ground-truth transitions (real hops rarely vetoed; constructed physically-impossible hops caught), and capped-additive corroboration vs noisy-OR on real correlated multi-camera sightings.
6. **Failure analysis as a feature** — cases where the system narrows to a set of look-alikes and *refuses* to assert an individual, and cases where it errs, each with its plain-English explanation.

## Architecture

```
                 (SIMULATED edge tier — asyncio tasks, not hardware)
  ┌────────────┐   ┌────────────┐        ┌────────────┐
  │ edge node  │   │ edge node  │  ...   │ edge node  │   one per camera
  │ detect →   │   │ ByteTrack →│        │ best-crop →│
  │ OSNet embed│   │ plate read │        │ class attrs│
  └─────┬──────┘   └─────┬──────┘        └─────┬──────┘
        │  POST /api/sightings   (localhost; a real deployment would put a
        │                         WireGuard/Tailscale mesh here — stubbed)
        ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ central server (FastAPI + SQLite audit tables + WebSocket)   │
  │   identity cascade: plate → class attrs → marks/3D-geometry  │
  │                     → ReID (tiebreaker only)                 │
  │   plausibility vetoes: plate / transit-time / attribute      │
  │   capped-additive corroboration (no noisy-OR)                │
  │   gated profile updates + gated 3D-model fusion (cargen),    │
  │     both reversible via snapshots                            │
  │   tracker: TENTATIVE→CONFIRMED→COASTING→LOST + prediction    │
  └───────────────┬──────────────────────────────────────────────┘
                  │ WebSocket
                  ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ operator console: live map · review queue (side-by-side      │
  │ crops + fact list) · dossier with rotatable 3D model,        │
  │ green=observed / red=guessed provenance overlay              │
  └──────────────────────────────────────────────────────────────┘
```

## Real vs simulated vs stubbed

**Real:** the reasoning layer end to end (cascade, all plausibility checks, corroboration math, gates, tracker) — pure logic, unit-tested; OSNet embeddings on actual pixels; the whole eval harness; dataset loaders; the cargen 3D reconstruction/fusion machinery (its own project, ~170 tests).

**Simulated (labeled):** the small synthetic world — used for controlled adversarial fixtures (deterministic correlated look-alikes that prove the independence-trap handling in unit tests) and as the always-runnable demo. Its plate reads and instance attributes are controlled noise channels, and pretrained YOLO cannot see its cartoon sprites, so detection falls back to sim boxes with per-observation provenance labels.

**Stubbed (labeled):** edge tier = local processes; mesh = the diagram above; instance attributes on real data = absent unless a dataset provides them; cargen's default prior backend here is a procedural stub — real 3D quality needs its SF3D/TRELLIS backends (GPU), behind capability checks with graceful CPU fallback.

**Cut (roadmap):** DVR timeline, camera-feed wall, on-device accelerators, drone specifics, training/fine-tuning loops.

## Design decisions

**Cascade over a single score.** A fused similarity score lets appearance outvote near-conclusive cheap checks — backwards for look-alikes, which are selected for maximal appearance similarity. Evidence is consulted in reliability order and ReID is a tiebreaker only: it ranks surviving candidates, it cannot create a match, and any veto is final. A plate match that fails the physics check flags a clone/clock-skew anomaly for a human instead of confirming.

**The independence trap.** Noisy-OR fusion assumes camera errors are independent; for look-alikes every camera makes the *same* error, so noisy-OR compounds correlated weakness into false certainty (measured on real CityFlow chains in RESULTS.md when present). Here, corroboration is additive with diminishing increments and appearance-only credit is hard-capped *below* the profile-update threshold — no number of appearance-only sightings can auto-update a profile; only a plate read or a human can. Updates snapshot their before-state and roll back losslessly.

**One anti-poisoning mechanism, twice.** The same gate guards the 3D model: a sighting's crop fuses into the target's cargen splat asset only on plate- or operator-confirmed events (cargen's pending-approval merge, auto-merge off), with per-splat provenance and pre-fusion snapshots. The dossier renders that provenance — green splats are real evidence, red are generative guesses — so evidence-vs-inference is visible, not asserted.

**3D geometry where 2D fails.** Cross-view matching is 2D ReID's worst case (front of car A embeds nearer the front of car B than the side of car A). The car3d bridge extracts view-invariant proportion ratios from the fused splat cloud and feeds them to the cascade's attribute tier — support/caution only, never a veto and never the tiebreaker, and withheld entirely until enough of the cloud is observed rather than guessed. Whether this actually buys cross-view precision is an open ablation, reported once real data + a real prior backend are in place.

**Explainability as the control surface.** Every decision emits a plain-English fact list, persisted in the audit tables and shown verbatim in the review queue. Operators act on reasons, not scores; reviewers can reconstruct any decision later.

## Limits — honest ones

- **Look-alikes identify a set, not an individual.** With shared class attributes and no distinguishing marks, appearance evidence stops at the set; the system refuses to guess past it (ambiguity → review). Anything claiming otherwise on appearance alone is overfitting or lying.
- **Calibration is per-distribution and goes stale.** The isotonic map is only meaningful for the camera/vehicle distribution it was fitted on; artifacts are versioned for exactly that reason. No production-accuracy claim is made anywhere.
- **The ablation's attribute channel uses ground-truth labels** — its delta is an upper bound on a real attribute head.
- **Single-crop 3D is rough by construction.** Traffic crops are terrible image-to-3D input (small, off-center, one view); model quality is bought with multiple confirmed sightings, and the stub prior's geometry means nothing at all — which is why geometry attrs gate on observed-fraction.
- The synthetic transit veto shares constants with its simulator; only the CityFlow validation tests it against reality.

## Explainability vs ethics

Per-decision explanations improve auditability and contestability — a real gap in deployed systems, and the specific thing critics correctly say ALPR networks lack. But explainability is necessary, not sufficient: consent, aggregation harm (many innocuous sightings compose into a movement profile), mission creep, retention, and independent oversight are structural properties of a deployment, not code. A perfectly explainable system can still be a mass-surveillance instrument. This project is a synthetic-plus-public-data demonstrator built to understand and critique these systems, not to operate them.

## Layout

```
datasets/     presence-gated loaders (VeRi-776, VehicleID, CityFlow)
eval/         retrieval metrics, hard-negative mining, reliability, ablation,
              cross-camera validation, RESULTS.md generator
sim/          synthetic world (fixtures + fallback demo)
perception/   detector glue, OSNet embedder, plate/attr channels
reasoning/    facts, profiles, plausibility, cascade, corroboration, gates
tracking/     lifecycle, smoother, prediction, FleetTracker
car3d/        cargen bridge: geometry attrs, gated 3D profile, renders
server/       FastAPI, SQLite, WebSocket, simulated edge feed
web/          operator console (Leaflet + plain JS)
calibration/  isotonic fit + versioned artifacts
tests/        pytest suites
```
