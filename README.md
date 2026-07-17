# Eyes Everywhere

A real-time, distributed vehicle-tracking and re-identification demo that runs **entirely on synthetic data**. A simulated city of eight roadside cameras streams sightings of procedurally-generated vehicles — including deliberately confusable look-alikes — into a central reasoning server, which decides *which sightings belong to which flagged target*, explains every decision in plain English, and asks a human whenever the evidence is merely "it looks similar."

> **Synthetic data only.** Every camera, vehicle, plate, GPS coordinate, and crop in this system is fabricated. Nothing connects to real cameras, real ALPR feeds, or any real person's data. The ML models are stock, off-the-shelf glue; the contribution — and the point — is the **reasoning layer**: the identity cascade, the physical-plausibility vetoes, the corroboration math that refuses to double-count correlated evidence, and the tracker lifecycle around them.

## Quick start

```
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate elsewhere)
pip install -r requirements.txt
pytest -m "not slow"            # ~130 tests, no model downloads needed
python demo.py                  # spins up server + sim feed, opens the console
```

`demo.py` flags two targets from a look-alike cluster (one by plate, one by appearance only) and replays the world at 8× speed. Watch the plate target confirm itself while the appearance-only target piles up review-queue items that wait for you. Re-fit the calibration artifact anytime with `python -m calibration.run`.

## Architecture

```
                 (SIMULATED edge tier — asyncio tasks, not hardware)
  ┌────────────┐   ┌────────────┐        ┌────────────┐
  │ edge node  │   │ edge node  │  ...   │ edge node  │     one per camera
  │  cam-nw    │   │  cam-n     │        │  cam-s     │
  │ ──────────
  │ sim render │   YOLO+ByteTrack (real, with honest fallback)
  │ OSNet embed│   plate read (controlled noise channel)
  │ class attrs│   instance attrs (sim-labeled ground truth)
  └─────┬──────┘   └─────┬──────┘        └─────┬──────┘
        │  POST /api/sightings  (localhost; a real deployment would put
        │                        WireGuard/Tailscale here — stubbed, see below)
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │ central server (FastAPI + SQLite)                       │
  │                                                         │
  │  identity cascade: plate → class attrs → marks → ReID   │
  │  plausibility vetoes: plate / transit-time / attribute  │
  │                        contradiction / corroboration    │
  │  capped-additive corroboration (no noisy-OR)            │
  │  gated profile updates w/ reversible snapshots          │
  │  tracker lifecycle: TENTATIVE→CONFIRMED→COASTING→LOST   │
  │  CV smoother + next-camera prediction                   │
  └──────────────┬──────────────────────────────────────────┘
                 │ WebSocket
                 ▼
  ┌─────────────────────────────────────────────────────────┐
  │ operator console (Leaflet + plain JS)                   │
  │  live map · review queue w/ side-by-side crops and the  │
  │  plain-English fact list · target dossier · audit trail │
  └─────────────────────────────────────────────────────────┘
```

## What's real vs. simulated

**Real (implemented and unit-tested):**
- The road graph with physical transit-time windows derived from distance and a speed envelope; Dijkstra lower bounds for multi-hop trips (`sim/road_graph.py`).
- The identity cascade, all four plausibility checks, capped-additive corroboration, the update gate with reversible snapshots, and the tracker lifecycle (`reasoning/`, `tracking/`) — pure logic, ~90 tests.
- OSNet appearance embeddings computed by the real torchreid model on the actual crop pixels; matching is max-similarity against a bounded per-target gallery.
- The isotonic calibration pipeline with hard-negative pairs, PR sweep, and versioned artifacts (`calibration/`).
- The FastAPI server, SQLite audit tables, WebSocket stream, and the console.

**Simulated / stubbed (labeled as such in code):**
- Vehicles are cartoon sprites. Look-alikes render near-identically *by construction* — that is the point, not a shortcut: it guarantees the embedding cannot separate them and the reasoning layer has to.
- Pretrained COCO YOLO does not recognize the sprites as cars (verified). The detector wrapper runs real YOLO+ByteTrack and falls back to the simulator's box when the model finds nothing; every observation records which path produced it (`detection_source`). On real footage the same code runs without the fallback.
- Plate "OCR" in the synthetic demo is a controlled error channel (miss rate, confusion-pair substitutions) applied to ground truth, labeled `source="sim"`. A real `fast-plate-ocr` adapter exists for the optional real-clip mode.
- Instance attributes (damage, stickers, racks) are simulator-labeled ground truth with a miss probability. No real detector backs them.
- Edge tier = local asyncio tasks. Mesh networking (WireGuard/Tailscale) = the diagram above and localhost sockets. There is no real mesh.
- OSNet ships with ImageNet weights by default; the VeRi-776 vehicle checkpoint requires a manual download (pass its path to `ReidEmbedder`).

**Cut (roadmap only):** DVR timeline, camera-feed wall, on-device accelerators, drone integration, continuous-learning loops.

## Design decisions

**Why a cascade instead of one similarity score.** A single fused score lets strong appearance similarity outvote cheap, near-conclusive checks — exactly backwards for look-alike vehicles, which are *designed* to max out appearance similarity. The cascade consults evidence in reliability order (plate → class attributes → distinguishing marks → ReID) and ReID is a **tiebreaker only**: it can rank surviving candidates but can never, by itself, create a match. A candidate with zero symbolic support scores zero regardless of embedding similarity, and any plausibility veto is final — a vehicle cannot cross town faster than the road network allows, no matter how good the crop looks. When a plate matches but physics vetoes, the system flags a plate-clone/clock-skew anomaly for a human instead of picking a side.

**The independence trap.** The textbook fusion rule for repeated detections, noisy-OR (`1 − Π(1−pᵢ)`), assumes camera errors are independent. For look-alikes they are correlated: if camera A mistakes the sibling Camry for the target, cameras B through D make the same mistake for the same reason. Under noisy-OR, six 60%-confident sightings of the *wrong* car compound to >99% belief. This system fuses additively with diminishing increments and a hard cap on appearance-only credit set *below* the profile-update threshold. Consequence, enforced by test: **no number of appearance-only corroborations can auto-update a target profile.** Crossing that line takes qualitatively independent evidence — a clean plate read or an explicit human confirmation — and every automated update snapshots its before-state so an operator rejection rolls it back losslessly.

**Why every decision is explainable.** Each association, veto, cap, and gate decision emits a plain-English fact list (`[+] support / [X] veto / [!] caution / [i] info`) that is stored in the audit tables and rendered verbatim in the review queue. This is not a UI garnish; it is the control surface. An operator shown "appearance similarity 0.95 — tiebreaker only" next to "plate unknown; plate evidence unavailable" makes a different (better) decision than one shown a bare 0.95, and a reviewer can audit weeks later why the system did what it did. If a decision can't articulate its facts, it doesn't happen.

## Limits — read this part

- **Look-alikes identify a set, not an individual.** When several vehicles share class attributes and carry no distinguishing marks, appearance evidence narrows candidates to that set and stops. The system is built to *refuse* to guess past that point (ambiguous matches go to review), which also means it cannot do what a single-score system falsely claims to do.
- **The calibration measures the simulator.** The isotonic map and PR sweep are computed on rendered sprite pairs. They say nothing about real-world ReID accuracy, and no accuracy claims of any kind are made here.
- **The perception stage is glue, partly bypassed.** YOLO mostly falls back on cartoon frames (recorded honestly per observation), plate reads are a noise model, and instance attributes are simulator labels. The end-to-end demo demonstrates the reasoning layer, not a perception benchmark.
- The simulator and the transit veto share the same physics constants, so the veto is exercised, not adversarially validated.

## Legal / ethical envelope

The same pipeline is an investigative tool or an instrument of mass surveillance depending entirely on things outside the code: who may flag a target and under what authority, how long data is retained, who audits the decisions, and whether the people observed have any say. Technical properties here — explainable decisions, human gates on weak evidence, reversible updates, audit trails — are necessary but nowhere near sufficient safeguards. This project is a synthetic demonstrator built to understand, and critique, how such systems reason; it must not be pointed at real people, and it never touches real data.

## Repo layout

```
sim/          synthetic world: graph, fleet, routes, emitter, sprite renderer
perception/   detector glue, OSNet embedder, plate channels, attributes
reasoning/    facts, profiles, plausibility checks, cascade, corroboration, gates
tracking/     lifecycle, smoother, next-camera prediction, FleetTracker
server/       FastAPI app, SQLite models, WS fan-out, simulated edge feed
web/          operator console (Leaflet + plain JS, no build step)
calibration/  pair dataset, isotonic fit, PR sweep, versioned artifacts
tests/        pytest suites (~130 tests; -m "not slow" skips model loads)
```
