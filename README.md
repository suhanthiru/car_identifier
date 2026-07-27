# Eyes Everywhere

A real-time, distributed vehicle-tracking and re-identification research demo whose headline deliverable is **honest, reproducible evaluation on public vehicle-ReID datasets** (VeRi-776, VehicleID, CityFlow). The system around the numbers — identity cascade, symbolic plausibility vetoes, capped-additive corroboration, a live operator console, and a per-target 3D model — exists to showcase and stress-test the method. Current results live in [RESULTS.md](RESULTS.md), regenerated end-to-end by one command.

> **Data envelope.** Real data means established public research datasets obtained under their research-use terms, and nothing else — no scraped feeds, no covert footage, no non-consented camera data. Until those datasets are downloaded (they require manual request forms; see [DATASETS.md](DATASETS.md)), every real-data section of RESULTS.md reads **PENDING**: the harness never substitutes synthetic numbers for missing real ones. The always-runnable demo uses a clearly-labeled synthetic world.

## Run it

```
python -m venv .venv
.venv\Scripts\python.exe start.py    # Linux/macOS: .venv/bin/python start.py
```

That's the whole thing. On a fresh checkout `start.py` finds nothing installed,
shows you what it's about to do, asks once, and sets the environment up itself:
the right torch build for this machine, the requirements, and the cargen 3D
bridge if it can find a checkout. Then it restarts into the demo.

After that it detects what's on the machine and runs the best demo it can: real
CityFlow footage if you have the dataset, otherwise the synthetic world, which
needs no downloads at all. It prints what it chose and why, opens the console,
and tells you if it's on CPU or GPU.

**No dataset, no GPU, and no cargen install required to see the system work.**

Addressing `.venv\Scripts\python.exe` directly is deliberate — it needs no
`activate`, so it can't accidentally install into your system Python, and on
Windows it sidesteps the PowerShell execution policy that blocks `Activate.ps1`
by default. Activate the venv first if you prefer; then plain `python start.py`
does the same thing.

### Setting up separately

`start.py` delegates all of this to `setup_env.py`, which is stdlib-only and
runs standalone:

```
python setup_env.py --check          # report what's installed, change nothing
python setup_env.py                  # install it
python setup_env.py --dry-run        # print the pip commands it would run
python start.py --check              # same report, from the launcher
python start.py --no-setup           # never install; fail if something's missing
```

**On torch and CUDA.** `pip install -r requirements.txt` on its own gets
whatever build ultralytics resolves, which is CPU-only on Windows and Linux.
A requirements file can't express a per-package `--index-url`, so `setup_env.py`
does it: if `nvidia-smi` reports a GPU it installs torch from PyTorch's CUDA
index *before* the requirements, and replaces an existing CPU build if it finds
one. No CUDA Toolkit install is needed — the wheels bundle their own runtime.
CUDA wheel tags move between releases; override with `EYES_TORCH_INDEX` if the
default 404s. On a machine with no NVIDIA GPU the CPU build is correct and the
step is skipped.

**On cargen.** The 3D bridge is a separate local repo, not a PyPI package, so
it can't be pinned in `requirements.txt`. `setup_env.py` searches for a checkout
alongside this one; set `EYES_CARGEN_PATH` if yours lives elsewhere. Not finding
it is not an error — everything except the 3D panel still runs.

cargen's `master` and `ee-adapter` branches have diverged, and this repo works
against either. `InsufficientDetail` — the refusal to reconstruct from a subject
too small to carry detail — exists only on `ee-adapter`; `car3d/compat.py` takes
cargen's class when present, defines an equivalent when not, and enforces the
same 64px bar itself so the protection never depends on which branch happens to
be checked out. That matters most on exactly this data: CityFlow crops run to a
39px short side at the low end.

`python start.py --check` reports whether the generative backends are real or
stubs. Stub geometry is a procedural sedan — the fusion, provenance, and audit
trail around it are real, but the shape is not a reconstruction:

```
3D backend  : installed, STUB geometry only — procedural sedan, ...
3D backend  : installed, real prior + segmenter          <- what you want
```

### Real 3D geometry (SF3D)

Optional, and a genuine install. Needs an NVIDIA GPU, VS Build Tools with the
C++ workload, and a Hugging Face account that has accepted the Stability AI
Community License on `stabilityai/stable-fast-3d` (the weights are gated;
`huggingface-cli login` once).

```
git clone https://github.com/Stability-AI/stable-fast-3d \
    <cargen>/third_party/stable-fast-3d
pip install numpy==1.26.4 einops==0.7.0 jaxtyping==0.2.31 omegaconf==2.3.0 \
    transformers==4.42.3 open_clip_torch==2.24.0 trimesh==4.4.1 \
    huggingface-hub==0.23.4 rembg==2.0.57 pynanoinstantmeshes==0.0.3 \
    gpytoolbox==0.2.0
```

Then the two C++ extensions, which must build inside the MSVC dev shell:

```
cmd /c 'call "...\VC\Auxiliary\Build\vcvars64.bat" && set USE_CUDA=0 && ^
        set DISTUTILS_USE_SDK=1 && ^
        python -m pip install ./uv_unwrapper ./texture_baker --no-build-isolation'
```

`DISTUTILS_USE_SDK=1` is not optional and is missing from most write-ups: with
the VC environment already activated, torch's `cpp_extension` refuses to build
without it and the failure looks unrelated to the cause. `USE_CUDA=0` builds
the texture baker for CPU, which avoids needing the full CUDA Toolkit — cargen
bridges the CPU baker to SF3D's GPU tensors itself. Override the checkout
location with `CARGEN_SF3D_PATH`.

**Two pins that matter.** `rembg` must stay at `2.0.57`: from 2.0.76 it
requires `numpy>=2.3`, which collides with this repo's pinned `1.26.4` and with
SF3D's own requirements. And `numpy` must stay at `1.26.4` — `setup_env.py`
restores it if a later install walks over it.

**Cost.** One reconstruction is ~98 s on an RTX 3080 Ti and produces ~120k
splats (the stub produces ~20k instantly). Fusion is queued off the ingest path
with a single worker, so the console stays responsive, but a target's 3D panel
takes a minute or two to appear after the gate opens. The test suite is
unaffected: `tests/conftest.py` pins cargen to its stub backends so
`pytest -m "not slow"` stays at ~65 s whether or not SF3D is installed.

**On datasets.** These are the one thing setup can't do for you: VeRi-776 and
VehicleID need a research-use request form, and CityFlow means accepting the
AIC2022 license. See [DATASETS.md](DATASETS.md), and mind that CityFlow's
detection globs `*/S*/c*/gt/gt.txt` — the `train/` level above `S01/` is
required or the scenario won't be found.

| | download | needed for |
|---|---|---|
| nothing | — | synthetic console + reasoning inspector + full test suite |
| CityFlow (AIC22 Track 1) | ~34 GB total, **0.73 GB** for scenario S01 alone | real-footage console, the CityFlow numbers in RESULTS.md |
| VeRi-776 | ~1 GB | the retrieval/calibration numbers in RESULTS.md |
| cargen | separate repo, editable install | the 3D reconstruction panel |
| FastReID checkpoint | 198 MB | the stronger appearance backbone (`--embedder fastreid`) |

Both datasets require a research-use request form — see [DATASETS.md](DATASETS.md).
Until they're present, every real-data section of RESULTS.md reads **PENDING**;
the harness never substitutes synthetic numbers for missing real ones.

### Everything else

```
pytest -m "not slow"                 # 298 tests, pure logic + fixtures, no datasets
python -m eval.run                   # regenerates RESULTS.md + figures (needs datasets)
python start.py --mode synthetic     # force the no-dataset demo
python start.py --mode cityflow      # force real footage; fails loudly if absent
python start.py --no-3d              # skip the 3D panel
python start.py --3d-identification  # let 3D geometry vote on identity (off by default)
python -m eval.run --embedder fastreid          # VeRi block on the strong backbone
python scripts/ablate_3d_cityflow.py --vehicles 95   # the measured 3D ablation
```

### Switching between synthetic and real data

Three levers, in increasing order of permanence. Nothing here needs a code
change; the mode is a function of what's on disk and what you ask for.

**1. `--mode`, per run.** The default is `auto`, which resolves to `cityflow`
if the dataset is found and `synthetic` otherwise. The explicit values differ
in one way that matters:

```
python start.py --mode synthetic     # always works, dataset present or not
python start.py --mode cityflow      # errors out if the data isn't found
```

Prefer naming the mode over relying on `auto`. `auto`'s fallback is friendly
but silent — a typo in the directory layout leaves you watching synthetic
cars and wondering why the footage looks wrong. `--mode cityflow` turns that
into an error message that prints the path it actually looked at.

**2. `EYES_CITYFLOW_ROOT`, per shell.** Auto-detection is just a presence
check on a path, so repointing the variable feeds the same command different
data:

```
$env:EYES_CITYFLOW_ROOT = "E:\datasets\CityFlow"           # this session
[Environment]::SetEnvironmentVariable('EYES_CITYFLOW_ROOT','E:\datasets\CityFlow','User')
```

Defined in `datasets/config.py`, default `data/datasets/CityFlow`. The same
pattern covers `EYES_VERI_ROOT` and `EYES_VEHICLEID_ROOT`, which feed
`python -m eval.run` rather than the console. Pointing the variable at a
nonexistent path is also the cleanest way to force synthetic without flags.

**3. Which one you got.** The banner answers before anything runs, and on a
failed switch the second line names the path it checked:

```
  mode        : synthetic world (no datasets needed)
  CityFlow    : no  - not found at data\datasets\CityFlow
```

#### The two modes are not one pipeline with different inputs

| | synthetic | cityflow |
|---|---|---|
| road graph | `build_default_world()` — fictional Gridville | `scen.to_road_graph()` — real camera GPS |
| feed | `run_feed`, generated sightings | `run_cityflow_feed`, decodes `vdo.avi` |
| database | `data/eyes.sqlite` | `data/eyes-cityflow.sqlite` |
| crops | `data/crops` | `data/crops-cityflow` |
| default speed | 8x real time | 4x |
| `world_source` | `"synthetic"` | `"real"` |

Separate databases are deliberate: switching modes never mixes the two
evidence sets, and each launch wipes its own DB and crops, so you always
start clean.

That last row carries the weight. `world_source` is served at
`GET /api/world_source`, and the console reads it to decide whether the map
draws our fabricated street network or a real basemap under real camera
positions. The two must never be mixed — real camera coordinates with
invented street names drawn on top would present fiction as fact. So the
mode is not a display toggle. It is a claim about what the coordinates mean.

For what stays real, simulated, or stubbed *within* each mode — the synthetic
world's controlled noise channels, the stubbed edge tier — see
[Real vs simulated vs stubbed](#real-vs-simulated-vs-stubbed) below.

### Noise you can ignore on first launch

Real mode prints an alarming-looking block from onnxruntime:

```
*************** EP Error ***************
EP Error ... RegisterTensorRTPluginsAsCustomOps Please install TensorRT ...
Falling back to ['CUDAExecutionProvider', 'CPUExecutionProvider'] and retrying.
... Error loading "onnxruntime_providers_cuda.dll" which depends on
    "cublasLt64_13.dll" which is missing
```

That is the plate reader (`fast-plate-ocr`), not the tracker or the embedders.
Its bundled onnxruntime wants TensorRT and a CUDA-13 runtime that the torch
wheels do not ship; it falls back to CPU and works. Plate OCR is cheap enough
that this costs nothing noticeable. Torch itself is unaffected — `start.py`'s
`compute:` line is the authority on whether the GPU is actually in use, and it
will still say `GPU - <your card>`.

The `MergeShapeInfo ... Falling back to lenient merge` warnings that follow are
from the same model and are equally harmless.

### On hardware

Runs on a laptop: the live console is a 0.6M-parameter appearance model plus
video decode, ~400–650 MB of RAM. Nothing requires a GPU.

If a CUDA-capable GPU **and** a CUDA build of torch are present, the embedders
use it automatically — no flag. `setup_env.py` installs that build when it sees
a GPU, and `start.py` prints which device it actually got, because a silent CPU
build is the failure mode you'd otherwise only notice as slowness. The GPU
matters most for the optional FastReID backbone (~40× heavier than the default)
and for cargen's real 3D backends, which need CUDA.

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

**Possible future embedder swaps:** the ReID embedder is intentionally off-the-shelf and swappable (`perception/embedder.py`) — the cascade's plausibility layer doesn't care how good the appearance vector is. Two upgrades worth doing once real data is flowing:
- **OSNet, VeRi-776-trained weights** (vs. today's ImageNet-pretrained default) — a same-architecture, same-size upgrade via the torchreid model zoo.
- **[FastReID](https://github.com/JDAI-CV/fast-reid)'s pretrained SBS(R50-ibn) checkpoint**, JD AI Research's production ReID platform. Its published VeRi-776 numbers (97.0% Rank-1, 81.9% mAP) are well above what OSNet-x0_25 should reach — trained end-to-end on vehicle identities with a much larger backbone (~24M vs ~2M params). Swapping it in as a second `Embedder` backend would let RESULTS.md report the cascade's ablation delta *on top of a nearer-SOTA embedding*, showing the reasoning layer still adds value (fewer look-alike false positives, honest refusals) even when the base embedder is already strong — a more convincing result than only demonstrating it against a weak embedder. Main integration cost: loading FastReID's checkpoint format standalone without pulling in its full training framework as a dependency (needs a spike to confirm feasibility).

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
- **The 3D geometry channel does not currently help identification — it hurts.** Now measured rather than assumed (`scripts/ablate_3d_cityflow.py`, 795 real CityFlow crops, real SF3D reconstructions, in RESULTS.md). Two findings. First, a yield ceiling: only **14%** of real crops clear the 64px subject bar and the observed-fraction trust gate, so the channel is silent on most sightings. Second, on the ones where it does speak it fired 9 extra attribute vetoes that removed **4 correct matches and 0 incorrect ones**, costing 1.6 points of precision. Single-crop proportion buckets are noisy enough on CCTV imagery to contradict on same-vehicle pairs about as often as on different-vehicle pairs — the failure mode that makes an attribute channel actively harmful rather than merely useless. The 3D panel remains valuable as an operator-facing artifact (provenance, reversible fusion, what-was-seen), so it stays on — but **the geometry channel no longer feeds identification decisions by default.** Reconstruction, export and the dossier are unchanged; what is gated is the promotion of geometry to evidence (profile attributes and the render-based shortlist verifier). Re-enable deliberately, per deployment and with numbers to justify it, via `--3d-identification` or `EYES_ENABLE_3D_IDENTIFICATION=1`. The indicated fix is more fused views per target, not a better prior.
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
