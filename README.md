# Eyes Everywhere

A real-time distributed vehicle-tracking and re-identification demo that runs **entirely on synthetic data**.

> **Synthetic data only.** Every camera, vehicle, plate, and sighting in this system is simulated.
> Nothing here connects to real cameras, real ALPR feeds, or any real person's data. This project
> exists to demonstrate — and critique — the *reasoning layer* of multi-camera re-identification
> systems: identity cascades, physical-plausibility vetoes, corroboration gating, and tracker
> lifecycle management. The ML models are off-the-shelf glue; the contribution is the logic
> around them.

## Status

Early scaffolding. Phases:

- [x] Phase 0 — repo, deps, skeleton
- [ ] Phase 1 — synthetic multi-camera world (road graph, transit windows, look-alike vehicles)
- [ ] Phase 2 — perception glue (YOLO + ByteTrack + ReID embed + plate OCR)
- [ ] Phase 3 — identity cascade + symbolic plausibility layer
- [ ] Phase 4 — capped-additive corroboration + gated profile updates
- [ ] Phase 5 — multi-target tracker lifecycle
- [ ] Phase 6 — central server + storage (FastAPI, SQLite, WebSocket)
- [ ] Phase 7 — operator console (Leaflet map, review queue, dossier)
- [ ] Phase 8 — isotonic calibration + threshold sweep
- [ ] Phase 9 — polish, docs, scripted end-to-end demo

## Layout

```
sim/          synthetic world: road graph, cameras, vehicles, event emitter
perception/   detection, dedup tracking, crops, embeddings, plate OCR
reasoning/    identity cascade, plausibility vetoes, corroboration
tracking/     multi-target tracker lifecycle + smoothing + prediction
server/       FastAPI app, DB models, WebSocket stream
web/          operator console (plain HTML/JS + Leaflet)
calibration/  isotonic calibration + threshold sweeps
tests/        pytest suites
```

## Run

```
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
pytest
```

Full run instructions land with the demo in Phase 9.
