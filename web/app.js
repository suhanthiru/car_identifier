/* Eyes Everywhere console — synthetic demo.
   Plain JS + Leaflet + one WebSocket. No build step. */
"use strict";

const STATE_COLORS = {
  tentative: "#e0a83c", confirmed: "#3ddc84", coasting: "#4da3ff", lost: "#77808c",
};
const TRAIL_LENGTH = 12;
const CLIP_FRAME_MS = 120;   // ~8fps loop for the sighting clip player

// Real weight constants (reasoning/weights.py) — never invented, so the
// confidence-breakdown popover is honest rather than illustrative.
const WEIGHT_LABELS = {
  plate: "plate", class_attrs: "class attrs", instance_marks: "distinguishing marks",
  geometry: "geometry", reid: "appearance (reid, tiebreaker)",
};

const map = L.map("map", { zoomControl: false, attributionControl: false });
let cameraMarkers = {};   // camera_id -> circleMarker
let targetMarkers = {};   // target_id -> marker
let targetTrails = {};    // target_id -> polyline
let trailPoints = {};     // target_id -> [[lat,lon],...]
let expectedRings = [];   // pulsing rings on expected-now cameras
let latestSnapshot = {};
let openDossierId = "";   // "" when the side panel is showing the targets list

/* ---------------------------------------------------------------- helpers */

async function api(path, opts) {
  const resp = await fetch(path, opts);
  if (!resp.ok) throw new Error(`${path}: HTTP ${resp.status}`);
  return resp.status === 204 ? null : resp.json();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtTime(t) { return `t+${Math.round(t)}s`; }

const FACT_ICON = { support: "+", veto: "✕", caution: "!", info: "i" };

function factRowHtml(f) {
  const icon = FACT_ICON[f.kind] || "i";
  const label = (f.check ? f.check.toUpperCase() + " · " : "") + f.kind.toUpperCase();
  return `<div class="fact-row ${f.kind}">
    <span class="fi">${icon}</span><span class="ft">${escapeHtml(f.text)}</span>
    <div class="fact-tip"><div class="ft-head">${escapeHtml(label)}</div>${escapeHtml(f.text)}</div>
  </div>`;
}

function confMeterHtml(score, breakdown) {
  const pct = Math.round(Math.min(1, score) * 100);
  const rows = Object.entries(breakdown || {}).map(([k, v]) =>
    `${WEIGHT_LABELS[k] || k} → +${v.toFixed(2)}`).join("<br>");
  return `<div class="conf-meter">
    <div class="conf-fill" style="width:${pct}%"></div>
    <div class="conf-tip"><div class="ft-head">Confidence breakdown (real weights)</div>
      ${rows || "no contributing signals"}<br>net (capped at 1.0): ${pct}%</div>
  </div>`;
}

/* ---------------------------------------------------------------- banner */

/** The provenance banner must describe what is actually on screen.
 *
 * index.html hard-codes the synthetic wording so that something honest
 * renders before the world_source round-trip completes; in real mode that
 * default is wrong in a way that matters — these are real vehicles, filmed
 * on real streets, used under a research licence — so it is replaced here.
 * The two states are deliberately worded to be unmistakable at a glance:
 * one says nothing on screen is real, the other says everything is.
 */
function setBanner(source) {
  const el = document.getElementById("banner-text");
  if (!el) return;
  const real = source === "real";
  el.textContent = real
    ? "REAL RESEARCH FOOTAGE — CityFlow / AI City Challenge, used under its "
      + "non-commercial research licence. These are real vehicles on real streets."
    : "SYNTHETIC DATA ONLY — every camera, vehicle and plate on this screen is simulated";
  document.getElementById("banner").classList.toggle("banner-real", real);
}

/* ------------------------------------------------------------------- map */

async function initMap() {
  const [cameras, adjacency, worldSource] = await Promise.all([
    api("/api/cameras"), api("/api/adjacency"),
    api("/api/world_source").catch(() => ({ source: "synthetic" })),
  ]);
  cameraMarkers = renderRoadMap(map, cameras, adjacency, worldSource.source).cameraMarkers;
  map.fitBounds(cameras.map((c) => [c.lat, c.lon]), { padding: [46, 46] });
  // Cameras in this world never go offline mid-run (synthetic) and there is
  // no per-camera live health feed for real mode either -- report the
  // honest "all present" count rather than fabricating a partial figure.
  document.getElementById("tb-cameras").textContent = `${cameras.length}/${cameras.length} CAMERAS`;
  document.getElementById("tb-subtitle").textContent =
    worldSource.source === "real" ? "REAL-DATA CONSOLE" : "SYNTHETIC RESEARCH CONSOLE";
  setBanner(worldSource.source);
  // Pause works in both worlds -- both feeds share server.feed.FeedClock.
  initFeedControl();
  initClock();
  initRestart();
  if (worldSource.source === "real") {
    initCityflowVehicleBrowser();
    initPipelineStrip();
    initTimeline();
    // Only real mode has footage behind a camera; in the synthetic world there
    // is nothing to show and a dead click would imply there was.
    Object.entries(cameraMarkers).forEach(([id, marker]) => {
      marker.on("click", () => openCameraView(id));
      marker.getElement && marker.getElement()?.classList.add("cam-clickable");
    });
    document.getElementById("map-hint")?.classList.remove("hidden");
  } else {
    document.getElementById("map-hint")?.classList.add("hidden");
  }
  document.querySelectorAll(".mo-close").forEach((b) => {
    b.onclick = () => closeOverlay(b.dataset.close);
  });
}

/* ------------------------------------------------------- clock and scale */

let latestStats = null;

function fmtClock(t) {
  const s = Math.max(0, Math.round(t));
  return `t+${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

/** Poll the clock and tallies once a second.
 *
 * Polled rather than pushed: these update on a fixed cadence regardless of
 * whether sightings are arriving, and the whole point is to show that time is
 * passing even during a quiet stretch of footage. A push-only clock would
 * freeze exactly when the operator most wants reassurance it hasn't.
 */
async function initClock() {
  const tick = async () => {
    const s = await api("/api/stats").catch(() => null);
    if (!s) return;
    latestStats = s;
    document.getElementById("tb-clock").textContent = fmtClock(s.sim_now);
    const total = s.footage_duration_s || 0;
    document.getElementById("tb-clock-total").textContent =
      total ? `/ ${fmtClock(total)}` : "";
    const fill = document.getElementById("tb-progress-fill");
    if (fill) fill.style.width = total ? `${Math.min(100, 100 * s.sim_now / total)}%` : "0%";
    const speed = document.getElementById("tb-speed");
    if (speed) {
      speed.textContent = s.paused ? "PAUSED" : "PLAYING";
      speed.classList.toggle("pill-paused", !!s.paused);
    }
    renderScaleStrip(s);
  };
  await tick();
  setInterval(tick, 1000);
}

function renderScaleStrip(s) {
  const strip = document.getElementById("scale-strip");
  if (!strip || s.world_source !== "real") return;
  strip.classList.remove("hidden");
  const sc = s.scale || {}, c = s.counters || {};
  const cell = (label, value, title) =>
    `<span class="ss-cell" title="${escapeHtml(title || "")}">` +
    `<b class="mono">${escapeHtml(String(value))}</b> ${escapeHtml(label)}</span>`;
  document.getElementById("ss-scale").innerHTML =
    `<span class="ss-title">${escapeHtml(s.scenario || "")} CONTAINS</span>` +
    cell("vehicles", sc.ground_truth_vehicles, "distinct vehicles in the ground truth") +
    cell("tracks", sc.ground_truth_tracks, "one per vehicle per camera it passed through") +
    cell("cameras", sc.cameras, "") +
    cell("transit routes", sc.transit_routes, "camera pairs with an observed hop");
  document.getElementById("ss-live").innerHTML =
    `<span class="ss-title">SO FAR</span>` +
    cell("sightings", c.sightings, "reported by the edge tier") +
    cell("vehicles seen", c.vehicles_seen, "distinct ground-truth vehicles observed") +
    cell("cross-camera hops", c.cross_camera_hops, "the same vehicle appearing at a new camera") +
    cell("reviews", c.reviews_raised, "sent to a human rather than asserted") +
    cell("refusals", c.refusals, "narrowed to a set and declined to name an individual");
}

async function initRestart() {
  const btn = document.getElementById("feed-restart");
  if (!btn) return;
  btn.onclick = async () => {
    if (!confirm("Replay from t=0?\n\nThis clears targets, reviews, alerts, "
                 + "crops and 3D models from this run.")) return;
    btn.disabled = true;
    btn.textContent = "⟲ RESTARTING";
    await api("/api/reset", { method: "POST" }).catch(() => null);
  };
}

async function initFeedControl() {
  const btn = document.getElementById("feed-toggle");
  if (!btn) return;
  // Server state, not local: the replay clock lives in the feed process, and
  // a reload or a second tab must show the truth rather than its own guess.
  const paint = (paused) => {
    btn.textContent = paused ? "▶ RESUME" : "⏸ PAUSE";
    btn.classList.toggle("paused", !!paused);
  };
  const sync = async () => {
    const s = await api("/api/feed_control").catch(() => null);
    if (s) paint(s.paused);
  };
  btn.onclick = async () => {
    const now = btn.textContent.includes("PAUSE");
    btn.disabled = true;
    try {
      const s = await api("/api/feed_control", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paused: now }),
      });
      paint(s.paused);
    } finally {
      btn.disabled = false;
    }
  };
  await sync();
  setInterval(sync, 5000);
}

async function initPipelineStrip() {
  const strip = document.getElementById("pipeline-strip");
  const refresh = async () => {
    const cfg = await api("/api/pipeline_config").catch(() => ({ plate_ocr: true }));
    renderPipelineDiagram(strip, { plateOcrEnabled: cfg.plate_ocr });
  };
  strip.classList.remove("hidden");
  await refresh();
  setInterval(refresh, 5000);
}

async function initCityflowVehicleBrowser() {
  const section = document.getElementById("cityflow-vehicles-section");
  const scenarios = await api("/api/cityflow/scenarios").catch(() => []);
  if (!scenarios.length) return;
  const scenario = scenarios[0];
  const all = await api(`/api/cityflow/${scenario}/vehicles`).catch(() => []);
  if (!all.length) return;
  document.getElementById("cityflow-scenario-tag").textContent = scenario;
  document.getElementById("flag-section").classList.add("hidden");
  section.classList.remove("hidden");

  // Filter on TIME, not on camera count.
  //
  // The obvious filter — "seen at 2+ cameras" — is a no-op on this dataset:
  // AIC22 Track 1 is a multi-camera benchmark, so its ground truth annotates
  // only vehicles that cross more than one camera. All 95 of S01's vehicles
  // qualify, as do 100% in every other scenario.
  //
  // What actually wastes an operator's time is flagging a car whose passage
  // has already gone by: there is no future sighting left to associate, so the
  // review queue stays empty and a working system looks broken. That is a
  // function of the clock, so the list is re-filtered as it advances.
  const toggle = document.getElementById("cf-upcoming-only");
  const render = () => {
    const now = latestStats ? latestStats.sim_now : 0;
    const upcomingOnly = !toggle || toggle.checked;
    const shown = all.filter(
      (v) => !upcomingOnly || (v.first_time_s || 0) >= now - 2);
    const count = document.getElementById("cf-vehicle-count");
    if (count) {
      count.textContent = upcomingOnly
        ? `${shown.length} of ${all.length} still to come`
        : `all ${all.length} vehicles`;
    }
    renderVehicleTiles(shown);
  };
  if (toggle) toggle.onchange = render;
  render();
  // Cheap enough to redo on the clock's cadence; keeps "still to come" true.
  setInterval(render, 3000);
}

function renderVehicleTiles(vehicles) {
  const grid = document.getElementById("cityflow-vehicles");
  grid.innerHTML = "";
  if (!vehicles.length) {
    // Once the replay passes the last vehicle this list empties, and a bare
    // grid reads as a broken panel rather than an exhausted one. Say which it
    // is, and name the two ways forward.
    grid.innerHTML = `<div class="vt-empty">Every vehicle in this scenario has
      already driven through. Untick <b>still to come</b> to browse them anyway,
      or <b>⟲ RESTART</b> to replay from t=0.</div>`;
    return;
  }
  vehicles.forEach((v) => {
    const tile = document.createElement("div");
    tile.className = "vehicle-tile";
    const img = v.thumbnail_b64
      ? `<img src="data:image/png;base64,${v.thumbnail_b64}" alt="vehicle ${v.vehicle_id}">`
      : `<div class="no-crop">no thumbnail</div>`;
    const cams = v.n_cameras || 1;
    // Shown as information, not as a filter: every annotated vehicle in this
    // dataset is multi-camera, and in S01 54 of 95 appear at ALL FIVE cameras
    // at once. The count is worth seeing because that overlap is surprising —
    // it is also why the transit windows collapse to their floor.
    const badge = `<span class="vt-cams${cams >= 2 ? "" : " vt-cams-single"}" `
      + `title="${escapeHtml((v.cameras || []).join(', '))}">${cams} cam${cams === 1 ? "" : "s"}</span>`;
    tile.innerHTML = `${img}<div class="vt-label">#${escapeHtml(String(v.vehicle_id))} · ${escapeHtml(v.first_camera)} · t+${Math.round(v.first_time_s)}s ${badge}</div>`;
    tile.onclick = () => flagCityflowVehicle(v);
    grid.appendChild(tile);
  });
}

async function flagCityflowVehicle(v) {
  await api("/api/targets", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      label: `vehicle ${v.vehicle_id} (real, first seen ${v.first_camera})`,
      // Seed from the FULL-RESOLUTION passage crops, not the thumbnail:
      // thumbnail_b64 is downscaled for the grid and would hand the
      // appearance model a blurrier car than the sightings it must match
      // later. gallery_b64[0] is that same first frame at full size, and the
      // rest cover the pose change across the passage.
      reference_crop_b64: (v.gallery_b64 && v.gallery_b64[0]) || v.thumbnail_b64 || "",
      reference_gallery_b64: v.gallery_b64 || [],
    }),
  });
}

/* ------------------------------------------------- camera view / timeline */

let cameraViewTimer = null;

/** Live view of one camera, refreshed against the replay clock.
 *
 * ~4 fps rather than the footage's 10: the frames are served by decoding the
 * AVI on demand, and the operator is reading a scene, not counting frames.
 * Deliberately shows NO detection boxes — see README/RESULTS on why drawing
 * ground-truth boxes here would flatter the system with the dataset's own
 * answer key.
 */
function openCameraView(cameraId) {
  closeOverlay("clip-view");
  const box = document.getElementById("camera-view");
  const img = document.getElementById("cv-frame");
  document.getElementById("cv-title").textContent = cameraId;
  document.getElementById("cv-sub").textContent = "live view";
  document.getElementById("cv-note").textContent =
    "Real footage at this camera, following the replay clock. No boxes drawn: "
    + "the dataset's ground truth would mark every vehicle perfectly and say "
    + "nothing about what the system concluded.";
  box.classList.remove("hidden");
  const note = document.getElementById("cv-note");
  const baseNote = note.textContent;
  // Cameras in a scenario neither start together nor run equally long — the
  // scenario timeline is the maximum across all of them — so near the end the
  // clock outruns the shorter videos and the endpoint has no frame to serve.
  // Say that, rather than leaving a broken image and letting it read as a
  // failure of the system.
  img.onerror = () => {
    img.removeAttribute("src");
    note.textContent = `No frame from ${cameraId} at this point in the replay — `
      + "this camera's footage has ended. Other cameras may still be running; "
      + "the scenario clock spans the longest of them.";
  };
  img.onload = () => { note.textContent = baseNote; };
  const tick = () => {
    const t = latestStats ? latestStats.sim_now : 0;
    img.src = `/api/cityflow/camera/${encodeURIComponent(cameraId)}/frame.jpg?t=${t.toFixed(2)}`;
  };
  tick();
  clearInterval(cameraViewTimer);
  // ~2 fps. The endpoint costs ~70 ms alone (18.6 ms per decoded frame; resize
  // and JPEG encode are ~2 ms together), but while the replay is decoding all
  // five cameras it measures ~450 ms. Polling faster than the server can answer
  // just queues requests, so the view would lag the clock instead of tracking
  // it — and a live view that drifts behind is worse than one that steps.
  cameraViewTimer = setInterval(tick, 500);
}

function closeOverlay(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add("hidden");
  if (id === "camera-view") { clearInterval(cameraViewTimer); cameraViewTimer = null; }
  if (id === "clip-view") { clearInterval(clipViewTimer); clipViewTimer = null; }
}

let clipViewTimer = null;

/** The flagged vehicle's own recorded passage, over the whole map.
 *
 * These are the crop frames the sighting already stored (`reference_clip`),
 * so this is footage of the car as the system saw it — not a re-render, not
 * the dataset's thumbnail.
 */
function openClipView(targetId, label, frames) {
  if (!frames || !frames.length) return;
  closeOverlay("camera-view");
  const box = document.getElementById("clip-view");
  const img = document.getElementById("clip-frame");
  document.getElementById("clip-title").textContent = label || targetId;
  document.getElementById("clip-sub").textContent =
    `${frames.length} frame${frames.length === 1 ? "" : "s"} from the associated passage`;
  document.getElementById("clip-note").textContent =
    "The sighting the tracker associated to this target, as recorded.";
  box.classList.remove("hidden");
  let i = 0;
  const tick = () => { img.src = frames[i % frames.length]; i += 1; };
  tick();
  clearInterval(clipViewTimer);
  clipViewTimer = setInterval(tick, CLIP_FRAME_MS);
}

async function initTimeline() {
  const scenarios = await api("/api/cityflow/scenarios").catch(() => []);
  if (!scenarios.length) return;
  const tl = await api(`/api/cityflow/${scenarios[0]}/timeline`).catch(() => null);
  if (!tl || !tl.duration_s) return;
  const panel = document.getElementById("timeline-panel");
  panel.classList.remove("hidden");
  const host = document.getElementById("timeline");
  host.innerHTML = "";
  tl.cameras.forEach((cam) => {
    const lane = document.createElement("div");
    lane.className = "tl-lane";
    const marks = (tl.lanes[cam] || []).map((p) => {
      const left = 100 * p.enter_s / tl.duration_s;
      const width = Math.max(0.35, 100 * (p.exit_s - p.enter_s) / tl.duration_s);
      return `<i class="tl-mark" style="left:${left.toFixed(3)}%;width:${width.toFixed(3)}%" `
           + `title="vehicle ${p.vehicle_id}: ${Math.round(p.enter_s)}-${Math.round(p.exit_s)}s"></i>`;
    }).join("");
    lane.innerHTML = `<span class="tl-cam mono">${escapeHtml(cam)}</span>`
                   + `<span class="tl-track">${marks}<i class="tl-playhead"></i></span>`;
    lane.querySelector(".tl-track").onclick = () => openCameraView(cam);
    host.appendChild(lane);
  });
  // One playhead per lane, inside the track. A single playhead over the whole
  // panel needs its offset expressed against the lane label's fixed width,
  // which cannot be written as a valid calc() (it ends up multiplying a
  // percentage by a percentage). Inside the track, `left: %` is exact.
  const heads = host.querySelectorAll(".tl-playhead");
  setInterval(() => {
    if (!latestStats) return;
    const pct = 100 * Math.min(1, latestStats.sim_now / tl.duration_s);
    heads.forEach((h) => { h.style.left = `${pct.toFixed(2)}%`; });
  }, 500);
}

/** Draw the cross-camera hop a vehicle just made, then fade it.
 *
 * A transient line between the two camera positions rather than a highlight on
 * a drawn road: in real mode no road network is drawn at all (the basemap has
 * the real streets, and overlaying our own would be fiction on top of fact), so
 * there is no edge to light. This claims only what it shows — the same vehicle
 * appeared at these two cameras, in this order.
 */
function flashHop(msg) {
  const a = cameraMarkers[msg.from_camera], b = cameraMarkers[msg.to_camera];
  if (!a || !b) return;
  const line = L.polyline([a.getLatLng(), b.getLatLng()], {
    color: "#8ee0a8", weight: 2, opacity: 0.9, dashArray: "5 5", interactive: false,
  }).addTo(map);
  let opacity = 0.9;
  const fade = setInterval(() => {
    opacity -= 0.07;
    if (opacity <= 0) { clearInterval(fade); map.removeLayer(line); return; }
    line.setStyle({ opacity });
  }, 60);
}

function flashContact(msg) {
  const blip = L.circleMarker([msg.lat, msg.lon], {
    radius: 5, color: "#9fb6d4", weight: 1, fillColor: "#9fb6d4", fillOpacity: 0.9,
    interactive: false,
  }).addTo(map);
  let opacity = 0.9;
  const fade = setInterval(() => {
    opacity -= 0.15;
    if (opacity <= 0) { map.removeLayer(blip); clearInterval(fade); return; }
    blip.setStyle({ fillOpacity: opacity, opacity });
  }, 350);
  const cam = cameraMarkers[msg.camera_id];
  if (cam) {
    cam.setStyle({ color: "#9fb6d4" });
    setTimeout(() => cam.setStyle({ color: "#3d5271" }), 700);
  }
}

function renderTargetsOnMap(targets) {
  expectedRings.forEach((r) => map.removeLayer(r));
  expectedRings = [];
  Object.entries(targets).forEach(([id, t]) => {
    if (!t.position) return;
    const color = STATE_COLORS[t.state] || "#77808c";
    const pos = [t.position.lat, t.position.lon];
    if (!targetMarkers[id]) {
      targetMarkers[id] = L.circleMarker(pos, {
        radius: 9, color, weight: 3, fillColor: color, fillOpacity: 0.35,
      }).addTo(map).bindTooltip("", { direction: "top", offset: [0, -8] });
      trailPoints[id] = [];
      targetTrails[id] = L.polyline([], {
        color, weight: 3, opacity: 0.5, dashArray: "2 6", interactive: false,
      }).addTo(map);
    }
    const marker = targetMarkers[id];
    marker.setLatLng(pos);
    marker.setStyle({ color, fillColor: color });
    marker.setTooltipContent(
      `${escapeHtml(t.label || id)} — ${t.state}, belief ${t.belief}`);
    const trail = trailPoints[id];
    const last = trail[trail.length - 1];
    if (!last || last[0] !== pos[0] || last[1] !== pos[1]) {
      trail.push(pos);
      if (trail.length > TRAIL_LENGTH) trail.shift();
      targetTrails[id].setLatLngs(trail);
      targetTrails[id].setStyle({ color });
    }
    (t.next_cameras || []).filter((p) => p.status === "expected-now").forEach((p) => {
      const cam = cameraMarkers[p.camera_id];
      if (!cam) return;
      expectedRings.push(L.circleMarker(cam.getLatLng(), {
        radius: 13, color, weight: 2, fill: false, dashArray: "3 5",
        interactive: false,
      }).addTo(map));
    });
  });
  Object.keys(targetMarkers).forEach((id) => {
    if (!targets[id]) {
      map.removeLayer(targetMarkers[id]);
      map.removeLayer(targetTrails[id]);
      delete targetMarkers[id]; delete targetTrails[id]; delete trailPoints[id];
    }
  });
}

/* --------------------------------------------------------------- sidebar */

function renderTargetList(targets) {
  const el = document.getElementById("targets");
  const entries = Object.entries(targets);
  el.innerHTML = entries.length ? "" : `<div class="alert-row">no targets flagged yet</div>`;
  entries.forEach(([id, t]) => {
    const card = document.createElement("div");
    card.className = `target-card ${t.state}`;
    card.innerHTML = `
      <span class="state" style="color:${STATE_COLORS[t.state]}">${t.state}</span>
      <b>${escapeHtml(t.label || id)}</b><br>
      <span style="color:var(--dim)">${t.plate ? "plate " + escapeHtml(t.plate) : "plate unknown"}
      ${t.last_seen ? " · last seen " + escapeHtml(t.last_seen.camera_id) : " · never seen"}</span>
      <div class="meter"><div style="width:${Math.round(t.belief * 100)}%"></div></div>`;
    card.onclick = () => openDossier(id);
    el.appendChild(card);
  });
  if (openDossierId && !targets[openDossierId]) showTargetsView();
}

/* ------------------------------------------------------ sighting clip player
   A clip is served as ordered PNG frames (see server/api.py). We flip an
   <img>'s src on a timer to loop it — no animated-image encoder, no new
   dependency, works everywhere. attachClipPlayers() must run after each
   innerHTML render (same pattern as .cf-try / .pd-mount wiring). */
const activeClips = [];   // { el, timer } for cleanup across re-renders

function clipPlayerHtml(frames, opts) {
  const { still = "", label = "", empty = "no clip yet" } = opts || {};
  const list = (frames && frames.length) ? frames : (still ? [still] : []);
  if (!list.length) return `<div class="clip-player empty">${escapeHtml(empty)}</div>`;
  const lbl = label ? `<span class="clip-label">${escapeHtml(label)}</span>` : "";
  return `<div class="clip-player" data-frames='${JSON.stringify(list)}'>
    <img src="${list[0]}" alt="${escapeHtml(label || "sighting clip")}">${lbl}</div>`;
}

function attachClipPlayers(root) {
  // Prune timers whose player was removed by a previous re-render.
  for (let k = activeClips.length - 1; k >= 0; k--) {
    if (!document.body.contains(activeClips[k].el)) {
      clearInterval(activeClips[k].timer);
      activeClips.splice(k, 1);
    }
  }
  root.querySelectorAll(".clip-player[data-frames]").forEach((el) => {
    let frames;
    try { frames = JSON.parse(el.getAttribute("data-frames")); } catch (e) { return; }
    const img = el.querySelector("img");
    if (!img || !frames || frames.length < 2) return;   // single frame = static
    frames.forEach((src) => { const pre = new Image(); pre.src = src; });
    let i = 0;
    const timer = setInterval(() => {
      i = (i + 1) % frames.length;
      img.src = frames[i];
    }, CLIP_FRAME_MS);
    activeClips.push({ el, timer });
  });
}

function sightingClipHtml(r) {
  return clipPlayerHtml(r.sighting_clip, {
    still: r.sighting_crop, label: "sighting", empty: "no clip" });
}

function reviewCardHtml(r) {
  const isAnomaly = r.kind === "anomaly";
  // Refusal-to-individuate (feature B) fires whenever the distinctiveness
  // floor is missed, whether that leaves 1 candidate (nothing else to
  // compare against yet) or several -- the "distinctiveness" caution fact
  // is the real signal, not the candidate count.
  const facts0 = (r.structured_facts && r.structured_facts.length)
    ? r.structured_facts
    : [];
  const isCandidate = facts0.some((f) => f.check === "distinctiveness")
    && (r.candidate_ids || []).length > 0;
  const facts = (r.structured_facts && r.structured_facts.length)
    ? r.structured_facts
    : (r.facts || "").split("\n").filter(Boolean).map((line) => ({
        kind: line.startsWith("[+]") ? "support" : line.startsWith("[X]") ? "veto"
          : line.startsWith("[!]") ? "caution" : "info",
        text: line.replace(/^\[[+X!i]\]\s*/, ""), check: "",
      }));

  if (isAnomaly) {
    const headline = facts.find((f) => f.check === "transit" || f.kind === "veto") || facts[0];
    return `<div class="review-card anomaly-collapsed" data-review="${r.review_id}">
      <div><div class="ac-title">AUTO-FLAGGED · ANOMALY</div>
        <div class="ac-sub">${escapeHtml(r.target_label || r.target_id)} — ${escapeHtml((headline && headline.text) || "")}</div></div>
      <div style="color:var(--dim);font-size:9px">▸</div>
    </div>`;
  }

  if (isCandidate) {
    const chips = r.candidate_ids.map((c) =>
      `<div class="candidate-chip">${escapeHtml(c)}</div>`).join("");
    const note = facts.find((f) => f.check === "distinctiveness");
    return `<div class="review-card candidate" data-review="${r.review_id}">
      <div class="rc-clip">${sightingClipHtml(r)}</div>
      <div class="candidate-badge">CANDIDATE SET · ${r.candidate_ids.length} VEHICLES</div>
      <div class="candidate-chips">${chips}</div>
      <div class="candidate-note">${escapeHtml((note && note.text) ||
        `Distinguishing marks insufficient (distinctiveness ${(r.distinctiveness ?? 0).toFixed(2)}). The system declines to assert an individual.`)}</div>
      <div class="fact-list">${facts.filter((f) => f !== note).map(factRowHtml).join("")}</div>
      <div class="rc-actions">
        <button class="btn-accept">Accept best</button>
        <button class="btn-reject">Reject</button>
      </div>
    </div>`;
  }

  return `<div class="review-card tentative" data-review="${r.review_id}">
    <div class="rc-head">
      <div class="rc-tag">NEEDS REVIEW</div>
      <div class="rc-score mono">${Math.round(r.score * 100)}%</div>
    </div>
    <div class="rc-clip">${sightingClipHtml(r)}</div>
    <div class="fact-list">${facts.map(factRowHtml).join("")}</div>
    ${confMeterHtml(r.score, r.score_breakdown)}
    ${(r.counterfactuals && r.counterfactuals.length) ? `
    <div class="counterfactuals">
      <div class="cf-label">What would change the outcome</div>
      ${r.counterfactuals.map((c) => `<div class="cf">→ ${escapeHtml(c)}</div>`).join("")}
    </div>` : ""}
    <div class="rc-actions">
      <button class="btn-accept">Accept</button>
      <button class="btn-reject">Reject</button>
    </div>
  </div>`;
}

async function refreshReviews() {
  const reviews = await api("/api/reviews");
  document.getElementById("review-count").textContent = reviews.length;
  const el = document.getElementById("reviews");
  el.innerHTML = reviews.length ? "" : `<div class="alert-row">queue empty</div>`;
  reviews.forEach((r) => {
    const wrap = document.createElement("div");
    wrap.innerHTML = reviewCardHtml(r);
    const card = wrap.firstElementChild;
    el.appendChild(card);
    if (card.classList.contains("anomaly-collapsed")) {
      card.onclick = () => expandAnomalyCard(card, r);
      return;
    }
    const accept = card.querySelector(".btn-accept");
    const reject = card.querySelector(".btn-reject");
    if (accept) accept.onclick = () => resolveReview(r.review_id, true);
    if (reject) reject.onclick = () => resolveReview(r.review_id, false);
  });
  attachClipPlayers(el);
}

function expandAnomalyCard(card, r) {
  card.classList.remove("anomaly-collapsed");
  card.classList.add("anomaly-expanded", "review-card");
  const facts = (r.structured_facts && r.structured_facts.length)
    ? r.structured_facts
    : [];
  card.innerHTML = `
    <div class="rc-head"><div class="rc-tag" style="color:var(--veto)">ANOMALY</div>
      <div class="rc-score mono">${Math.round(r.score * 100)}%</div></div>
    <div class="rc-clip">${sightingClipHtml(r)}</div>
    <div class="fact-list">${facts.map(factRowHtml).join("")}</div>
    <div class="rc-actions">
      <button class="btn-accept">Accept</button>
      <button class="btn-reject">Reject</button>
    </div>`;
  card.querySelector(".btn-accept").onclick = () => resolveReview(r.review_id, true);
  card.querySelector(".btn-reject").onclick = () => resolveReview(r.review_id, false);
  attachClipPlayers(card);
}

async function resolveReview(reviewId, accept) {
  await api(`/api/reviews/${reviewId}/resolve`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ accept }),
  });
  refreshReviews();
}

async function refreshAudit() {
  let audit;
  try { audit = await api("/api/audit?limit=8"); } catch (e) { return; }
  const badge = document.getElementById("audit-badge");
  badge.textContent = audit.verified ? `intact · ${audit.length}` : "TAMPERED";
  badge.style.background = audit.verified ? "var(--confirmed)" : "var(--veto)";
  const el = document.getElementById("audit");
  const rows = audit.entries.slice().reverse();
  el.innerHTML = rows.length ? "" : `<div class="alert-row">no entries yet</div>`;
  rows.forEach((e) => {
    const row = document.createElement("div");
    row.className = "alert-row";
    row.innerHTML = `<b>#${e.seq}</b> ${escapeHtml(e.action)}
      <span style="color:var(--dim)">${escapeHtml(e.actor)}</span>
      <span style="float:right;font-family:Consolas,monospace">${escapeHtml(e.entry_hash)}</span>`;
    el.appendChild(row);
  });
  const latest = rows[0];
  const status = document.getElementById("fs-chain-status");
  const text = document.getElementById("fs-chain-text");
  status.textContent = audit.verified ? "✓" : "✕";
  status.className = audit.verified ? "ok" : "bad";
  text.textContent = latest
    ? `chain ${audit.verified ? "verified" : "TAMPERED"} · #${latest.seq} · ${latest.entry_hash}`
    : "no entries yet";
  document.getElementById("tb-chain").textContent = latest ? `#${latest.seq} ${latest.entry_hash}` : "—";
}

document.getElementById("fs-audit-toggle").onclick = () =>
  document.getElementById("audit-drawer").classList.remove("hidden");
document.getElementById("audit-close").onclick = () =>
  document.getElementById("audit-drawer").classList.add("hidden");

function pushAlert(msg) {
  // Alerts feed folded into the audit drawer's context in the redesigned
  // console; the review queue + dossier already surface the same events
  // with fuller context, so this just keeps the audit chain fresh.
}

/* --------------------------------------------------------------- dossier */

function showTargetsView() {
  openDossierId = "";
  document.getElementById("side-title").textContent = "TARGETS";
  document.getElementById("side-back").classList.add("hidden");
  document.getElementById("side-targets").classList.remove("hidden");
  document.getElementById("side-dossier").classList.add("hidden");
}

function showDossierView() {
  document.getElementById("side-title").textContent = "TARGET DOSSIER";
  document.getElementById("side-back").classList.remove("hidden");
  document.getElementById("side-targets").classList.add("hidden");
  document.getElementById("side-dossier").classList.remove("hidden");
}

async function openDossier(targetId) {
  openDossierId = targetId;
  const d = await api(`/api/targets/${targetId}`);
  let model3d = { exists: false };
  try { model3d = await api(`/api/targets/${targetId}/model3d`); } catch (e) { /* optional */ }
  const live = d.live || {};
  const attrs = Object.entries(d.class_attrs).map(([k, v]) =>
    `<div class="trait-row"><span class="tk">${escapeHtml(k)}</span><span class="tv">${escapeHtml(v)}</span></div>`).join("");
  const marks = Object.entries(d.instance_attrs)
    .filter(([k]) => !k.startsWith("geom3d:"))
    .map(([k, v]) =>
      `<div class="trait-row"><span class="tk">${escapeHtml(k)}</span><span class="tv">${escapeHtml(v)}</span></div>`).join("");
  const updates = d.profile_updates.map((u) =>
    `<tr><td>v${u.version}</td><td>${fmtTime(u.timestamp_s)}</td>
     <td>${escapeHtml(u.reason)}</td></tr>`).join("");
  const chain = d.corroboration_chain.slice(-6).reverse().map((c) => `
    <div class="timeline-item">
      <div class="dot" style="background:${c.verdict === "confirmed" ? "var(--confirmed)" : "var(--tentative)"}"></div>
      <div class="ti-cam mono">${fmtTime(c.timestamp_s)}</div>
      <div class="ti-sub">${escapeHtml(c.verdict)} · belief ${c.belief_after.toFixed(2)}</div>
    </div>`).join("");

  // Top of the dossier: a toggle between the 3D reconstruction and the clip
  // of the sighting where the system first locked onto this target. Defaults
  // to the clip so it's useful even with 3D off (the plain synthetic demo).
  const clipPane = clipPlayerHtml(d.reference_clip, {
    still: d.reference_crop ? `/api/crops/${d.reference_crop}` : "",
    empty: "no sighting yet" });
  const model3dPane = model3d.exists ? `
      <div class="dossier-section-label">Reconstruction (fused from confirmed sightings only)</div>
      <div class="dossier-recon"><img src="${model3d.turntable}" alt="turntable with provenance overlay"></div>
      <div class="dossier-legend">
        <span><span class="sw sw-good"></span>confirmed (${Math.round(model3d.observed_fraction * 100)}% of structure)</span>
        <span><span class="sw sw-guess"></span>generative-prior guess</span>
      </div>
      <div class="d-sub" style="margin-top:4px">${model3d.observations} fused observation(s) · ${model3d.n_splats} splats
        ${model3d.geometry && model3d.geometry.trustworthy
          ? ` · ${escapeHtml(model3d.geometry.body_profile)}, ${escapeHtml(model3d.geometry.length_class)} (L/W ${model3d.geometry.lw_ratio})`
          : " · geometry withheld: too little confirmed structure"}`
    : `<div class="dossier-recon-empty">3D model not reconstructed yet.
        <span style="color:var(--dim)">Enable <code>EYES_ENABLE_3D</code> and confirm sightings to
        build one from fused observations.</span></div>`;
  // The clip is 150px wide in this panel and is the most informative thing in
  // the dossier — it is the actual footage of the car. Offer it full size over
  // the map, which is the only surface big enough to be worth looking at.
  const canMaximize = (d.reference_clip || []).length > 0;
  const dossierTop = `
    <div class="dossier-top">
      <div class="dossier-toggle">
        <button class="dossier-tab active" data-pane="clip">Targeting clip</button>
        <button class="dossier-tab" data-pane="model3d">3D model</button>
        ${canMaximize ? `<button class="dossier-maximize" id="clip-maximize"
            title="Play this passage full size over the map">⤢ enlarge</button>` : ""}
      </div>
      <div class="dossier-pane" data-pane="clip">${clipPane}</div>
      <div class="dossier-pane hidden" data-pane="model3d">${model3dPane}</div>
    </div>`;

  document.getElementById("side-dossier").innerHTML = `
    <div class="dossier-header">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span class="d-id">${escapeHtml(d.label || d.target_id)}</span>
        <span class="d-state ${escapeHtml(live.state || "lost")}">${escapeHtml((live.state || "?").toUpperCase())}</span>
      </div>
      <div class="d-sub">${d.target_id} · belief ${live.belief ?? 0} ·
        profile v${live.profile_version ?? 0} · gallery ${d.gallery_size} crops
        ${d.plate ? " · plate " + escapeHtml(d.plate) : ""}</div>
    </div>
    ${dossierTop}
    <div>
      <div class="dossier-section-label">Sighting history</div>
      <div class="timeline">${chain || '<div class="ti-sub">no associations yet</div>'}</div>
    </div>
    <div>
      <div class="dossier-section-label">Class attributes</div>
      <div class="traits-box">${attrs || '<div class="trait-row"><span class="tk">none recorded</span></div>'}</div>
    </div>
    <div>
      <div class="dossier-section-label">Distinguishing marks</div>
      <div class="traits-box">${marks || '<div class="trait-row"><span class="tk">none recorded</span></div>'}</div>
    </div>
    <div>
      <div class="dossier-section-label">Profile updates (all gated + reversible)</div>
      <table class="dossier-table">${updates || '<tr><td>none — the gate has not opened</td></tr>'}</table>
    </div>
    <div class="dossier-audit">
      <div class="dossier-audit-head"><span>AUDIT</span>
        <button class="link-btn" onclick="document.getElementById('audit-drawer').classList.remove('hidden')">view full chain ▸</button></div>
      ${model3d.exists ? `<div class="dossier-audit-entry"><a href="${model3d.exports.splat}" style="color:var(--accent)">model.splat</a> · <a href="${model3d.exports.provenance_ply}" style="color:var(--accent)">provenance .ply</a></div>` : ""}
    </div>`;
  const dossierEl = document.getElementById("side-dossier");
  dossierEl.querySelectorAll(".dossier-tab").forEach((tab) => {
    tab.onclick = () => {
      dossierEl.querySelectorAll(".dossier-tab").forEach(
        (t) => t.classList.toggle("active", t === tab));
      dossierEl.querySelectorAll(".dossier-pane").forEach(
        (p) => p.classList.toggle("hidden", p.dataset.pane !== tab.dataset.pane));
    };
  });
  attachClipPlayers(dossierEl);
  const maximize = document.getElementById("clip-maximize");
  if (maximize) {
    maximize.onclick = () => openClipView(
      d.target_id, d.label || d.target_id, d.reference_clip);
  }
  showDossierView();
}

document.getElementById("side-back").onclick = showTargetsView;

/* ------------------------------------------------------------- flag form */

document.getElementById("flag-form").onsubmit = async (e) => {
  e.preventDefault();
  const classAttrs = {};
  const body = document.getElementById("flag-body").value;
  const color = document.getElementById("flag-color").value;
  if (body) classAttrs.body_type = body;
  if (color) classAttrs.color = color;
  await api("/api/targets", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      label: document.getElementById("flag-label").value,
      plate: document.getElementById("flag-plate").value.trim().toUpperCase(),
      class_attrs: classAttrs,
    }),
  });
  e.target.reset();
};

/* ------------------------------------------------------------- websocket */

function connect() {
  const ws = new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/console`);
  ws.onmessage = (raw) => {
    const msg = JSON.parse(raw.data);
    if (msg.type === "snapshot") {
      latestSnapshot = msg.targets || {};
      renderTargetsOnMap(latestSnapshot);
      renderTargetList(latestSnapshot);
      if (openDossierId && latestSnapshot[openDossierId]) openDossier(openDossierId);
    } else if (msg.type === "contact") {
      flashContact(msg);
    } else if (msg.type === "hop") {
      flashHop(msg);
    } else if (msg.type === "resetting") {
      // Clear immediately rather than waiting for the rewind to land: the
      // panels still hold the previous pass's targets and reviews, and leaving
      // them on screen beside a clock about to jump back to zero is the exact
      // past-footage-next-to-present-conclusions confusion the reset exists to
      // avoid.
      onResetting();
    } else if (msg.type === "reset_done") {
      onResetDone();
    } else {
      pushAlert(msg);
      if (["review", "anomaly", "association", "rejection"].includes(msg.type)) {
        refreshReviews();
      }
      refreshAudit();
    }
  };
  ws.onclose = () => setTimeout(connect, 1500);
}

/* ----------------------------------------------------------------- reset */

function onResetting() {
  ["camera-view", "clip-view"].forEach(closeOverlay);
  latestSnapshot = {};
  openDossierId = "";
  showTargetsView();
  renderTargetsOnMap({});
  renderTargetList({});
  ["reviews", "targets", "audit"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = "";
  });
  const btn = document.getElementById("feed-restart");
  if (btn) { btn.disabled = true; btn.textContent = "⟲ RESTARTING"; }
}

function onResetDone() {
  const btn = document.getElementById("feed-restart");
  if (btn) { btn.disabled = false; btn.textContent = "⟲ RESTART"; }
  refreshReviews();
  refreshAudit();
  // The vehicle browser is keyed to the clock (it hides passages already gone
  // by), so it has to be rebuilt against the rewound timeline.
  initCityflowVehicleBrowser();
}

// initMap owns the clock, restart and timeline init: the timeline only exists
// in real mode and it already knows which mode this is. Calling them here too
// would double every poll interval.
initMap().then(() => {
  connect();
  refreshReviews();
  refreshAudit();
  setInterval(refreshReviews, 5000);
  setInterval(refreshAudit, 5000);
});
