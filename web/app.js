/* Eyes Everywhere console — synthetic demo.
   Plain JS + Leaflet + one WebSocket. No build step. */
"use strict";

const STATE_COLORS = {
  tentative: "#e0a83c", confirmed: "#3ddc84", coasting: "#4da3ff", lost: "#77808c",
};
const TRAIL_LENGTH = 12;

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
  if (worldSource.source === "real") {
    initCityflowVehicleBrowser();
    initPipelineStrip();
  }
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
  const vehicles = await api(`/api/cityflow/${scenario}/vehicles`).catch(() => []);
  if (!vehicles.length) return;
  document.getElementById("cityflow-scenario-tag").textContent = scenario;
  document.getElementById("flag-section").classList.add("hidden");
  section.classList.remove("hidden");
  const grid = document.getElementById("cityflow-vehicles");
  grid.innerHTML = "";
  vehicles.forEach((v) => {
    const tile = document.createElement("div");
    tile.className = "vehicle-tile";
    const img = v.thumbnail_b64
      ? `<img src="data:image/png;base64,${v.thumbnail_b64}" alt="vehicle ${v.vehicle_id}">`
      : `<div class="no-crop">no thumbnail</div>`;
    tile.innerHTML = `${img}<div class="vt-label">#${escapeHtml(String(v.vehicle_id))} · ${escapeHtml(v.first_camera)} · t+${Math.round(v.first_time_s)}s</div>`;
    tile.onclick = () => flagCityflowVehicle(v);
    grid.appendChild(tile);
  });
}

async function flagCityflowVehicle(v) {
  await api("/api/targets", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      label: `vehicle ${v.vehicle_id} (real, first seen ${v.first_camera})`,
    }),
  });
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

  const sightingImg = r.sighting_crop
    ? `<img src="${r.sighting_crop}" alt="sighting crop">`
    : `<div class="no-crop">no crop</div>`;
  const refImg = r.reference_crop
    ? `<img src="${r.reference_crop}" alt="reference crop">`
    : `<div class="no-crop">no reference yet</div>`;

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
    <div class="rc-crops">
      <figure>${sightingImg}</figure>
      <span class="rc-vs">vs</span>
      <figure>${refImg}</figure>
    </div>
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
    <div class="fact-list">${facts.map(factRowHtml).join("")}</div>
    <div class="rc-actions">
      <button class="btn-accept">Accept</button>
      <button class="btn-reject">Reject</button>
    </div>`;
  card.querySelector(".btn-accept").onclick = () => resolveReview(r.review_id, true);
  card.querySelector(".btn-reject").onclick = () => resolveReview(r.review_id, false);
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
    ${model3d.exists ? `
    <div>
      <div class="dossier-section-label">Reconstruction (fused from confirmed sightings only)</div>
      <div class="dossier-recon"><img src="${model3d.turntable}" alt="turntable with provenance overlay"></div>
      <div class="dossier-legend">
        <span><span class="sw sw-good"></span>confirmed (${Math.round(model3d.observed_fraction * 100)}% of structure)</span>
        <span><span class="sw sw-guess"></span>generative-prior guess</span>
      </div>
      <div class="d-sub" style="margin-top:4px">${model3d.observations} fused observation(s) · ${model3d.n_splats} splats
        ${model3d.geometry && model3d.geometry.trustworthy
          ? ` · ${escapeHtml(model3d.geometry.body_profile)}, ${escapeHtml(model3d.geometry.length_class)} (L/W ${model3d.geometry.lw_ratio})`
          : " · geometry withheld: too little confirmed structure"}</div>
    </div>` : (d.reference_crop ? `
    <div class="dossier-recon"><img src="/api/crops/${d.reference_crop}" style="width:160px"></div>` : "")}
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

initMap().then(() => {
  connect();
  refreshReviews();
  refreshAudit();
  setInterval(refreshReviews, 5000);
  setInterval(refreshAudit, 5000);
});
