/* Eyes Everywhere console — synthetic demo.
   Plain JS + Leaflet + one WebSocket. No build step. */
"use strict";

const STATE_COLORS = {
  tentative: "#e0a83c", confirmed: "#3ddc84", coasting: "#4da3ff", lost: "#77808c",
};
const TRAIL_LENGTH = 12;

const map = L.map("map", { zoomControl: false, attributionControl: false });
let cameraMarkers = {};   // camera_id -> circleMarker
let targetMarkers = {};   // target_id -> marker
let targetTrails = {};    // target_id -> polyline
let trailPoints = {};     // target_id -> [[lat,lon],...]
let expectedRings = [];   // pulsing rings on expected-now cameras
let latestSnapshot = {};

/* ---------------------------------------------------------------- helpers */

async function api(path, opts) {
  const resp = await fetch(path, opts);
  if (!resp.ok) throw new Error(`${path}: HTTP ${resp.status}`);
  return resp.status === 204 ? null : resp.json();
}

function factsHtml(text) {
  return (text || "").split("\n").map((line) => {
    const cls = line.startsWith("[+]") ? "f-support"
      : line.startsWith("[X]") ? "f-veto"
      : line.startsWith("[!]") ? "f-caution" : "f-info";
    return `<span class="${cls}">${escapeHtml(line)}</span>`;
  }).join("\n");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtTime(t) { return `t+${Math.round(t)}s`; }

/* ------------------------------------------------------------------- map */

async function initMap() {
  const cameras = await api("/api/cameras");
  const adjacency = await api("/api/adjacency");
  const byId = {};
  cameras.forEach((c) => { byId[c.camera_id] = c; });

  const seen = new Set();
  adjacency.forEach((e) => {
    const key = [e.src, e.dst].sort().join("|");
    if (seen.has(key)) return;
    seen.add(key);
    L.polyline(
      [[byId[e.src].lat, byId[e.src].lon], [byId[e.dst].lat, byId[e.dst].lon]],
      { color: "#243247", weight: 2, interactive: false }).addTo(map);
  });
  cameras.forEach((c) => {
    cameraMarkers[c.camera_id] = L.circleMarker([c.lat, c.lon], {
      radius: 7, color: "#3d5271", weight: 2, fillColor: "#16202e", fillOpacity: 1,
    }).addTo(map).bindTooltip(c.name, {
      permanent: true, direction: "bottom", offset: [0, 8], className: "cam-label",
    });
  });
  map.fitBounds(cameras.map((c) => [c.lat, c.lon]), { padding: [46, 46] });
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
}

async function refreshReviews() {
  const reviews = await api("/api/reviews");
  document.getElementById("review-count").textContent = reviews.length;
  const el = document.getElementById("reviews");
  el.innerHTML = reviews.length ? "" : `<div class="alert-row">queue empty</div>`;
  reviews.forEach((r) => {
    const card = document.createElement("div");
    const refused = (r.facts || "").includes("Cannot assert individual");
    card.className = `review-card${r.kind === "anomaly" ? " anomaly" : ""}${refused ? " refused" : ""}`;
    const sightingImg = r.sighting_crop
      ? `<img src="${r.sighting_crop}" alt="sighting crop">`
      : `<div class="no-crop">no crop</div>`;
    const refImg = r.reference_crop
      ? `<img src="${r.reference_crop}" alt="reference crop">`
      : `<div class="no-crop">no reference yet</div>`;
    const prefix = r.kind === "anomaly" ? "ANOMALY — "
      : refused ? "CANDIDATE SET — " : "";
    card.innerHTML = `
      <b>${prefix}${escapeHtml(r.target_label || r.target_id)}</b>
      <span style="float:right;color:var(--dim)">score ${r.score.toFixed(2)}</span>
      <div class="crops">
        <figure>${sightingImg}<figcaption>this sighting</figcaption></figure>
        <figure>${refImg}<figcaption>target reference</figcaption></figure>
      </div>
      <div class="facts">${factsHtml(r.facts)}</div>
      <div class="actions">
        <button class="btn-accept">Accept match</button>
        <button class="btn-reject">Reject</button>
      </div>`;
    card.querySelector(".btn-accept").onclick = () => resolveReview(r.review_id, true);
    card.querySelector(".btn-reject").onclick = () => resolveReview(r.review_id, false);
    el.appendChild(card);
  });
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
}

function pushAlert(msg) {
  const el = document.getElementById("alerts");
  const row = document.createElement("div");
  row.className = `alert-row kind-${msg.type}`;
  const target = msg.target_id ? ` ${msg.target_id}` : "";
  const extra = msg.detail && msg.detail.verdict ? ` (${msg.detail.verdict})`
    : msg.detail && msg.detail.to ? ` -> ${msg.detail.to}` : "";
  row.innerHTML = `<b>${escapeHtml(msg.type)}</b>${escapeHtml(target + extra)}
    <span style="float:right">${fmtTime(msg.timestamp_s || 0)}</span>`;
  el.prepend(row);
  while (el.children.length > 40) el.removeChild(el.lastChild);
}

/* --------------------------------------------------------------- dossier */

async function openDossier(targetId) {
  const d = await api(`/api/targets/${targetId}`);
  let model3d = { exists: false };
  try { model3d = await api(`/api/targets/${targetId}/model3d`); } catch (e) { /* optional */ }
  const live = d.live || {};
  const attrs = Object.entries(d.class_attrs).map(([k, v]) =>
    `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(v)}</td></tr>`).join("");
  const marks = Object.entries(d.instance_attrs)
    .filter(([k]) => !k.startsWith("geom3d:"))  // shown in the 3D section
    .map(([k, v]) =>
      `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(v)}</td></tr>`).join("");
  const updates = d.profile_updates.map((u) =>
    `<tr><td>v${u.version}</td><td>${fmtTime(u.timestamp_s)}</td>
     <td>${escapeHtml(u.reason)}</td></tr>`).join("");
  const chain = d.corroboration_chain.slice(-6).map((c) =>
    `<tr><td>${fmtTime(c.timestamp_s)}</td><td>${escapeHtml(c.verdict)}</td>
     <td>belief ${c.belief_after.toFixed(2)}</td></tr>`).join("");
  document.getElementById("dossier-body").innerHTML = `
    <h3>${escapeHtml(d.label || d.target_id)}</h3>
    <div class="sub">${d.target_id} · state ${escapeHtml(live.state || "?")} ·
      belief ${live.belief ?? 0} · profile v${live.profile_version ?? 0} ·
      gallery ${d.gallery_size} crops</div>
    ${d.reference_crop ? `<img src="/api/crops/${d.reference_crop}" style="width:160px;border-radius:4px">` : ""}
    <table><tr><th colspan="2">Class attributes</th></tr>${attrs ||
      "<tr><td colspan=2>none recorded</td></tr>"}</table>
    <table><tr><th colspan="2">Distinguishing marks (sim-labeled)</th></tr>${marks ||
      "<tr><td colspan=2>none recorded</td></tr>"}</table>
    ${model3d.exists ? `
    <h4 style="margin-top:12px">3D model (fused from confirmed sightings only)</h4>
    <div class="sub">${model3d.observations} fused observation(s) ·
      ${model3d.n_splats} splats ·
      ${Math.round(model3d.observed_fraction * 100)}% confirmed by real sightings
      ${model3d.geometry && model3d.geometry.trustworthy
        ? ` · profile ${escapeHtml(model3d.geometry.body_profile)},
            ${escapeHtml(model3d.geometry.length_class)}
            (L/W ${model3d.geometry.lw_ratio})`
        : " · geometry withheld: too little confirmed structure"}</div>
    <img src="${model3d.turntable}" style="width:100%;border-radius:4px"
         alt="turntable with provenance overlay">
    <div class="sub"><span style="color:#3ddc84">green</span> = confirmed from
      sightings · <span style="color:#ff5d5d">red</span> = generative-prior guess ·
      downloads: <a href="${model3d.exports.splat}">model.splat</a>,
      <a href="${model3d.exports.provenance_ply}">provenance .ply</a></div>` : ""}
    <table><tr><th colspan="3">Profile updates (all gated + reversible)</th></tr>${updates ||
      "<tr><td colspan=3>none — the gate has not opened</td></tr>"}</table>
    <table><tr><th colspan="3">Recent corroboration chain</th></tr>${chain ||
      "<tr><td colspan=3>no associations yet</td></tr>"}</table>`;
  document.getElementById("dossier").classList.remove("hidden");
}

document.getElementById("dossier-close").onclick = () =>
  document.getElementById("dossier").classList.add("hidden");
document.getElementById("dossier").onclick = (e) => {
  if (e.target.id === "dossier") e.currentTarget.classList.add("hidden");
};

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
