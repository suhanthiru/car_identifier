/* Eyes Everywhere — reasoning inspector (sandbox).
   Runs the real cascade on hand-built inputs via /api/inspect/evaluate.
   No sim, no DB, no audit trail — just "what would the system conclude". */
"use strict";

let CAMERAS = [];
let targetCounter = 0;

async function api(path, opts) {
  const resp = await fetch(path, opts);
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`${path}: HTTP ${resp.status} ${body}`);
  }
  return resp.json();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function factsHtml(facts) {
  const prefix = { support: "[+]", veto: "[X]", caution: "[!]", info: "[i]" };
  const cls = { support: "f-support", veto: "f-veto", caution: "f-caution", info: "f-info" };
  return facts.map((f) =>
    `<span class="${cls[f.kind] || "f-info"}">${prefix[f.kind] || "[i]"} ${escapeHtml(f.text)}</span>`
  ).join("\n");
}

function parseMarks(text) {
  const out = {};
  text.split("\n").forEach((line) => {
    const idx = line.indexOf(":");
    if (idx < 0) return;
    const k = line.slice(0, idx).trim();
    const v = line.slice(idx + 1).trim();
    if (k && v) out[k] = v;
  });
  return out;
}

function marksToText(attrs) {
  return Object.entries(attrs || {}).map(([k, v]) => `${k}: ${v}`).join("\n");
}

/* ------------------------------------------------------------------ setup */

async function loadCameras() {
  CAMERAS = await api("/api/cameras");
  document.getElementById("s-camera").innerHTML = CAMERAS.map((c) =>
    `<option value="${c.camera_id}">${escapeHtml(c.name)} (${c.camera_id})</option>`).join("");
}

function cameraOptionsHtml(selected) {
  const none = `<option value="" ${!selected ? "selected" : ""}>— none (fresh flag) —</option>`;
  const opts = CAMERAS.map((c) =>
    `<option value="${c.camera_id}" ${c.camera_id === selected ? "selected" : ""}>` +
    `${escapeHtml(c.name)} (${c.camera_id})</option>`).join("");
  return none + opts;
}

/* --------------------------------------------------------- target forms */

function addTargetForm(preset) {
  const p = preset || {};
  const id = p.target_id || `t${(targetCounter += 1)}`;
  const div = document.createElement("div");
  div.className = "target-form";
  div.dataset.targetId = id;
  const hasGallery = p.reid_similarity !== undefined && p.reid_similarity !== null;
  const reidVal = hasGallery ? p.reid_similarity : 0.5;
  div.innerHTML = `
    <button type="button" class="remove-target" title="remove">&times;</button>
    <h3>${escapeHtml(id)}</h3>
    <label>Label
      <input class="t-label" value="${escapeHtml(p.label || "Test target")}">
    </label>
    <div class="field-grid">
      <label>Plate (blank = unknown)
        <input class="t-plate" value="${escapeHtml(p.plate || "")}" placeholder="e.g. ABC-1234">
      </label>
    </div>
    <div class="field-grid four">
      <label>Make <input class="t-make" value="${escapeHtml(p.make || "")}"></label>
      <label>Model <input class="t-model" value="${escapeHtml(p.model || "")}"></label>
      <label>Body <input class="t-body" value="${escapeHtml(p.body || "")}"></label>
      <label>Color <input class="t-color" value="${escapeHtml(p.color || "")}"></label>
    </div>
    <label>Distinguishing marks (mark:value, one per line)
      <textarea class="t-marks" rows="2">${escapeHtml(marksToText(p.instance_attrs))}</textarea>
    </label>
    <div class="field-grid">
      <label>Last seen camera
        <select class="t-last-camera">${cameraOptionsHtml(p.last_seen_camera_id)}</select>
      </label>
      <label>Last seen timestamp (s)
        <input class="t-last-time" type="number" step="1" value="${p.last_seen_timestamp_s ?? ""}">
      </label>
    </div>
    <div class="reid-toggle">
      <input type="checkbox" class="t-has-gallery" ${hasGallery ? "checked" : ""}>
      target has a confirmed appearance gallery (simulate ReID similarity)
    </div>
    <div class="reid-row">
      <input type="range" class="t-reid" min="-1" max="1" step="0.01"
             value="${reidVal}" ${hasGallery ? "" : "disabled"}>
      <span class="reid-value">${reidVal.toFixed(2)}</span>
    </div>`;
  div.querySelector(".remove-target").onclick = () => div.remove();
  const toggle = div.querySelector(".t-has-gallery");
  const slider = div.querySelector(".t-reid");
  const val = div.querySelector(".reid-value");
  toggle.onchange = () => { slider.disabled = !toggle.checked; };
  slider.oninput = () => { val.textContent = parseFloat(slider.value).toFixed(2); };
  document.getElementById("targets-form").appendChild(div);
  return div;
}

function clearTargets() {
  document.getElementById("targets-form").innerHTML = "";
  targetCounter = 0;
}

function collectTargets() {
  return [...document.querySelectorAll(".target-form")].map((div) => {
    const classAttrs = {};
    const make = div.querySelector(".t-make").value.trim();
    const model = div.querySelector(".t-model").value.trim();
    const body = div.querySelector(".t-body").value.trim();
    const color = div.querySelector(".t-color").value.trim();
    if (make) classAttrs.make = make;
    if (model) classAttrs.model = model;
    if (body) classAttrs.body_type = body;
    if (color) classAttrs.color = color;
    const hasGallery = div.querySelector(".t-has-gallery").checked;
    const lastCam = div.querySelector(".t-last-camera").value;
    const lastTime = div.querySelector(".t-last-time").value;
    return {
      target_id: div.dataset.targetId,
      label: div.querySelector(".t-label").value.trim() || div.dataset.targetId,
      plate: div.querySelector(".t-plate").value.trim().toUpperCase(),
      class_attrs: classAttrs,
      instance_attrs: parseMarks(div.querySelector(".t-marks").value),
      last_seen_camera_id: lastCam || "",
      last_seen_timestamp_s: lastCam && lastTime !== "" ? parseFloat(lastTime) : null,
      reid_similarity: hasGallery ? parseFloat(div.querySelector(".t-reid").value) : null,
    };
  });
}

function collectSighting() {
  const classAttrs = {};
  const make = document.getElementById("s-make").value.trim();
  const model = document.getElementById("s-model").value.trim();
  const body = document.getElementById("s-body").value.trim();
  const color = document.getElementById("s-color").value.trim();
  if (make) classAttrs.make = make;
  if (model) classAttrs.model = model;
  if (body) classAttrs.body_type = body;
  if (color) classAttrs.color = color;
  return {
    camera_id: document.getElementById("s-camera").value,
    timestamp_s: parseFloat(document.getElementById("s-time").value || "0"),
    plate_text: document.getElementById("s-plate").value.trim().toUpperCase(),
    plate_confidence: parseFloat(document.getElementById("s-plate-conf").value || "0.9"),
    class_attrs: classAttrs,
    instance_attrs: parseMarks(document.getElementById("s-marks").value),
  };
}

/* ------------------------------------------------------------- rendering */

function signalsTableHtml(signals) {
  if (!signals) return `<div class="hint">no signals</div>`;
  const rows = Object.entries(signals).map(([k, v]) => {
    let display = v, cls = "";
    if (typeof v === "boolean") { cls = v ? "set-true" : "set-false"; display = v; }
    else if (v === null) { display = "—"; }
    else if (typeof v === "number") { display = Number.isInteger(v) ? v : v.toFixed(3); }
    else if (Array.isArray(v)) { display = v.length ? v.join(", ") : "—"; }
    return `<tr><td>${escapeHtml(k)}</td><td class="${cls}">${escapeHtml(String(display))}</td></tr>`;
  }).join("");
  return `<table class="signals-table">${rows}</table>`;
}

function decisionCardHtml(d, isBest, floor) {
  const distPct = Math.round(Math.max(0, Math.min(1, d.distinctiveness)) * 100);
  const floorPct = Math.round(Math.max(0, Math.min(1, floor)) * 100);
  const cfHtml = d.counterfactuals.length ? `
    <div class="counterfactuals">
      <div class="cf-label">What would change the outcome</div>
      ${d.counterfactuals.map((c) => `<div class="cf">&rarr; ${escapeHtml(c.text)}</div>`).join("")}
    </div>` : "";
  const candidateIds = d.candidate_ids || [];
  const candidateHtml = d.refused_to_individuate ? `
    <div class="candidate-set">Cannot assert an individual. Candidate set:
      <b>${escapeHtml(candidateIds.length ? candidateIds.join(", ") : d.target_id)}</b></div>` : "";
  return `
    <div class="decision-card verdict-${d.verdict}${isBest ? " is-best" : ""}">
      <div class="decision-head">
        <span class="title">${escapeHtml(d.label)}${isBest ? " &#9733;" : ""}</span>
        <span class="verdict-badge verdict-${d.verdict}">${escapeHtml(d.verdict)}</span>
      </div>
      <div class="metric-row">
        <span>score <b>${d.score.toFixed(2)}</b></span>
        <span>tier <b>${escapeHtml(d.deciding_tier)}</b></span>
        <span>reid sim <b>${d.reid_similarity.toFixed(2)}</b></span>
        ${d.anomaly ? `<span style="color:var(--veto)"><b>ANOMALY</b></span>` : ""}
        ${d.requires_review ? `<span>requires review</span>` : ""}
      </div>
      <div class="hint">distinctiveness ${d.distinctiveness.toFixed(2)} (floor ${floor.toFixed(2)})</div>
      <div class="dist-meter">
        <div class="fill" style="width:${distPct}%"></div>
        <div class="floor-mark" style="left:${floorPct}%"></div>
      </div>
      ${candidateHtml}
      <div class="facts">${factsHtml(d.facts)}</div>
      ${cfHtml}
      <details class="signals-detail">
        <summary>raw signals (${Object.keys(d.signals || {}).length} fields)</summary>
        ${signalsTableHtml(d.signals)}
      </details>
    </div>`;
}

async function evaluate() {
  const targets = collectTargets();
  if (!targets.length) {
    alert("Add at least one target profile.");
    return;
  }
  const body = {
    sighting: collectSighting(),
    targets,
    distinctiveness_floor: parseFloat(document.getElementById("floor-slider").value),
  };
  const resultsEl = document.getElementById("results");
  let result;
  try {
    result = await api("/api/inspect/evaluate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    document.getElementById("results-empty").classList.add("hidden");
    resultsEl.innerHTML =
      `<div class="candidate-set" style="border-color:var(--veto)">${escapeHtml(e.message)}</div>`;
    return;
  }
  document.getElementById("results-empty").classList.add("hidden");
  const floor = body.distinctiveness_floor;
  try {
    const marginNote = result.all_decisions.length > 1
      ? `<div class="margin-note">margin between top two candidates: ${result.margin.toFixed(3)}</div>` : "";
    resultsEl.innerHTML = marginNote +
      result.all_decisions.map((d) => decisionCardHtml(d, d.target_id === result.best.target_id, floor)).join("");
  } catch (e) {
    // A render bug must be visible, not a silently stale panel from the
    // previous evaluation — that's worse than no answer at all here.
    resultsEl.innerHTML =
      `<div class="candidate-set" style="border-color:var(--veto)">` +
      `Render error: ${escapeHtml(e.message)}</div>`;
  }
}

/* -------------------------------------------------------------- presets */

const PRESETS = {
  plate: {
    sighting: { camera: null, time: 1000, plate: "ABC-1234", plateConf: 0.95,
               make: "Toyota", model: "Camry", body: "sedan", color: "silver", marks: "" },
    targets: [{ label: "Silver Camry — case 12", plate: "ABC-1234",
               make: "Toyota", model: "Camry", body: "sedan", color: "silver" }],
  },
  transit: {
    sighting: { camera: "cam-e", time: 1005, plate: "ABC-1234", plateConf: 0.95,
               make: "Toyota", model: "Camry", body: "sedan", color: "silver", marks: "" },
    targets: [{ label: "Silver Camry — case 12", plate: "ABC-1234",
               make: "Toyota", model: "Camry", body: "sedan", color: "silver",
               last_seen_camera_id: "cam-nw", last_seen_timestamp_s: 1000 }],
  },
  refusal: {
    sighting: { camera: null, time: 1000, plate: "", plateConf: 0.9,
               make: "Toyota", model: "Camry", body: "sedan", color: "silver", marks: "" },
    targets: [{ label: "Generic silver Camry", plate: "",
               make: "Toyota", model: "Camry", body: "sedan", color: "silver",
               reid_similarity: 0.97 }],
  },
  ambiguous: {
    sighting: { camera: null, time: 1000, plate: "", plateConf: 0.9,
               make: "Toyota", model: "Camry", body: "sedan", color: "silver",
               marks: "accessory: roof rack" },
    targets: [
      { target_id: "a", label: "Look-alike A", plate: "", instance_attrs: { accessory: "roof rack" },
        make: "Toyota", model: "Camry", body: "sedan", color: "silver", reid_similarity: 0.9 },
      { target_id: "b", label: "Look-alike B", plate: "", instance_attrs: { accessory: "roof rack" },
        make: "Toyota", model: "Camry", body: "sedan", color: "silver", reid_similarity: 0.88 },
    ],
  },
};

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  const s = p.sighting;
  document.getElementById("s-camera").value = s.camera || CAMERAS[0]?.camera_id || "";
  document.getElementById("s-time").value = s.time;
  document.getElementById("s-plate").value = s.plate;
  document.getElementById("s-plate-conf").value = s.plateConf;
  document.getElementById("s-make").value = s.make;
  document.getElementById("s-model").value = s.model;
  document.getElementById("s-body").value = s.body;
  document.getElementById("s-color").value = s.color;
  document.getElementById("s-marks").value = s.marks;
  clearTargets();
  p.targets.forEach((t) => addTargetForm(t));
  evaluate();
}

/* ---------------------------------------------------------------- init */

document.getElementById("evaluate").onclick = evaluate;
document.getElementById("add-target").onclick = () => addTargetForm();
document.getElementById("floor-slider").oninput = (e) => {
  document.getElementById("floor-value").textContent = parseFloat(e.target.value).toFixed(2);
};
document.querySelectorAll(".preset").forEach((btn) => {
  btn.onclick = () => applyPreset(btn.dataset.preset);
});

loadCameras().then(() => {
  addTargetForm({ label: "Test target", plate: "ABC-1234",
                  make: "Toyota", model: "Camry", body: "sedan", color: "silver" });
});
