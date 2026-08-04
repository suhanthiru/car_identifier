/* Guided tour of the console. Plain JS, no framework, no build step — same
 * rules as app.js, and loaded after it so it can read app.js's globals
 * (latestStats, clockNow, showcaseById) and drive the same controls an
 * operator would.
 *
 * WHY THIS EXISTS. The best explanatory writing in this project was buried in
 * `title=` attributes: invisible unless you happened to hover the right pixel.
 * Someone opening this cold gets five dense panels and no way in. This walks
 * them through it, and — the part that matters for a system that makes claims
 * about vehicles — states plainly what it will and will not conclude.
 *
 * TWO CONSTRAINTS SHAPE THE IMPLEMENTATION.
 *
 * 1. ANCHOR TO SHELLS, NOT CONTENTS. Most panels rebuild wholesale on a timer
 *    — #targets on every snapshot paint (up to ~60/s during a burst),
 *    #reasoning-body every 2s, #reviews and #audit every 5s, #ss-scale every
 *    1s. A spotlight held on a generated child is pointing at a detached node
 *    within a quarter of a second. Every step below targets an element
 *    declared statically in index.html. The exception is a .vehicle-tile,
 *    which is safe because renderVehicleTiles reconciles rather than rebuilds.
 *
 * 2. DRIVE THE SERVER, NOT THE CLIENT MIRROR. Pausing by writing `feedPaused`
 *    lasts until the 1s stats poll overwrites it. Pause goes through
 *    POST /api/feed_control like the button does.
 *
 * Progress is persisted because two paths reload the page out from under us:
 * reconcileScenario's drift check and a scenario switch.
 */
"use strict";

const TOUR_SEEN_KEY = "eyes.tour.seen";
const TOUR_STEP_KEY = "eyes.tour.step";

/* localStorage throws in private mode — the same try/catch idiom the cascade
 * panel's collapse memory uses. A tour that cannot remember it was dismissed
 * is a nuisance; a tour that throws on load breaks the console. */
function tourGet(key) {
  try { return localStorage.getItem(key); } catch (e) { return null; }
}
function tourSet(key, value) {
  try { localStorage.setItem(key, value); } catch (e) { /* private mode */ }
}

/* Steps. `anchor` is a selector resolved fresh every paint, so a panel that
 * appears late (the cascade panel only exists once a target is followed) is
 * skipped rather than pointed at nothing. `when` gates a step on the console
 * actually being in the right mode. */
const TOUR_STEPS = [
  {
    anchor: null,
    title: "What this is",
    body: `<p>This is a <b>vehicle re-identification console</b>. Cameras see
      cars, the cars drive out of view, and the system tries to work out when
      a car at one camera is the same car it saw at another.</p>
      <p>Everything on this screen is running on real footage from a public
      benchmark dataset — real cameras, real vehicles, real ground truth.</p>`,
    note: `The interesting question is not "can it match cars". It is
      "when should it refuse to". Most of what you are about to see is the
      system declining to conclude things.`,
  },
  {
    anchor: "#feed-toggle",
    title: "The replay starts paused",
    body: `<p>Time is stopped. Nothing has been ingested yet, and the clock only
      advances once you press this.</p>`,
    note: `Deliberate: a replay that ran while nobody was looking would burn
      through the footage before you had picked anything to follow.`,
    act: `Leave it paused for now — the tour will start it when there is
      something to watch.`,
  },
  {
    anchor: "#cityflow-vehicles",
    title: "The cars you can follow",
    body: `<p>Every tile here is a real vehicle from the footage. Clicking one
      flags it as a target — the car you want the system to find again.</p>
      <p>The filter above is set to <b>the interesting ones</b>: a fixed
      shortlist, each with a stated reason and a prediction you can check.</p>`,
    note: `That shortlist is ranked from the dataset's own ground truth, never
      from what the system concluded. Picking the cars we already knew would
      work would be rigging the demo.`,
    scroll: true,
  },
  {
    anchor: "#cityflow-vehicles .vehicle-tile.vt-showcase",
    title: "Pick one",
    body: `<p>Hover any showcase tile and it tells you what it should make the
      system do. Some are here because they work. At least one is here because
      it <i>doesn't</i>, and that is worth seeing too.</p>`,
    act: `Click a tile to flag it. It will appear in SELECTED above.`,
    scroll: true,
  },
  {
    anchor: "#split-selected",
    title: "Your targets",
    body: `<p>Flagged cars live here, above the browse list rather than buried
      under it. Each shows what the system has done about it so far —
      how many sightings it compared, and how many it declined.</p>`,
    scroll: true,
  },
  {
    anchor: "#reasoning-panel",
    title: "How it decides",
    body: `<p>This is the cascade, and it runs in a fixed order:
      <b>plate → class attributes → distinguishing marks → appearance</b>.</p>
      <p>Appearance — the neural network everyone reaches for first — comes
      <b>last</b>, and its contribution is capped at 0.30 out of 1.0.</p>`,
    note: `Appearance similarity can only ever break a tie, never carry a
      decision. Two silver sedans look alike to any embedding; that is exactly
      the case the system must not get confidently wrong.`,
  },
  {
    anchor: "#reasoning-panel",
    title: "Why every car reads 0.22",
    body: `<p>Distinctiveness asks: <i>how uniquely does this evidence name one
      car?</i> A clean plate read scores 1.0. Matching class attributes score
      <code>0.20 / 0.90 = 0.22</code>.</p>
      <p>This dataset has no readable plates and no make/model annotation. So
      the only evidence that ever fires is "silver sedan matches silver sedan",
      and that is worth 0.22 on every car, every time.</p>`,
    note: `0.22 sits below the 0.30 floor required to name an individual. So
      the system returns a candidate SET and refuses to point at one car. The
      number is constant because the evidence really is that weak — and saying
      so is more useful than a confidence figure that looks decisive.`,
  },
  {
    anchor: "#queue-panel",
    title: "What it refuses to do alone",
    body: `<p>When the system cannot justify a conclusion, it puts the decision
      here instead of making it. Nothing auto-confirms on this dataset.</p>
      <p>Each card shows the evidence for and against, and what would have had
      to be different to flip the answer.</p>`,
    note: `A system that acts on vehicle identity should be able to say why,
      and be overruled. The queue is where a human stays in the loop.`,
  },
  {
    anchor: "#map-panel",
    title: "Space and time, not pixels",
    body: `<p>The map and the timeline below it carry the one signal that is
      independent of appearance: <b>could this car physically have got
      there in that time?</b></p>
      <p>The travel windows come from hops real vehicles were actually observed
      to make in this footage — not from a guessed speed limit.</p>`,
    note: `Independence is the point. Look-alikes defeat every appearance
      signal at once, because they genuinely look alike. They do not defeat
      physics.`,
  },
  {
    anchor: "#footer-strip",
    title: "The audit log",
    body: `<p>Every decision is written to a hash-chained log. Change any entry
      after the fact and the chain stops verifying.</p>`,
    note: `If a system like this is ever used to justify a real-world action,
      "what did it conclude, when, and on what evidence" has to survive being
      asked months later.`,
  },
  {
    anchor: null,
    title: "What this cannot show you",
    body: `<p>Being straight about the limits: this dataset has no readable
      plates, no make or model labels, and no distinguishing marks. So plate
      matching, plate contradiction and mark-based vetoes — real parts of the
      cascade — never fire here.</p>
      <p>They are not hidden. The <a href="inspector.html">reasoning
      inspector</a> runs the <i>same</i> cascade on cases you build by hand,
      with plates and marks, so you can see those paths work.</p>`,
    note: `The limits you are looking at are properties of the data, not
      of the system — and the console is built to say which is which.`,
  },
];

let tourIndex = 0;
let tourActive = false;
let tourEls = null;
let tourTimer = null;

function tourBuild() {
  if (tourEls) return tourEls;
  const root = document.createElement("div");
  root.id = "tour-root";
  root.className = "hidden";
  const shades = [0, 1, 2, 3].map(() => {
    const s = document.createElement("div");
    s.className = "tour-shade";
    root.appendChild(s);
    return s;
  });
  const ring = document.createElement("div");
  ring.className = "tour-ring";
  root.appendChild(ring);
  const bubble = document.createElement("div");
  bubble.className = "tour-bubble";
  bubble.setAttribute("role", "dialog");
  bubble.setAttribute("aria-modal", "true");
  bubble.setAttribute("aria-live", "polite");
  bubble.tabIndex = -1;
  root.appendChild(bubble);
  document.body.appendChild(root);
  tourEls = { root, shades, ring, bubble };
  return tourEls;
}

/* Lay the four shades around `rect`, leaving it uncovered and clickable. */
function tourShade(rect) {
  const { shades } = tourEls;
  const W = window.innerWidth;
  const H = window.innerHeight;
  const pad = 4;
  const r = rect
    ? { top: rect.top - pad, left: rect.left - pad,
        right: rect.right + pad, bottom: rect.bottom + pad }
    : { top: 0, left: 0, right: 0, bottom: 0 };
  const boxes = rect ? [
    { left: 0, top: 0, width: W, height: Math.max(0, r.top) },
    { left: 0, top: r.bottom, width: W, height: Math.max(0, H - r.bottom) },
    { left: 0, top: r.top, width: Math.max(0, r.left), height: Math.max(0, r.bottom - r.top) },
    { left: r.right, top: r.top, width: Math.max(0, W - r.right), height: Math.max(0, r.bottom - r.top) },
  ] : [
    { left: 0, top: 0, width: W, height: H },
    { left: 0, top: 0, width: 0, height: 0 },
    { left: 0, top: 0, width: 0, height: 0 },
    { left: 0, top: 0, width: 0, height: 0 },
  ];
  boxes.forEach((b, i) => {
    const s = shades[i];
    s.style.left = `${b.left}px`;
    s.style.top = `${b.top}px`;
    s.style.width = `${b.width}px`;
    s.style.height = `${b.height}px`;
  });
}

/* Put the bubble on whichever side of the target has room, and point the
 * arrow back at the target's centre. */
function tourPlace(rect, step) {
  const { bubble } = tourEls;
  const W = window.innerWidth;
  const H = window.innerHeight;
  const bw = bubble.offsetWidth;
  const bh = bubble.offsetHeight;
  const gap = 14;
  if (!rect) {
    bubble.dataset.side = "center";
    bubble.style.left = `${Math.round((W - bubble.offsetWidth) / 2)}px`;
    bubble.style.top = `${Math.round((H - bh) / 2)}px`;
    return;
  }
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  let side;
  if (rect.bottom + gap + bh < H) side = "bottom";
  else if (rect.top - gap - bh > 0) side = "top";
  else if (rect.left - gap - bw > 0) side = "left";
  else side = "right";
  let left;
  let top;
  if (side === "bottom" || side === "top") {
    left = clamp(rect.left + rect.width / 2 - bw / 2, 8, W - bw - 8);
    top = side === "bottom" ? rect.bottom + gap : rect.top - gap - bh;
    const ax = clamp(rect.left + rect.width / 2 - left, 14, bw - 14);
    bubble.style.setProperty("--tour-arrow", `${Math.round(ax) - 5}px`);
  } else {
    top = clamp(rect.top + rect.height / 2 - bh / 2, 8, H - bh - 8);
    left = side === "left" ? rect.left - gap - bw : rect.right + gap;
    const ay = clamp(rect.top + rect.height / 2 - top, 14, bh - 14);
    bubble.style.setProperty("--tour-arrow", `${Math.round(ay) - 5}px`);
  }
  // data-side names where the bubble sits; the arrow is drawn on the opposite
  // edge, which the CSS selectors already account for.
  bubble.dataset.side = side;
  bubble.style.left = `${Math.round(left)}px`;
  bubble.style.top = `${Math.round(top)}px`;
}

function tourResolve(step) {
  if (!step.anchor) return null;
  try { return document.querySelector(step.anchor); } catch (e) { return null; }
}

/* Recomputes geometry only. Called on a timer and on scroll/resize, so it must
 * not rebuild the bubble — that would drop focus and restart CSS animations
 * several times a second. */
function tourReposition() {
  if (!tourActive) return;
  const step = TOUR_STEPS[tourIndex];
  if (!step) return;
  const el = tourResolve(step);
  const visible = el && el.offsetParent !== null;
  const rect = visible ? el.getBoundingClientRect() : null;
  // A zero-size rect means the element exists but is not laid out (a hidden
  // panel, or a headless viewport). Treat it as absent rather than spotlighting
  // the top-left corner.
  const usable = rect && rect.width > 1 && rect.height > 1 ? rect : null;
  tourShade(usable);
  tourEls.ring.style.display = usable ? "block" : "none";
  if (usable) {
    tourEls.ring.style.left = `${usable.left - 4}px`;
    tourEls.ring.style.top = `${usable.top - 4}px`;
    tourEls.ring.style.width = `${usable.width + 8}px`;
    tourEls.ring.style.height = `${usable.height + 8}px`;
    tourEls.ring.classList.toggle("tour-ring-act", Boolean(step.act));
  }
  tourPlace(usable, step);
}

function tourRender() {
  const step = TOUR_STEPS[tourIndex];
  if (!step) { tourEnd(true); return; }
  const { bubble } = tourEls;
  const last = tourIndex === TOUR_STEPS.length - 1;
  bubble.innerHTML = `
    <h4>${step.title}</h4>
    ${step.body}
    ${step.act ? `<p class="tour-do">${step.act}</p>` : ""}
    ${step.note ? `<p class="tour-note">${step.note}</p>` : ""}
    <div class="tour-foot">
      <span class="tour-step">${tourIndex + 1} / ${TOUR_STEPS.length}</span>
      <button class="tour-skip" data-tour="end">skip</button>
      <span class="tour-spacer"></span>
      ${tourIndex ? `<button data-tour="back">← back</button>` : ""}
      <button class="tour-primary" data-tour="${last ? "end" : "next"}">${last ? "Done" : "Next →"}</button>
    </div>`;
  bubble.querySelectorAll("[data-tour]").forEach((b) => {
    b.onclick = () => {
      const what = b.dataset.tour;
      if (what === "end") tourEnd(true);
      else if (what === "back") tourGo(tourIndex - 1);
      else tourGo(tourIndex + 1);
    };
  });
  const el = tourResolve(step);
  if (step.scroll && el && el.scrollIntoView) {
    el.scrollIntoView({ block: "nearest", inline: "nearest" });
  }
  tourReposition();
  bubble.focus();
}

function tourGo(i) {
  tourIndex = Math.max(0, Math.min(TOUR_STEPS.length - 1, i));
  tourSet(TOUR_STEP_KEY, String(tourIndex));
  tourRender();
}

function tourStart(fromButton) {
  tourBuild();
  tourActive = true;
  tourEls.root.classList.remove("hidden");
  // Restarting from the button starts over; a reload mid-tour resumes.
  const saved = parseInt(tourGet(TOUR_STEP_KEY) || "0", 10);
  tourIndex = fromButton ? 0 : (Number.isFinite(saved) ? saved : 0);
  if (tourIndex >= TOUR_STEPS.length) tourIndex = 0;
  // Paint BEFORE the request, not after. Awaiting the pause first meant a
  // click on ? TOUR showed nothing until a round-trip completed, which reads
  // as a dead button — and on a stalled server it never appeared at all.
  tourRender();
  if (!tourTimer) tourTimer = setInterval(tourReposition, 400);
  // Stop the world while it talks. Through the server, not the client mirror:
  // the 1s stats poll would overwrite a local flag within a second.
  api("/api/feed_control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paused: true }),
  }).catch(() => { /* the tour is still worth showing on a dead server */ });
}

function tourEnd(seen) {
  tourActive = false;
  if (tourEls) tourEls.root.classList.add("hidden");
  if (tourTimer) { clearInterval(tourTimer); tourTimer = null; }
  if (seen) {
    tourSet(TOUR_SEEN_KEY, "1");
    tourSet(TOUR_STEP_KEY, "0");
  }
}

/* Escape closes, arrows navigate. The console's only other document-level key
 * handler belongs to the clip overlay and is attached only while that overlay
 * is open, so defer to it rather than fight it. */
document.addEventListener("keydown", (ev) => {
  if (!tourActive) return;
  const clip = document.getElementById("clip-view");
  if (clip && !clip.classList.contains("hidden")) return;
  if (ev.key === "Escape") { ev.preventDefault(); tourEnd(true); }
  else if (ev.key === "ArrowRight" || ev.key === "Enter") { ev.preventDefault(); tourGo(tourIndex + 1); }
  else if (ev.key === "ArrowLeft") { ev.preventDefault(); tourGo(tourIndex - 1); }
});

window.addEventListener("resize", tourReposition);
// Capture phase: the panels that scroll are inner elements, and scroll does
// not bubble.
window.addEventListener("scroll", tourReposition, true);

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("tour-start");
  if (btn) btn.onclick = () => tourStart(true);
  if (tourGet(TOUR_SEEN_KEY) === "1") return;
  // Wait for the console to finish its first paint, so the panels the tour
  // points at exist and have been laid out.
  setTimeout(() => { if (!tourActive) tourStart(false); }, 1200);
});
