/* Eyes Everywhere — shared pipeline-tier diagram (PLATE -> CLASS ATTRS ->
   INSTANCE ATTRS/GEOMETRY -> REID). Read-only visual, same dark palette as
   roadmap.js. Two uses:
   - the main console's status strip (real-data mode only), reflecting the
     live GET /api/pipeline_config toggle;
   - the Cascade Inspector, which additionally highlights whichever tier
     actually decided the currently-evaluated case (MatchDecision.
     deciding_tier). The real cascade only distinguishes "plate",
     "attributes" and "reid" tiers (reasoning/cascade.py) -- CLASS ATTRS
     and INSTANCE ATTRS/GEOMETRY are drawn as two nodes for readability
     but both light up together for an "attributes" decision, since the
     system itself does not distinguish between them at that level. */
"use strict";

const PIPELINE_TIERS = [
  { id: "plate", label: "PLATE" },
  { id: "attributes", label: "CLASS ATTRS" },
  { id: "geometry", label: "INSTANCE / GEOMETRY" },
  { id: "reid", label: "REID · TIEBREAKER" },
];

/**
 * container: a DOM element to render into.
 * opts.plateOcrEnabled: false dims/dashes/badges the PLATE node "OFF" --
 *   real-clip mode only (RealPerceptor honors this; the synthetic world's
 *   SimulatedPlateReader always reads regardless, so omit this option --
 *   leave it true -- outside real mode).
 * opts.decidingTier: optional deciding_tier string ("plate"|"attributes"|
 *   "reid"|"none") highlighting which tier actually decided one case.
 */
function renderPipelineDiagram(container, opts) {
  const { plateOcrEnabled = true, decidingTier = null } = opts || {};
  const nodesHtml = PIPELINE_TIERS.map((tier, i) => {
    const isPlate = tier.id === "plate";
    const off = isPlate && !plateOcrEnabled;
    const deciding = tier.id === decidingTier
      || (decidingTier === "attributes" && tier.id === "geometry");
    const cls = ["pd-node"];
    if (off) cls.push("pd-off");
    if (deciding) cls.push("pd-deciding");
    const badge = off ? `<span class="pd-badge">OFF</span>` : "";
    const arrow = i < PIPELINE_TIERS.length - 1 ? `<div class="pd-arrow">→</div>` : "";
    return `<div class="${cls.join(" ")}">${tier.label}${badge}</div>${arrow}`;
  }).join("");
  container.innerHTML = `<div class="pd-row">${nodesHtml}</div>`;
}
