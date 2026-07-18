/* Eyes Everywhere — shared "stylized synthetic road map" renderer.

   Not a real-world basemap. This project's camera network is entirely
   fictional ("Gridville" — see sim/road_graph.py), so pulling in real map
   tiles (OSM/Mapbox) would render real streets and real place names under
   fake camera positions, which is exactly the kind of thing this project
   is careful never to do (see the SYNTHETIC DATA ONLY banner). Instead
   this draws a clean, minimal, self-consistent road map in the app's own
   dark palette — real-map visual language (curved two-lane roads,
   street-name labels, intersection markers) without pretending any of it
   is a real place. Street names are fabricated and deterministic (hashed
   from the camera ids), same spirit as sim/render.py's per-camera tint.

   Used by both web/app.js (the live console) and web/inspector.js (the
   sandbox's stage map) so the two look consistent.
*/
"use strict";

const ROAD_ASPHALT = "#1c2531";
const ROAD_CENTERLINE = "#3a4a63";
const CAMERA_RING = "#3d5271";
const CAMERA_FILL = "#16202e";

const STREET_NAMES = [
  "Elm", "Grant", "Harbor", "Union", "Foundry", "Cascade", "Meridian",
  "Birchwood", "Prairie", "Anchor", "Cedar", "Vale", "Kestrel", "Marlow",
  "Hollow", "Beacon", "Orchard", "Summit", "Coral", "Wren",
];
const STREET_SUFFIXES = ["Ave", "St", "Blvd", "Rd", "Ln", "Way"];

function hashStr(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function edgeStreetName(srcId, dstId) {
  const h = hashStr([srcId, dstId].sort().join("|"));
  const name = STREET_NAMES[h % STREET_NAMES.length];
  const suffix = STREET_SUFFIXES[Math.floor(h / STREET_NAMES.length) % STREET_SUFFIXES.length];
  return `${name} ${suffix}`;
}

/* Bend an otherwise dead-straight edge into a gentle curve — purely a
   cosmetic path for the polyline. The reasoning layer's transit-time math
   (sim/road_graph.py, haversine + a road factor) never sees this; it
   reasons over the straight-line camera coordinates, unchanged. */
function curvedPath(a, b, seedKey) {
  const h = hashStr(seedKey);
  const sign = (h % 2 === 0) ? 1 : -1;
  const mag = 0.08 + (h % 100) / 100 * 0.10; // 8-18% perpendicular bow
  const dLat = b.lat - a.lat, dLon = b.lon - a.lon;
  const midLat = (a.lat + b.lat) / 2 - dLon * mag * sign;
  const midLon = (a.lon + b.lon) / 2 + dLat * mag * sign;
  return [[a.lat, a.lon], [midLat, midLon], [b.lat, b.lon]];
}

function trafficLightIcon() {
  return L.divIcon({
    className: "traffic-light-icon",
    html: '<div class="tl-body"><i class="tl-r"></i><i class="tl-y"></i><i class="tl-g"></i></div>',
    iconSize: [8, 16],
    // Anchor at the icon's bottom-right so the glyph sits up-and-left of
    // the intersection point instead of covering the camera marker.
    iconAnchor: [10, 20],
  });
}

// Free, no-API-key dark basemap. Only ever used when the graph's
// coordinates are REAL (world_source === "real") — see the module header.
const REAL_TILE_URL = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";
const REAL_TILE_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> ' +
  'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>';

/**
 * Draw the map's static layer onto an already-created Leaflet map, then the
 * camera nodes on top. Returns { cameraMarkers } so callers layer their own
 * dynamic markers (live contacts, state colors, highlight rings) on top —
 * this module only owns "does the map read as a place."
 *
 * `source` is "synthetic" (default) or "real", from GET /api/world_source:
 *   - synthetic: our own fabricated road network — curved roads, fabricated
 *     street-name labels, no tile layer. This is fiction and must look like
 *     a clean diagram wearing map styling, not like it's claiming to be a
 *     real place.
 *   - real: the camera coordinates are genuine GPS (e.g. from a CityFlow
 *     scenario's calibration). A real dark basemap tile layer goes underneath
 *     instead — it already has real streets and real names, so we do NOT
 *     draw our fabricated roads/labels on top of real ones.
 * Traffic-light glyphs at each camera/intersection are drawn either way:
 * they annotate "this is a camera node," not a claim about real signal
 * equipment, so they're honest in both modes.
 */
function renderRoadMap(map, cameras, adjacency, source) {
  const isReal = source === "real";
  if (isReal) {
    L.tileLayer(REAL_TILE_URL, {
      attribution: REAL_TILE_ATTRIBUTION, maxZoom: 20, subdomains: "abcd",
    }).addTo(map);
    // Maps are created with attributionControl:false for the clean look the
    // synthetic mode wants; real tiles legally require attribution, so add
    // it back here rather than asking every caller to remember.
    L.control.attribution({ prefix: false, position: "bottomright" }).addTo(map);
  } else {
    const byId = {};
    cameras.forEach((c) => { byId[c.camera_id] = c; });
    const seen = new Set();
    (adjacency || []).forEach((e) => {
      const key = [e.src, e.dst].sort().join("|");
      if (seen.has(key)) return;
      seen.add(key);
      const a = byId[e.src], b = byId[e.dst];
      if (!a || !b) return;
      const path = curvedPath(a, b, key);
      L.polyline(path, {
        color: ROAD_ASPHALT, weight: 7, opacity: 1, interactive: false,
        lineCap: "round", lineJoin: "round",
      }).addTo(map);
      L.polyline(path, {
        color: ROAD_CENTERLINE, weight: 1.5, opacity: 0.85, dashArray: "6 8",
        interactive: false, lineCap: "round",
      }).addTo(map);
      L.tooltip({
        permanent: true, direction: "center", className: "street-label", interactive: false,
      }).setLatLng(path[1]).setContent(edgeStreetName(e.src, e.dst)).addTo(map);
    });
  }

  const cameraMarkers = {};
  cameras.forEach((c) => {
    L.marker([c.lat, c.lon], {
      icon: trafficLightIcon(), interactive: false, keyboard: false,
    }).addTo(map);
    cameraMarkers[c.camera_id] = L.circleMarker([c.lat, c.lon], {
      radius: 7, color: CAMERA_RING, weight: 2, fillColor: CAMERA_FILL, fillOpacity: 1,
    }).addTo(map).bindTooltip(c.name, {
      permanent: true, direction: "bottom", offset: [0, 8], className: "cam-label",
    });
  });

  return { cameraMarkers, isReal };
}
