import { Camera } from "./camera.mjs";

const element = id => document.getElementById(id);
const canvas = element("map"), context = canvas.getContext("2d");
const viewport = element("viewport"), overview = element("overview");
const camera = new Camera();
const raster = document.createElement("canvas");
const rasterContext = raster.getContext("2d");
const maps = new Map(), pointers = new Map();
let catalog = [], values = null, entry = null, hover = null;
let width = 0, height = 0, request = 0, frame = null;
const plot = () => ({ x: 72, y: 44, width: Math.max(20, width - 96), height: Math.max(20, height - 76) });
const center = () => ({ x: plot().x + plot().width / 2, y: plot().y + plot().height / 2 });
const colorStops = [[20,12,56], [75,40,124], [158,66,109], [231,121,81], [252,221,160]];

function color(value) {
  const scaled = Math.min(1, value / +element("maximum").value);
  const t = element("scale").value === "power" ? scaled ** 0.4 : scaled;
  const i = Math.min(3, Math.floor(t * 4)), fraction = t * 4 - i;
  return colorStops[i].map((channel, c) => Math.round(channel + fraction * (colorStops[i + 1][c] - channel)));
}

function probability(query, key) {
  const [layers, heads, size] = entry.shape;
  const offset = (+element("layer").value * heads + +element("head").value) * size * size;
  return values[offset + query * size + key];
}

function renderRaster() {
  if (!values) return;
  const size = entry.shape[2];
  raster.width = raster.height = size;
  const image = rasterContext.createImageData(size, size);
  const masked = getComputedStyle(document.body).getPropertyValue("--masked").trim();
  // Read the theme's masked color through the canvas to support CSS color formats.
  rasterContext.fillStyle = masked;
  rasterContext.fillRect(0, 0, 1, 1);
  const gray = rasterContext.getImageData(0, 0, 1, 1).data;
  for (let q = 0; q < size; q++) for (let k = 0; k < size; k++) {
    const rgb = k > q ? gray : color(probability(q, k));
    const index = (q * size + k) * 4;
    image.data.set([rgb[0], rgb[1], rgb[2], 255], index);
  }
  rasterContext.putImageData(image, 0, 0);
  element("heading").textContent = `Layer ${+element("layer").value + 1} · Head ${+element("head").value + 1}`;
  element("maximum-label").textContent = `${+element("maximum").value * 100}%`;
  canvas.setAttribute("aria-label", `${entry.run}, ${element("heading").textContent}. Attention by query and key position.`);
  redraw();
}

function redraw() {
  if (frame !== null) return;
  frame = requestAnimationFrame(() => { frame = null; draw(); });
}

function draw() {
  const theme = getComputedStyle(document.body), area = plot();
  const ratio = Math.min(devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.fillStyle = theme.getPropertyValue("--panel");
  context.fillRect(0, 0, width, height);
  if (!values) return;
  const size = camera.size, extent = size * camera.scale;
  context.save();
  context.beginPath(); context.rect(area.x, area.y, area.width, area.height); context.clip();
  context.imageSmoothingEnabled = false;
  context.drawImage(raster, camera.x, camera.y, extent, extent);
  if (hover) {
    context.strokeStyle = theme.getPropertyValue("--fg"); context.lineWidth = 1;
    context.strokeRect(camera.x + hover.key * camera.scale, camera.y + hover.query * camera.scale,
                       camera.scale, camera.scale);
  }
  context.restore();
  context.fillStyle = theme.getPropertyValue("--fg");
  context.strokeStyle = theme.getPropertyValue("--border");
  context.lineWidth = 1; context.strokeRect(area.x, area.y, area.width, area.height);
  context.font = "12px system-ui"; context.textAlign = "center";
  context.fillText("Key position · token being read", area.x + area.width / 2, 19);
  context.save(); context.translate(16, area.y + area.height / 2); context.rotate(-Math.PI / 2);
  context.fillText("Query position · token doing the reading", 0, 0); context.restore();
  const spacing = Math.max(1, 2 ** Math.ceil(Math.log2(48 / camera.scale)));
  const ticks = [];
  for (let i = 0; i < size; i += spacing) ticks.push(i);
  if ((size - 1 - ticks.at(-1)) * camera.scale > 40) ticks.push(size - 1);
  for (const tick of ticks) {
    const x = camera.x + (tick + 0.5) * camera.scale;
    const y = camera.y + (tick + 0.5) * camera.scale;
    if (x >= area.x + 7 && x <= area.x + area.width - 7) context.fillText(tick, x, area.y - 9);
    if (y >= area.y + 7 && y <= area.y + area.height - 7) {
      context.textAlign = "right"; context.fillText(tick, area.x - 8, y + 4); context.textAlign = "center";
    }
  }
  element("zoom-label").textContent = `${Math.round(camera.scale / camera.fitScale(area) * 100)}%`;
  const small = overview.getContext("2d");
  small.imageSmoothingEnabled = false;
  small.drawImage(raster, 0, 0, overview.width, overview.height);
  const first = camera.world(area.x, area.y), last = camera.world(area.x + area.width, area.y + area.height);
  const x = Math.max(0, first.x), y = Math.max(0, first.y);
  const right = Math.min(size, last.x), bottom = Math.min(size, last.y);
  small.strokeStyle = "white"; small.lineWidth = 3;
  small.strokeRect(x, y, Math.max(0, right - x), Math.max(0, bottom - y));
  small.strokeStyle = "black"; small.lineWidth = 1;
  small.strokeRect(x, y, Math.max(0, right - x), Math.max(0, bottom - y));
}

function options(id, count) {
  const select = element(id), old = +select.value || 0;
  select.replaceChildren(...Array.from({ length: count }, (_, i) => {
    const option = document.createElement("option"); option.value = i; option.textContent = i + 1; return option;
  }));
  select.value = Math.min(old, count - 1);
}

async function checkedFetch(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Could not load ${url} (HTTP ${response.status}). Check the local server output.`);
  return response;
}

async function selectCheckpoint() {
  const token = ++request, selected = catalog[+element("checkpoint").value];
  if (!selected) return;
  element("status").textContent = "Loading attention maps…";
  try {
    const key = `${selected.id}-${selected.version}`;
    if (!maps.has(key)) {
      const response = await checkedFetch(`/api/maps/${selected.id}`);
      const buffer = await response.arrayBuffer();
      const expected = selected.shape.reduce((a, b) => a * b, 1);
      if (buffer.byteLength !== expected * 4) throw new Error("Map size differs from its checkpoint metadata. Refresh probes.");
      const view = new DataView(buffer), array = new Float32Array(expected);
      for (let i = 0; i < expected; i++) array[i] = view.getFloat32(i * 4, true);
      maps.set(key, array);
    }
    if (token !== request) return;
    entry = selected; values = maps.get(key); camera.size = entry.shape[2];
    options("layer", entry.shape[0]); options("head", entry.shape[1]);
    element("metadata").textContent = `${entry.run} · step ${entry.checkpoint_step.toLocaleString()} · ${entry.sequences} windows · ${entry.position_encoding} positions`;
    element("status").textContent = "";
    hover = null; camera.fit(plot()); renderRaster();
  } catch (error) {
    if (token === request) element("status").textContent = error.message;
  }
}

async function refresh() {
  element("reload").disabled = true;
  ++request;
  try {
    const response = await checkedFetch("/api/checkpoints");
    catalog = await response.json();
    if (!catalog.length) throw new Error("No attention probes found.");
    const prior = entry?.run;
    element("checkpoint").replaceChildren(...catalog.map((row, i) => {
      const option = document.createElement("option"); option.value = i; option.textContent = row.run; return option;
    }));
    element("checkpoint").value = Math.max(0, catalog.findIndex(row => row.run === prior));
    maps.clear(); await selectCheckpoint();
  } catch (error) { element("status").textContent = error.message; }
  finally { element("reload").disabled = false; }
}

function point(event) {
  const rect = canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}
function resetReadout() { hover = null; element("readout").textContent = "Hover a cell to inspect it."; }
function inspect(position) {
  const area = plot(), world = camera.world(position.x, position.y);
  const key = Math.floor(world.x), query = Math.floor(world.y);
  if (position.x < area.x || position.y < area.y || position.x > area.x + area.width ||
      position.y > area.y + area.height || key < 0 || query < 0 || key >= camera.size || query >= camera.size) {
    resetReadout(); redraw(); return;
  }
  hover = { key, query };
  element("readout").textContent = `Query ${query} → key ${key}\n${key > query ? "Masked future position" : (probability(query, key) * 100).toFixed(6) + "% probability"}\nDistance ${query - key} tokens`;
  redraw();
}
function zoom(factor, anchor = center()) {
  if (!values) return;
  camera.zoom(camera.scale * factor, anchor, plot()); hover = null; redraw();
}
canvas.addEventListener("wheel", event => {
  if (!values) return;
  event.preventDefault();
  const dy = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? height : 1);
  zoom(Math.exp(-Math.max(-200, Math.min(200, dy)) * .005), point(event));
}, { passive: false });
canvas.addEventListener("pointerdown", event => {
  if (!values || event.button > 0) return;
  viewport.focus({ preventScroll: true }); event.preventDefault();
  pointers.set(event.pointerId, point(event)); canvas.setPointerCapture(event.pointerId);
  canvas.style.cursor = "grabbing";
});
canvas.addEventListener("pointermove", event => {
  if (!values) return;
  const current = point(event), previous = pointers.get(event.pointerId);
  if (!previous) { inspect(current); return; }
  if (pointers.size === 2) {
    const other = [...pointers.entries()].find(([id]) => id !== event.pointerId)[1];
    const distance = p => Math.hypot(p.x - other.x, p.y - other.y);
    const oldCenter = { x: (previous.x + other.x) / 2, y: (previous.y + other.y) / 2 };
    if (distance(previous) > 1) camera.zoom(camera.scale * distance(current) / distance(previous), oldCenter, plot());
    camera.pan((current.x - previous.x) / 2, (current.y - previous.y) / 2, plot());
  } else camera.pan(current.x - previous.x, current.y - previous.y, plot());
  pointers.set(event.pointerId, current); hover = null; redraw();
});
function release(event) {
  pointers.delete(event.pointerId); canvas.style.cursor = pointers.size ? "grabbing" : "grab";
}
for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) canvas.addEventListener(event, release);
canvas.addEventListener("pointerleave", () => { if (!pointers.size) { resetReadout(); redraw(); } });
canvas.addEventListener("dblclick", () => { camera.fit(plot()); redraw(); });
overview.addEventListener("pointerdown", event => {
  if (!values) return;
  const rect = overview.getBoundingClientRect();
  camera.center({ x: (event.clientX - rect.left) / rect.width * camera.size,
                  y: (event.clientY - rect.top) / rect.height * camera.size }, plot());
  redraw();
});
viewport.addEventListener("keydown", event => {
  if (!values) return;
  const moves = { ArrowLeft: [80, 0], ArrowRight: [-80, 0], ArrowUp: [0, 80], ArrowDown: [0, -80] };
  if (moves[event.key]) { camera.pan(...moves[event.key], plot()); redraw(); }
  else if (event.key === "+" || event.key === "=") zoom(1.5);
  else if (event.key === "-") zoom(1 / 1.5);
  else if (event.key.toLowerCase() === "f") { camera.fit(plot()); redraw(); }
  else return;
  event.preventDefault();
});
element("checkpoint").onchange = selectCheckpoint;
for (const id of ["layer", "head", "maximum", "scale"]) element(id).onchange = () => { resetReadout(); renderRaster(); };
element("zoom-in").onclick = () => zoom(1.5);
element("zoom-out").onclick = () => zoom(1 / 1.5);
element("fit").onclick = () => { camera.fit(plot()); resetReadout(); redraw(); };
element("reload").onclick = refresh;
element("fullscreen").onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await viewport.requestFullscreen();
  } catch (error) { element("status").textContent = `Fullscreen unavailable. ${error.message}`; }
};
new ResizeObserver(() => {
  const old = width && height ? camera.world(center().x, center().y) : null;
  width = viewport.clientWidth; height = viewport.clientHeight;
  if (old) camera.center(old, plot()); else camera.fit(plot());
  redraw();
}).observe(viewport);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderRaster);
refresh();
