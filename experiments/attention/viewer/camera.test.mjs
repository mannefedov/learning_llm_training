import assert from "node:assert/strict";
import test from "node:test";
import { Camera } from "./camera.mjs";

const plot = { x: 72, y: 44, width: 900, height: 600 };

test("zoom preserves the point under the pointer when away from pan bounds", () => {
  const camera = new Camera();
  camera.fit(plot);
  const anchor = { x: 500, y: 300 };
  const before = camera.world(anchor.x, anchor.y);
  camera.zoom(camera.scale * 8, anchor, plot);
  const after = camera.world(anchor.x, anchor.y);
  assert.ok(Math.abs(after.x - before.x) < 1e-10);
  assert.ok(Math.abs(after.y - before.y) < 1e-10);
});

test("pan limits allow both corner cells inside the viewport", () => {
  const camera = new Camera();
  camera.fit(plot);
  camera.zoom(camera.scale * 8, { x: 500, y: 300 }, plot);
  for (const point of [{ x: .5, y: .5 }, { x: 255.5, y: 255.5 }]) {
    camera.center(point, plot);
    const x = camera.x + point.x * camera.scale;
    const y = camera.y + point.y * camera.scale;
    assert.ok(x >= plot.x && x <= plot.x + plot.width);
    assert.ok(y >= plot.y && y <= plot.y + plot.height);
  }
  camera.pan(1e6, 1e6, plot);
  assert.ok(camera.world(plot.x, plot.y).x <= 0);
  camera.pan(-2e6, -2e6, plot);
  assert.ok(camera.world(plot.x + plot.width, plot.y + plot.height).x >= 256);
});

test("fit contains all four edges on wide and narrow layouts", () => {
  for (const area of [plot, { x: 72, y: 44, width: 120, height: 220 }]) {
    const camera = new Camera();
    camera.fit(area);
    assert.ok(camera.x >= area.x && camera.y >= area.y);
    assert.ok(camera.x + 256 * camera.scale <= area.x + area.width);
    assert.ok(camera.y + 256 * camera.scale <= area.y + area.height);
  }
});
