// Coordinates are CSS pixels. The camera maps key/query indices to screen space.
export class Camera {
  constructor() { this.x = 0; this.y = 0; this.scale = 1; this.size = 256; }

  fitScale(plot) { return Math.min(plot.width, plot.height) * 0.94 / this.size; }

  fit(plot) {
    this.scale = this.fitScale(plot);
    this.x = plot.x + (plot.width - this.size * this.scale) / 2;
    this.y = plot.y + (plot.height - this.size * this.scale) / 2;
  }

  world(x, y) { return { x: (x - this.x) / this.scale, y: (y - this.y) / this.scale }; }

  constrain(plot) {
    const extent = this.size * this.scale;
    function bound(origin, length, value) {
      if (extent <= length) return origin + (length - extent) / 2;
      return Math.max(origin + length - extent - 20, Math.min(origin + 20, value));
    }
    this.x = bound(plot.x, plot.width, this.x);
    this.y = bound(plot.y, plot.height, this.y);
  }

  pan(dx, dy, plot) { this.x += dx; this.y += dy; this.constrain(plot); }

  zoom(scale, anchor, plot) {
    const point = this.world(anchor.x, anchor.y);
    this.scale = Math.max(this.fitScale(plot) * 0.5, Math.min(64, scale));
    this.x = anchor.x - point.x * this.scale;
    this.y = anchor.y - point.y * this.scale;
    this.constrain(plot);
  }

  center(point, plot) {
    this.x = plot.x + plot.width / 2 - point.x * this.scale;
    this.y = plot.y + plot.height / 2 - point.y * this.scale;
    this.constrain(plot);
  }
}
