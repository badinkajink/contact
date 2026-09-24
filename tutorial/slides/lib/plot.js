/* plot.js: plotting for the curriculum decks, on top of fig.js.
 *
 * Load after fig.js as a classic script:
 *   <script src="lib/fig.js"></script>
 *   <script src="lib/plot.js"></script>
 * It adds methods to Fig.prototype and defines the global `Plot`. Style: Times, black 1 px
 * axes, the web-safe palette in Fig.C. Everything is SVG except heat maps and colorbars,
 * which are drawn into an offscreen canvas and embedded as a data-URL <image> (works from
 * file://).
 *
 * ---------------------------------------------------------------- coordinates
 * Fig keeps one scale for x and y (circles stay round). Plots usually need two:
 *
 *   const F = Plot.fig("#svg", { w: 540, h: 380, xlim: [-2, 2], ylim: [0, 8] });
 *
 * maps xlim x ylim onto the rectangle inside `margin` (px; default {l: 62, r: 16, t: 14,
 * b: 52}; a number sets all four) with separate x and y scales. `equal: true` keeps one scale
 * (the box is centred and the limits are the minimum shown). Plot.fig returns an ordinary Fig:
 * every fig.js method (arrow, text, label, dot, handle, in3, group, ...) works in these world
 * coordinates. F.limits({xlim, ylim, margin, equal}) re-maps an existing figure the same way
 * (the strip chart calls it every frame). After limits(), F.sx and F.sy are the px-per-unit
 * scales and F.s = min(F.sx, F.sy) sizes fig.js circles.
 *
 * The plot box is the world rectangle that plotting methods clip to. F.bounds() returns it as
 * {x0, x1, y0, y1}: the limits after Plot.fig / F.limits, the axes range after F.axes, and
 * otherwise the whole visible world rect of the figure. Methods that sample a function over a
 * rectangle (contour, heat, region, quiver) default to the plot box and accept {xlim, ylim}.
 *
 * Draw order is paint order: heat and region first, then axes, then curves and labels.
 * Every drawing method returns the SVG element it made (a <g> for composites) with computed
 * data attached as properties; F.contour returns its polylines. To make any of them a build
 * step, draw into a group: F.group({class: "step"}).curve(...).
 *
 * Text: axis labels, tick labels given as strings, legend names and bar labels are upright;
 * $...$ segments are italic math with fig.js sub/superscripts: "$J(z_k)$", "time $t$ (s)",
 * "$x^2$". Numbers use a true minus sign (U+2212).
 *
 * ---------------------------------------------------------------- axes
 * F.axes({xlim, ylim, xticks, yticks, xstep, ystep, xfmt, yfmt, xlabel, ylabel, grid, frame,
 *         arrows, origin, clip, size, lsize, color, gridColor, ylabelAt}) -> <g>
 *   xlim, ylim   world range of the axes (default F.bounds()); also becomes the plot box
 *                unless clip: false.
 *   xticks       undefined: automatic round ticks (1-2-5 steps, thinned so labels never touch);
 *                a number: about that many ticks; an array of values; an array of
 *                [value, "label"] pairs (for pi labels: [[Math.PI, "$π$"]]);
 *                false or []: no ticks. yticks the same.
 *   xstep        exact tick spacing (overrides the automatic choice). ystep the same.
 *   xfmt(v)      tick label formatter. Default: fixed decimals from the tick step.
 *   xlabel       text under the x axis (box style) or above the right end (origin style).
 *   ylabel       text left of the y axis, rotated (box style; ylabelAt: "top" puts it
 *                horizontal above the axis) or right of the top end (origin style).
 *   grid         light gray lines at every tick. frame: full rectangle around the box.
 *   origin       true or [x, y]: math-style axes through that point (clamped to the box),
 *                ticks across the lines, no label at the crossing, arrows on by default.
 *                Without origin: box style, axes along the bottom and left edges, ticks out.
 *   arrows       arrowheads 9 px past the positive ends.
 *   size         tick label size (18 px); lsize axis label size (21 px).
 *
 * F.hline(y, {color, width, dash, label}) / F.vline(x, ...): reference line across the box.
 * F.legend(items, {at, size}) -> <g>. items: [{name, color, width, dash, marker}] (marker:
 *   true draws a dot instead of a line). at: "nw" (default), "ne", "sw", "se", or [px, py].
 *
 * ---------------------------------------------------------------- curves and points
 * F.curve(f, x0, x1, {n, color, width, dash, opacity, clip, jumps}) -> <path>
 *   The graph of y = f(x) sampled at n + 1 points (n = 400), clipped to the plot box.
 *   x0, x1 default to the box. Non-finite values break the line. A segment whose two ends
 *   lie beyond opposite edges (the asymptotes of tan x or 1/x) is dropped; jumps: true keeps
 *   it. clip: false disables clipping. path.lines holds the clipped polylines.
 * F.param(fn, t0, t1, {n, ...}) -> <path>   fn(t) = [x, y], same options.
 * F.polyline(pts, {color, width, dash, close, fill, fillOpacity, stroke}) -> <path> or <g>
 *   Clipped polyline. With fill, the closed polygon is clipped (Sutherland-Hodgman) and filled;
 *   stroke: "none" leaves only the fill.
 * F.area(f, x0, x1, {base, fill, opacity, n}) -> <path>   shades between y = f(x) and
 *   y = base (a number, default 0, or a function of x), clipped to the box.
 * F.scatter(pts, {r, fill, stroke, width, opacity, shape}) -> <g>
 *   r in px (4). fill may be a function (p, i) -> color. shape: "circle" (default),
 *   "square", "x", "+". Points outside the box are skipped.
 *
 * ---------------------------------------------------------------- fields over a rectangle
 * F.contour(f, levels, {xlim, ylim, nx, ny, color, colors, cmap, width, dash, opacity,
 *           labels, lsize}) -> polylines
 *   Marching squares on an nx x ny grid (default one cell per 5 px), saddles resolved by the
 *   cell-centre value. levels: an array, or a number n for about n levels at round values.
 *   colors: an array per level or a function (level, k) -> color; cmap: a colormap name
 *   (levels spread over t in [0.25, 1]); color: one color (default ink). labels: true or a
 *   function level -> string; one label per level on its longest line, with a white halo,
 *   placed so labels do not overlap each other or the box edge.
 *   Returns a flat array of polylines ([[x, y], ...]) with .level and .closed on each, and
 *   .levels, .range ([min, max] of f on the grid) and .el (the <g>) on the array.
 *   Plot.contours(f, levels, {xlim, ylim, nx, ny}) computes the same without drawing.
 * F.heat(f, {xlim, ylim, nx, ny, cmap, vmin, vmax, opacity, smooth}) -> <image>
 *   f at cell centres (default one cell per 3 px) through the colormap (default "heat");
 *   vmin / vmax default to the data range; NaN is transparent. smooth: false shows blocky
 *   cells. image.vmin, .vmax, .cmap feed F.colorbar({of: image}).
 * F.colorbar({of, cmap, vmin, vmax, at, len, w, ticks, label, size}) -> <g>
 *   Vertical bar; default 14 px right of the plot box and as tall as it, so leave
 *   margin.r of about 80 px. at: [px, py] top-left corner; len: height in px.
 * F.region(pred, {xlim, ylim, nx, ny, fill, opacity, stroke, width, dash}) -> <g>
 *   Shades the set {(x, y): pred(x, y)} (default pale yellow at opacity 0.7) and strokes its
 *   boundary (default ink, 1.5 px; stroke: "none" to omit). Boundary points are found by
 *   bisection on grid edges, so the edge is exact to 1/65536 of a cell; corners of the set are
 *   cut within one cell (default 4 px). For an exact polygon use F.poly. g.lines holds the
 *   boundary polylines.
 * F.quiver(field, {xlim, ylim, n, scale, color, cmap, width, head, pivot}) -> <g>
 *   field(x, y) = [u, v] at the centres of an n x n grid (n = 12, or [nx, ny]). scale: world
 *   units of arrow per unit of field; default: the longest arrow is 0.85 of a cell. color:
 *   a color or a function (x, y, u, v) -> color; cmap colors by magnitude. pivot: "mid"
 *   (default) or "tail". g.scale is the scale used.
 * F.grid2(A, {n, color, axisColor, width, original, square, e1Color, e2Color, labels}) -> <g>
 *   A = [[a, b], [c, d]]. Draws the images of the lines x = i and y = i (|i| <= n; default
 *   enough to cover the box) under v -> A v, the images of the two axes darker, the arrows
 *   A e1 (red) and A e2 (green) with labels (default ["Ae_1", "Ae_2"] in fig.js label syntax;
 *   false for none). original: true draws the unmapped grid in light gray underneath;
 *   square: true (or a color) shades the image of the unit square.
 *
 * ---------------------------------------------------------------- 3D
 * F.surface3(view, f, {xlim, ylim, n, zscale, zlim, fill, stroke, width, opacity, shade})
 *   -> <g>. view from Fig.view3(az, el) (needs view.depth for ordering). Quads of z = f(x, y)
 *   over an n x n grid (n = 24, or [nx, ny]) drawn far-to-near (painter's order by the depth
 *   of each quad centre). z is multiplied by zscale for display. fill: a color (pale yellow
 *   default), a colormap name (colored by z over zlim, default the data range) or a function
 *   (x, y, z) -> color. shade (0.35) darkens faces turned away from the viewer. stroke: mesh
 *   color (ink; "none" for none), width 0.5.
 * F.axes3(view, {len, labels, origin, color, width, neg, lsize}) -> <g>
 *   Arrows along x, y, z from origin; len a number or [lx, ly, lz]; labels ["x", "y", "z"]
 *   or false; neg: true adds dashed negative half-axes. Painter's order is the caller's:
 *   draw axes3 before surface3 to hide them behind the surface, after to draw them on top.
 *
 * ---------------------------------------------------------------- statistics
 * F.hist(samples, {bins, range, density, fill, stroke, width, opacity}) -> <g>
 *   bins: a count (default ceil(sqrt(N)), at most 60) or an array of edges; range [a, b]
 *   (default data min..max); the last bin includes its right edge; samples outside are
 *   ignored. density: true divides by (count in range x bin width), so the bars have area 1.
 *   g.edges, g.counts, g.heights. Plot.histogram(samples, opts) computes without drawing.
 * F.bars(values, labels, {at, width, base, fill, stroke, values, fmt, size}) -> <g>
 *   Bar i centred at x = at[i] (default i), width 0.7, from base (0) to values[i]. fill: a
 *   color, an array, or a function (v, i) -> color. labels are written under the plot box
 *   where x tick labels go, so call F.axes({xticks: false, ...}). values: true writes each
 *   value above its bar (fmt(v) formats it).
 *
 * ---------------------------------------------------------------- colormaps
 * Plot.colormap(name) -> cmap, with cmap(t) = "#rrggbb" for t in [0, 1] (clamped) and
 *   cmap.rgb(t) = [r, g, b]. Names: "heat" (sequential: white, pale yellow, orange, dark red;
 *   CIE lightness falls monotonically from 100 to 19), "diverging" (alias "div": dark blue,
 *   white at 0.5, dark red), "gray" (white to dark gray). Also accepts an array of colors
 *   (evenly spaced) or of [t, color] stops, or a function t -> color. Plot.colormaps lists
 *   the names. For a diverging map centred on 0 pass vmin = -vmax.
 *
 * ---------------------------------------------------------------- strip chart
 * Plot.strip(svgOrSelector, {w, h, series, window, ylim, ylabel, xlabel, margin, grid,
 *            legend, size}) -> {push(t, values), clear(), draw(), fig}
 *   A rolling time series. series: [{name, color, width, dash}] (default colors red, blue,
 *   green, purple, orange, teal). push(t, values) appends one sample (values: an array in
 *   series order, or a number for one series; null or NaN leaves a gap) and redraws on the
 *   next animation frame. The x axis shows [t - window, t] (window = 10 s), starting at the
 *   first t. ylim: fixed [a, b]; omitted, it grows to round values around everything pushed
 *   since the last clear(). A t smaller than the last one clears the chart (a reset
 *   simulation). draw() redraws now. xlabel defaults to "time $t$ (s)".
 *
 * ---------------------------------------------------------------- helpers
 * Plot.ticks(lo, hi, n) -> round tick values; Plot.fmt(v, decimals?) -> label string;
 * Plot.linspace(a, b, n); Plot.clip(pts, box, jumps) -> clipped polylines;
 * Plot.rgb(color) -> [r, g, b] or null; Plot.hex([r, g, b]) -> "#rrggbb".
 */
(function () {
  "use strict";
  const Fig = window.Fig;
  if (!Fig) throw new Error("plot.js: load lib/fig.js before lib/plot.js");
  const C = Fig.C, el = Fig.el, P = Fig.prototype;
  const MINUS = "−";
  const Plot = {};

  // ---------------------------------------------------------------- small utilities
  const isNum = (v) => typeof v === "number" && isFinite(v);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const snap = (v) => Math.floor(v) + 0.5;
  const SX = (F) => F.sx || F.s, SY = (F) => F.sy || F.s;
  const hasOwn = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
  const f2 = (v) => v.toFixed(2);
  // The object in a group chain that owns the coordinate transform (see F.limits).
  function owner(F) {
    let o = F;
    while (o && !hasOwn(o, "X") && !hasOwn(o, "svg")) o = Object.getPrototypeOf(o);
    return o || F;
  }
  function safe(f) {
    return function () {
      try { return f.apply(null, arguments); } catch (e) { return NaN; }
    };
  }

  function strokeOf(o, dflt, wdflt) {
    return {
      stroke: o.color || o.stroke || dflt || C.ink,
      "stroke-width": o.width || wdflt || 1.5,
      "stroke-dasharray": o.dash === true ? "6 5" : o.dash || null,
      "stroke-linecap": o.cap || "round",
      "stroke-linejoin": "round",
      opacity: o.opacity,
    };
  }

  // Width of a string in Times, in px (an estimate; SVG text cannot be measured while hidden).
  function textW(s, size) {
    let w = 0, skip = false;
    for (const ch of String(s)) {
      if (ch === "$" || ch === "{" || ch === "}" || ch === "​") continue;
      if (ch === "_" || ch === "^") { skip = true; continue; }
      let c;
      if (ch >= "0" && ch <= "9") c = 0.5;
      else if (ch === "." || ch === "," || ch === " " || ch === ":" || ch === ";") c = 0.25;
      else if (ch === MINUS || ch === "+" || ch === "=" || ch === "<" || ch === ">") c = 0.564;
      else if (ch === "-" || ch === "(" || ch === ")" || ch === "[" || ch === "]") c = 0.333;
      else if (ch === "i" || ch === "l" || ch === "j" || ch === "t" || ch === "f" || ch === "r") c = 0.3;
      else if (ch === "m" || ch === "w" || ch === "M" || ch === "W") c = 0.8;
      else if (ch >= "A" && ch <= "Z") c = 0.68;
      else c = 0.48;
      w += skip ? c * 0.7 : c;
      skip = false;
    }
    return w * size;
  }

  // fig.js label syntax (f_n, f_{t1}, x^2) into tspans; same shifts as fig.js fillLabel.
  function mathInto(t, s, size) {
    let i = 0, buf = "";
    const flush = () => { if (buf) { el("tspan", {}, t).textContent = buf; buf = ""; } };
    const group = () => {
      if (s[i] === "{") {
        let j = s.indexOf("}", i);
        if (j < 0) j = s.length;
        const g = s.slice(i + 1, j);
        i = j + 1;
        return g;
      }
      return s[i++] || "";
    };
    while (i < s.length) {
      const c = s[i];
      if (c === "_" || c === "^") {
        flush();
        i++;
        const g = group();
        const shift = c === "_" ? 0.3 : -0.42;
        el("tspan", { dy: shift + "em", "font-size": Math.round(size * 0.7) }, t).textContent = g;
        el("tspan", { dy: (-shift * 0.7) + "em", "font-size": size }, t).textContent = "​";
      } else {
        buf += c;
        i++;
      }
    }
    flush();
  }
  // Upright text with $...$ segments set as italic math.
  function rich(t, s, size) {
    s = String(s);
    if (s.indexOf("$") < 0) { t.textContent = s; return t; }
    s.split("$").forEach((part, k) => {
      if (!part) return;
      if (k % 2) mathInto(el("tspan", { "font-style": "italic" }, t), part, size);
      else el("tspan", {}, t).textContent = part;
    });
    return t;
  }
  function txt(parent, x, y, s, o) {
    o = o || {};
    const size = o.size || 18;
    const t = el("text", {
      x: f2(x), y: f2(y),
      "font-size": size,
      "text-anchor": o.anchor || "middle",
      "dominant-baseline": o.baseline || "middle",
      fill: o.color || C.ink,
      transform: o.rotate ? "rotate(" + o.rotate + " " + f2(x) + " " + f2(y) + ")" : null,
    }, parent);
    if (o.halo) {
      t.setAttribute("stroke", o.halo === true ? "#ffffff" : o.halo);
      t.setAttribute("stroke-width", 4);
      t.setAttribute("stroke-linejoin", "round");
      t.setAttribute("paint-order", "stroke");
    }
    return rich(t, s, size);
  }

  // ---------------------------------------------------------------- numbers and ticks
  function decimalsOf(v) {
    for (let d = 0; d < 7; d++) {
      const s = v * Math.pow(10, d);
      if (Math.abs(s - Math.round(s)) < 1e-6 * Math.max(1, Math.abs(s))) return d;
    }
    return 6;
  }
  function fixed(v, dec) {
    let s = v.toFixed(dec);
    if (parseFloat(s) === 0) s = (0).toFixed(dec);
    return s.replace("-", MINUS);
  }
  Plot.fmt = function (v, dec) {
    if (dec !== undefined) return fixed(v, dec);
    if (!isNum(v)) return String(v);
    if (v === 0) return "0";
    const a = Math.abs(v);
    const s = a >= 1e5 || a < 1e-3 ? v.toExponential(2) : String(parseFloat(v.toPrecision(4)));
    return s.replace("-", MINUS);
  };
  function niceStep(raw) {
    if (!(raw > 0) || !isFinite(raw)) return 1;
    const e = Math.floor(Math.log10(raw)), b = Math.pow(10, e), f = raw / b;
    return (f <= 1 + 1e-9 ? 1 : f <= 2 + 1e-9 ? 2 : f <= 5 + 1e-9 ? 5 : 10) * b;
  }
  function bumpStep(st) {
    const e = Math.floor(Math.log10(st) + 1e-9), b = Math.pow(10, e), f = Math.round(st / b);
    return (f === 1 ? 2 : f === 2 ? 5 : 10) * b;
  }
  function multiples(lo, hi, st) {
    const eps = (hi - lo) * 1e-9, out = [];
    const k0 = Math.ceil((lo - eps) / st), k1 = Math.floor((hi + eps) / st);
    for (let k = k0; k <= k1 && out.length < 1000; k++) out.push(parseFloat((k * st).toFixed(12)));
    return out;
  }
  Plot.ticks = function (lo, hi, n) {
    return multiples(lo, hi, niceStep((hi - lo) / Math.max(1, n || 5)));
  };
  Plot.linspace = function (a, b, n) {
    const out = [];
    for (let i = 0; i < n; i++) out.push(n === 1 ? a : a + (b - a) * i / (n - 1));
    return out;
  };

  // Tick list [{v, s}] for one axis. pxLen: axis length in px.
  function makeTicks(spec, step, lim, pxLen, size, fmt, isX) {
    if (spec === false || spec === null) return [];
    const lo = lim[0], hi = lim[1], eps = (hi - lo) * 1e-9;
    const inR = (v) => v >= lo - eps && v <= hi + eps;
    if (Array.isArray(spec)) {
      if (!spec.length) return [];
      if (Array.isArray(spec[0])) return spec.filter((p) => inR(p[0])).map((p) => ({ v: p[0], s: String(p[1]) }));
      const dec = Math.max.apply(null, spec.map(decimalsOf));
      return spec.filter(inR).map((v) => ({ v, s: fmt ? fmt(v) : fixed(v, dec) }));
    }
    const label = (list, st) => {
      const dec = decimalsOf(st);
      return list.map((v) => ({ v, s: fmt ? fmt(v) : fixed(v, dec) }));
    };
    if (step) return label(multiples(lo, hi, step), step);
    const target = typeof spec === "number" ? spec : Math.max(2, Math.floor(pxLen / (isX ? 90 : 55)));
    let st = niceStep((hi - lo) / Math.max(1, target));
    let list = label(multiples(lo, hi, st), st);
    for (let guard = 0; guard < 12 && list.length > 2; guard++) {
      const gap = st * pxLen / (hi - lo);
      const need = isX ? Math.max.apply(null, list.map((t) => textW(t.s, size))) + 14 : size + 8;
      if (gap >= need) break;
      st = bumpStep(st);
      list = label(multiples(lo, hi, st), st);
    }
    return list;
  }

  // ---------------------------------------------------------------- coordinates
  function aX(x) { return this.ox + x * this.sx; }
  function aY(y) { return this.oy - y * this.sy; }
  function aWorld(px, py) { return [(px - this.ox) / this.sx, (this.oy - py) / this.sy]; }
  function margins(m) {
    const d = { l: 62, r: 16, t: 14, b: 52 };
    if (m === undefined || m === null) return d;
    if (typeof m === "number") return { l: m, r: m, t: m, b: m };
    return Object.assign(d, m);
  }

  P.limits = function (o) {
    o = o || {};
    const xl = o.xlim, yl = o.ylim;
    if (!xl || !yl || !(xl[1] > xl[0]) || !(yl[1] > yl[0]))
      throw new Error("plot.js: limits need xlim [a, b] and ylim [c, d] with a < b and c < d");
    const m = margins(o.margin);
    const W = this.w - m.l - m.r, H = this.h - m.t - m.b;
    let sx = W / (xl[1] - xl[0]), sy = H / (yl[1] - yl[0]);
    let ox = m.l - xl[0] * sx, oy = m.t + yl[1] * sy;
    if (o.equal) {
      const s = Math.min(sx, sy), cx = (xl[0] + xl[1]) / 2, cy = (yl[0] + yl[1]) / 2;
      sx = sy = s;
      ox = m.l + W / 2 - cx * s;
      oy = m.t + H / 2 + cy * s;
    }
    this.sx = sx; this.sy = sy; this.s = Math.min(sx, sy); this.ox = ox; this.oy = oy;
    this.X = aX; this.Y = aY; this.world = aWorld;
    this.margin = m;
    this.plotBox = { x0: xl[0], x1: xl[1], y0: yl[0], y1: yl[1] };
    return this;
  };

  Plot.fig = function (target, o) {
    o = o || {};
    const F = Fig(target, { w: o.w, h: o.h, fontSize: o.fontSize });
    return F.limits(o);
  };

  P.bounds = function () {
    if (this.plotBox) return Object.assign({}, this.plotBox);
    const a = this.world(0, this.h), b = this.world(this.w, 0);
    return { x0: Math.min(a[0], b[0]), x1: Math.max(a[0], b[0]), y0: Math.min(a[1], b[1]), y1: Math.max(a[1], b[1]) };
  };
  function rectFrom(F, o) {
    const B = F.bounds();
    if (o.xlim) { B.x0 = o.xlim[0]; B.x1 = o.xlim[1]; }
    if (o.ylim) { B.y0 = o.ylim[0]; B.y1 = o.ylim[1]; }
    return B;
  }
  function pxSize(F, B) {
    return [Math.abs(F.X(B.x1) - F.X(B.x0)), Math.abs(F.Y(B.y0) - F.Y(B.y1))];
  }

  // ---------------------------------------------------------------- clipping
  function clipSeg(a, b, B) {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const ex = (B.x1 - B.x0) * 1e-9, ey = (B.y1 - B.y0) * 1e-9;
    const p = [-dx, dx, -dy, dy];
    const q = [a[0] - (B.x0 - ex), B.x1 + ex - a[0], a[1] - (B.y0 - ey), B.y1 + ey - a[1]];
    let t0 = 0, t1 = 1;
    for (let k = 0; k < 4; k++) {
      if (p[k] === 0) { if (q[k] < 0) return null; continue; }
      const r = q[k] / p[k];
      if (p[k] < 0) { if (r > t1) return null; if (r > t0) t0 = r; }
      else { if (r < t0) return null; if (r < t1) t1 = r; }
    }
    return [t0, t1];
  }
  const lerp2 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  const okPt = (p) => p && isNum(p[0]) && isNum(p[1]);

  function clipPolyline(pts, B, jumps) {
    const out = [];
    let cur = null;
    if (!B) {
      pts.forEach((p) => {
        if (okPt(p)) { if (!cur) out.push(cur = []); cur.push(p); } else cur = null;
      });
      return out.filter((l) => l.length > 1);
    }
    for (let i = 1; i < pts.length; i++) {
      const a = pts[i - 1], b = pts[i];
      if (!okPt(a) || !okPt(b)) { cur = null; continue; }
      if (jumps && ((a[1] > B.y1 && b[1] < B.y0) || (a[1] < B.y0 && b[1] > B.y1))) { cur = null; continue; }
      const c = clipSeg(a, b, B);
      if (!c) { cur = null; continue; }
      const p = c[0] > 0 ? lerp2(a, b, c[0]) : a, q = c[1] < 1 ? lerp2(a, b, c[1]) : b;
      if (!cur || c[0] > 0) { cur = [p]; out.push(cur); }
      cur.push(q);
      if (c[1] < 1) cur = null;
    }
    return out;
  }
  Plot.clip = function (pts, box, jumps) { return clipPolyline(pts, box, !!jumps); };

  // Sutherland-Hodgman: clip a closed polygon to the box.
  function clipPolygon(pts, B) {
    let out = pts.filter(okPt);
    const planes = [
      [(p) => p[0] >= B.x0, (a, b) => lerp2(a, b, (B.x0 - a[0]) / (b[0] - a[0]))],
      [(p) => p[0] <= B.x1, (a, b) => lerp2(a, b, (B.x1 - a[0]) / (b[0] - a[0]))],
      [(p) => p[1] >= B.y0, (a, b) => lerp2(a, b, (B.y0 - a[1]) / (b[1] - a[1]))],
      [(p) => p[1] <= B.y1, (a, b) => lerp2(a, b, (B.y1 - a[1]) / (b[1] - a[1]))],
    ];
    for (const [inside, cut] of planes) {
      const inp = out;
      out = [];
      for (let i = 0; i < inp.length; i++) {
        const cur = inp[i], prev = inp[(i + inp.length - 1) % inp.length];
        if (inside(cur)) {
          if (!inside(prev)) out.push(cut(prev, cur));
          out.push(cur);
        } else if (inside(prev)) out.push(cut(prev, cur));
      }
      if (!out.length) break;
    }
    return out;
  }

  function dOf(F, lines, close) {
    return lines.map((l) => l.map((p, i) => (i ? "L" : "M") + f2(F.X(p[0])) + " " + f2(F.Y(p[1]))).join(" ") +
      (close ? " Z" : "")).join(" ");
  }
  function drawLines(F, pts, o, jumps, wdflt) {
    const B = o.clip === false ? null : F.bounds();
    const lines = clipPolyline(pts, B, jumps);
    const e = el("path", Object.assign({ d: dOf(F, lines), fill: "none" }, strokeOf(o, C.ink, wdflt)), F.root);
    e.lines = lines;
    return e;
  }

  // ---------------------------------------------------------------- axes
  P.axes = function (o) {
    o = o || {};
    const b = this.bounds();
    const xl = o.xlim || [b.x0, b.x1], yl = o.ylim || [b.y0, b.y1];
    if (o.clip !== false) owner(this).plotBox = { x0: xl[0], x1: xl[1], y0: yl[0], y1: yl[1] };
    const size = o.size || 18, lsize = o.lsize || 21, color = o.color || C.ink;
    const g = el("g", { class: "plot-axes" }, this.root);
    const X0 = this.X(xl[0]), X1 = this.X(xl[1]), Y0 = this.Y(yl[0]), Y1 = this.Y(yl[1]);
    const org = o.origin ? (Array.isArray(o.origin) ? o.origin : [0, 0]) : null;
    const math = !!org;
    const arrows = o.arrows === undefined ? math : !!o.arrows;
    const ow = [org ? clamp(org[0], xl[0], xl[1]) : xl[0], org ? clamp(org[1], yl[0], yl[1]) : yl[0]];
    const ay = snap(this.Y(ow[1])), ax = snap(this.X(ow[0])); // px row of the x axis, px column of the y axis
    const xt = makeTicks(o.xticks, o.xstep, xl, X1 - X0, size, o.xfmt, true);
    const yt = makeTicks(o.yticks, o.ystep, yl, Y0 - Y1, size, o.yfmt, false);
    const line = (x1, y1, x2, y2, c, w) =>
      el("line", { x1: f2(x1), y1: f2(y1), x2: f2(x2), y2: f2(y2), stroke: c, "stroke-width": w }, g);
    const ext = arrows ? 9 : 0;
    if (o.grid) {
      const gc = o.gridColor || "#dddddd";
      xt.forEach((t) => { const x = snap(this.X(t.v)); line(x, Y0, x, Y1, gc, 1); });
      yt.forEach((t) => { const y = snap(this.Y(t.v)); line(X0, y, X1, y, gc, 1); });
    }
    if (o.frame) {
      el("rect", { x: f2(snap(X0)), y: f2(snap(Y1)), width: f2(X1 - X0), height: f2(Y0 - Y1),
        fill: "none", stroke: color, "stroke-width": 1 }, g);
    }
    line(X0, ay, X1 + ext, ay, color, 1);
    line(ax, Y0, ax, Y1 - ext, color, 1);
    if (arrows) {
      el("polygon", { points: [X1 + ext, ay, X1 + ext - 9, ay - 3.6, X1 + ext - 9, ay + 3.6].map(f2).join(" "), fill: color }, g);
      el("polygon", { points: [ax, Y1 - ext, ax - 3.6, Y1 - ext + 9, ax + 3.6, Y1 - ext + 9].map(f2).join(" "), fill: color }, g);
    }
    const tout = math ? 4 : (o.ticklen === undefined ? 5 : o.ticklen), tin = math ? 4 : 0;
    const spanx = xl[1] - xl[0], spany = yl[1] - yl[0];
    xt.forEach((t) => {
      if (math && Math.abs(t.v - ow[0]) < 1e-9 * spanx) return;
      const x = snap(this.X(t.v));
      if (tout || tin) line(x, ay - tin, x, ay + tout, color, 1);
      const hw = textW(t.s, size) / 2;
      txt(g, clamp(x, hw + 2, this.w - hw - 2), ay + tout + 4 + size * 0.5, t.s, { size, color });
    });
    let maxw = 0;
    yt.forEach((t) => {
      if (math && Math.abs(t.v - ow[1]) < 1e-9 * spany) return;
      const y = snap(this.Y(t.v));
      if (tout || tin) line(ax - tout, y, ax + tin, y, color, 1);
      maxw = Math.max(maxw, textW(t.s, size));
      txt(g, ax - tout - 5, clamp(y, size * 0.5 + 1, this.h - size * 0.5 - 1), t.s, { size, color, anchor: "end" });
    });
    if (o.xlabel) {
      if (math) txt(g, X1 + ext, ay - lsize * 0.5 - 6, o.xlabel, { size: lsize, color, anchor: "end" });
      else txt(g, (X0 + X1) / 2, ay + tout + 4 + (xt.length ? size + 6 : 0) + lsize * 0.5, o.xlabel, { size: lsize, color });
    }
    if (o.ylabel) {
      if (math) txt(g, ax + 10, Y1 - ext + lsize * 0.5, o.ylabel, { size: lsize, color, anchor: "start" });
      else if (o.ylabelAt === "top") txt(g, ax, Y1 - ext - lsize * 0.5 - 4, o.ylabel, { size: lsize, color, anchor: "middle" });
      else {
        const x = Math.max(lsize * 0.6, ax - tout - 5 - maxw - 8 - lsize * 0.5);
        txt(g, x, (Y0 + Y1) / 2, o.ylabel, { size: lsize, color, rotate: -90 });
      }
    }
    return g;
  };

  function refLine(F, a, b, o) {
    o = o || {};
    const e = drawLines(F, [a, b], Object.assign({ width: 1.2, color: C.gray, dash: "6 5" }, o), false);
    if (o.label) {
      const B = F.bounds(), vertical = a[0] === b[0];
      const x = vertical ? F.X(a[0]) + 5 : F.X(B.x1) - 4, y = vertical ? F.Y(B.y1) + 12 : F.Y(a[1]) - 12;
      txt(F.root, x, y, o.label, { size: o.lsize || 18, color: o.lcolor || o.color || C.ink, anchor: vertical ? "start" : "end" });
    }
    return e;
  }
  P.hline = function (y, o) { const B = this.bounds(); return refLine(this, [B.x0, y], [B.x1, y], o); };
  P.vline = function (x, o) { const B = this.bounds(); return refLine(this, [x, B.y0], [x, B.y1], o); };

  P.legend = function (items, o) {
    o = o || {};
    const size = o.size || 18, row = size + 6;
    const wmax = Math.max.apply(null, items.map((it) => textW(it.name || "", size)).concat([0]));
    const bw = 12 + 24 + 8 + wmax + 12, bh = items.length * row + 8;
    const B = this.bounds();
    const L = this.X(B.x0), R = this.X(B.x1), T = this.Y(B.y1), Bt = this.Y(B.y0);
    const at = o.at || "nw";
    let x, y;
    if (Array.isArray(at)) { x = at[0]; y = at[1]; }
    else {
      x = at.indexOf("e") >= 0 ? R - bw - 8 : L + 8;
      y = at.indexOf("s") >= 0 ? Bt - bh - 8 : T + 8;
    }
    const g = el("g", { class: "plot-legend" }, this.root);
    el("rect", { x: f2(x), y: f2(y), width: f2(bw), height: f2(bh), fill: "#ffffff", "fill-opacity": 0.85, stroke: "none" }, g);
    items.forEach((it, k) => {
      const cy = y + 4 + row * (k + 0.5), c = it.color || C.ink;
      if (it.marker) el("circle", { cx: f2(x + 24), cy: f2(cy), r: 4, fill: c }, g);
      else el("line", Object.assign({ x1: f2(x + 12), y1: f2(cy), x2: f2(x + 36), y2: f2(cy) },
        strokeOf({ color: c, width: it.width || 2, dash: it.dash })), g);
      txt(g, x + 44, cy, it.name || "", { size, anchor: "start" });
    });
    return g;
  };

  // ---------------------------------------------------------------- curves and points
  P.curve = function (f, x0, x1, o) {
    if (x0 !== null && typeof x0 === "object") { o = x0; x0 = undefined; x1 = undefined; }
    o = o || {};
    const B = this.bounds();
    if (x0 === undefined || x0 === null) x0 = B.x0;
    if (x1 === undefined || x1 === null) x1 = B.x1;
    const n = o.n || 400, g = safe(f), pts = [];
    for (let i = 0; i <= n; i++) { const x = x0 + (x1 - x0) * i / n; pts.push([x, g(x)]); }
    return drawLines(this, pts, o, o.jumps !== true, 2);
  };

  P.param = function (fn, t0, t1, o) {
    o = o || {};
    const n = o.n || 400, pts = [];
    for (let i = 0; i <= n; i++) {
      let p;
      try { p = fn(t0 + (t1 - t0) * i / n); } catch (e) { p = null; }
      pts.push(p);
    }
    return drawLines(this, pts, o, false, 2);
  };

  P.polyline = function (pts, o) {
    o = o || {};
    const B = o.clip === false ? null : this.bounds();
    const ring = o.close && pts.length ? pts.concat([pts[0]]) : pts;
    let fillEl = null, lineEl = null;
    if (o.fill && o.fill !== "none") {
      const poly = B ? clipPolygon(pts, B) : pts.filter(okPt);
      fillEl = el("path", { d: poly.length > 2 ? dOf(this, [poly], true) : "", fill: o.fill,
        "fill-opacity": o.fillOpacity, stroke: "none" }, this.root);
    }
    if (o.stroke !== "none") lineEl = drawLines(this, ring, o, false, 1.5);
    if (fillEl && lineEl) {
      const g = el("g", {}, this.root);
      g.appendChild(fillEl);
      g.appendChild(lineEl);
      g.lines = lineEl.lines;
      return g;
    }
    return lineEl || fillEl;
  };

  P.area = function (f, x0, x1, o) {
    if (x0 !== null && typeof x0 === "object") { o = x0; x0 = undefined; x1 = undefined; }
    o = o || {};
    const B = this.bounds();
    if (x0 === undefined || x0 === null) x0 = B.x0;
    if (x1 === undefined || x1 === null) x1 = B.x1;
    const n = o.n || 200, g = safe(f);
    const base = typeof o.base === "function" ? safe(o.base) : () => (o.base || 0);
    const top = [], bot = [];
    for (let i = 0; i <= n; i++) {
      const x = x0 + (x1 - x0) * i / n;
      top.push([x, g(x)]);
      bot.push([x, base(x)]);
    }
    const poly = clipPolygon(top.concat(bot.reverse()), B);
    return el("path", { d: poly.length > 2 ? dOf(this, [poly], true) : "", fill: o.fill || C.cone,
      "fill-opacity": o.opacity === undefined ? 0.7 : o.opacity, stroke: "none" }, this.root);
  };

  P.scatter = function (pts, o) {
    o = o || {};
    const B = o.clip === false ? null : this.bounds();
    const g = el("g", { class: "plot-scatter" }, this.root);
    const r = o.r || 4, shape = o.shape || "circle";
    pts.forEach((p, i) => {
      if (!okPt(p)) return;
      if (B && (p[0] < B.x0 || p[0] > B.x1 || p[1] < B.y0 || p[1] > B.y1)) return;
      const x = this.X(p[0]), y = this.Y(p[1]);
      const fill = typeof o.fill === "function" ? o.fill(p, i) : o.fill === undefined ? C.ink : o.fill;
      const st = { stroke: o.stroke || "none", "stroke-width": o.width || 1, opacity: o.opacity };
      if (shape === "square") el("rect", Object.assign({ x: f2(x - r), y: f2(y - r), width: 2 * r, height: 2 * r, fill }, st), g);
      else if (shape === "x" || shape === "+") {
        const k = shape === "x" ? r * 0.8 : r;
        const d = shape === "x"
          ? "M" + f2(x - k) + " " + f2(y - k) + "L" + f2(x + k) + " " + f2(y + k) + "M" + f2(x - k) + " " + f2(y + k) + "L" + f2(x + k) + " " + f2(y - k)
          : "M" + f2(x - k) + " " + f2(y) + "L" + f2(x + k) + " " + f2(y) + "M" + f2(x) + " " + f2(y - k) + "L" + f2(x) + " " + f2(y + k);
        el("path", { d, fill: "none", stroke: o.stroke || (typeof fill === "string" && fill !== "none" ? fill : C.ink),
          "stroke-width": o.width || 2, "stroke-linecap": "round", opacity: o.opacity }, g);
      } else el("circle", Object.assign({ cx: f2(x), cy: f2(y), r, fill }, st), g);
    });
    return g;
  };

  // ---------------------------------------------------------------- marching squares
  // Cell corners c0 (i, j), c1 (i+1, j), c2 (i+1, j+1), c3 (i, j+1): counter-clockwise in world
  // coordinates. Cell edges e0 bottom, e1 right, e2 top, e3 left; edge k runs from corner k to
  // corner k+1. Edge ids: 2v for the horizontal edge (i, j)-(i+1, j), 2v + 1 for the vertical
  // edge (i, j)-(i, j+1), where v = j (nx + 1) + i. Each crossing is keyed by its edge id, so
  // neighbouring cells share it exactly and segments chain without tolerances.
  const SEGS = [null, [[3, 0]], [[0, 1]], [[3, 1]], [[1, 2]], null, [[0, 2]], [[3, 2]],
    [[2, 3]], [[0, 2]], null, [[1, 2]], [[1, 3]], [[0, 1]], [[3, 0]], null];
  function cellEdges(W, i, j) {
    const v = j * W + i;
    return [2 * v, 2 * (v + 1) + 1, 2 * (v + W), 2 * v + 1];
  }
  function cellCase(flag, W, i, j) {
    const v = j * W + i, f0 = flag[v], f1 = flag[v + 1], f2_ = flag[v + W + 1], f3 = flag[v + W];
    if (f0 > 1 || f1 > 1 || f2_ > 1 || f3 > 1) return -1;
    return f0 | (f1 << 1) | (f2_ << 2) | (f3 << 3);
  }
  // flag: 1 inside, 0 outside, 2 invalid (the cell is skipped). centerIn(i, j) decides saddles.
  function march(nx, ny, flag, centerIn, edgePt) {
    const W = nx + 1, A = [], Bn = [];
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const k = cellCase(flag, W, i, j);
        if (k <= 0 || k === 15) continue;
        let segs;
        if (k === 5) segs = centerIn(i, j) ? [[0, 1], [2, 3]] : [[3, 0], [1, 2]];
        else if (k === 10) segs = centerIn(i, j) ? [[3, 0], [1, 2]] : [[0, 1], [2, 3]];
        else segs = SEGS[k];
        const e = cellEdges(W, i, j);
        for (const s of segs) { A.push(e[s[0]]); Bn.push(e[s[1]]); }
      }
    }
    const adj = new Map();
    const add = (id, s) => { const l = adj.get(id); if (l) l.push(s); else adj.set(id, [s]); };
    for (let s = 0; s < A.length; s++) { add(A[s], s); add(Bn[s], s); }
    const used = new Uint8Array(A.length), lines = [];
    const walk = (chain) => {
      for (;;) {
        const id = chain[chain.length - 1];
        let nxt = -1;
        for (const t of adj.get(id)) if (!used[t]) { nxt = t; break; }
        if (nxt < 0) return;
        used[nxt] = 1;
        chain.push(A[nxt] === id ? Bn[nxt] : A[nxt]);
      }
    };
    for (let s = 0; s < A.length; s++) {
      if (used[s]) continue;
      used[s] = 1;
      const chain = [A[s], Bn[s]];
      walk(chain);
      chain.reverse();
      walk(chain);
      const closed = chain.length > 3 && chain[0] === chain[chain.length - 1];
      const pts = [];
      chain.forEach((id) => {
        const p = edgePt(id), q = pts[pts.length - 1];
        if (!q || q[0] !== p[0] || q[1] !== p[1]) pts.push(p);
      });
      if (pts.length < 2) continue;
      pts.closed = closed;
      lines.push(pts);
    }
    return lines;
  }
  function gridAxes(xl, yl, nx, ny) {
    const xs = (i) => (i >= nx ? xl[1] : xl[0] + (xl[1] - xl[0]) * i / nx);
    const ys = (j) => (j >= ny ? yl[1] : yl[0] + (yl[1] - yl[0]) * j / ny);
    return [xs, ys];
  }
  function niceLevels(lo, hi, n) {
    if (!(hi > lo)) return [];
    const st = niceStep((hi - lo) / Math.max(1, n));
    return multiples(lo, hi, st).filter((v) => v > lo && v < hi);
  }

  Plot.contours = function (f, levels, o) {
    o = o || {};
    const xl = o.xlim || [-1, 1], yl = o.ylim || [-1, 1];
    const nx = o.nx || 80, ny = o.ny || nx, W = nx + 1;
    const [xs, ys] = gridAxes(xl, yl, nx, ny);
    const g = safe(f), V = new Float64Array(W * (ny + 1));
    let lo = Infinity, hi = -Infinity;
    for (let j = 0; j <= ny; j++) {
      for (let i = 0; i <= nx; i++) {
        const v = g(xs(i), ys(j));
        const ok = isNum(v);
        V[j * W + i] = ok ? v : NaN;
        if (ok) { if (v < lo) lo = v; if (v > hi) hi = v; }
      }
    }
    if (typeof levels === "number") levels = niceLevels(lo, hi, levels);
    levels = (levels || []).slice();
    const out = [], flag = new Uint8Array(V.length);
    levels.forEach((L) => {
      for (let k = 0; k < V.length; k++) flag[k] = V[k] !== V[k] ? 2 : V[k] >= L ? 1 : 0;
      const centerIn = (i, j) => {
        const v = j * W + i;
        return (V[v] + V[v + 1] + V[v + W] + V[v + W + 1]) / 4 >= L;
      };
      const cache = new Map();
      const edgePt = (id) => {
        let p = cache.get(id);
        if (p) return p;
        const v = id >> 1, i = v % W, j = (v - i) / W, vert = id & 1;
        const a = V[v], b = V[vert ? v + W : v + 1], t = (L - a) / (b - a);
        p = vert ? [xs(i), ys(j) + t * (ys(j + 1) - ys(j))] : [xs(i) + t * (xs(i + 1) - xs(i)), ys(j)];
        cache.set(id, p);
        return p;
      };
      march(nx, ny, flag, centerIn, edgePt).forEach((l) => { l.level = L; out.push(l); });
    });
    out.levels = levels;
    out.range = [lo, hi];
    return out;
  };

  function levelColors(o, levels) {
    const n = levels.length;
    if (typeof o.colors === "function") return levels.map((L, k) => o.colors(L, k));
    if (Array.isArray(o.colors)) return levels.map((L, k) => o.colors[k % o.colors.length]);
    if (o.cmap) {
      const cm = Plot.colormap(o.cmap);
      return levels.map((L, k) => cm(n > 1 ? 0.25 + 0.75 * k / (n - 1) : 1));
    }
    return levels.map(() => o.color || C.ink);
  }

  function placeLabels(F, g, lines, levels, colors, o) {
    const size = o.lsize || 17;
    const fmt = typeof o.labels === "function" ? o.labels : (v) => Plot.fmt(v);
    const B = F.bounds();
    const bx0 = Math.max(F.X(B.x0), 0), bx1 = Math.min(F.X(B.x1), F.w);
    const by0 = Math.max(F.Y(B.y1), 0), by1 = Math.min(F.Y(B.y0), F.h);
    const placed = [];
    const fr = [0.5, 0.3, 0.7, 0.15, 0.85, 0.4, 0.6, 0.05, 0.95, 0.22, 0.78];
    levels.forEach((L, k) => {
      const s = fmt(L), w = textW(s, size) + 8, h = size + 2;
      const cands = lines.filter((l) => l.level === L).map((l) => {
        const px = l.map((p) => [F.X(p[0]), F.Y(p[1])]), cum = [0];
        for (let i = 1; i < px.length; i++) cum.push(cum[i - 1] + Math.hypot(px[i][0] - px[i - 1][0], px[i][1] - px[i - 1][1]));
        return { px, cum, len: cum[cum.length - 1] };
      }).sort((a, b) => b.len - a.len).slice(0, 3);
      const at = (c, t) => {
        const target = t * c.len;
        let i = 1;
        while (i < c.cum.length - 1 && c.cum[i] < target) i++;
        const seg = c.cum[i] - c.cum[i - 1] || 1, u = (target - c.cum[i - 1]) / seg;
        return lerp2(c.px[i - 1], c.px[i], clamp(u, 0, 1));
      };
      const fits = (x, y) => x - w / 2 >= bx0 + 2 && x + w / 2 <= bx1 - 2 && y - h / 2 >= by0 + 2 && y + h / 2 <= by1 - 2 &&
        !placed.some((q) => Math.abs(q[0] - x) < (q[2] + w) / 2 + 2 && Math.abs(q[1] - y) < (q[3] + h) / 2 + 1);
      for (const c of cands) {
        if (c.len < w * 1.5) continue;
        let done = false;
        for (const t of fr) {
          const p = at(c, t);
          if (fits(p[0], p[1])) {
            placed.push([p[0], p[1], w, h]);
            txt(g, p[0], p[1], s, { size, color: colors[k], halo: true });
            done = true;
            break;
          }
        }
        if (done) break;
      }
    });
  }

  P.contour = function (f, levels, o) {
    o = o || {};
    const B = rectFrom(this, o), [pw, ph] = pxSize(this, B);
    const nx = o.nx || clamp(Math.round(pw / 5), 10, 200), ny = o.ny || clamp(Math.round(ph / 5), 10, 200);
    const lines = Plot.contours(f, levels, { xlim: [B.x0, B.x1], ylim: [B.y0, B.y1], nx, ny });
    const lv = lines.levels, colors = levelColors(o, lv);
    const g = el("g", { class: "plot-contour" }, this.root);
    lv.forEach((L, k) => {
      const ls = lines.filter((l) => l.level === L);
      if (!ls.length) return;
      el("path", Object.assign({ d: ls.map((l) => dOf(this, [l], l.closed)).join(" "), fill: "none" },
        strokeOf({ color: colors[k], width: o.width || 1.3, dash: o.dash, opacity: o.opacity })), g);
    });
    if (o.labels) placeLabels(this, g, lines, lv, colors, o);
    lines.el = g;
    return lines;
  };

  // ---------------------------------------------------------------- colors and colormaps
  function rgbOf(c) {
    if (Array.isArray(c)) return c.slice(0, 3);
    const s = String(c).trim();
    let m = /^#([0-9a-f]{3})$/i.exec(s);
    if (m) return m[1].split("").map((h) => parseInt(h + h, 16));
    m = /^#([0-9a-f]{6})$/i.exec(s);
    if (m) return [0, 2, 4].map((k) => parseInt(m[1].slice(k, k + 2), 16));
    m = /^rgba?\(([^)]+)\)$/i.exec(s);
    if (m) return m[1].split(",").slice(0, 3).map((x) => parseFloat(x));
    return null;
  }
  function hex(rgb) {
    return "#" + rgb.map((v) => {
      const s = Math.round(clamp(v, 0, 255)).toString(16);
      return s.length < 2 ? "0" + s : s;
    }).join("");
  }
  Plot.rgb = rgbOf;
  Plot.hex = hex;

  const MAPS = {
    heat: [[0, "#ffffff"], [0.12, "#ffff99"], [0.4, "#ff9933"], [0.7, "#cc3300"], [1, "#660000"]],
    diverging: [[0, "#003380"], [0.25, "#6f94d0"], [0.5, "#ffffff"], [0.75, "#e0806a"], [1, "#800000"]],
    gray: [[0, "#ffffff"], [1, "#333333"]],
  };
  MAPS.div = MAPS.diverging;
  Plot.colormaps = ["heat", "diverging", "gray"];

  Plot.colormap = function (name) {
    if (typeof name === "function") {
      if (name.rgb) return name;
      const f = (t) => name(clamp(isNum(t) ? t : 0, 0, 1));
      f.rgb = (t) => rgbOf(f(t)) || [0, 0, 0];
      Object.defineProperty(f, "name", { value: "custom" });
      return f;
    }
    let stops;
    if (Array.isArray(name)) {
      stops = Array.isArray(name[0]) ? name : name.map((c, k) => [name.length > 1 ? k / (name.length - 1) : 0, c]);
    } else {
      stops = MAPS[name || "heat"];
      if (!stops) throw new Error("plot.js: unknown colormap '" + name + "' (have " + Plot.colormaps.join(", ") + ")");
    }
    const S = stops.map((s) => [s[0], rgbOf(s[1]) || [0, 0, 0]]);
    const rgb = (t) => {
      t = clamp(isNum(t) ? t : 0, 0, 1);
      if (t <= S[0][0]) return S[0][1].slice();
      for (let k = 1; k < S.length; k++) {
        if (t <= S[k][0]) {
          const u = (t - S[k - 1][0]) / (S[k][0] - S[k - 1][0] || 1);
          return [0, 1, 2].map((c) => S[k - 1][1][c] + (S[k][1][c] - S[k - 1][1][c]) * u);
        }
      }
      return S[S.length - 1][1].slice();
    };
    const f = (t) => hex(rgb(t));
    f.rgb = rgb;
    Object.defineProperty(f, "name", { value: typeof name === "string" ? name : "custom" });
    f.stops = stops;
    return f;
  };

  // A canvas of nx x ny pixels -> data URL. pix(c, r) returns [r, g, b] or null (transparent).
  function canvasURL(nx, ny, pix) {
    const cv = document.createElement("canvas");
    cv.width = nx;
    cv.height = ny;
    const ctx = cv.getContext("2d"), im = ctx.createImageData(nx, ny), d = im.data;
    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) {
        const k = 4 * (r * nx + c), v = pix(c, r);
        if (v) { d[k] = v[0]; d[k + 1] = v[1]; d[k + 2] = v[2]; d[k + 3] = 255; }
      }
    }
    ctx.putImageData(im, 0, 0);
    return cv.toDataURL("image/png");
  }
  function image(parent, x, y, w, h, url, o) {
    const im = el("image", {
      x: f2(x), y: f2(y), width: f2(w), height: f2(h),
      preserveAspectRatio: "none", opacity: o.opacity, href: url,
    }, parent);
    im.style.imageRendering = o.smooth === false ? "pixelated" : "auto";
    return im;
  }

  P.heat = function (f, o) {
    o = o || {};
    const B = rectFrom(this, o), [pw, ph] = pxSize(this, B);
    const nx = o.nx || clamp(Math.ceil(pw / 3), 4, 400), ny = o.ny || clamp(Math.ceil(ph / 3), 4, 400);
    const g = safe(f), V = new Float64Array(nx * ny);
    let lo = Infinity, hi = -Infinity;
    for (let r = 0; r < ny; r++) {
      const y = B.y1 - (r + 0.5) * (B.y1 - B.y0) / ny;
      for (let c = 0; c < nx; c++) {
        const v = g(B.x0 + (c + 0.5) * (B.x1 - B.x0) / nx, y);
        V[r * nx + c] = isNum(v) ? v : NaN;
        if (isNum(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
      }
    }
    const vmin = o.vmin !== undefined ? o.vmin : isFinite(lo) ? lo : 0;
    let vmax = o.vmax !== undefined ? o.vmax : isFinite(hi) ? hi : 1;
    if (!(vmax > vmin)) vmax = vmin + 1;
    const cm = Plot.colormap(o.cmap || "heat");
    const url = canvasURL(nx, ny, (c, r) => {
      const v = V[r * nx + c];
      return v === v ? cm.rgb((v - vmin) / (vmax - vmin)) : null;
    });
    const x = Math.min(this.X(B.x0), this.X(B.x1)), y = Math.min(this.Y(B.y0), this.Y(B.y1));
    const im = image(this.root, x, y, pw, ph, url, o);
    im.vmin = vmin;
    im.vmax = vmax;
    im.cmap = cm;
    return im;
  };

  P.colorbar = function (o) {
    o = o || {};
    const src = o.of || {};
    const cm = Plot.colormap(o.cmap || src.cmap || "heat");
    const vmin = o.vmin !== undefined ? o.vmin : src.vmin !== undefined ? src.vmin : 0;
    const vmax = o.vmax !== undefined ? o.vmax : src.vmax !== undefined ? src.vmax : 1;
    const B = this.bounds(), size = o.size || 17;
    const w = o.w || 14;
    const at = o.at || [this.X(B.x1) + 14, this.Y(B.y1)];
    const len = o.len || Math.abs(this.Y(B.y0) - this.Y(B.y1));
    const g = el("g", { class: "plot-colorbar" }, this.root);
    const n = 128;
    image(g, at[0], at[1], w, len, canvasURL(1, n, (c, r) => cm.rgb(1 - (r + 0.5) / n)), {});
    el("rect", { x: f2(snap(at[0])), y: f2(snap(at[1])), width: w, height: f2(len), fill: "none", stroke: C.ink, "stroke-width": 1 }, g);
    const ticks = makeTicks(o.ticks, o.step, [vmin, vmax], len, size, o.fmt, false);
    ticks.forEach((t) => {
      const y = snap(at[1] + len * (1 - (t.v - vmin) / (vmax - vmin)));
      el("line", { x1: f2(at[0] + w), y1: f2(y), x2: f2(at[0] + w + 4), y2: f2(y), stroke: C.ink, "stroke-width": 1 }, g);
      txt(g, at[0] + w + 7, clamp(y, size * 0.5 + 1, this.h - size * 0.5 - 1), t.s, { size, anchor: "start" });
    });
    if (o.label) txt(g, at[0] + w / 2, at[1] - size * 0.5 - 5, o.label, { size, anchor: "middle" });
    return g;
  };

  P.region = function (pred, o) {
    o = o || {};
    const B = rectFrom(this, o), [pw, ph] = pxSize(this, B);
    const nx = o.nx || clamp(Math.round(pw / 4), 8, 300), ny = o.ny || clamp(Math.round(ph / 4), 8, 300), W = nx + 1;
    const [xs, ys] = gridAxes([B.x0, B.x1], [B.y0, B.y1], nx, ny);
    const test = (x, y) => { try { return !!pred(x, y); } catch (e) { return false; } };
    const flag = new Uint8Array(W * (ny + 1));
    for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) flag[j * W + i] = test(xs(i), ys(j)) ? 1 : 0;
    const cc = new Map();
    const centerIn = (i, j) => {
      const key = j * W + i;
      if (!cc.has(key)) cc.set(key, test((xs(i) + xs(i + 1)) / 2, (ys(j) + ys(j + 1)) / 2));
      return cc.get(key);
    };
    const cache = new Map();
    const edgePt = (id) => {
      let p = cache.get(id);
      if (p) return p;
      const v = id >> 1, i = v % W, j = (v - i) / W, vert = id & 1;
      let a = [xs(i), ys(j)], b = vert ? [xs(i), ys(j + 1)] : [xs(i + 1), ys(j)];
      if (!flag[v]) { const t = a; a = b; b = t; }
      for (let k = 0; k < 16; k++) {
        const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        if (test(m[0], m[1])) a = m; else b = m;
      }
      p = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      cache.set(id, p);
      return p;
    };
    const corner = (i, j, c) => [xs(c === 1 || c === 2 ? i + 1 : i), ys(c >= 2 ? j + 1 : j)];
    // Fill: every inside piece as a counter-clockwise polygon in one path, so shared cell edges
    // cancel under the nonzero rule and no seams show. Runs of full cells merge into one rect.
    const parts = [];
    for (let j = 0; j < ny; j++) {
      let run = -1;
      for (let i = 0; i <= nx; i++) {
        const k = i < nx ? cellCase(flag, W, i, j) : -1;
        if (k === 15) { if (run < 0) run = i; continue; }
        if (run >= 0) {
          parts.push([[xs(run), ys(j)], [xs(i), ys(j)], [xs(i), ys(j + 1)], [xs(run), ys(j + 1)]]);
          run = -1;
        }
        if (k <= 0) continue;
        const e = cellEdges(W, i, j);
        if ((k === 5 || k === 10) && !centerIn(i, j)) {
          if (k === 5) {
            parts.push([corner(i, j, 0), edgePt(e[0]), edgePt(e[3])]);
            parts.push([edgePt(e[1]), corner(i, j, 2), edgePt(e[2])]);
          } else {
            parts.push([edgePt(e[0]), corner(i, j, 1), edgePt(e[1])]);
            parts.push([edgePt(e[2]), corner(i, j, 3), edgePt(e[3])]);
          }
          continue;
        }
        const poly = [];
        for (let c = 0; c < 4; c++) {
          const a = (k >> c) & 1, b = (k >> ((c + 1) % 4)) & 1;
          if (a) poly.push(corner(i, j, c));
          if (a !== b) poly.push(edgePt(e[c]));
        }
        parts.push(poly);
      }
    }
    const g = el("g", { class: "plot-region" }, this.root);
    el("path", { d: parts.map((p) => dOf(this, [p], true)).join(" "), fill: o.fill || C.cone,
      "fill-opacity": o.opacity === undefined ? 0.7 : o.opacity, "fill-rule": "nonzero", stroke: "none" }, g);
    const lines = march(nx, ny, flag, centerIn, edgePt);
    if (o.stroke !== "none") {
      el("path", Object.assign({ d: lines.map((l) => dOf(this, [l], l.closed)).join(" "), fill: "none" },
        strokeOf({ color: o.stroke || C.ink, width: o.width || 1.5, dash: o.dash })), g);
    }
    g.lines = lines;
    return g;
  };

  P.quiver = function (field, o) {
    o = o || {};
    const B = rectFrom(this, o), [pw, ph] = pxSize(this, B);
    const nx = Array.isArray(o.n) ? o.n[0] : o.n || 12, ny = Array.isArray(o.n) ? o.n[1] : nx;
    const sx = SX(this), sy = SY(this), items = [];
    let maxPx = 0, maxMag = 0;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const x = B.x0 + (i + 0.5) * (B.x1 - B.x0) / nx, y = B.y0 + (j + 0.5) * (B.y1 - B.y0) / ny;
        let v;
        try { v = field(x, y); } catch (e) { v = null; }
        if (!okPt(v)) continue;
        const L = Math.hypot(v[0] * sx, v[1] * sy);
        maxPx = Math.max(maxPx, L);
        maxMag = Math.max(maxMag, Math.hypot(v[0], v[1]));
        items.push([x, y, v[0], v[1]]);
      }
    }
    const cell = Math.min(pw / nx, ph / ny);
    const scale = o.scale !== undefined ? o.scale : maxPx > 0 ? 0.85 * cell / maxPx : 1;
    const cm = o.cmap ? Plot.colormap(o.cmap) : null;
    const G = this.group({ class: "plot-quiver" }), w = o.width || 1.4;
    items.forEach(([x, y, u, v]) => {
      const a = o.pivot === "tail" ? [x, y] : [x - 0.5 * scale * u, y - 0.5 * scale * v];
      const b = [a[0] + scale * u, a[1] + scale * v];
      const Lpx = Math.hypot((b[0] - a[0]) * sx, (b[1] - a[1]) * sy);
      if (Lpx < 0.5) return;
      const color = typeof o.color === "function" ? o.color(x, y, u, v)
        : cm ? cm(0.3 + 0.7 * Math.hypot(u, v) / (maxMag || 1)) : o.color || C.ink;
      G.arrow(a, b, { color, width: w, headpx: Math.min(o.head || 7, Lpx * 0.5) });
    });
    G.root.scale = scale;
    return G.root;
  };

  P.grid2 = function (A, o) {
    o = o || {};
    const a = A[0][0], b = A[0][1], c = A[1][0], d = A[1][1];
    const M = (p) => [a * p[0] + b * p[1], c * p[0] + d * p[1]];
    const B = this.bounds();
    let n = o.n;
    if (!n) {
      const det = a * d - b * c;
      n = 10;
      if (Math.abs(det) > 1e-9) {
        const inv = (p) => [(d * p[0] - b * p[1]) / det, (-c * p[0] + a * p[1]) / det];
        const cs = [[B.x0, B.y0], [B.x1, B.y0], [B.x1, B.y1], [B.x0, B.y1]].map(inv);
        n = Math.ceil(Math.max.apply(null, cs.map((q) => Math.max(Math.abs(q[0]), Math.abs(q[1]))))) + 1;
      }
      n = clamp(n, 1, 60);
    }
    const G = this.group({ class: "plot-grid2" });
    if (o.original) {
      const ls = [];
      for (let i = -n; i <= n; i++) { ls.push([i, -n], [i, n], null, [-n, i], [n, i], null); }
      drawLines(G, ls, { color: o.originalColor || "#cccccc", width: 1 }, false);
    }
    if (o.square) {
      G.polyline([[0, 0], M([1, 0]), M([1, 1]), M([0, 1])], { close: true, fill: o.square === true ? C.cone : o.square,
        fillOpacity: 0.8, stroke: "none" });
    }
    const ls = [], axes = [];
    for (let i = -n; i <= n; i++) {
      if (i === 0) continue;
      ls.push(M([i, -n]), M([i, n]), null, M([-n, i]), M([n, i]), null);
    }
    axes.push(M([0, -n]), M([0, n]), null, M([-n, 0]), M([n, 0]));
    drawLines(G, ls, { color: o.color || "#6699cc", width: o.width || 1 }, false);
    drawLines(G, axes, { color: o.axisColor || C.blue, width: (o.width || 1) + 0.8 }, false);
    const labels = o.labels === false ? [null, null] : o.labels || ["Ae_1", "Ae_2"];
    G.arrow([0, 0], M([1, 0]), { color: o.e1Color || C.red, width: 3, label: labels[0] || undefined, lsize: o.lsize || 20 });
    G.arrow([0, 0], M([0, 1]), { color: o.e2Color || C.green, width: 3, label: labels[1] || undefined, lsize: o.lsize || 20 });
    return G.root;
  };

  // ---------------------------------------------------------------- 3D
  const sub3 = (p, q) => [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
  const cross3 = (p, q) => [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
  const dot3 = (p, q) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
  const unit3 = (p) => { const l = Math.hypot(p[0], p[1], p[2]) || 1; return [p[0] / l, p[1] / l, p[2] / l]; };

  P.surface3 = function (view, f, o) {
    o = o || {};
    const xl = o.xlim || [-1, 1], yl = o.ylim || [-1, 1];
    const nx = Array.isArray(o.n) ? o.n[0] : o.n || 24, ny = Array.isArray(o.n) ? o.n[1] : nx, W = nx + 1;
    const zs = o.zscale === undefined ? 1 : o.zscale, g = safe(f);
    const pts = [], Z = [];
    let lo = Infinity, hi = -Infinity;
    for (let j = 0; j <= ny; j++) {
      for (let i = 0; i <= nx; i++) {
        const x = xl[0] + (xl[1] - xl[0]) * i / nx, y = yl[0] + (yl[1] - yl[0]) * j / ny, z = g(x, y);
        Z.push(z);
        pts.push([x, y, zs * z]);
        if (isNum(z)) { if (z < lo) lo = z; if (z > hi) hi = z; }
      }
    }
    const cm = typeof o.fill === "string" && MAPS[o.fill] ? Plot.colormap(o.fill) : null;
    const zr = o.zlim || [lo, hi];
    const shade = o.shade === undefined ? 0.35 : o.shade;
    const light = unit3(o.light || (view.toward ? [view.toward[0], view.toward[1], view.toward[2] + 0.6] : [0, 0, 1]));
    const quads = [];
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const v = j * W + i, idx = [v, v + 1, v + W + 1, v + W];
        if (!idx.every((k) => isNum(Z[k]))) continue;
        const q = idx.map((k) => pts[k]);
        const zc = (Z[idx[0]] + Z[idx[1]] + Z[idx[2]] + Z[idx[3]]) / 4;
        const cx = (q[0][0] + q[2][0]) / 2, cy = (q[0][1] + q[2][1]) / 2;
        const ctr = [cx, cy, (q[0][2] + q[1][2] + q[2][2] + q[3][2]) / 4];
        let color = typeof o.fill === "function" ? o.fill(cx, cy, zc)
          : cm ? cm((zc - zr[0]) / (zr[1] - zr[0] || 1)) : o.fill || C.cone;
        if (shade > 0) {
          const nrm = unit3(cross3(sub3(q[2], q[0]), sub3(q[3], q[1])));
          const k = 1 - shade * (1 - Math.abs(dot3(nrm, light)));
          const rgb = rgbOf(color);
          if (rgb) color = hex(rgb.map((x) => x * k));
        }
        quads.push({ q, color, depth: view.depth ? view.depth(ctr) : 0 });
      }
    }
    quads.sort((p, q) => p.depth - q.depth);
    const G = this.group({ class: "plot-surface" }), T = G.in3(view);
    const stroke = o.stroke === undefined ? C.ink : o.stroke;
    quads.forEach((Q) => T.poly(Q.q, { fill: Q.color, color: stroke, width: o.width || 0.5, fillOpacity: o.opacity }));
    G.root.zrange = [lo, hi];
    return G.root;
  };

  P.axes3 = function (view, o) {
    o = o || {};
    const L = Array.isArray(o.len) ? o.len : [o.len || 1, o.len || 1, o.len || 1];
    const labels = o.labels === false ? [null, null, null] : o.labels || ["x", "y", "z"];
    const O = o.origin || [0, 0, 0];
    const G = this.group({ class: "plot-axes3" }), T = G.in3(view);
    const color = o.color || C.ink, width = o.width || 1.5;
    for (let k = 0; k < 3; k++) {
      const tip = O.slice(), neg = O.slice();
      tip[k] += L[k];
      neg[k] -= L[k] * (o.neg === true ? 0.6 : o.neg || 0);
      if (o.neg) T.line(O, neg, { color, width: 1, dash: "5 4" });
      T.arrow(O, tip, { color, width, label: labels[k] || undefined, lsize: o.lsize || 20 });
    }
    return G.root;
  };

  // ---------------------------------------------------------------- statistics
  Plot.histogram = function (samples, o) {
    o = o || {};
    const xs = Array.from(samples).filter(isNum);
    let edges;
    if (Array.isArray(o.bins)) edges = o.bins.slice();
    else {
      let lo = o.range ? o.range[0] : Math.min.apply(null, xs.length ? xs : [0]);
      let hi = o.range ? o.range[1] : Math.max.apply(null, xs.length ? xs : [1]);
      if (!(hi > lo)) { lo -= 0.5; hi += 0.5; }
      const k = o.bins || clamp(Math.ceil(Math.sqrt(xs.length)), 1, 60);
      edges = Plot.linspace(lo, hi, k + 1);
    }
    const k = edges.length - 1, counts = new Array(k).fill(0);
    const e0 = edges[0], ek = edges[k];
    let n = 0;
    xs.forEach((x) => {
      if (x < e0 || x > ek) return;
      let a = 0, b = k;
      while (b - a > 1) { const m = (a + b) >> 1; if (x >= edges[m]) a = m; else b = m; }
      counts[a]++;
      n++;
    });
    const heights = counts.map((c, i) => (o.density ? (n ? c / (n * (edges[i + 1] - edges[i])) : 0) : c));
    return { edges, counts, heights, n };
  };

  function clipRect(B, x0, x1, y0, y1) {
    const a = Math.max(Math.min(x0, x1), B.x0), b = Math.min(Math.max(x0, x1), B.x1);
    const c = Math.max(Math.min(y0, y1), B.y0), d = Math.min(Math.max(y0, y1), B.y1);
    return a < b && c < d ? [[a, c], [b, c], [b, d], [a, d]] : null;
  }

  P.hist = function (samples, o) {
    o = o || {};
    const H = Plot.histogram(samples, o), B = this.bounds();
    const g = el("g", { class: "plot-hist" }, this.root);
    H.heights.forEach((h, i) => {
      if (!(h > 0)) return;
      const r = clipRect(B, H.edges[i], H.edges[i + 1], 0, h);
      if (!r) return;
      el("path", { d: dOf(this, [r], true), fill: o.fill || "#cccccc", "fill-opacity": o.opacity,
        stroke: o.stroke || C.ink, "stroke-width": o.width || 1, "stroke-linejoin": "miter" }, g);
    });
    g.edges = H.edges;
    g.counts = H.counts;
    g.heights = H.heights;
    g.n = H.n;
    return g;
  };

  P.bars = function (values, labels, o) {
    if (labels && !Array.isArray(labels)) { o = labels; labels = null; }
    o = o || {};
    const at = o.at || values.map((v, i) => i), w = o.width || 0.7, base = o.base || 0;
    const B = this.bounds(), size = o.size || 18;
    const g = el("g", { class: "plot-bars" }, this.root);
    const fmt = o.fmt || ((v) => Plot.fmt(v));
    values.forEach((v, i) => {
      const fill = typeof o.fill === "function" ? o.fill(v, i) : Array.isArray(o.fill) ? o.fill[i % o.fill.length] : o.fill || "#cccccc";
      const r = isNum(v) ? clipRect(B, at[i] - w / 2, at[i] + w / 2, base, v) : null;
      if (r) {
        el("path", { d: dOf(this, [r], true), fill, stroke: o.stroke || C.ink, "stroke-width": 1, "stroke-linejoin": "miter" }, g);
      }
      if (o.values && isNum(v)) {
        const y = clamp(v, B.y0, B.y1), up = v >= base;
        txt(g, this.X(at[i]), this.Y(y) + (up ? -(size * 0.5 + 4) : size * 0.5 + 4), fmt(v), { size });
      }
      if (labels && labels[i] !== undefined && labels[i] !== null) {
        txt(g, this.X(at[i]), this.Y(B.y0) + 9 + size * 0.5, labels[i], { size });
      }
    });
    return g;
  };

  // ---------------------------------------------------------------- strip chart
  const SERIES = [C.red, C.blue, C.green, C.purple, C.orange, C.teal];
  Plot.strip = function (target, o) {
    o = o || {};
    const win = o.window || 10;
    const series = (o.series || [{ name: "" }]).map((s, k) => Object.assign({ color: SERIES[k % SERIES.length], width: 2 }, s));
    const F = Plot.fig(target, { w: o.w || 520, h: o.h || 240, xlim: [0, win], ylim: o.ylim || [-1, 1], margin: o.margin });
    let T = [], V = series.map(() => []), t0 = null, lo = Infinity, hi = -Infinity, raf = 0;
    const yrange = () => {
      if (o.ylim) return o.ylim;
      if (!(hi >= lo)) return [-1, 1];
      if (hi - lo < 1e-12) return [lo - 1, hi + 1];
      const st = niceStep((hi - lo) / 4);
      let a = Math.floor(lo / st - 1e-9) * st, b = Math.ceil(hi / st + 1e-9) * st;
      if (b - a < 1e-12) b = a + st;
      return [a, b];
    };
    function draw() {
      if (raf && typeof cancelAnimationFrame === "function") cancelAnimationFrame(raf);
      raf = 0;
      const tn = T.length ? T[T.length - 1] : 0;
      const a = t0 === null ? 0 : Math.max(t0, tn - win);
      F.limits({ xlim: [a, a + win], ylim: yrange(), margin: o.margin });
      F.clear();
      F.axes({ xlabel: o.xlabel === undefined ? "time $t$ (s)" : o.xlabel, ylabel: o.ylabel, grid: o.grid !== false, size: o.size });
      series.forEach((s, k) => F.polyline(T.map((t, i) => [t, V[k][i]]), { color: s.color, width: s.width, dash: s.dash }));
      if (o.legend !== false && series.some((s) => s.name)) F.legend(series, { at: o.legend || "nw", size: o.size });
    }
    function reset() {
      T = [];
      V = series.map(() => []);
      t0 = null;
      lo = Infinity;
      hi = -Infinity;
    }
    function push(t, vals) {
      if (!isNum(t)) return;
      if (!Array.isArray(vals)) vals = [vals];
      if (T.length && t < T[T.length - 1]) reset();
      if (t0 === null) t0 = t;
      T.push(t);
      series.forEach((s, k) => {
        const v = vals[k] === null || vals[k] === undefined ? NaN : +vals[k];
        V[k].push(v);
        if (isNum(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
      });
      let cut = 0;
      while (cut < T.length - 1 && T[cut + 1] < t - win) cut++;
      if (cut) { T.splice(0, cut); V.forEach((v) => v.splice(0, cut)); }
      if (typeof requestAnimationFrame !== "function") draw();
      else if (!raf) raf = requestAnimationFrame(draw);
    }
    function clear() { reset(); draw(); }
    draw();
    return { push, clear, draw, fig: F, get t() { return T.slice(); }, get values() { return V.map((v) => v.slice()); } };
  };

  window.Plot = Plot;
})();
