/* fig.js — a tiny helper for drawing retro physics figures as inline SVG.
 *
 *   const F = Fig("#my-svg", { w: 520, h: 320, xlim: [-3, 3], ylim: [-0.5, 3] });
 *   F.ground(-3, 3, 0);
 *   F.rect([0, 0.5], 1.4, 1, { fill: "#ddd" });
 *   F.arrow([0, 0], [0, 1.2], { color: Fig.C.normal, label: "N" });
 *
 * World coordinates have x to the right and y UP. Sizes given in world units
 * unless the option name ends in "px". Labels passed to label() / {label}
 * understand f_n, f_{t1}, x^2 for sub/superscripts and are set in italic.
 */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";

  // Web-safe colors, on theme.
  const C = {
    ink: "#000000",
    force: "#cc0000",
    normal: "#0000cc",
    friction: "#008800",
    cone: "#ffff99",
    gray: "#999999",
    light: "#e6e6e6",
    faint: "#f2f2f2",
    purple: "#990099",
    orange: "#cc6600",
    teal: "#008080",
    red: "#cc0000",
    blue: "#0000cc",
    green: "#008800",
  };

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  // Parse "f_{t1}^2" style labels into tspans (sub/sup via dy, which every browser supports).
  function fillLabel(textEl, s, size) {
    let i = 0, buf = "";
    const flush = () => {
      if (buf) { el("tspan", {}, textEl).textContent = buf; buf = ""; }
    };
    const group = () => {
      if (s[i] === "{") {
        const j = s.indexOf("}", i);
        const g = s.slice(i + 1, j);
        i = j + 1;
        return g;
      }
      return s[i++];
    };
    while (i < s.length) {
      const c = s[i];
      if (c === "_" || c === "^") {
        flush();
        i++;
        const g = group();
        const shift = c === "_" ? 0.3 : -0.42;
        const t = el("tspan", { dy: shift + "em", "font-size": Math.round(size * 0.7) }, textEl);
        t.textContent = g;
        // Return to the baseline: the dy is in the *next* tspan's em units.
        const back = el("tspan", { dy: (-shift * 0.7) + "em", "font-size": size }, textEl);
        back.textContent = "​";
      } else {
        buf += c;
        i++;
      }
    }
    flush();
  }

  function Fig(target, opt) {
    if (!(this instanceof Fig)) return new Fig(target, opt);
    opt = opt || {};
    let svg = typeof target === "string" ? document.querySelector(target) : target;
    if (!svg) throw new Error("fig.js: no element " + target);
    if (svg.tagName.toLowerCase() !== "svg") svg = el("svg", {}, svg);
    this.svg = svg;
    this.w = opt.w || +svg.getAttribute("width") || 480;
    this.h = opt.h || +svg.getAttribute("height") || 320;
    svg.setAttribute("width", this.w);
    svg.setAttribute("height", this.h);
    svg.setAttribute("viewBox", "0 0 " + this.w + " " + this.h);
    svg.setAttribute("xmlns", NS);
    if (opt.xlim && opt.ylim) {
      const pad = opt.pad === undefined ? 0 : opt.pad;
      const sx = (this.w - 2 * pad) / (opt.xlim[1] - opt.xlim[0]);
      const sy = (this.h - 2 * pad) / (opt.ylim[1] - opt.ylim[0]);
      this.s = Math.min(sx, sy);
      const cx = (opt.xlim[0] + opt.xlim[1]) / 2, cy = (opt.ylim[0] + opt.ylim[1]) / 2;
      this.ox = this.w / 2 - cx * this.s;
      this.oy = this.h / 2 + cy * this.s;
    } else {
      this.s = opt.scale || 50;
      this.ox = opt.origin ? opt.origin[0] : this.w / 2;
      this.oy = opt.origin ? opt.origin[1] : this.h / 2;
    }
    this.fontSize = opt.fontSize || 22;
    this.root = el("g", {}, svg);
  }

  Fig.C = C;
  Fig.el = el;
  Fig.deg = (d) => (d * Math.PI) / 180;
  Fig.dir = (a) => [Math.cos(a), Math.sin(a)];

  const P = Fig.prototype;

  // ---- coordinates ---------------------------------------------------------
  P.X = function (x) { return this.ox + x * this.s; };
  P.Y = function (y) { return this.oy - y * this.s; };
  P.px = function (p) { return [this.X(p[0]), this.Y(p[1])]; };
  P.world = function (px, py) { return [(px - this.ox) / this.s, (this.oy - py) / this.s]; };
  P.clear = function () { this.root.innerHTML = ""; return this; };
  P.group = function (attrs) {
    const g = Object.create(this);
    g.root = el("g", attrs || {}, this.root);
    return g;
  };

  function stroke(o, dflt) {
    return {
      stroke: o.color || o.stroke || dflt || C.ink,
      "stroke-width": o.width || 1.5,
      "stroke-dasharray": o.dash === true ? "6 5" : o.dash || null,
      "stroke-linecap": o.cap || "round",
      "stroke-linejoin": "round",
      opacity: o.opacity,
    };
  }

  // ---- primitives ----------------------------------------------------------
  P.line = function (a, b, o) {
    o = o || {};
    const A = this.px(a), B = this.px(b);
    return el("line", Object.assign({ x1: A[0], y1: A[1], x2: B[0], y2: B[1] }, stroke(o)), this.root);
  };

  P.path = function (pts, o) {
    o = o || {};
    const d = pts.map((p, i) => (i ? "L" : "M") + this.X(p[0]).toFixed(2) + " " + this.Y(p[1]).toFixed(2)).join(" ");
    return el("path", Object.assign({
      d: d + (o.close ? " Z" : ""),
      fill: o.fill || "none",
      "fill-opacity": o.fillOpacity,
    }, stroke(o)), this.root);
  };

  P.poly = function (pts, o) {
    o = Object.assign({ fill: C.light, close: true }, o || {});
    if (o.stroke === undefined && o.color === undefined) o.color = C.ink;
    if (o.stroke === "none") o.color = "none";
    return this.path(pts, o);
  };

  P.circle = function (c, r, o) {
    o = o || {};
    const A = this.px(c);
    return el("circle", Object.assign({
      cx: A[0], cy: A[1], r: r * this.s,
      fill: o.fill || "none", "fill-opacity": o.fillOpacity,
    }, stroke(o)), this.root);
  };

  P.ellipse = function (c, rx, ry, o) {
    o = o || {};
    const A = this.px(c);
    return el("ellipse", Object.assign({
      cx: A[0], cy: A[1], rx: rx * this.s, ry: ry * this.s,
      fill: o.fill || "none", "fill-opacity": o.fillOpacity,
      transform: o.angle ? "rotate(" + (-o.angle * 180 / Math.PI) + " " + A[0] + " " + A[1] + ")" : null,
    }, stroke(o)), this.root);
  };

  P.dot = function (c, o) {
    o = o || {};
    const A = this.px(c);
    return el("circle", {
      cx: A[0], cy: A[1], r: o.rpx || 4,
      fill: o.fill || o.color || C.ink,
      stroke: o.stroke || "none",
      "stroke-width": o.width || 1,
    }, this.root);
  };

  P.rect = function (c, w, h, o) {
    o = o || {};
    const a = o.angle || 0, ca = Math.cos(a), sa = Math.sin(a);
    const corners = [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
      .map(([x, y]) => [c[0] + ca * x - sa * y, c[1] + sa * x + ca * y]);
    const e = this.poly(corners, Object.assign({ fill: "#dddddd" }, o));
    e.corners = corners;
    return e;
  };

  // Plain upright text. o: {size, color, anchor: start|middle|end, bold, italic, dx, dy (px)}
  P.text = function (p, s, o) {
    o = o || {};
    const A = this.px(p);
    const size = o.size || this.fontSize;
    const t = el("text", {
      x: A[0] + (o.dx || 0), y: A[1] + (o.dy || 0),
      "font-size": size,
      "text-anchor": o.anchor || "start",
      "dominant-baseline": o.baseline || "middle",
      fill: o.color || C.ink,
      "font-style": o.italic ? "italic" : null,
      "font-weight": o.bold ? "bold" : null,
    }, this.root);
    if (o.italic || o.math) fillLabel(t, String(s), size);
    else t.textContent = s;
    return t;
  };

  // Math-ish italic label with sub/superscripts.
  P.label = function (p, s, o) {
    return this.text(p, s, Object.assign({ italic: true, anchor: "middle" }, o || {}));
  };

  // Arrow from a to b. o: {color, width, headpx, label, lpos: 'end'|'mid'|'start', loff: [dx,dy] px, lsize}
  P.arrow = function (a, b, o) {
    o = o || {};
    const color = o.color || C.ink;
    const w = o.width || 3;
    const A = this.px(a), B = this.px(b);
    const dx = B[0] - A[0], dy = B[1] - A[1];
    const L = Math.hypot(dx, dy);
    const g = el("g", {}, this.root);
    if (L < 1e-6) return g;
    const ux = dx / L, uy = dy / L;
    const head = Math.min(o.headpx || 8 + 2.6 * w, L * 0.6);
    const hw = head * 0.42;
    const S = [B[0] - ux * head * 0.8, B[1] - uy * head * 0.8];
    el("line", Object.assign({ x1: A[0], y1: A[1], x2: S[0], y2: S[1] }, stroke({ color, width: w, dash: o.dash })), g);
    el("polygon", {
      points: [
        B[0], B[1],
        B[0] - ux * head - uy * hw, B[1] - uy * head + ux * hw,
        B[0] - ux * head + uy * hw, B[1] - uy * head - ux * hw,
      ].map((v) => v.toFixed(2)).join(" "),
      fill: color,
      stroke: color,
      "stroke-width": 1,
      "stroke-linejoin": "round",
    }, g);
    if (o.label) {
      let q;
      if (o.lpos === "mid") q = [(A[0] + B[0]) / 2, (A[1] + B[1]) / 2];
      else if (o.lpos === "start") q = [A[0] - ux * 16, A[1] - uy * 16];
      else q = [B[0] + ux * 16, B[1] + uy * 16];
      const off = o.loff || [0, 0];
      const t = el("text", {
        x: q[0] + off[0], y: q[1] + off[1],
        "font-size": o.lsize || this.fontSize,
        "text-anchor": o.anchor || "middle",
        "dominant-baseline": "middle",
        fill: o.lcolor || color,
        "font-style": o.upright ? null : "italic",
        "font-weight": o.lbold ? "bold" : null,
      }, g);
      fillLabel(t, o.label, o.lsize || this.fontSize);
    }
    return g;
  };

  // Hatched surface of a solid along a->b. The solid (and the hatching) is on the
  // right-hand side of the direction a->b, in world coordinates.
  P.surface = function (a, b, o) {
    o = o || {};
    const A = this.px(a), B = this.px(b);
    const L = Math.hypot(B[0] - A[0], B[1] - A[1]);
    const g = el("g", {}, this.root);
    if (L < 1e-9) return g;
    const wx = b[0] - a[0], wy = b[1] - a[1], wl = Math.hypot(wx, wy);
    const u = [wx / wl, -wy / wl];          // direction, screen coords
    const n = [wy / wl, wx / wl];           // world right-hand normal (wy, -wx), screen coords
    const gap = o.gappx || 13, len = (o.hatchpx || 11) * 0.72;
    for (let t = gap / 2; t < L; t += gap) {
      const x = A[0] + u[0] * t, y = A[1] + u[1] * t;
      el("line", {
        x1: x, y1: y,
        x2: x + (n[0] - u[0]) * len, y2: y + (n[1] - u[1]) * len,
        stroke: o.color || C.ink, "stroke-width": 1,
      }, g);
    }
    el("line", { x1: A[0], y1: A[1], x2: B[0], y2: B[1], stroke: o.color || C.ink, "stroke-width": o.width || 2 }, g);
    return g;
  };

  // A floor from x0 to x1 at height y (solid below).
  P.ground = function (x0, x1, y, o) { return this.surface([x0, y], [x1, y], o); };

  // 2D friction cone (a wedge) at apex, axis angle `dir` (rad), half-angle `half`, length `len`.
  P.wedge = function (apex, dir, half, len, o) {
    o = Object.assign({ fill: C.cone, fillOpacity: 0.8, color: C.ink, width: 1.2 }, o || {});
    const pts = [apex];
    const n = o.flat ? 1 : Math.max(2, Math.ceil(half * 24));
    for (let i = 0; i <= n; i++) {
      const a = dir - half + (2 * half * i) / n;
      pts.push([apex[0] + len * Math.cos(a), apex[1] + len * Math.sin(a)]);
    }
    return this.poly(pts, o);
  };

  // Angle marker: arc of radius rpx around c from world angle a0 to a1 (CCW).
  P.arc = function (c, rpx, a0, a1, o) {
    o = o || {};
    const A = this.px(c);
    const n = Math.max(2, Math.ceil(Math.abs(a1 - a0) * 20));
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const a = a0 + ((a1 - a0) * i) / n;
      pts.push([A[0] + rpx * Math.cos(a), A[1] - rpx * Math.sin(a)]);
    }
    const g = el("g", {}, this.root);
    el("path", Object.assign({
      d: pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(2) + " " + p[1].toFixed(2)).join(" "),
      fill: "none",
    }, stroke({ color: o.color, width: o.width || 1.3, dash: o.dash })), g);
    if (o.label) {
      const am = (a0 + a1) / 2, r = rpx + (o.lgap || 14);
      const t = el("text", {
        x: A[0] + r * Math.cos(am), y: A[1] - r * Math.sin(am),
        "font-size": o.lsize || this.fontSize * 0.9,
        "text-anchor": "middle", "dominant-baseline": "middle",
        fill: o.color || C.ink, "font-style": "italic",
      }, g);
      fillLabel(t, o.label, o.lsize || this.fontSize * 0.9);
    }
    return g;
  };

  // Small right-angle mark at corner c between unit directions u and v.
  P.rightAngle = function (c, u, v, o) {
    o = o || {};
    const k = (o.px || 10) / this.s;
    return this.path([
      [c[0] + u[0] * k, c[1] + u[1] * k],
      [c[0] + (u[0] + v[0]) * k, c[1] + (u[1] + v[1]) * k],
      [c[0] + v[0] * k, c[1] + v[1] * k],
    ], { width: 1, color: o.color });
  };

  // Draggable handle at world point p. onDrag(worldPoint) is called while dragging.
  P.handle = function (p, onDrag, o) {
    o = o || {};
    const A = this.px(p);
    const h = el("circle", {
      cx: A[0], cy: A[1], r: o.rpx || 9,
      fill: o.fill || "#ffffff", stroke: o.color || C.ink, "stroke-width": 2,
      class: "handle",
    }, this.root);
    const self = this;
    h.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      h.setPointerCapture(e.pointerId);
      const move = (ev) => {
        const pt = self.svg.createSVGPoint();
        pt.x = ev.clientX; pt.y = ev.clientY;
        const q = pt.matrixTransform(self.svg.getScreenCTM().inverse());
        onDrag(self.world(q.x, q.y));
      };
      const up = () => {
        h.removeEventListener("pointermove", move);
        h.removeEventListener("pointerup", up);
      };
      h.addEventListener("pointermove", move);
      h.addEventListener("pointerup", up);
    });
    return h;
  };

  // ---- 3D (orthographic) ---------------------------------------------------
  // view3(az, el): z is up; az rotates about z, el tilts the camera up.
  Fig.view3 = function (az, elev) {
    const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(elev), se = Math.sin(elev);
    const ex = [-sa, ca, 0];
    const ey = [-se * ca, -se * sa, ce];
    const ez = [ce * ca, ce * sa, se]; // toward the viewer
    const f = (p) => [p[0] * ex[0] + p[1] * ex[1] + p[2] * ex[2], p[0] * ey[0] + p[1] * ey[1] + p[2] * ey[2]];
    f.depth = (p) => p[0] * ez[0] + p[1] * ez[1] + p[2] * ez[2];
    f.toward = ez;
    return f;
  };

  // Circle in 3D: center c, unit axis n, radius r -> list of 3D points.
  Fig.ring3 = function (c, n, r, k) {
    k = k || 72;
    let a = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const dot = a[0] * n[0] + a[1] * n[1] + a[2] * n[2];
    a = [a[0] - dot * n[0], a[1] - dot * n[1], a[2] - dot * n[2]];
    const la = Math.hypot(a[0], a[1], a[2]);
    a = a.map((x) => x / la);
    const b = [n[1] * a[2] - n[2] * a[1], n[2] * a[0] - n[0] * a[2], n[0] * a[1] - n[1] * a[0]];
    const pts = [];
    for (let i = 0; i <= k; i++) {
      const t = (2 * Math.PI * i) / k;
      pts.push([0, 1, 2].map((j) => c[j] + r * (Math.cos(t) * a[j] + Math.sin(t) * b[j])));
    }
    return pts;
  };

  // Bind a 3D view to this figure: F.in3(view).arrow([x,y,z], [x,y,z], opts), etc.
  P.in3 = function (view) {
    const F = this;
    const T = {
      view,
      p: (q) => view(q),
      line: (a, b, o) => F.line(view(a), view(b), o),
      arrow: (a, b, o) => F.arrow(view(a), view(b), o),
      path: (pts, o) => F.path(pts.map(view), o),
      poly: (pts, o) => F.poly(pts.map(view), o),
      dot: (a, o) => F.dot(view(a), o),
      label: (a, s, o) => F.label(view(a), s, o),
      text: (a, s, o) => F.text(view(a), s, o),
      ring: (c, n, r, o) => F.path(Fig.ring3(c, n, r), Object.assign({ close: true }, o || {})),
      // An open (ice-cream) cone: apex, unit axis, height, rim radius.
      cone: (apex, axis, h, r, o) => {
        o = Object.assign({ fill: C.cone, inside: "#f5f5c0", color: C.ink, width: 1.3 }, o || {});
        const top = apex.map((x, i) => x + h * axis[i]);
        const rim3 = Fig.ring3(top, axis, r, 180).slice(0, 180);
        const R = rim3.map(view), A = view(apex), Tp = view(top);
        const ref = Math.atan2(Tp[1] - A[1], Tp[0] - A[0]);
        const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
        const ang = R.map((q) => wrap(Math.atan2(q[1] - A[1], q[0] - A[0]) - ref));
        let lo = 0, hi = 0;
        ang.forEach((a, i) => { if (a < ang[lo]) lo = i; if (a > ang[hi]) hi = i; });
        // The front arc between the two silhouette points is the one nearer the viewer.
        const arc = (from, to) => { const out = []; for (let i = from; ; i = (i + 1) % R.length) { out.push(i); if (i === to) break; } return out; };
        const a1 = arc(lo, hi), a2 = arc(hi, lo);
        const depth = (idx) => idx.reduce((s, i) => s + view.depth(rim3[i]), 0) / idx.length;
        const front = depth(a1) > depth(a2) ? a1 : a2;
        F.poly([A].concat(front.map((i) => R[i])), { fill: o.fill, fillOpacity: o.fillOpacity, stroke: "none" });
        F.poly(R, { fill: o.inside, fillOpacity: o.fillOpacity, color: o.color, width: o.width });
        F.line(A, R[lo], { color: o.color, width: o.width });
        F.line(A, R[hi], { color: o.color, width: o.width });
        return { rim: rim3, top };
      },
      // An open pyramid: apex plus 3D rim vertices (in order).
      pyramid: (apex, verts, o) => {
        o = Object.assign({ fill: C.cone, color: C.ink, width: 1.3 }, o || {});
        const A = view(apex), R = verts.map(view);
        F.poly(R, { fill: o.fill, fillOpacity: o.fillOpacity == null ? 0.55 : o.fillOpacity, color: o.color, width: o.width });
        R.forEach((q) => F.line(A, q, { color: o.color, width: o.width, dash: o.dash }));
      },
    };
    return T;
  };

  window.Fig = Fig;
})();
