/* contact.js — the little bit of contact mechanics the live figures need.
 *
 * Everything here is mirrored (and tested) in the companion notebooks:
 * tutorial/04_friction_cones.py and tutorial/06_force_closure.py.
 */
(function () {
  "use strict";

  const V = {
    add: (a, b) => a.map((x, i) => x + b[i]),
    sub: (a, b) => a.map((x, i) => x - b[i]),
    mul: (a, k) => a.map((x) => x * k),
    dot: (a, b) => a.reduce((s, x, i) => s + x * b[i], 0),
    norm: (a) => Math.hypot.apply(null, a),
    unit: (a) => { const n = Math.hypot.apply(null, a); return n > 0 ? a.map((x) => x / n) : a.slice(); },
    cross2: (a, b) => a[0] * b[1] - a[1] * b[0],
    cross3: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
    rot: (a, t) => [Math.cos(t) * a[0] - Math.sin(t) * a[1], Math.sin(t) * a[0] + Math.cos(t) * a[1]],
    perp: (a) => [-a[1], a[0]],
    // Unsigned angle between two vectors, radians.
    angle: (a, b) => Math.atan2(Math.abs(V.cross2(a, b)), V.dot(a, b)),
  };
  const deg = (r) => (r * 180) / Math.PI;
  const rad = (d) => (d * Math.PI) / 180;

  // ---- 2D friction cone -----------------------------------------------------
  // f and inward normal n are 2D vectors. Returns {fn, ft, theta, alpha, mode}, where
  // mode is zero | pull | inside | edge | outside. "edge" means within angTol (rad) of the
  // cone's boundary, so a dragged arrow can actually land on it.
  function classify2(f, n, mu, tol, angTol) {
    tol = tol === undefined ? 1e-9 : tol;
    angTol = angTol === undefined ? 0.015 : angTol;
    n = V.unit(n);
    const t = [n[1], -n[0]];
    const fn = V.dot(f, n), ft = V.dot(f, t);
    const alpha = Math.atan(mu);
    const theta = Math.atan2(Math.abs(ft), fn);
    let mode;
    if (Math.hypot(fn, ft) < tol) mode = "zero";
    else if (fn < -tol) mode = "pull";
    else if (Math.abs(theta - alpha) <= angTol) mode = "edge";
    else if (theta < alpha) mode = "inside";
    else mode = "outside";
    return { fn, ft, theta, alpha, mode };
  }

  // ---- friction "slices" (the set of allowed tangential forces per unit normal force)
  // A model is {kind: 'circle'|'ellipse'|'poly', mu, mu2, verts}.
  function regularPolygon(k, r, phase) {
    const v = [];
    for (let j = 0; j < k; j++) {
      const a = (phase || 0) + (2 * Math.PI * j) / k;
      v.push([r * Math.cos(a), r * Math.sin(a)]);
    }
    return v;
  }

  // Inscribed k-gon has vertices on the circle; circumscribed has edge midpoints on it.
  function pyramid(k, mu, how, phase) {
    const r = how === "circumscribed" ? mu / Math.cos(Math.PI / k) : mu;
    const ph = phase === undefined ? (how === "circumscribed" ? Math.PI / k : 0) : phase;
    return { kind: "poly", mu, verts: regularPolygon(k, r, ph), k, how };
  }

  const MODELS = {
    circle: (mu) => ({ kind: "circle", mu }),
    diamond: (mu) => pyramid(4, mu, "inscribed", 0),     // |ft1| + |ft2| <= mu fn
    box: (mu) => pyramid(4, mu, "circumscribed"),         // |ft1|, |ft2| <= mu fn
    octagon: (mu) => pyramid(8, mu, "inscribed", 0),
    ellipse: (mu, mu2) => ({ kind: "ellipse", mu, mu2: mu2 === undefined ? mu / 3 : mu2 }),
  };

  // Maximum dissipation: the point of the slice that makes f.v most negative.
  function mdp(model, v) {
    const s = V.norm(v);
    if (s < 1e-12) return [0, 0];
    if (model.kind === "circle") return V.mul(v, -model.mu / s);
    if (model.kind === "ellipse") {
      const a = model.mu, b = model.mu2;
      const m = [a * a * v[0], b * b * v[1]];
      const d = Math.hypot(a * v[0], b * v[1]);
      return [-m[0] / d, -m[1] / d];
    }
    // Polygon: best vertex; if two tie (v along a facet normal) return the facet midpoint.
    let best = Infinity, arg = [];
    model.verts.forEach((p, i) => {
      const d = V.dot(p, v);
      if (d < best - 1e-9 * s) { best = d; arg = [i]; }
      else if (Math.abs(d - best) <= 1e-9 * s) arg.push(i);
    });
    if (arg.length === 1) return model.verts[arg[0]].slice();
    return V.mul(arg.reduce((acc, i) => V.add(acc, model.verts[i]), [0, 0]), 1 / arg.length);
  }

  // Euclidean projection onto the slice (scaled by `scale`).
  function project(model, z, scale) {
    if (model.kind === "circle") {
      const r = model.mu * scale, n = V.norm(z);
      return n <= r ? z.slice() : V.mul(z, r / n);
    }
    if (model.kind === "ellipse") {
      const a = model.mu * scale, b = model.mu2 * scale;
      if ((z[0] / a) ** 2 + (z[1] / b) ** 2 <= 1) return z.slice();
      // f_i = a_i^2 z_i / (a_i^2 + t), find t >= 0 on the boundary by bisection.
      const g = (t) => (a * z[0] / (a * a + t)) ** 2 + (b * z[1] / (b * b + t)) ** 2 - 1;
      let lo = 0, hi = Math.max(a, b) * V.norm(z) + 1;
      while (g(hi) > 0) hi *= 2;
      for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; g(m) > 0 ? (lo = m) : (hi = m); }
      return [a * a * z[0] / (a * a + hi), b * b * z[1] / (b * b + hi)];
    }
    const P = model.verts.map((p) => V.mul(p, scale));
    // Inside test (convex, CCW).
    let inside = true;
    for (let i = 0; i < P.length; i++) {
      const a = P[i], b = P[(i + 1) % P.length];
      if (V.cross2(V.sub(b, a), V.sub(z, a)) < 0) { inside = false; break; }
    }
    if (inside) return z.slice();
    let best = null, bd = Infinity;
    for (let i = 0; i < P.length; i++) {
      const a = P[i], b = P[(i + 1) % P.length], e = V.sub(b, a);
      const t = Math.max(0, Math.min(1, V.dot(V.sub(z, a), e) / V.dot(e, e)));
      const q = V.add(a, V.mul(e, t)), d = V.norm(V.sub(z, q));
      if (d < bd) { bd = d; best = q; }
    }
    return best;
  }

  // A puck sliding on a floor: implicit (maximum-dissipation) time stepping.
  // Each step picks the friction force in the slice that minimizes |v_next|,
  // i.e. the projection of -m v / h onto mu*m*g*slice. Returns [[x, y], ...].
  function slidePuck(model, v0, opt) {
    opt = opt || {};
    const g = opt.g || 9.81, h = opt.h || 0.002, tmax = opt.tmax || 10;
    let x = [0, 0], v = v0.slice();
    const path = [x.slice()];
    for (let t = 0; t < tmax; t += h) {
      const f = project(model, V.mul(v, -1 / h), g); // per unit mass
      v = V.add(v, V.mul(f, h));
      x = V.add(x, V.mul(v, h));
      path.push(x.slice());
      if (V.norm(v) < 1e-9) break;
    }
    return path;
  }

  // ---- planar grasps ---------------------------------------------------------
  // Nguyen: contacts p1, p2 with INWARD normals n1, n2.
  function nguyen(p1, n1, p2, n2, mu) {
    const d = V.unit(V.sub(p2, p1));
    const b1 = V.angle(d, n1), b2 = V.angle(V.mul(d, -1), n2);
    const a = Math.atan(mu);
    return { ok: b1 < a && b2 < a, beta1: b1, beta2: b2, alpha: a };
  }

  // The normal-angle shortcut: angle between the two inward normals, measured from "exactly opposite".
  function normalAngleTest(n1, n2, mu) {
    const ang = V.angle(n1, V.mul(n2, -1));
    return { ok: ang <= 2 * Math.atan(mu), angle: ang };
  }

  // Primitive (edge) wrenches [fx, fy, tau/lambda] of planar frictional contacts.
  function edgeWrenches2(contacts, mu, lambda) {
    lambda = lambda || 1;
    const a = Math.atan(mu), out = [];
    contacts.forEach(({ p, n }) => {
      const u = V.unit(n);
      [-a, a].forEach((s) => {
        const f = V.mul(V.rot(u, s), 1 / Math.cos(a)); // unit normal component
        out.push([f[0], f[1], V.cross2(p, f) / lambda]);
      });
    });
    return out;
  }

  // Does a finite set of vectors in R^3 positively span R^3? Exact for small sets:
  // if the cone they generate is not everything, one of its facets is spanned by a
  // pair of generators, whose cross product is then a separating normal.
  function positivelySpans3(ws, tol) {
    tol = tol || 1e-9;
    if (rank3(ws, 1e-9) < 3) return false;
    for (let i = 0; i < ws.length; i++) {
      for (let j = i + 1; j < ws.length; j++) {
        const u = V.cross3(ws[i], ws[j]);
        const nu = V.norm(u);
        if (nu < 1e-12) continue;
        let pos = false, neg = false;
        for (const w of ws) {
          const d = V.dot(w, u) / nu;
          if (d > tol) pos = true;
          else if (d < -tol) neg = true;
          if (pos && neg) break;
        }
        if (!(pos && neg)) return false;
      }
    }
    return true;
  }

  function rank3(ws, tol) {
    let best = 0;
    for (let i = 0; i < ws.length; i++) {
      if (V.norm(ws[i]) > tol) best = Math.max(best, 1);
      for (let j = i + 1; j < ws.length; j++) {
        const c = V.cross3(ws[i], ws[j]);
        if (V.norm(c) > tol) {
          best = Math.max(best, 2);
          for (let k = j + 1; k < ws.length; k++) if (Math.abs(V.dot(c, ws[k])) > tol) return 3;
        }
      }
    }
    return best;
  }

  // Ferrari-Canny epsilon (L1 version) for points in R^3: distance from the origin
  // to the nearest facet of their convex hull, or 0 if the origin is not inside.
  // Brute force over triples; fine for the ~10 points a planar grasp produces.
  function epsilon3(ws) {
    if (rank3(ws, 1e-9) < 3) return 0;
    let eps = Infinity;
    const n = ws.length;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) for (let k = j + 1; k < n; k++) {
      const u = V.cross3(V.sub(ws[j], ws[i]), V.sub(ws[k], ws[i]));
      const nu = V.norm(u);
      if (nu < 1e-12) continue;
      const c = V.dot(u, ws[i]) / nu; // plane: dot(u/|u|, x) = c
      let pos = false, neg = false;
      for (let m = 0; m < n; m++) {
        const d = V.dot(u, ws[m]) / nu - c;
        if (d > 1e-9) pos = true; else if (d < -1e-9) neg = true;
      }
      if (pos === neg) continue;        // not a supporting plane (or a degenerate one)
      // Signed distance from the origin into the hull across this facet.
      eps = Math.min(eps, pos ? -c : c);
    }
    return eps > 1e-12 && eps < Infinity ? eps : 0;
  }

  // 2D convex hull (monotone chain), CCW.
  function hull2(pts) {
    const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (p.length < 3) return p;
    const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo = [], up = [];
    for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
    for (let i = p.length - 1; i >= 0; i--) {
      const q = p[i];
      while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop();
      up.push(q);
    }
    return lo.slice(0, -1).concat(up.slice(0, -1));
  }

  // Radius of the largest origin-centered circle inside a CCW convex polygon (0 if outside).
  function inradius2(h) {
    let r = Infinity;
    for (let i = 0; i < h.length; i++) {
      const a = h[i], b = h[(i + 1) % h.length], e = V.sub(b, a);
      const d = V.cross2(e, V.mul(a, -1)) / V.norm(e); // >0 when origin is to the left
      r = Math.min(r, d);
    }
    return Math.max(0, r);
  }

  // Intersect a convex polygon with the half-plane {x : dot(a, x) <= b}.
  function clip(poly, a, b) {
    const out = [];
    for (let i = 0; i < poly.length; i++) {
      const P = poly[i], Q = poly[(i + 1) % poly.length];
      const dp = V.dot(a, P) - b, dq = V.dot(a, Q) - b;
      if (dp <= 0) out.push(P);
      if ((dp < 0 && dq > 0) || (dp > 0 && dq < 0)) {
        const t = dp / (dp - dq);
        out.push(V.add(P, V.mul(V.sub(Q, P), t)));
      }
    }
    return out;
  }

  // ---- shapes for the grasp explorers (outward normals, s in [0, 1)) --------
  const Shapes = {
    disk: (R) => ({
      name: "disk",
      at: (s) => { const t = 2 * Math.PI * s; return { p: [R * Math.cos(t), R * Math.sin(t)], n: [Math.cos(t), Math.sin(t)] }; },
      outline: () => Array.from({ length: 121 }, (_, i) => { const t = (2 * Math.PI * i) / 120; return [R * Math.cos(t), R * Math.sin(t)]; }),
    }),
    ellipse: (a, b) => ({
      name: "ellipse",
      at: (s) => { const t = 2 * Math.PI * s; return { p: [a * Math.cos(t), b * Math.sin(t)], n: V.unit([b * Math.cos(t), a * Math.sin(t)]) }; },
      outline: () => Array.from({ length: 121 }, (_, i) => { const t = (2 * Math.PI * i) / 120; return [a * Math.cos(t), b * Math.sin(t)]; }),
    }),
    polygon: (verts, name) => {
      const L = verts.map((v, i) => V.norm(V.sub(verts[(i + 1) % verts.length], v)));
      const tot = L.reduce((x, y) => x + y, 0);
      return {
        name: name || "polygon",
        at: (s) => {
          let d = (((s % 1) + 1) % 1) * tot;
          for (let i = 0; i < verts.length; i++) {
            if (d <= L[i] || i === verts.length - 1) {
              const a = verts[i], b = verts[(i + 1) % verts.length], e = V.unit(V.sub(b, a));
              return { p: V.add(a, V.mul(e, Math.min(d, L[i]))), n: [e[1], -e[0]] }; // CCW -> outward = right normal
            }
            d -= L[i];
          }
        },
        outline: () => verts.concat([verts[0]]),
      };
    },
  };
  Shapes.box = (w, h) => Shapes.polygon([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], "box");
  // Isosceles wedge ("watermelon seed") pointing up, half-angle beta at the tip.
  Shapes.wedge = (beta, H) => {
    const bw = H * Math.tan(beta);
    return Shapes.polygon([[-bw, -H / 2], [bw, -H / 2], [0, H / 2]], "wedge");
  };

  window.Contact = {
    V, deg, rad, classify2, regularPolygon, pyramid, MODELS, mdp, project, slidePuck,
    nguyen, normalAngleTest, edgeWrenches2, positivelySpans3, rank3, epsilon3, hull2, inradius2,
    clip, Shapes,
  };
})();
