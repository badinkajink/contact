/* 01_linear_maps.js: the algorithms behind the live figures of deck 01
 * (01_vectors_and_matrices.html, "Vectors, matrices, and linear maps").
 *
 * Load after lib/num.js:
 *   <script src="lib/num.js"></script>
 *   <script src="lib/algo/01_linear_maps.js"></script>
 * It defines one global, window.LinearMaps. Every function with a Python counterpart in
 * tutorial/00_math_toolkit.py (section "Deck 01", names d01_*) is checked against it by
 * tools/twins/01.json (uv run tools/twins.py 01). Matrices are arrays of rows, as in num.js.
 *
 * Vectors
 *   length(v)                 Euclidean length.                          (d01_vector_length)
 *   coords2(u, w, x)          [c1, c2] with c1 u + c2 w = x, or null when u, w are parallel.
 *                                                                        (d01_coords)
 *   indepDet(h)               {det, rank} of [v1 v2 v3], v3 = (1, 1, 2 + h). (d01_independence_det)
 *   dotAngle(a, b)            {dot, norm_a, norm_b, cos, deg}; cos, deg null for a zero vector.
 *                                                                        (d01_dot_angle)
 *   pnorm(x, p)               p-norm for p >= 1, p = Infinity for max |x_i|. (d01_pnorm)
 *   pnorms(x)                 [|x|_1, |x|_2, |x|_inf].                   (d01_pnorms)
 *   unitBall(p, k)            k + 1 points of the curve |x|_p = 1 (closed), for drawing.
 * Matrices as maps
 *   det2(A), inv2(A)          2x2 determinant; inverse or null.          (d01_det2, d01_inv2)
 *   compose(B, A)             B A, the matrix of "first A, then B".      (d01_compose)
 *   transposeCheck(A, x, y)   [(A x) . y, x . (A^T y)].                  (d01_transpose_check)
 *   parallelogramArea(a, c, b, d)  {box, ac, bd, 2bc, area, det}.        (d01_parallelogram_area)
 *   nonsquare(t, s)           x(t) = (2, 1, 0) + t (-1, -1, 1) under A23, B32 s. (d01_nonsquare_maps)
 *   hull2(points)             convex hull of 2D points, counter-clockwise (drawing helper).
 * Systems, rank, solution sets
 *   rank(A)                   Num.rank of a matrix (a vector counts as one row). (d01_rank)
 *   classify2(A, b)           {status: 'one' | 'none' | 'infinite', x, rankA, rankAb}.
 *                             'infinite' returns the shortest solution.  (d01_classify_2x2)
 *   rrefSteps(A)              Num.rref(A) with steps as [op, i, j, k, text] rows:
 *                             swap [op, i, j, 0], scale [op, i, i, k], add [op, i, j, k].
 *                                                                        (d01_rref_steps)
 *   consistency(A, b)         {rankA, rankAb, consistent}.               (d01_consistency)
 *   nullBasis(A)              special solutions (one per free column), as rows. (d01_null_basis)
 *   fourSubspaces(A)          {rank, pivots, col, row, null, left_null, dims}. (d01_four_subspaces)
 *   solutionSet(A, b)         {consistent, particular (free variables 0), null}. (d01_solution_set)
 *   tableLegs(W, t, half)     leg forces of the four-legged table.       (d01_table_legs)
 *   goalDims(G)               n - rank(G).                               (d01_goal_dims)
 *   stacked(N, G, bG)         {rankN, rankNG, rankAug, consistent, dimSol, nav}. (d01_stacked_ranks)
 */
(function () {
  "use strict";
  const Num = window.Num;
  if (!Num) throw new Error("01_linear_maps.js: load lib/num.js first");

  const as2d = (A) => (Array.isArray(A[0]) ? A : [A]);
  const z0 = (v) => v + 0; // -0 -> 0

  // ------------------------------------------------------------------ vectors
  function length(v) {
    return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  }

  function coords2(u, w, x) {
    const det = u[0] * w[1] - u[1] * w[0];
    const scale = Math.max(1, ...[u[0], u[1], w[0], w[1]].map(Math.abs)) ** 2;
    if (Math.abs(det) <= 1e-12 * scale) return null;
    return [(x[0] * w[1] - x[1] * w[0]) / det, (u[0] * x[1] - u[1] * x[0]) / det];
  }

  function indepDet(h) {
    const V = Num.transpose([[1, 0, 1], [0, 1, 1], [1, 1, 2 + h]]);
    return { det: Num.det(V), rank: Num.rank(V) };
  }

  function dotAngle(a, b) {
    const dot = Num.dot(a, b), na = length(a), nb = length(b);
    if (na === 0 || nb === 0) return { dot, norm_a: na, norm_b: nb, cos: null, deg: null };
    const c = Math.max(-1, Math.min(1, dot / (na * nb)));
    return { dot, norm_a: na, norm_b: nb, cos: c, deg: (Math.acos(c) * 180) / Math.PI };
  }

  function pnorm(x, p) {
    const ax = x.map(Math.abs);
    const m = Math.max(...ax);
    if (p === Infinity) return m;
    if (p < 1) throw new Error("pnorm: the p-norm is a norm only for p >= 1");
    if (m === 0) return 0;
    return m * Math.pow(ax.reduce((s, v) => s + Math.pow(v / m, p), 0), 1 / p);
  }

  function pnorms(x) {
    return [pnorm(x, 1), pnorm(x, 2), pnorm(x, Infinity)];
  }

  // Points of {x : |x|_p = 1}: each direction (cos t, sin t) divided by its p-norm.
  function unitBall(p, k) {
    k = k || 360;
    const pts = [];
    for (let i = 0; i <= k; i++) {
      const t = (2 * Math.PI * i) / k, d = [Math.cos(t), Math.sin(t)];
      const r = pnorm(d, p);
      pts.push([d[0] / r, d[1] / r]);
    }
    return pts;
  }

  // ------------------------------------------------------------------ matrices as maps
  function det2(A) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
  }

  function inv2(A) {
    const d = det2(A);
    if (d === 0) return null;
    return [[A[1][1] / d, -A[0][1] / d], [-A[1][0] / d, A[0][0] / d]];
  }

  function compose(B, A) {
    if (B[0].length !== A.length) throw new Error("compose: shapes do not compose");
    return Num.matmul(B, A);
  }

  function transposeCheck(A, x, y) {
    return [Num.dot(Num.matvec(A, x), y), Num.dot(x, Num.matvec(Num.transpose(A), y))];
  }

  function parallelogramArea(a, c, b, d) {
    const box = (a + b) * (c + d);
    return { box, ac: a * c, bd: b * d, "2bc": 2 * b * c, area: box - a * c - b * d - 2 * b * c, det: a * d - b * c };
  }

  const A23 = [[1, 0, 1], [0, 1, 1]];
  function nonsquare(t, s) {
    t = t || 0;
    s = s || [1, 0.5];
    const x = [2 - t, 1 - t, t];
    const cube = [];
    for (const i of [0, 1]) for (const j of [0, 1]) for (const k of [0, 1]) cube.push(Num.matvec(A23, [i, j, k]));
    const key = (p) => p.join(",");
    const seen = new Map();
    cube.forEach((p) => seen.set(key(p), p));
    const image = Array.from(seen.values()).sort((p, q) => p[0] - q[0] || p[1] - q[1]);
    const B32 = Num.transpose(A23);
    return {
      x, Ax: Num.matvec(A23, x), cube_image: image, Bs: Num.matvec(B32, s),
      rank_A23: Num.rank(A23), rank_B32: Num.rank(B32), e3_offset: 1,
    };
  }

  // Andrew's monotone chain; returns the hull counter-clockwise without repeating the start.
  function hull2(points) {
    const P = points.slice().sort((p, q) => p[0] - q[0] || p[1] - q[1]);
    if (P.length < 3) return P;
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lower = [], upper = [];
    for (const p of P) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
      lower.push(p);
    }
    for (let i = P.length - 1; i >= 0; i--) {
      const p = P[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
      upper.push(p);
    }
    return lower.slice(0, -1).concat(upper.slice(0, -1));
  }

  // ------------------------------------------------------------------ systems
  function rank(A) {
    const M = as2d(A);
    return M.length && M[0].length ? Num.rank(M) : 0;
  }

  function classify2(A, b) {
    const rA = rank(A), rAb = rank(Num.hstack(A, b));
    if (rA === 2) return { status: "one", x: Num.solve(A, b), rankA: rA, rankAb: rAb };
    if (rA === rAb) return { status: "infinite", x: Num.matvec(Num.pinv(A), b), rankA: rA, rankAb: rAb };
    return { status: "none", x: null, rankA: rA, rankAb: rAb };
  }

  function rrefSteps(A) {
    const r = Num.rref(A);
    const steps = r.steps.map((s) =>
      s.op === "swap" ? ["swap", s.i, s.j, 0, s.text]
        : s.op === "scale" ? ["scale", s.i, s.i, s.k, s.text]
          : ["add", s.i, s.j, s.k, s.text]);
    return { R: r.R, pivots: r.pivots, rank: r.rank, steps, tex: r.steps.map((s) => s.tex), matrices: r.steps.map((s) => s.matrix) };
  }

  function consistency(A, b) {
    const rA = rank(A), rAb = rank(Num.hstack(A, b));
    return { rankA: rA, rankAb: rAb, consistent: rA === rAb };
  }

  function nullBasis(A) {
    const n = A[0].length;
    const r = Num.rref(A);
    const out = [];
    for (let f = 0; f < n; f++) {
      if (r.pivots.includes(f)) continue;
      const x = new Array(n).fill(0);
      x[f] = 1;
      r.pivots.forEach((p, i) => { x[p] = z0(-r.R[i][f]); });
      out.push(x);
    }
    return out;
  }

  function fourSubspaces(A) {
    const m = A.length, n = A[0].length;
    const r = Num.rref(A);
    return {
      rank: r.rank, pivots: r.pivots,
      col: r.pivots.map((j) => Num.col(A, j)),
      row: r.R.slice(0, r.rank),
      null: nullBasis(A),
      left_null: nullBasis(Num.transpose(A)),
      dims: { col: r.rank, row: r.rank, null: n - r.rank, left_null: m - r.rank },
    };
  }

  function solutionSet(A, b) {
    const n = A[0].length;
    const r = Num.rref(Num.hstack(A, b));
    if (r.pivots.includes(n)) return { consistent: false, particular: null, null: nullBasis(A) };
    const x = new Array(n).fill(0);
    r.pivots.forEach((p, i) => { x[p] = r.R[i][n]; });
    return { consistent: true, particular: x, null: nullBasis(A) };
  }

  function tableLegs(W, t, half) {
    W = W === undefined ? 100 : W;
    t = t || 0;
    const h = half === undefined ? 0.5 : half;
    const E = [[1, 1, 1, 1], [h, h, -h, -h], [h, -h, -h, h]];
    const f = [W / 4 + t, W / 4 - t, W / 4 + t, W / 4 - t];
    const Ef = Num.matvec(E, f);
    return {
      E, rank: Num.rank(E), f, residual: [Ef[0] - W, Ef[1], Ef[2]],
      pushing: f.every((v) => v >= 0), t_range: [-W / 4, W / 4],
    };
  }

  function goalDims(G) {
    const M = as2d(G);
    return M[0].length - rank(M);
  }

  function stacked(N, G, bG) {
    N = as2d(N);
    G = as2d(G);
    bG = Array.isArray(bG) ? bG : [bG];
    const n = N[0].length;
    const NG = Num.vstack(N, G);
    const aug = Num.hstack(NG, new Array(N.length).fill(0).concat(bG));
    const rN = rank(N), rNG = rank(NG), rAug = rank(aug);
    const cons = rNG === rAug;
    return { rankN: rN, rankNG: rNG, rankAug: rAug, consistent: cons, dimSol: cons ? n - rNG : null, nav: rNG - rN };
  }

  window.LinearMaps = {
    length, coords2, indepDet, dotAngle, pnorm, pnorms, unitBall,
    det2, inv2, compose, transposeCheck, parallelogramArea, nonsquare, hull2,
    rank, classify2, rrefSteps, consistency, nullBasis, fourSubspaces, solutionSet,
    tableLegs, goalDims, stacked,
  };
})();
