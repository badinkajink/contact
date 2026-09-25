/* 02_eigen_ls.js — the algorithms behind the live figures of deck 02
 * (02_eigenvalues_and_least_squares.html): projections, Gram-Schmidt, 2x2 eigenvalues and
 * SVD, quadratic forms, Cholesky, conditioning of the drawer, the crashing index, line
 * fitting, the four-legged table, and the scalar Schur complement.
 *
 * Load after lib/num.js:
 *   <script src="lib/num.js"></script>
 *   <script src="lib/algo/02_eigen_ls.js"></script>
 * Defines one global, window.EigenLS. Every function returns a plain object whose keys match
 * the Python twin in tutorial/00_math_toolkit.py (section "Deck 02"), checked by
 *   uv run tools/twins.py 02
 *
 *   twin (00_math_toolkit.py)      JavaScript
 *   projection_onto_line(a, b)     EigenLS.projectLine(a, b)
 *   gram_schmidt_qr(vectors)       EigenLS.gramSchmidt(vectors)
 *   eigen_2x2(A)                   EigenLS.eig2(A)
 *   quadratic_from_eigen(...)      EigenLS.fromEigen(l1, l2, thetaDeg)
 *   quadratic_form_axes(A, c)      EigenLS.quadAxes(A, c)
 *   cholesky_2x2_steps(A, b)       EigenLS.chol2(A, b)
 *   svd_circle_ellipse(A)          EigenLS.svd2(A)
 *   nearly_parallel_rows(phi, db)  EigenLS.twoRows(phiDeg, db)
 *   drawer_conditioning(th, eps)   EigenLS.drawer(thetaDeg, epsDeg, b)
 *   crashing_index(J, C)           EigenLS.crashingIndex(J, C)
 *   line_fit_normal_equations      EigenLS.fitLine(t, y)
 *   table_force_distribution       EigenLS.tableForces(W, com, legs, t)
 *   schur_minimize_out(a, b, c)    EigenLS.schur2(a, b, c)
 *
 * Sign convention (as lib/num.js and the notebook): an eigenvector or singular vector is
 * scaled so that its entry of largest magnitude is positive, the first one when two tie to
 * within 1e-12. Missing results are null (Python None); an infinite condition number is
 * Infinity.
 */
(function () {
  "use strict";
  const N = window.Num;
  if (!N) throw new Error("02_eigen_ls.js: load lib/num.js first");
  const rad = (d) => (d * Math.PI) / 180;
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const hyp = (v) => Math.sqrt(dot(v, v));

  // Largest-magnitude entry positive (first one on a tie within 1e-12).
  function fixSign(v) {
    let k = 0;
    for (let i = 0; i < v.length; i++) if (Math.abs(v[i]) > Math.abs(v[k]) + 1e-12) k = i;
    return v[k] < 0 ? v.map((x) => -x) : v.slice();
  }

  // ------------------------------------------------------------------ part 1
  function projectLine(a, b) {
    const xhat = dot(a, b) / dot(a, a);
    const p = a.map((x) => xhat * x);
    const e = b.map((x, i) => x - p[i]);
    const aa = dot(a, a);
    return {
      xhat, p, e, e_dot_a: dot(e, a), dist: hyp(e),
      P: a.map((ai) => a.map((aj) => (ai * aj) / aa)),
    };
  }

  // Classical Gram-Schmidt on a list of independent vectors: Q (rows), R, residuals w.
  function gramSchmidt(vectors) {
    const k = vectors.length;
    const Q = [], W = [];
    const R = N.zeros(k, k);
    vectors.forEach((a, j) => {
      let w = a.slice();
      Q.forEach((q, i) => {
        R[i][j] = dot(q, a);
        w = w.map((x, t) => x - R[i][j] * q[t]);
      });
      R[j][j] = hyp(w);
      W.push(w);
      Q.push(w.map((x) => x / R[j][j]));
    });
    return { Q, R, w: W };
  }

  // ------------------------------------------------------------------ part 2
  // Eigenvalues of a 2x2 matrix from lambda^2 - tr lambda + det = 0.
  function eig2(A) {
    const tr = A[0][0] + A[1][1];
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    const disc = (tr * tr) / 4 - det;
    const out = { tr, det, disc };
    if (disc < -1e-14 * Math.max(1, tr * tr)) {
      return Object.assign(out, { real: false, values: null, vectors: null, complex: [tr / 2, Math.sqrt(-disc)] });
    }
    const s = Math.sqrt(Math.max(disc, 0));
    const vals = [tr / 2 + s, tr / 2 - s];
    const amax = Math.max(1, ...A.flat().map(Math.abs));
    const vecs = [];
    vals.forEach((lam) => {
      const M = [[A[0][0] - lam, A[0][1]], [A[1][0], A[1][1] - lam]];
      const r = hyp(M[0]) >= hyp(M[1]) ? M[0] : M[1];
      let v;
      if (hyp(r) < 1e-12 * amax) v = vecs.length ? [0, 1] : [1, 0]; // A = lambda I
      else v = [-r[1] / hyp(r), r[0] / hyp(r)];
      const k = Math.abs(v[0]) >= Math.abs(v[1]) - 1e-12 ? 0 : 1;
      vecs.push(v[k] > 0 ? v : v.map((x) => -x));
    });
    return Object.assign(out, { real: true, values: vals, vectors: vecs, complex: null });
  }

  // Symmetric A = V diag(l1, l2) V^T with the first eigenvector at thetaDeg.
  function fromEigen(l1, l2, thetaDeg) {
    const t = rad(thetaDeg), c = Math.cos(t), s = Math.sin(t);
    const V = [[c, -s], [s, c]];
    return N.matmul(V, N.diag([l1, l2]), N.transpose(V));
  }

  // Shape of J(z) = 1/2 z^T A z: kind, and the semi-axes sqrt(2c / l_i) of the level set J = c.
  function quadAxes(A, c) {
    c = c === undefined ? 1 : c;
    const e = eig2(A);
    const [l1, l2] = e.values;
    const tol = 1e-9 * Math.max(1, Math.abs(l1), Math.abs(l2));
    let kind;
    if (l2 > tol) kind = "bowl";
    else if (l1 < -tol) kind = "cap";
    else if (l1 > tol && l2 < -tol) kind = "saddle";
    else if (Math.abs(l1) <= tol && Math.abs(l2) <= tol) kind = "flat";
    else kind = l1 > tol ? "valley" : "ridge";
    const axes = [l1, l2].map((l) => (l > tol ? Math.sqrt((2 * c) / l) : null));
    return { values: [l1, l2], vectors: e.vectors, kind, semi_axes: axes, kappa: l2 > tol ? l1 / l2 : null };
  }

  // Cholesky of a symmetric 2x2 matrix entry by entry, with the two triangular solves.
  function chol2(A, b) {
    const out = { L11: null, L21: null, L22sq: null, L22: null, ok: false, L: null, w: null, z: null };
    if (A[0][0] <= 0) return out;
    const L11 = Math.sqrt(A[0][0]);
    const L21 = A[1][0] / L11;
    const L22sq = A[1][1] - L21 * L21;
    Object.assign(out, { L11, L21, L22sq });
    if (L22sq <= 1e-14 * Math.max(1, Math.abs(A[1][1]))) return out;
    const L22 = Math.sqrt(L22sq);
    Object.assign(out, { L22, ok: true, L: [[L11, 0], [L21, L22]] });
    if (b) {
      const w1 = b[0] / L11, w2 = (b[1] - L21 * w1) / L22;
      const z2 = w2 / L22, z1 = (w1 - L21 * z2) / L11;
      Object.assign(out, { w: [w1, w2], z: [z1, z2] });
    }
    return out;
  }

  // ------------------------------------------------------------------ part 3
  // SVD of a 2x2 matrix: sigma_1 from the eigenvalues of A^T A, sigma_2 = |det A| / sigma_1,
  // u_2 = s * rot90(u_1) with s = sign(det A det V) (as the Python twin).
  function svd2(A) {
    const AtA = N.matmul(N.transpose(A), A);
    const E = N.eigSym(AtA); // values descending
    const vs = E.vectors.map((v) => fixSign(v));
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    const s1 = Math.sqrt(Math.max(E.values[0], 0));
    const s2 = s1 > 0 ? Math.abs(det) / s1 : 0;
    const u1 = s1 > 0 ? N.matvec(A, vs[0]).map((x) => x / s1) : [1, 0];
    const detV = vs[0][0] * vs[1][1] - vs[0][1] * vs[1][0];
    const sg = det * detV < 0 ? -1 : 1;
    const u2 = [-sg * u1[1], sg * u1[0]];
    const cols = (a, b) => [[a[0], b[0]], [a[1], b[1]]];
    return {
      S: [s1, s2], U: cols(u1, u2), V: cols(vs[0], vs[1]),
      cond: s2 > 0 ? s1 / s2 : Infinity, det, AtA_eigs: [E.values[0], E.values[1]],
    };
  }

  // Two unit rows phiDeg apart (the first at 45 deg), z* = (1, 1), then b_2 moved by db.
  function twoRows(phiDeg, db) {
    const a1 = rad(45), a2 = rad(45 + phiDeg);
    const A = [[Math.cos(a1), Math.sin(a1)], [Math.cos(a2), Math.sin(a2)]];
    const b = [A[0][0] + A[0][1], A[1][0] + A[1][1] + db];
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (Math.abs(det) < 1e-12) return { A, b, cond: Infinity, z: null, shift: null };
    const z = [(b[0] * A[1][1] - A[0][1] * b[1]) / det, (A[0][0] * b[1] - A[1][0] * b[0]) / det];
    return { A, b, cond: N.cond(A), z, shift: Math.hypot(z[0] - 1, z[1] - 1) };
  }

  // The drawer: modeled rail constraint (0, 1) . v = 0, true rail turned by epsDeg, hand
  // command (cos th, sin th) . v = b.
  function drawer(thetaDeg, epsDeg, b) {
    epsDeg = epsDeg || 0;
    b = b === undefined ? 1 : b;
    const th = rad(thetaDeg), ep = rad(epsDeg);
    const c = [Math.cos(th), Math.sin(th)];
    const Am = [[0, 1], c];
    const At = [[-Math.sin(ep), Math.cos(ep)], c];
    const out = {
      cond: N.cond(Am), cond_true: N.cond(At),
      cond_formula: Math.tan(rad(45 + thetaDeg / 2)),
      speed_model: Math.abs(Math.cos(th)) > 1e-12 ? b / Math.cos(th) : null,
      T: [[Math.sin(th), -Math.cos(th)], [Math.cos(th), Math.sin(th)]],
    };
    const ct = Math.cos(th - ep);
    if (Math.abs(ct) < 1e-12) return Object.assign(out, { feasible: false, v_true: null, speed_true: null, ratio: null });
    const v = [(b / ct) * Math.cos(ep), (b / ct) * Math.sin(ep)];
    return Object.assign(out, {
      feasible: true, v_true: v, speed_true: b / ct,
      ratio: out.speed_model !== null ? Math.cos(th) / ct : null,
    });
  }

  // Hou & Mason 2021 eq. 9: cond([J_hat; C_hat]).
  function crashingIndex(J, C) {
    const Jh = N.rowspace(J);
    const Ch = C.map((r) => r.map((x) => x / hyp(r)));
    return { raw: N.cond(N.vstack(J, C)), index: N.cond(N.vstack(Jh, Ch)), rows_J_hat: Jh.length };
  }

  // ------------------------------------------------------------------ part 4
  // Least-squares line y = c + d t by the normal equations.
  function fitLine(t, y) {
    const n = t.length;
    const st = t.reduce((s, x) => s + x, 0), stt = dot(t, t), sy = y.reduce((s, x) => s + x, 0), sty = dot(t, y);
    const AtA = [[n, st], [st, stt]], Aty = [sy, sty];
    const det = n * stt - st * st;
    const c = (stt * sy - st * sty) / det, d = (n * sty - st * sy) / det;
    const e = y.map((yi, i) => yi - c - d * t[i]);
    return {
      AtA, Aty, c, d, residuals: e, sse: dot(e, e),
      Ate: [e.reduce((s, x) => s + x, 0), dot(t, e)],
    };
  }

  // Four-legged table: minimum-norm leg forces and the line of all equilibrium distributions.
  function tableForces(W, com, legs, t) {
    W = W === undefined ? 100 : W;
    com = com || [0.2, 0.1];
    legs = legs || [[1, 1], [-1, 1], [-1, -1], [1, -1]];
    t = t || 0;
    const M = [legs.map(() => 1), legs.map((p) => p[0]), legs.map((p) => p[1])];
    const rhs = [W, W * com[0], W * com[1]];
    const MMt = N.matmul(M, N.transpose(M));
    const fp = N.matvec(N.transpose(M), N.solve(MMt, rhs));
    let nv = N.nullspace(M)[0];
    const m = Math.max(...nv.map(Math.abs));
    nv = nv.map((x) => x / m);
    const f = fp.map((x, i) => x + t * nv[i]);
    let lo = -Infinity, hi = Infinity;
    nv.forEach((ni, i) => {
      if (ni > 1e-12) lo = Math.max(lo, -fp[i] / ni);
      if (ni < -1e-12) hi = Math.min(hi, -fp[i] / ni);
    });
    return {
      MMt, f_plus: fp, norm_plus: hyp(fp), null: nv, f, norm: hyp(f),
      residual: N.matvec(M, f).map((x, i) => x - rhs[i]), t_range: [lo, hi],
    };
  }

  // Minimize 1/2 (a x^2 + 2 b x u + c u^2) over u: s = a - b^2 / c, u* = -(b / c) x.
  function schur2(a, b, c) {
    const tr = a + c, det = a * c - b * b;
    const r = Math.sqrt(Math.max((tr * tr) / 4 - det, 0));
    const eigs = [tr / 2 + r, tr / 2 - r];
    if (c <= 0) return { s: null, gain: null, eigs, pd: false, c_pos: false };
    const s = a - (b * b) / c;
    return { s, gain: -b / c, eigs, pd: s > 0, c_pos: true };
  }

  window.EigenLS = {
    projectLine, gramSchmidt, eig2, fromEigen, quadAxes, chol2,
    svd2, twoRows, drawer, crashingIndex, fitLine, tableForces, schur2, fixSign,
  };
})();
