/* 03_derivatives.js: the algorithms behind the live figures of deck 03
 * (03_derivatives_and_jacobians.html, "Derivatives, gradients, and Jacobians").
 *
 * Load after lib/num.js:
 *   <script src="lib/num.js"></script>
 *   <script src="lib/algo/03_derivatives.js"></script>
 * It defines one global, window.Derivatives. Every function with a Python counterpart in
 * tutorial/00_math_toolkit.py (section "Deck 03", names d03_*) is checked against it by
 * tools/twins/03.json (uv run tools/twins.py 03). Angles are in radians unless a name says deg.
 * The quartic and the valley function use multiplications only (z * z * z * z), in the same
 * order as the Python, so both produce the same doubles.
 *
 * One variable (the quartic J(z) = z^4 - 4 z^2 + z + 4)
 *   quartic(z)                    J(z).                                        (d03_quartic)
 *   quarticDerivs(z)              [J, J', J'', J''', J''''].                   (d03_quartic_derivs)
 *   secantSlope(z0, h)            (J(z0 + h) - J(z0)) / h.                     (d03_secant_slope)
 *   secantTable(z0, hs)           [{h, slope, error}].                         (d03_secant_table)
 *   stationaryPoints()            [{z, J, d2J}] at the three roots of J'.      (d03_stationary_points)
 *   inflectionPoints()            [-sqrt(2/3), sqrt(2/3)].                     (d03_inflection_points)
 *   taylorCoeffs(z0)              [J^(k)(z0) / k!], k = 0..4.                  (d03_taylor_coeffs)
 *   taylorPoly(z0, delta, n)      degree-n Taylor polynomial at z0 + delta.    (d03_taylor_poly)
 *   taylorErrorTable(z0, deltas)  [{delta, J, lin, quad, err_lin, err_quad, ratio_lin, ratio_quad}].
 *                                                                              (d03_taylor_error_table)
 *   sincosCheck(theta, h)         {dcos_fd, minus_sin, dsin_fd, cos}.          (d03_sincos_check)
 *   rotationDerivative(theta)     {dR, RS}.                                    (d03_rotation_derivative)
 * Two variables (the valley function J(z) = z1^4 - 4 z1^2 + z1 + 4 + 2 (z2 - z1)^2)
 *   valley(z), valleyGrad(z), valleyHess(z)                     (d03_valley, _grad, _hess)
 *   valleyStationary()            [{z, J, eig}].                               (d03_valley_stationary)
 *   partialSlices(z)              {J, grad, fd}.                               (d03_partial_slices)
 *   directional(z, phiDeg)        {d, D, fd, norm_grad, cos}.                  (d03_directional_derivative)
 *   steepest(z)                   {ascent, descent, level, rate} (deg).        (d03_steepest_directions)
 *   levelTangent(z)               {tangent, dot, J, J_plus, J_minus}.          (d03_level_tangent)
 *   sliceCurvature(z, phiDeg)     {g1, g2, fd2}.                               (d03_slice_curvature)
 *   hessianEigen(z)               {values (descending), vectors}.              (d03_hessian_eigen)
 *   valleyModels(z0, delta)       {J, lin, quad}.                              (d03_valley_models)
 *   quadModel(z0)                 function delta -> quadratic model value (drawing helper).
 * Matrix calculus
 *   fdGradient(f, x, h)           central-difference gradient.                 (d03_fd_gradient)
 *   gradLinear(b, x)              {f, grad, fd}.                               (d03_grad_linear)
 *   gradQuadratic(A, x)           {f, grad, hess, fd}.                         (d03_grad_quadratic)
 *   gradLeastSquares(A, b, x)     {f, r, grad, hess, AtB, xstar, fd}.          (d03_grad_least_squares)
 * Jacobians, linearization, constraints
 *   armFK(theta, l), armJacobian(theta, l) -> {J, det}                         (d03_arm_fk, _jacobian)
 *   armFirstOrder(theta, dtheta, l) -> {true, lin}                             (d03_arm_first_order)
 *   armDistanceGradient(theta, target, l) -> {p, e, w, grad, fd}               (d03_arm_distance_gradient)
 *   pendulumStep(x, u, h), pendulumAB(xbar, ubar, h) -> {A, B, det}            (d03_pendulum_step, _AB)
 *   pendulumRollouts(dtheta0Deg, thetaBarDeg, T, h) -> {theta_nl, theta_lin, max_gap_deg}
 *                                                                              (d03_pendulum_rollouts)
 *   beadConstraint(q, qdot) -> {Phi, J, Jqdot}                                 (d03_bead_constraint)
 *   pivotConstraint(q), pivotJacobian(theta) -> {J, null}, pivotResidual(theta, qdot),
 *   pivotPose(theta)                                                           (d03_pivot_*)
 * Finite differences
 *   fdForward(z0, h), fdCentral(z0, h)                                         (d03_fd_forward, _central)
 *   fdSteps()                     91 steps 1 .. 1e-15 (7, 5, 3, 2, 1.5, 1 x 10^-e). (d03_fd_steps)
 *   fdErrorCurve(z0, hs)          {h, forward, central} absolute errors.       (d03_fd_error_curve)
 *   fdJacobian(f, x, h, central)  matrix (array of rows).                      (d03_fd_jacobian)
 *   fdPendulumAB(xbar, ubar, h, eps) -> {A_fd, B_fd, A, B, max_err}            (d03_fd_pendulum_AB)
 */
(function () {
  "use strict";
  const Num = window.Num;
  if (!Num) throw new Error("03_derivatives.js: load lib/num.js first");
  const D = {};
  const DEG = Math.PI / 180;

  // ---------------------------------------------------------------- one variable
  D.quartic = function (z) {
    return z * z * z * z - 4.0 * z * z + z + 4.0;
  };
  D.quarticDerivs = function (z) {
    return [D.quartic(z), 4.0 * z * z * z - 8.0 * z + 1.0, 12.0 * z * z - 8.0, 24.0 * z, 24.0];
  };
  D.secantSlope = function (z0, h) {
    return (D.quartic(z0 + h) - D.quartic(z0)) / h;
  };
  D.secantTable = function (z0, hs) {
    if (z0 === undefined) z0 = 1.0;
    hs = hs || [0.5, 0.1, 0.01, 0.001];
    const d = D.quarticDerivs(z0)[1];
    return hs.map((h) => ({ h, slope: D.secantSlope(z0, h), error: D.secantSlope(z0, h) - d }));
  };
  D.stationaryPoints = function () {
    return [-1.5, 0.0, 1.5].map((z) => {
      for (let it = 0; it < 60; it++) {
        const d = D.quarticDerivs(z);
        const step = d[1] / d[2];
        z -= step;
        if (Math.abs(step) < 1e-15) break;
      }
      const d = D.quarticDerivs(z);
      return { z, J: d[0], d2J: d[2] };
    });
  };
  D.inflectionPoints = function () {
    const r = Math.sqrt(2.0 / 3.0);
    return [-r, r];
  };
  D.taylorCoeffs = function (z0) {
    const d = D.quarticDerivs(z0), fact = [1, 1, 2, 6, 24];
    return d.map((v, k) => v / fact[k]);
  };
  D.taylorPoly = function (z0, delta, degree) {
    const c = D.taylorCoeffs(z0);
    let s = 0.0, p = 1.0;
    for (let k = 0; k <= degree; k++) { s += c[k] * p; p *= delta; }
    return s;
  };
  D.taylorErrorTable = function (z0, deltas) {
    if (z0 === undefined) z0 = 1.0;
    deltas = deltas || [0.2, 0.1, 0.05, 0.025];
    const rows = deltas.map((d) => {
      const J = D.quartic(z0 + d), lin = D.taylorPoly(z0, d, 1), quad = D.taylorPoly(z0, d, 2);
      return { delta: d, J, lin, quad, err_lin: J - lin, err_quad: J - quad };
    });
    for (let i = 1; i < rows.length; i++) {
      rows[i].ratio_lin = rows[i - 1].err_lin / rows[i].err_lin;
      rows[i].ratio_quad = rows[i - 1].err_quad / rows[i].err_quad;
    }
    return rows;
  };
  D.sincosCheck = function (theta, h) {
    if (h === undefined) h = 1e-5;
    return {
      dcos_fd: (Math.cos(theta + h) - Math.cos(theta - h)) / (2 * h), minus_sin: -Math.sin(theta),
      dsin_fd: (Math.sin(theta + h) - Math.sin(theta - h)) / (2 * h), cos: Math.cos(theta),
    };
  };
  D.rotationDerivative = function (theta) {
    const c = Math.cos(theta), s = Math.sin(theta);
    return { dR: [[-s, -c], [c, -s]], RS: Num.matmul([[c, -s], [s, c]], [[0, -1], [1, 0]]) };
  };

  // ---------------------------------------------------------------- two variables
  D.valley = function (z) {
    const z1 = z[0], z2 = z[1], e = z2 - z1;
    return z1 * z1 * z1 * z1 - 4.0 * z1 * z1 + z1 + 4.0 + 2.0 * e * e;
  };
  D.valleyGrad = function (z) {
    const z1 = z[0], z2 = z[1];
    return [4.0 * z1 * z1 * z1 - 8.0 * z1 + 1.0 - 4.0 * (z2 - z1), 4.0 * (z2 - z1)];
  };
  D.valleyHess = function (z) {
    const z1 = z[0];
    return [[12.0 * z1 * z1 - 4.0, -4.0], [-4.0, 4.0]];
  };
  // Eigenvalues of a symmetric 2x2 matrix, descending, with unit eigenvectors whose entry of
  // largest magnitude is positive (the notebook's and Num.eigSym's convention).
  function eig2(H) {
    const a = H[0][0], b = H[0][1], d = H[1][1];
    const m = (a + d) / 2, r = Math.hypot((a - d) / 2, b);
    const vals = [m + r, m - r];
    const vecs = vals.map((l) => {
      let v = Math.abs(b) > 1e-300 ? [b, l - a] : (Math.abs(a - l) <= Math.abs(d - l) ? [1, 0] : [0, 1]);
      if (Math.abs(b) > 1e-300 && Math.hypot(v[0], v[1]) < 1e-12 * (Math.abs(l) + 1)) v = [l - d, b];
      const n = Math.hypot(v[0], v[1]);
      v = [v[0] / n, v[1] / n];
      const k = Math.abs(v[0]) >= Math.abs(v[1]) ? 0 : 1;
      if (v[k] < 0) v = [-v[0], -v[1]];
      return v;
    });
    return { values: vals, vectors: vecs };
  }
  D.eig2 = eig2;
  D.valleyStationary = function () {
    return D.stationaryPoints().map((p) => {
      const z = [p.z, p.z];
      return { z, J: D.valley(z), eig: eig2(D.valleyHess(z)).values };
    });
  };
  D.partialSlices = function (z) {
    const h = 1e-6, z1 = z[0], z2 = z[1];
    const fd1 = (D.valley([z1 + h, z2]) - D.valley([z1 - h, z2])) / (2 * h);
    const fd2 = (D.valley([z1, z2 + h]) - D.valley([z1, z2 - h])) / (2 * h);
    return { J: D.valley(z), grad: D.valleyGrad(z), fd: [fd1, fd2] };
  };
  D.directional = function (z, phiDeg) {
    const phi = phiDeg * DEG, d = [Math.cos(phi), Math.sin(phi)], g = D.valleyGrad(z), t = 1e-6;
    const fd = (D.valley([z[0] + t * d[0], z[1] + t * d[1]]) - D.valley([z[0] - t * d[0], z[1] - t * d[1]])) / (2 * t);
    const Dd = g[0] * d[0] + g[1] * d[1], ng = Math.hypot(g[0], g[1]);
    return { d, D: Dd, fd, norm_grad: ng, cos: ng > 0 ? Dd / ng : null };
  };
  D.steepest = function (z) {
    const g = D.valleyGrad(z), a = Math.atan2(g[1], g[0]) / DEG;
    const wrap = (x) => { let y = (x + 180.0) % 360.0; if (y < 0) y += 360.0; return y - 180.0; };
    const lv = [wrap(a + 90.0), wrap(a - 90.0)].sort((p, q) => p - q);
    return { ascent: a, descent: wrap(a + 180.0), level: lv, rate: Math.hypot(g[0], g[1]) };
  };
  D.levelTangent = function (z) {
    const g = D.valleyGrad(z), n = Math.hypot(g[0], g[1]), t = [-g[1] / n, g[0] / n], s = 1e-3;
    return {
      tangent: t, dot: g[0] * t[0] + g[1] * t[1], J: D.valley(z),
      J_plus: D.valley([z[0] + s * t[0], z[1] + s * t[1]]), J_minus: D.valley([z[0] - s * t[0], z[1] - s * t[1]]),
    };
  };
  D.sliceCurvature = function (z, phiDeg) {
    const phi = phiDeg * DEG, d = [Math.cos(phi), Math.sin(phi)], H = D.valleyHess(z), g = D.valleyGrad(z), t = 1e-4;
    const at = (s) => D.valley([z[0] + s * d[0], z[1] + s * d[1]]);
    const Hd = [H[0][0] * d[0] + H[0][1] * d[1], H[1][0] * d[0] + H[1][1] * d[1]];
    return { g1: g[0] * d[0] + g[1] * d[1], g2: d[0] * Hd[0] + d[1] * Hd[1], fd2: (at(t) - 2 * at(0) + at(-t)) / (t * t) };
  };
  D.hessianEigen = function (z) {
    return eig2(D.valleyHess(z));
  };
  D.valleyModels = function (z0, delta) {
    const g = D.valleyGrad(z0), H = D.valleyHess(z0), J0 = D.valley(z0);
    const lin = J0 + g[0] * delta[0] + g[1] * delta[1];
    const q = delta[0] * (H[0][0] * delta[0] + H[0][1] * delta[1]) + delta[1] * (H[1][0] * delta[0] + H[1][1] * delta[1]);
    return { J: D.valley([z0[0] + delta[0], z0[1] + delta[1]]), lin, quad: lin + 0.5 * q };
  };
  D.quadModel = function (z0) {
    const g = D.valleyGrad(z0), H = D.valleyHess(z0), J0 = D.valley(z0);
    return (dx, dy) => J0 + g[0] * dx + g[1] * dy + 0.5 * (H[0][0] * dx * dx + 2 * H[0][1] * dx * dy + H[1][1] * dy * dy);
  };

  // ---------------------------------------------------------------- matrix calculus
  D.fdGradient = function (f, x, h) {
    if (h === undefined) h = 1e-6;
    return x.map((_, k) => {
      const xp = x.slice(), xm = x.slice();
      xp[k] += h; xm[k] -= h;
      return (f(xp) - f(xm)) / (2 * h);
    });
  };
  D.gradLinear = function (b, x) {
    return { f: Num.dot(b, x), grad: b.slice(), fd: D.fdGradient((y) => Num.dot(b, y), x) };
  };
  D.gradQuadratic = function (A, x) {
    const S = Num.add(A, Num.transpose(A));
    return { f: Num.dot(x, Num.matvec(A, x)), grad: Num.matvec(S, x), hess: S, fd: D.fdGradient((y) => Num.dot(y, Num.matvec(A, y)), x) };
  };
  D.gradLeastSquares = function (A, b, x) {
    const r = Num.sub(Num.matvec(A, x), b), At = Num.transpose(A), AtA = Num.matmul(At, A), Atb = Num.matvec(At, b);
    const f = (y) => { const s = Num.sub(Num.matvec(A, y), b); return 0.5 * Num.dot(s, s); };
    return { f: 0.5 * Num.dot(r, r), r, grad: Num.matvec(At, r), hess: AtA, AtB: Atb, xstar: Num.solve(AtA, Atb), fd: D.fdGradient(f, x) };
  };

  // ---------------------------------------------------------------- Jacobians
  const L0 = [1.0, 1.0];
  D.armFK = function (theta, l) {
    l = l || L0;
    const t1 = theta[0], t2 = theta[1];
    return [l[0] * Math.cos(t1) + l[1] * Math.cos(t1 + t2), l[0] * Math.sin(t1) + l[1] * Math.sin(t1 + t2)];
  };
  D.armJacobian = function (theta, l) {
    l = l || L0;
    const t1 = theta[0], t2 = theta[1];
    const s1 = Math.sin(t1), c1 = Math.cos(t1), s12 = Math.sin(t1 + t2), c12 = Math.cos(t1 + t2);
    const J = [[-l[0] * s1 - l[1] * s12, -l[1] * s12], [l[0] * c1 + l[1] * c12, l[1] * c12]];
    return { J, det: J[0][0] * J[1][1] - J[0][1] * J[1][0] };
  };
  D.armFirstOrder = function (theta, dtheta, l) {
    const p0 = D.armFK(theta, l), p1 = D.armFK([theta[0] + dtheta[0], theta[1] + dtheta[1]], l);
    const J = D.armJacobian(theta, l).J;
    return { true: [p1[0] - p0[0], p1[1] - p0[1]], lin: Num.matvec(J, dtheta) };
  };
  D.armDistanceGradient = function (theta, target, l) {
    const p = D.armFK(theta, l), e = [p[0] - target[0], p[1] - target[1]], J = D.armJacobian(theta, l).J;
    const w = (th) => { const q = D.armFK(th, l); return 0.5 * ((q[0] - target[0]) ** 2 + (q[1] - target[1]) ** 2); };
    return { p, e, w: 0.5 * (e[0] * e[0] + e[1] * e[1]), grad: Num.matvec(Num.transpose(J), e), fd: D.fdGradient(w, theta) };
  };

  const PEND = { g: 9.81, l: 1.0, m: 1.0 };
  D.PEND = PEND;
  D.pendulumStep = function (x, u, h) {
    if (h === undefined) h = 0.01;
    const a = -(PEND.g / PEND.l) * Math.sin(x[0]) + u / (PEND.m * PEND.l * PEND.l);
    const om1 = x[1] + h * a;
    return [x[0] + h * om1, om1];
  };
  D.pendulumAB = function (xbar, ubar, h) {
    if (h === undefined) h = 0.01;
    const k = (PEND.g / PEND.l) * Math.cos(xbar[0]), b = 1.0 / (PEND.m * PEND.l * PEND.l);
    const A = [[1.0 - h * h * k, h], [-h * k, 1.0]], B = [[h * h * b], [h * b]];
    return { A, B, det: A[0][0] * A[1][1] - A[0][1] * A[1][0] };
  };
  D.pendulumRollouts = function (dtheta0Deg, thetaBarDeg, T, h) {
    if (thetaBarDeg === undefined) thetaBarDeg = 0.0;
    if (T === undefined) T = 2.0;
    if (h === undefined) h = 0.01;
    const tb = thetaBarDeg * DEG, A = D.pendulumAB([tb, 0.0], 0.0, h).A;
    let x = [tb + dtheta0Deg * DEG, 0.0], dx = [dtheta0Deg * DEG, 0.0];
    const nl = [x[0] / DEG], lin = [(tb + dx[0]) / DEG];
    const n = Math.round(T / h);
    for (let k = 0; k < n; k++) {
      x = D.pendulumStep(x, 0.0, h);
      dx = [A[0][0] * dx[0] + A[0][1] * dx[1], A[1][0] * dx[0] + A[1][1] * dx[1]];
      nl.push(x[0] / DEG);
      lin.push((tb + dx[0]) / DEG);
    }
    let gap = 0;
    for (let k = 0; k < nl.length; k++) gap = Math.max(gap, Math.abs(nl[k] - lin[k]));
    return { theta_nl: nl, theta_lin: lin, max_gap_deg: gap };
  };

  D.beadConstraint = function (q, qdot) {
    const J = [2 * q[0], 2 * q[1]];
    return { Phi: q[0] * q[0] + q[1] * q[1] - 1.0, J, Jqdot: J[0] * qdot[0] + J[1] * qdot[1] };
  };

  const BLOCK = { a: 0.1, b: 0.05 };
  D.BLOCK = BLOCK;
  D.pivotConstraint = function (q, p0) {
    p0 = p0 || [0, 0];
    const th = q[2], c = [-BLOCK.a, -BLOCK.b], co = Math.cos(th), si = Math.sin(th);
    return [q[0] + co * c[0] - si * c[1] - p0[0], q[1] + si * c[0] + co * c[1] - p0[1]];
  };
  D.pivotJacobian = function (theta) {
    const c = [-BLOCK.a, -BLOCK.b], co = Math.cos(theta), si = Math.sin(theta);
    const col = [-si * c[0] - co * c[1], co * c[0] - si * c[1]];
    return { J: [[1.0, 0.0, col[0]], [0.0, 1.0, col[1]]], null: [-col[0], -col[1], 1.0] };
  };
  D.pivotResidual = function (theta, qdot) {
    return Num.matvec(D.pivotJacobian(theta).J, qdot);
  };
  D.pivotPose = function (theta) {
    const c = [-BLOCK.a, -BLOCK.b], co = Math.cos(theta), si = Math.sin(theta);
    return [-(co * c[0] - si * c[1]), -(si * c[0] + co * c[1]), theta];
  };
  // World coordinates of the four corners of the block at configuration q (drawing helper),
  // counter-clockwise from the lower-left corner.
  D.pivotCorners = function (q) {
    const co = Math.cos(q[2]), si = Math.sin(q[2]), a = BLOCK.a, b = BLOCK.b;
    return [[-a, -b], [a, -b], [a, b], [-a, b]].map(([u, v]) => [q[0] + co * u - si * v, q[1] + si * u + co * v]);
  };

  // ---------------------------------------------------------------- finite differences
  D.fdForward = function (z0, h) {
    return (D.quartic(z0 + h) - D.quartic(z0)) / h;
  };
  D.fdCentral = function (z0, h) {
    return (D.quartic(z0 + h) - D.quartic(z0 - h)) / (2.0 * h);
  };
  D.fdSteps = function () {
    const hs = [];
    for (let e = 0; e < 16; e++) {
      for (const m of ["7", "5", "3", "2", "1.5", "1"]) {
        const h = parseFloat(m + "e-" + e);
        if (h <= 1.0) hs.push(h);
      }
    }
    return hs;
  };
  D.fdErrorCurve = function (z0, hs) {
    if (z0 === undefined) z0 = 1.0;
    hs = hs || D.fdSteps();
    const d = D.quarticDerivs(z0)[1];
    return {
      h: hs.slice(), forward: hs.map((h) => Math.abs(D.fdForward(z0, h) - d)),
      central: hs.map((h) => Math.abs(D.fdCentral(z0, h) - d)),
    };
  };
  D.fdJacobian = function (f, x, h, central) {
    if (h === undefined) h = 1e-6;
    const f0 = f(x), cols = [];
    for (let j = 0; j < x.length; j++) {
      const xp = x.slice(); xp[j] += h;
      if (central) {
        const xm = x.slice(); xm[j] -= h;
        const a = f(xp), b = f(xm);
        cols.push(a.map((v, i) => (v - b[i]) / (2 * h)));
      } else {
        const a = f(xp);
        cols.push(a.map((v, i) => (v - f0[i]) / h));
      }
    }
    return Num.transpose(cols);
  };
  D.fdPendulumAB = function (xbar, ubar, h, eps) {
    if (ubar === undefined) ubar = 0.0;
    if (h === undefined) h = 0.01;
    if (eps === undefined) eps = 1e-6;
    const JF = D.fdJacobian((v) => D.pendulumStep([v[0], v[1]], v[2], h), [xbar[0], xbar[1], ubar], eps);
    const an = D.pendulumAB(xbar, ubar, h);
    const exact = [[an.A[0][0], an.A[0][1], an.B[0][0]], [an.A[1][0], an.A[1][1], an.B[1][0]]];
    let err = 0;
    for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) err = Math.max(err, Math.abs(JF[i][j] - exact[i][j]));
    return { A_fd: JF.map((r) => r.slice(0, 2)), B_fd: JF.map((r) => r.slice(2)), A: an.A, B: an.B, max_err: err };
  };

  window.Derivatives = D;
})();
