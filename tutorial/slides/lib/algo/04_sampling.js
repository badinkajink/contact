/* 04_sampling.js: the algorithms behind the live figures of deck 04
 * (04_probability_and_sampling.html, "Probability, Gaussians, and sampling").
 *
 * Load after lib/num.js:
 *   <script src="lib/num.js"></script>
 *   <script src="lib/algo/04_sampling.js"></script>
 * It defines one global, window.Sampling. Every function with a Python counterpart in
 * tutorial/00_math_toolkit.py (section "Deck 04", names d04_*) is checked against it by
 * tools/twins/04.json (uv run tools/twins.py 04). Random draws come from Num.rng(seed) in the
 * order the notebook's D04Rng reproduces: a function that takes a seed makes its own generator,
 * draws uniforms or normals in one sequence, and never mixes the two kinds.
 *
 * Discrete probability
 *   twoDiceCounts()                 {2: 1, ..., 7: 6, ..., 12: 1}          (d04_two_dice_counts)
 *   maxTwoDiceCounts()              {1: 1, 2: 3, ..., 6: 11}               (d04_max_two_dice_counts)
 *   pmfMoments(values, probs)       {mean, var, std, EX2}                  (d04_pmf_moments)
 *   atLeastOne(p, k)                1 - (1 - p)^k                          (d04_at_least_one)
 *   dieRolls(n, seed, dice = 1)     n sums of `dice` dice, 1 + floor(6 u)  (d04_die_rolls)
 *   sampleStats(x)                  {n, mean, var (1/N), var_unbiased}     (d04_sample_stats)
 *   runningMean(x)                  mean of the first N values             (d04_running_mean)
 * Continuous distributions
 *   uniformMoments(a, b)            {height, mean, var, std}               (d04_uniform_moments)
 *   gaussPdf(x, m = 0, s = 1)       Gaussian density                       (d04_gauss_pdf)
 *   erf(x), erfc(x)                 to about 1e-15 (series, continued fraction)
 *   normalCdf(x)                    P(Z <= x)                              (d04_normal_cdf)
 *   probWithin(k)                   erf(k / sqrt 2)                        (d04_prob_within)
 *   uniformStream(seed, n)          first n uniforms of Num.rng(seed)      (d04_uniform_stream)
 *   boxMuller(u1, u2)               {r, theta, deg, z1, z2}                (d04_box_muller)
 *   radiusTail(r)                   exp(-r^2 / 2)                          (d04_radius_tail)
 *   radiusTailSamples(n, seed, rs)  fractions of n 2D draws beyond each r  (d04_radius_tail_samples)
 *   normalPairs(n, seed)            n points (z1, z2) of Num.rng(seed) normals, in order
 * Random vectors
 *   sampleCov(points, unbiased)     {mean, cov, corr}                      (d04_sample_cov)
 *   directionalVariance(S, deg)     a^T S a                                (d04_directional_variance)
 *   projectedVariance(S, deg, n, seed)  {theory, sample}                   (d04_projected_variance)
 *   covEllipse(S)                   {var, std, angle, v1}                  (d04_cov_ellipse)
 *   covFromAxes(s1, s2, deg)        V diag(s1^2, s2^2) V^T                 (d04_cov_from_axes)
 *   ellipseProb(k)                  1 - exp(-k^2 / 2)                      (d04_ellipse_prob)
 *   ellipseFractions(s1, s2, deg, n, seed, ks)                             (d04_ellipse_fractions)
 *   chol2(S)                        [[l11, 0], [l21, l22]] or null         (d04_chol2)
 *   mvnSamples(m, S, n, seed)       n points m + L eps                     (d04_mvn_samples)
 *   mvnPdf(x, m, S)                 2D Gaussian density                    (d04_mvn_pdf)
 *   ellipsePoints(m, S, k, n = 120) closed polyline of the k-sigma ellipse (drawing helper)
 * Monte Carlo
 *   mcPi(n, seed)                   {n, inside, estimate}                  (d04_mc_pi)
 *   mcPiRms(ns, trials, seed0)      RMS error per N                        (d04_mc_pi_rms)
 *   smoothedStep(z, sigma)          Phi(-z / sigma)                        (d04_smoothed_step)
 *   smoothedStepMC(zs, sigma, n, seed)                                     (d04_smoothed_step_mc)
 * Weights and sampling-based search
 *   weightedMean(x, w)                                                     (d04_weighted_mean)
 *   softmaxWeights(J, lam)          shifted by min J                       (d04_softmax_weights)
 *   ess(w)                          1 / sum w^2                            (d04_ess)
 *   lambdaForEss(J, target, lo, hi, iters)   bisection on log lambda       (d04_lambda_for_ess)
 *   tailEstimates(n, seed, shift, t)         plain and importance sampling (d04_tail_estimates)
 *   tailRelativeStd(n, shift, t)                                           (d04_tail_relative_std)
 *   eliteFit(z, J, frac)            {K, idx, mean, cov}; z numbers or points (d04_elite_fit)
 *   cemCost(z)                      two-basin cost of slide elite-refit-live (d04_cem_cost)
 *   cemRun({m0, s0, frac, n, iters, seed})   history; each record also keeps its samples and
 *                                   costs for drawing                      (d04_cem_run)
 *   quartic(z), quarticStationary() roots of 4 z^3 - 8 z + 1              (d04_quartic_stationary)
 *   restartSuccess(k, lo, hi, trials, seed)                                (d04_restart_success)
 *   hou2021TestCount()              {planar, spatial, problems}            (d04_hou2021_test_count)
 *   directions(n, seed, method)     angles; method 'gauss' or 'square'     (d04_directions)
 *   diagonalFraction(ang, halfDeg)                                         (d04_diagonal_fraction)
 */
(function () {
  "use strict";
  const SQ2 = Math.SQRT2, SQPI = Math.sqrt(Math.PI);
  const range = (n) => Array.from({ length: n }, (_, i) => i);
  const sum = (a) => a.reduce((s, x) => s + x, 0);

  // ------------------------------------------------------------------ discrete probability
  function twoDiceCounts() {
    const c = {};
    for (let s = 2; s <= 12; s++) c[s] = 0;
    for (let a = 1; a <= 6; a++) for (let b = 1; b <= 6; b++) c[a + b]++;
    return c;
  }
  function maxTwoDiceCounts() {
    const c = {};
    for (let k = 1; k <= 6; k++) c[k] = 0;
    for (let a = 1; a <= 6; a++) for (let b = 1; b <= 6; b++) c[Math.max(a, b)]++;
    return c;
  }
  function pmfMoments(values, probs) {
    const x = Array.from(values), p = Array.from(probs);
    const m = sum(x.map((v, i) => p[i] * v));
    const v = sum(x.map((t, i) => p[i] * (t - m) * (t - m)));
    return { mean: m, var: v, std: Math.sqrt(v), EX2: sum(x.map((t, i) => p[i] * t * t)) };
  }
  const atLeastOne = (p, k) => 1 - Math.pow(1 - p, k);
  function dieRolls(n, seed, dice) {
    dice = dice || 1;
    const R = Num.rng(seed), out = new Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let d = 0; d < dice; d++) s += 1 + Math.floor(6 * R.uniform());
      out[i] = s;
    }
    return out;
  }
  function sampleStats(x) {
    const n = x.length, m = sum(x) / n;
    let ss = 0;
    for (const t of x) ss += (t - m) * (t - m);
    return { n, mean: m, var: ss / n, var_unbiased: n > 1 ? ss / (n - 1) : NaN };
  }
  function runningMean(x) {
    const out = new Array(x.length);
    let s = 0;
    for (let i = 0; i < x.length; i++) { s += x[i]; out[i] = s / (i + 1); }
    return out;
  }

  // ------------------------------------------------------------------ continuous distributions
  const uniformMoments = (a, b) => ({ height: 1 / (b - a), mean: (a + b) / 2, var: (b - a) * (b - a) / 12, std: (b - a) / Math.sqrt(12) });
  function gaussPdf(x, m, s) {
    m = m || 0; s = s === undefined ? 1 : s;
    const z = (x - m) / s;
    return Math.exp(-0.5 * z * z) / (s * Math.sqrt(2 * Math.PI));
  }
  // erf by the all-positive series erf(x) = 2/sqrt(pi) e^{-x^2} sum_n (2x^2)^n x / (2n+1)!! for
  // |x| < 3, and erfc by its continued fraction beyond; both reach about 1e-15.
  function erfcCF(x) {                  // x >= 3
    let f = x;
    for (let n = 80; n >= 1; n--) f = x + (n / 2) / f;
    return Math.exp(-x * x) / (SQPI * f);
  }
  function erf(x) {
    const ax = Math.abs(x);
    if (ax >= 3) return Math.sign(x) * (1 - erfcCF(ax));
    let term = ax, s = ax;
    for (let n = 1; n < 200 && term > 1e-17 * s; n++) { term *= 2 * ax * ax / (2 * n + 1); s += term; }
    return Math.sign(x) * 2 / SQPI * Math.exp(-ax * ax) * s;
  }
  function erfc(x) {
    if (x >= 3) return erfcCF(x);
    if (x <= -3) return 2 - erfcCF(-x);
    return 1 - erf(x);
  }
  const normalCdf = (x) => 0.5 * erfc(-x / SQ2);
  const probWithin = (k) => erf(k / SQ2);
  function uniformStream(seed, n) { const R = Num.rng(seed); return range(n).map(() => R.uniform()); }
  function boxMuller(u1, u2) {
    const r = Math.sqrt(-2 * Math.log(u1)), th = 2 * Math.PI * u2;
    return { r, theta: th, deg: th * 180 / Math.PI, z1: r * Math.cos(th), z2: r * Math.sin(th) };
  }
  const radiusTail = (r) => Math.exp(-0.5 * r * r);
  function normalPairs(n, seed) {
    const R = Num.rng(seed), out = new Array(n);
    for (let i = 0; i < n; i++) { const a = R.normal(); out[i] = [a, R.normal()]; }
    return out;
  }
  function radiusTailSamples(n, seed, rs) {
    rs = rs || [1, 2, 3];
    const P = normalPairs(n, seed).map((p) => Math.hypot(p[0], p[1]));
    return rs.map((r) => P.filter((q) => q > r).length / n);
  }

  // ------------------------------------------------------------------ random vectors
  function sampleCov(points, unbiased) {
    const n = points.length;
    const m = [0, 1].map((j) => sum(points.map((p) => p[j])) / n);
    const S = [[0, 0], [0, 0]];
    for (const p of points) {
      const d = [p[0] - m[0], p[1] - m[1]];
      S[0][0] += d[0] * d[0]; S[0][1] += d[0] * d[1]; S[1][1] += d[1] * d[1];
    }
    const k = unbiased ? n - 1 : n;
    S[0][0] /= k; S[0][1] /= k; S[1][1] /= k; S[1][0] = S[0][1];
    return { mean: m, cov: S, corr: S[0][1] / Math.sqrt(S[0][0] * S[1][1]) };
  }
  function directionalVariance(S, deg) {
    const a = [Math.cos(deg * Math.PI / 180), Math.sin(deg * Math.PI / 180)];
    return a[0] * (S[0][0] * a[0] + S[0][1] * a[1]) + a[1] * (S[1][0] * a[0] + S[1][1] * a[1]);
  }
  function projectedVariance(S, deg, n, seed) {
    const a = [Math.cos(deg * Math.PI / 180), Math.sin(deg * Math.PI / 180)];
    const X = mvnSamples([0, 0], S, n, seed).map((x) => x[0] * a[0] + x[1] * a[1]);
    return { theory: directionalVariance(S, deg), sample: sampleStats(X).var };
  }
  // Closed-form eigen of a symmetric 2x2: the larger eigenvalue first, eigenvector with its
  // largest entry positive, angle in (-90, 90].
  function covEllipse(S) {
    const a = S[0][0], b = S[0][1], d = S[1][1];
    const mid = (a + d) / 2, rad = Math.hypot((a - d) / 2, b);
    const l1 = mid + rad, l2 = mid - rad;
    let v;
    if (Math.abs(b) < 1e-15 * (Math.abs(a) + Math.abs(d) + 1e-300)) v = a >= d ? [1, 0] : [0, 1];
    else if (a >= d) { v = [l1 - d, b]; } else { v = [b, l1 - a]; }
    const nv = Math.hypot(v[0], v[1]);
    v = [v[0] / nv, v[1] / nv];
    const big = Math.abs(v[0]) >= Math.abs(v[1]) ? 0 : 1;
    if (v[big] < 0) v = [-v[0], -v[1]];
    let ang = Math.atan2(v[1], v[0]) * 180 / Math.PI;
    if (ang <= -90) ang += 180;
    if (ang > 90) ang -= 180;
    return { var: [l1, l2], std: [Math.sqrt(Math.max(l1, 0)), Math.sqrt(Math.max(l2, 0))], angle: ang, v1: v };
  }
  function covFromAxes(s1, s2, deg) {
    const c = Math.cos(deg * Math.PI / 180), s = Math.sin(deg * Math.PI / 180);
    const p = s1 * s1, q = s2 * s2;
    return [[c * c * p + s * s * q, c * s * (p - q)], [c * s * (p - q), s * s * p + c * c * q]];
  }
  const ellipseProb = (k) => 1 - Math.exp(-0.5 * k * k);
  function ellipseFractions(s1, s2, deg, n, seed, ks) {
    ks = ks || [1, 2, 3];
    const c = Math.cos(deg * Math.PI / 180), s = Math.sin(deg * Math.PI / 180);
    const A = [[c * s1, -s * s2], [s * s1, c * s2]];
    const Si = Num.inv(covFromAxes(s1, s2, deg));
    const q = normalPairs(n, seed).map((e) => {
      const x = [A[0][0] * e[0] + A[0][1] * e[1], A[1][0] * e[0] + A[1][1] * e[1]];
      return x[0] * (Si[0][0] * x[0] + Si[0][1] * x[1]) + x[1] * (Si[1][0] * x[0] + Si[1][1] * x[1]);
    });
    return ks.map((k) => q.filter((t) => t <= k * k).length / n);
  }
  function chol2(S) {
    if (!(S[0][0] > 0)) return null;
    const l11 = Math.sqrt(S[0][0]), l21 = S[1][0] / l11, r = S[1][1] - l21 * l21;
    if (!(r > 0)) return null;
    return [[l11, 0], [l21, Math.sqrt(r)]];
  }
  function mvnSamples(m, S, n, seed) {
    const L = chol2(S);
    if (!L) throw new Error("Sampling.mvnSamples: covariance is not positive definite");
    return normalPairs(n, seed).map((e) => [m[0] + L[0][0] * e[0], m[1] + L[1][0] * e[0] + L[1][1] * e[1]]);
  }
  function mvnPdf(x, m, S) {
    const det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
    const d = [x[0] - m[0], x[1] - m[1]];
    const q = (S[1][1] * d[0] * d[0] - (S[0][1] + S[1][0]) * d[0] * d[1] + S[0][0] * d[1] * d[1]) / det;
    return Math.exp(-0.5 * q) / (2 * Math.PI * Math.sqrt(det));
  }
  function ellipsePoints(m, S, k, n) {
    n = n || 120;
    const L = chol2(S);
    if (!L) return [];
    return range(n + 1).map((i) => {
      const t = 2 * Math.PI * i / n, e = [k * Math.cos(t), k * Math.sin(t)];
      return [m[0] + L[0][0] * e[0], m[1] + L[1][0] * e[0] + L[1][1] * e[1]];
    });
  }

  // ------------------------------------------------------------------ Monte Carlo
  function mcPi(n, seed) {
    const R = Num.rng(seed);
    let inside = 0;
    for (let i = 0; i < n; i++) { const x = R.uniform(), y = R.uniform(); if (x * x + y * y <= 1) inside++; }
    return { n, inside, estimate: 4 * inside / n };
  }
  function mcPiRms(ns, trials, seed0) {
    trials = trials || 200; seed0 = seed0 === undefined ? 1 : seed0;
    return ns.map((n) => {
      let s = 0;
      for (let t = 0; t < trials; t++) { const e = mcPi(n, seed0 + t).estimate - Math.PI; s += e * e; }
      return Math.sqrt(s / trials);
    });
  }
  const smoothedStep = (z, sigma) => normalCdf(-z / sigma);
  function smoothedStepMC(zs, sigma, n, seed) {
    const R = Num.rng(seed), e = range(n).map(() => R.normal());
    return zs.map((z) => e.filter((t) => z + sigma * t < 0).length / n);
  }

  // ------------------------------------------------------------------ weights
  function weightedMean(x, w) { return sum(x.map((v, i) => w[i] * v)) / sum(w); }
  function softmaxWeights(J, lam) {
    const jm = Math.min(...J), e = J.map((j) => Math.exp(-(j - jm) / lam)), s = sum(e);
    return e.map((v) => v / s);
  }
  const ess = (w) => 1 / sum(w.map((v) => v * v));
  function lambdaForEss(J, target, lo, hi, iters) {
    let a = Math.log(lo || 1e-3), b = Math.log(hi || 1e3);
    iters = iters || 100;
    for (let i = 0; i < iters; i++) {
      const c = 0.5 * (a + b);
      if (ess(softmaxWeights(J, Math.exp(c))) < target) a = c; else b = c;
    }
    return Math.exp(0.5 * (a + b));
  }
  function tailEstimates(n, seed, shift, t) {
    shift = shift === undefined ? 3 : shift; t = t === undefined ? 3 : t;
    const R = Num.rng(seed);
    const x = range(n).map(() => R.normal()), y = range(n).map(() => shift + R.normal());
    let hp = 0, hi = 0, si = 0;
    for (const v of x) if (v > t) hp++;
    for (const v of y) if (v > t) { hi++; si += Math.exp(-0.5 * v * v + 0.5 * (v - shift) * (v - shift)); }
    return { plain: hp / n, hits_plain: hp, is: si / n, hits_is: hi, exact: 1 - normalCdf(t) };
  }
  function tailRelativeStd(n, shift, t) {
    shift = shift === undefined ? 3 : shift; t = t === undefined ? 3 : t;
    const p = 1 - normalCdf(t), m2 = Math.exp(shift * shift) * (1 - normalCdf(t + shift));
    const plain = Math.sqrt(p * (1 - p) / n) / p, imp = Math.sqrt((m2 - p * p) / n) / p;
    return { p, plain, is: imp, ratio_n: (plain / imp) * (plain / imp) };
  }
  function eliteFit(z, J, frac) {
    const pts = z.map((v) => (Array.isArray(v) ? v : [v]));
    const d = pts[0].length, K = Math.floor(frac * J.length + 1e-9);
    const idx = range(J.length).sort((i, j) => J[i] - J[j] || i - j).slice(0, K);
    const m = range(d).map((j) => sum(idx.map((i) => pts[i][j])) / K);
    const C = range(d).map((a) => range(d).map((b) => sum(idx.map((i) => (pts[i][a] - m[a]) * (pts[i][b] - m[b]))) / K));
    return { K, idx, mean: m, cov: C };
  }
  const CA = [1.4, 0.9], CB = [-1.4, -0.9], CPHI = 35 * Math.PI / 180;
  function cemCost(z) {
    const d0 = z[0] - CA[0], d1 = z[1] - CA[1];
    const u = Math.cos(CPHI) * d0 + Math.sin(CPHI) * d1, v = -Math.sin(CPHI) * d0 + Math.cos(CPHI) * d1;
    const qa = (u / 1.2) * (u / 1.2) + (v / 0.3) * (v / 0.3);
    const qb = 1 + 0.3 * ((z[0] - CB[0]) * (z[0] - CB[0]) + (z[1] - CB[1]) * (z[1] - CB[1]));
    return Math.min(qa, qb);
  }
  function cemRun(o) {
    o = o || {};
    const m0 = o.m0 || [0, 2], s0 = o.s0 === undefined ? 1.5 : o.s0, frac = o.frac === undefined ? 0.2 : o.frac;
    const n = o.n || 40, iters = o.iters === undefined ? 10 : o.iters, seed = o.seed === undefined ? 1 : o.seed;
    const R = Num.rng(seed);
    let m = m0.slice(), S = [[s0 * s0, 0], [0, s0 * s0]];
    const hist = [];
    for (let it = 0; it < iters; it++) {
      const L = Num.chol([[S[0][0] + 1e-12, S[0][1]], [S[1][0], S[1][1] + 1e-12]]) || [[Math.sqrt(Math.max(S[0][0], 0) + 1e-12), 0], [0, Math.sqrt(Math.max(S[1][1], 0) + 1e-12)]];
      const Z = range(n).map(() => {
        const e0 = R.normal(), e1 = R.normal();
        return [m[0] + L[0][0] * e0, m[1] + L[1][0] * e0 + L[1][1] * e1];
      });
      const J = Z.map(cemCost);
      const fit = eliteFit(Z, J, frac);
      hist.push({ mean: m.slice(), cov: S.map((r) => r.slice()), best: Math.min(...J), elites: fit.idx, samples: Z, costs: J });
      m = fit.mean; S = fit.cov;
    }
    hist.push({ mean: m.slice(), cov: S.map((r) => r.slice()), best: cemCost(m), elites: [], samples: [], costs: [] });
    return hist;
  }

  // ------------------------------------------------------------------ restarts and test problems
  const quartic = (z) => z * z * z * z - 4 * z * z + z + 4;
  // z^3 - 2 z + 1/4 = 0 has three real roots: z_k = 2 sqrt(2/3) cos(phi/3 - 2 pi k / 3).
  function quarticStationary() {
    const p = -2, q = 0.25, A = 2 * Math.sqrt(-p / 3);
    const phi = Math.acos((3 * q / (2 * p)) * Math.sqrt(-3 / p));
    const z = [0, 1, 2].map((k) => A * Math.cos(phi / 3 - 2 * Math.PI * k / 3)).sort((a, b) => a - b);
    return { z, J: z.map(quartic), J2: z.map((t) => 12 * t * t - 8) };
  }
  function restartSuccess(k, lo, hi, trials, seed) {
    lo = lo === undefined ? -2 : lo; hi = hi === undefined ? 2 : hi;
    trials = trials || 1000; seed = seed === undefined ? 1 : seed;
    const zmax = quarticStationary().z[1], p = (zmax - lo) / (hi - lo);
    const R = Num.rng(seed);
    let ok = 0;
    for (let t = 0; t < trials; t++) {
      let hit = false;
      for (let j = 0; j < k; j++) if (lo + (hi - lo) * R.uniform() < zmax) hit = true;
      if (hit) ok++;
    }
    return { zmax, p, theory: 1 - Math.pow(1 - p, k), empirical: ok / trials };
  }
  function hou2021TestCount() {
    const planar = [[1, 2, 2, 1], [2, 1, 2, 1]], spatial = [[1, 2, 3, 3], [2, 3, 3, 3], [3, 3, 3, 3]];
    const cnt = (rows) => sum(rows.map((r) => r[1] * r[2] * r[3]));
    const a = cnt(planar), b = cnt(spatial);
    return { planar: a, spatial: b, problems: 1000 * (a + b) };
  }
  function directions(n, seed, method) {
    const R = Num.rng(seed);
    return range(n).map(() => {
      let x, y;
      if (method === "gauss") { x = R.normal(); y = R.normal(); } else { x = 2 * R.uniform() - 1; y = 2 * R.uniform() - 1; }
      return Math.atan2(y, x);
    });
  }
  function diagonalFraction(ang, halfDeg) {
    halfDeg = halfDeg === undefined ? 15 : halfDeg;
    const mod = (a, b) => ((a % b) + b) % b;
    return ang.filter((a) => Math.abs(mod(a * 180 / Math.PI, 90) - 45) <= halfDeg).length / ang.length;
  }

  window.Sampling = {
    twoDiceCounts, maxTwoDiceCounts, pmfMoments, atLeastOne, dieRolls, sampleStats, runningMean,
    uniformMoments, gaussPdf, erf, erfc, normalCdf, probWithin, uniformStream, boxMuller, radiusTail,
    radiusTailSamples, normalPairs,
    sampleCov, directionalVariance, projectedVariance, covEllipse, covFromAxes, ellipseProb,
    ellipseFractions, chol2, mvnSamples, mvnPdf, ellipsePoints,
    mcPi, mcPiRms, smoothedStep, smoothedStepMC,
    weightedMean, softmaxWeights, ess, lambdaForEss, tailEstimates, tailRelativeStd, eliteFit,
    cemCost, cemRun, quartic, quarticStationary, restartSuccess, hou2021TestCount, directions,
    diagonalFraction,
  };
})();
