// Random degenerate test problems for tools/twins/num.json, drawn from Num.rng(seed).
// num_ref.py's gen_lp / gen_qp repeat every draw in the same order with the Mulberry32 port,
// so both sides build identical problems. All entries are small integers (plus 0.05 on the
// diagonal of some H), and 60-70 % of the rows pass through one integer vertex v, which makes
// the vertices degenerate. Load after lib/num.js.
window.NUM_GEN = (function () {
  const ints = (g, n, k, off) => Array.from({ length: n }, () => g.int(k) - off);
  const dotI = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

  // LP: n = 2..5 variables, 1..7 inequality rows, equality rows when seed % 3 == 0 (a doubled
  // copy of the first when seed % 5 == 0), bounds by seed % 4: default x >= 0, free,
  // per-variable [-3, 3] / [null, 3] / [-3, null], or [-4, 4].
  function lp(seed) {
    const g = Num.rng(seed);
    const n = 2 + g.int(4), m = 1 + g.int(7);
    const v = ints(g, n, 5, 2);
    const Aub = Array.from({ length: m }, () => ints(g, n, 7, 3));
    const bub = Aub.map((a) => dotI(a, v) + (g.uniform() < 0.6 ? 0 : 1 + g.int(2)));
    const p = { c: ints(g, n, 7, 3), Aub, bub };
    if (seed % 3 === 0) {
      const E = Array.from({ length: 1 + g.int(2) }, () => ints(g, n, 5, 2));
      if (seed % 5 === 0) E.push(E[0].map((x) => 2 * x));
      p.Aeq = E;
      p.beq = E.map((e) => dotI(e, v));
    }
    const kind = seed % 4;
    if (kind === 1) p.bounds = [null, null];
    else if (kind === 2) p.bounds = Array.from({ length: n }, () => [[-3, 3], [null, 3], [-3, null]][g.int(3)]);
    else if (kind === 3) p.bounds = [-4, 4];
    return p;
  }

  // QP: n = 2..4, H = B B^T (+ 0.05 I unless seed % 3 == 0, so some H are only semidefinite),
  // n + 1 .. n + 3 rows (70 % through v) plus the box |x_j - v_j| <= 3; seed % 4 == 1 adds a
  // copy of row 0 and twice row 1; seed % 4 == 2 adds row 0 twice as an equality.
  function qp(seed) {
    const g = Num.rng(seed);
    const n = 2 + g.int(3);
    const B = Array.from({ length: n }, () => ints(g, n, 5, 2));
    const H = Num.matmul(B, Num.transpose(B));
    if (seed % 3 !== 0) for (let i = 0; i < n; i++) H[i][i] += 0.05;
    const v = ints(g, n, 5, 2);
    const Aub = [], bub = [];
    const m = n + 1 + g.int(3);
    for (let i = 0; i < m; i++) {
      const a = ints(g, n, 7, 3);
      if (a.every((x) => x === 0)) a[0] = 1;
      Aub.push(a);
      bub.push(dotI(a, v) + (g.uniform() < 0.7 ? 0 : 1 + g.int(2)));
    }
    for (let j = 0; j < n; j++) {
      const e = new Array(n).fill(0);
      e[j] = 1;
      Aub.push(e, e.map((x) => -x));
      bub.push(v[j] + 3, -v[j] + 3);
    }
    if (seed % 4 === 1) {
      Aub.push(Aub[0].slice(), Aub[1].map((x) => 2 * x));
      bub.push(bub[0], 2 * bub[1]);
    }
    const p = { H, g: ints(g, n, 13, 6), Aub, bub };
    if (seed % 4 === 2) {
      p.Aeq = [Aub[0].slice(), Aub[0].slice()];
      p.beq = [bub[0], bub[0]];
    }
    return p;
  }

  return { lp, qp };
})();
