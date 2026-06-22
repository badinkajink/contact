# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.23.3",
#     "numpy",
#     "matplotlib",
# ]
# ///
"""Companion notebook for Part I (Ch. 1): Optimization Fundamentals.

Every numerical table in the tutorial's optimization chapter is
reproduced and verified here: gradient descent vs. Newton on the
quartic, backtracking line search, penalty and log-barrier methods,
Lagrangian duality, and a 2D LCP solved by mode enumeration.

Run locally:        marimo edit 01_optimization_fundamentals.py
Export for web:     marimo export html-wasm 01_optimization_fundamentals.py -o site/
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Optimization Fundamentals — Numerical Companion

    This notebook verifies every worked example in **Chapter 1** of the
    trajectory-optimization tutorial. The running test function is the quartic

    $$J(z) = z^4 - 4z^2 + z + 4,$$

    which has a global minimum near $z\approx -1.47$, a local minimum near
    $z \approx 1.35$, and a local max near $z \approx 0.13$ — enough structure
    to exhibit every pathology we care about.
    """)
    return


@app.cell
def _():
    # The running quartic and its derivatives
    def J(z):
        return z**4 - 4 * z**2 + z + 4

    def dJ(z):
        return 4 * z**3 - 8 * z + 1

    def d2J(z):
        return 12 * z**2 - 8

    return J, d2J, dJ


@app.cell
def _(J, np, plt):
    _z = np.linspace(-2.5, 2.5, 400)
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.plot(_z, J(_z), lw=2)
    _ax.set_xlabel("z"); _ax.set_ylabel("J(z)")
    _ax.set_title("The running quartic: two minima and a saddle-like local max")
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Gradient descent vs. Newton's method

    Both start at $z_0 = 2$. Watch the *convergence-rate signature*:
    gradient descent shrinks $|J'|$ by a roughly constant factor
    (linear convergence), while Newton roughly **squares** the error
    each step (quadratic convergence — the exponent doubles).
    """)
    return


@app.cell
def _(d2J, dJ):
    def run_gradient_descent(z0=2.0, alpha=0.05, iters=12):
        rows, z = [], z0
        for k in range(iters):
            g = dJ(z)
            rows.append((k, z, g))
            z = z - alpha * g
        return rows

    def run_newton(z0=2.0, iters=6):
        rows, z = [], z0
        for k in range(iters):
            g, H = dJ(z), d2J(z)
            rows.append((k, z, g, H))
            z = z - g / H
        return rows

    gd_rows = run_gradient_descent()
    newton_rows = run_newton()
    return gd_rows, newton_rows


@app.cell
def _(gd_rows, mo, newton_rows):
    _gd = "\n".join(f"| {k} | {z:.4f} | {g:.4f} |" for k, z, g in gd_rows[:8])
    _nw = "\n".join(
        f"| {k} | {z:.4f} | {g:.4g} | {H:.2f} |" for k, z, g, H in newton_rows
    )
    mo.md(
        f"""
    **Gradient descent** (α = 0.05) — compare tutorial Worked Example 1.2:

    | k | z_k | J'(z_k) |
    |---|------|---------|
    {_gd}

    **Newton** — compare tutorial Worked Example 1.2b (note |J'|: 17 → 4.0 → 0.62 → 0.028 → 1e-4, digits doubling):

    | k | z_k | J'(z_k) | J''(z_k) |
    |---|------|---------|----------|
    {_nw}
    """
    )
    return


@app.cell
def _(d2J, dJ, mo):
    # Newton's failure mode: start near the local MAX at z≈0.13
    _z = 0.0
    _trace = []
    for _k in range(4):
        _trace.append((_k, _z, d2J(_z)))
        _z = _z - dJ(_z) / d2J(_z)
    _rows = "\n".join(f"| {k} | {z:.4f} | {H:.2f} |" for k, z, H in _trace)
    mo.md(
        f"""
        ## 2. Newton's failure mode: negative curvature

        Starting from $z_0 = 0$, where $J''(0) = -8 < 0$, Newton happily
        converges to the **local maximum** — it finds stationary points, not minima:

        | k | z_k | J''(z_k) |
        |---|------|----------|
        {_rows}

        The fix (Levenberg–Marquardt): replace $H$ with $H + \\mu$ until positive.
        This same trick regularizes $Q_{{uu}}$ in iLQR.
        """
    )
    return


@app.cell
def _(J, dJ, mo):
    # Backtracking / Armijo line search demo from the tutorial
    def armijo_accepts(z, d, alpha, c1=1e-4):
        return J(z + alpha * d) <= J(z) + c1 * alpha * dJ(z) * d

    _z0, _d = 2.0, -dJ(2.0)  # steepest descent direction = -17
    _lines = []
    for _a in [0.3, 0.2, 0.1, 0.05]:
        _znew = _z0 + _a * _d
        _lines.append(
            f"| {_a} | {_znew:.2f} | {J(_znew):.2f} | "
            f"{'accept' if armijo_accepts(_z0, _d, _a) else 'REJECT'} |"
        )
    _tab = "\n".join(_lines)
    mo.md(
        f"""
        ## 3. Backtracking line search (Armijo)

        At $z_0=2$, direction $d=-J'(2)=-17$. Accept iff
        $J(z+\\alpha d) \\le J(z) + c_1 \\alpha J'(z) d$, $c_1 = 10^{{-4}}$:

        | α | z_new | J(z_new) | Armijo |
        |---|-------|----------|--------|
        {_tab}

        Matches the tutorial: α = 0.3 overshoots into the far wall and is rejected;
        smaller steps pass. iLQR's forward pass is this test in trajectory space.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Constrained optimization: penalty vs. barrier

    Problem (tutorial Example 1.3):
    $\min z_1^2 + z_2^2 \;\; \text{s.t.} \;\; z_1 + z_2 \ge 1$,
    with solution $z^\star = (1/2, 1/2)$, multiplier $\mu^\star = 1$.

    - **Quadratic penalty** approaches from the *infeasible* side,
      violation $\approx 1/\rho$ — this is exactly MuJoCo-style soft
      contact (penetration ↔ violation, stiffness ↔ ρ).
    - **Log barrier** approaches from the *strictly feasible* side along
      the central path, error $\approx 1/(4t)$ — this is IPOPT.
    """)
    return


@app.cell
def _(mo, np):
    # closed-form stationary points derived in the tutorial, verified numerically
    def penalty_z(rho):  # min z1²+z2² + (ρ/2)max(1-z1-z2,0)², symmetric z
        return rho / (2 + 2 * rho)

    def barrier_z(t):  # min z1²+z2² - (1/t)ln(z1+z2-1), symmetric z
        return 0.25 * (1 + np.sqrt(1 + 4 / t))

    _pen = "\n".join(
        f"| {r} | {penalty_z(r):.4f} | {1 - 2*penalty_z(r):+.4f} |"
        for r in [1, 10, 100, 1000]
    )
    _bar = "\n".join(
        f"| {t} | {barrier_z(t):.5f} | {barrier_z(t)-0.5:.5f} |"
        for t in [1, 10, 100, 1000]
    )
    mo.md(
        f"""
    | ρ (penalty) | z | constraint violation |
    |---|---|---|
    {_pen}

    | t (barrier) | z | distance to z* |
    |---|---|---|
    {_bar}

    Both columns match the tutorial tables; note opposite approach directions.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Lagrangian duality, verified

    Dual function for Example 1.3: $d(\mu) = \mu - \mu^2/2$, maximized at
    $\mu^\star = 1$ with $d^\star = 1/2 = J^\star$ (strong duality).
    """)

    return


@app.cell
def _(np):
    _mu = np.linspace(0, 2, 100)
    _d = _mu - _mu**2 / 2
    _i = np.argmax(_d)
    print(f"argmax_mu d(mu) = {_mu[_i]:.3f},  d* = {_d[_i]:.3f}  (primal J* = 0.5)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. A 2D LCP by mode enumeration

    Solve $0 \le \lambda \perp G\lambda + g \ge 0$ with
    $G = \begin{bmatrix}2&1\\1&2\end{bmatrix}$, $g = (-1, -3)$
    (tutorial Worked Example 1.4a). With $n$ components there are $2^n$
    modes; each mode guess reduces the LCP to a linear solve.
    """)
    return


@app.cell
def _(np):
    def solve_lcp_enumeration(G, g, tol=1e-12):
        """Brute-force LCP solver: try all 2^n active sets."""
        n = len(g)
        results = []
        for mask in range(2**n):
            active = [bool(mask >> i & 1) for i in range(n)]  # active: λ_i free, w_i=0
            lam = np.zeros(n)
            idx = [i for i in range(n) if active[i]]
            if idx:
                Gaa = G[np.ix_(idx, idx)]
                lam[idx] = np.linalg.solve(Gaa, -g[idx])
            w = G @ lam + g
            ok = np.all(lam >= -tol) and np.all(w >= -tol) and abs(lam @ w) < 1e-9
            results.append((active, lam.copy(), w.copy(), ok))
        return results

    G_demo = np.array([[2.0, 1.0], [1.0, 2.0]])
    g_demo = np.array([-1.0, -3.0])
    lcp_modes = solve_lcp_enumeration(G_demo, g_demo)
    return (lcp_modes,)


@app.cell
def _(lcp_modes, mo):
    _rows = []
    for active, lam, w, ok in lcp_modes:
        name = ", ".join("active" if a else "inactive" for a in active)
        _rows.append(
            f"| ({name}) | ({lam[0]:.3g}, {lam[1]:.3g}) | "
            f"({w[0]:.3g}, {w[1]:.3g}) | {'✓ SOLUTION' if ok else '×'} |"
        )
    _tab = "\n".join(_rows)
    mo.md(
        f"""
    | mode (λ₁, λ₂) | λ | w = Gλ + g | valid? |
    |---|---|---|---|
    {_tab}

    Exactly one of the four modes is consistent: $\\lambda^\\star = (0, 3/2)$ —
    contact 2 carries the load, contact 1 separates. Three modes had to be
    tried and rejected; with friction this becomes $3^{{n_c}}$ modes, the
    combinatorics every contact solver is implicitly searching.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    **Next:** `02_lqr_ilqr_sliding_block.py` applies all of this along a
    trajectory: the Riccati recursion is Newton's method + Schur complements,
    time step by time step.
    """)
    return


if __name__ == "__main__":
    app.run()
