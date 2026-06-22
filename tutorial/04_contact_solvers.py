# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "matplotlib"]
# ///
"""Companion notebook for Part III (Ch. 7-8): Contact solvers.

Verifies: the 1D ball-on-floor LCP, the 2D coupled-contact LCP, a working
PGS solver, the closed-form second-order-cone projection (stick / slip /
separate), the sliding-block-with-friction simulation against the analytic
-mu*g deceleration, and the 1D complementarity-free (ComFree) closed-form
contact force from the tutorial's worked example (beta = 10, lambda_N = mg).

Run locally:        marimo edit 04_contact_solvers.py
Export for web:     marimo export html-wasm 04_contact_solvers.py -o site/
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        # Contact Solvers: LCP, PGS, Cone Projection, ComFree

        Contact forces are Lagrange multipliers; contact solvers are
        constrained-optimization algorithms. Everything below is Chapter 1
        machinery wearing physics clothing.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Ball on floor: the 1×1 LCP

        A unit point mass resting on the floor under gravity, time step
        $h = 0.01$. The Delassus "matrix" is $G = 1/m$, the free velocity is
        $v^f = v - hg$, and the LCP is
        $0 \le \lambda \perp G\lambda + v^f \ge 0$ in the impulse $\lambda$.
        """
    )
    return


@app.cell
def _(mo, np):
    def ball_floor_lcp(v, m=1.0, g=10.0, h=0.01):
        """Returns (impulse, post-velocity) for one normal contact."""
        G = 1.0 / m
        vf = v - h * g
        # mode 1: separated (lambda = 0): need vf >= 0
        if vf >= 0:
            return 0.0, vf
        # mode 2: active (post-velocity = 0): lambda = -vf / G
        lam = -vf / G
        return lam, 0.0

    _resting = ball_floor_lcp(v=0.0)
    _rising = ball_floor_lcp(v=+1.0)
    _falling = ball_floor_lcp(v=-1.0)
    mo.md(
        f"""
| initial v | impulse λ | post-velocity | mode |
|---|---|---|---|
| 0 (resting) | {_resting[0]:.3f} | {_resting[1]:.3f} | active — λ = h·mg supports the weight |
| +1 (rising) | {_rising[0]:.3f} | {_rising[1]:.3f} | separating — floor cannot pull |
| −1 (falling) | {_falling[0]:.3f} | {_falling[1]:.3f} | active — inelastic stop |

The complementarity does the mode selection automatically: no
`if touching:` logic, the math *is* the if-statement.
"""
    )
    return ball_floor_lcp,


@app.cell
def _(mo, np):
    mo.md(
        r"""
        ## 2. The 2D coupled LCP, by enumeration and by PGS

        $G = \begin{bmatrix}2&1\\1&2\end{bmatrix}$, $g = (-1,-3)$,
        solution $\lambda^\star = (0, 3/2)$ (tutorial Worked Example 1.4a).
        PGS reaches the same answer by per-contact "negotiation":
        $\lambda_i \leftarrow \max(0, \lambda_i - (G\lambda + g)_i / G_{ii})$.
        """
    )

    def pgs(G, g, iters=50, lam0=None):
        n = len(g)
        lam = np.zeros(n) if lam0 is None else lam0.copy()
        trace = [lam.copy()]
        for _ in range(iters):
            for i in range(n):
                w_i = G[i] @ lam + g[i]
                lam[i] = max(0.0, lam[i] - w_i / G[i, i])
            trace.append(lam.copy())
        return lam, np.array(trace)

    G2 = np.array([[2.0, 1.0], [1.0, 2.0]])
    g2 = np.array([-1.0, -3.0])
    lam_pgs, pgs_trace = pgs(G2, g2)
    print("PGS solution:", np.round(lam_pgs, 6), "  (exact: [0, 1.5])")
    print("complementarity check  lam . (G lam + g) =",
          float(lam_pgs @ (G2 @ lam_pgs + g2)))
    return G2, g2, lam_pgs, pgs, pgs_trace


@app.cell
def _(pgs_trace, plt):
    _fig, _ax = plt.subplots(figsize=(6.5, 3))
    _ax.plot(pgs_trace[:, 0], label="λ₁")
    _ax.plot(pgs_trace[:, 1], label="λ₂")
    _ax.axhline(1.5, color="gray", ls=":")
    _ax.set_xlabel("PGS sweep"); _ax.set_ylabel("impulse")
    _ax.set_title("PGS: per-contact negotiation to consensus")
    _ax.legend(); _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Projection onto the friction cone (stick / slip / separate)

        The closed-form second-order-cone projection — the workhorse inside
        ADMM contact solvers. Three branches, three physical modes.
        """
    )
    return


@app.cell
def _(np):
    def project_soc(lam_n, lam_t, mu):
        """Project (lam_n, lam_t in R^k) onto the cone ||lam_t|| <= mu*lam_n."""
        lam_t = np.atleast_1d(np.asarray(lam_t, dtype=float))
        s = np.linalg.norm(lam_t)
        if s <= mu * lam_n:                      # inside: keep (stick)
            return lam_n, lam_t, "stick (inside cone)"
        if mu * s <= -lam_n:                     # in polar cone: zero (separate)
            return 0.0, np.zeros_like(lam_t), "separate (project to tip)"
        coef = (lam_n + mu * s) / (1 + mu**2)    # surface: radial shrink (slip)
        return coef, coef * mu * lam_t / s, "slip (project to surface)"

    soc_cases = [
        project_soc(1.0, [0.3], mu=0.5),   # inside
        project_soc(-1.0, [0.2], mu=0.5),  # polar
        project_soc(1.0, [2.0], mu=0.5),   # surface
    ]
    for n, t, mode in soc_cases:
        print(f"lam_N = {n:+.3f}, lam_T = {np.round(t,3)}   -> {mode}")
    return project_soc, soc_cases


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Sliding block with Coulomb friction vs. analytic solution

        A 0.01 kg cube sliding at 2 m/s with $\mu = 0.5$ should decelerate at
        exactly $\mu g = 4.9$ m/s² and stop at $t \approx 0.41$ s (the NCP
        ground truth from the tutorial's ComFree example). We simulate with
        the per-step 1D friction LCP: tangential impulse is the smaller of
        "what stops the slip" and "the Coulomb budget $\mu \lambda_N$".
        """
    )
    return


@app.cell
def _(np, plt):
    def simulate_sliding_cube(v0=2.0, m=0.01, mu=0.5, g=9.8, h=0.002, t_end=0.6):
        ts, vs = [0.0], [v0]
        v = v0
        steps = int(t_end / h)
        for k in range(steps):
            lam_n = m * g * h                       # normal impulse (resting contact)
            lam_t_stick = m * abs(v)                # impulse that would stop slip
            lam_t = min(lam_t_stick, mu * lam_n)    # Coulomb budget
            v = v - np.sign(v) * lam_t / m
            ts.append((k + 1) * h); vs.append(v)
        return np.array(ts), np.array(vs)

    ts_cube, vs_cube = simulate_sliding_cube()
    _stop = ts_cube[np.argmax(vs_cube <= 1e-9)]
    _fig, _ax = plt.subplots(figsize=(6.5, 3))
    _ax.plot(ts_cube, vs_cube, lw=2, label="LCP simulation")
    _ax.plot(ts_cube, np.maximum(2.0 - 0.5 * 9.8 * ts_cube, 0), "--",
             label="analytic v₀ − μg·t")
    _ax.set_xlabel("t [s]"); _ax.set_ylabel("v [m/s]")
    _ax.set_title(f"Coulomb deceleration; stops at t = {_stop:.3f} s (analytic 0.408 s)")
    _ax.legend(); _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return simulate_sliding_cube, ts_cube, vs_cube


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. The complementarity-free (ComFree) closed form, 1D

        Tutorial worked example: block of mass 1 kg resting on the floor,
        $h = 0.01$, gravity 10. The dual-cone violation is
        $\mathrm{val} = \tilde J Q^{-1} b + \phi/h$-style quantity; with
        $K = Q$ the closed form $\beta = \max(-K \cdot \mathrm{val},\, 0)$
        gives $\beta = 10$, i.e. a contact **force**
        $\lambda_N = \beta / h = 1000$... wait, let's let the numbers speak —
        the point is that $\beta/h$ recovers exactly the impulse that LCP
        produced in §1, with *zero iterations*.
        """
    )
    return


@app.cell
def _(ball_floor_lcp, mo):
    # 1D ComFree per the tutorial's worked example conventions
    _h, _m, _g, _v, _phi = 0.01, 1.0, 10.0, 0.0, 0.0
    _Q = _m / _h**2
    _b = _m * _v / _h - _m * _g          # generalized momentum/force term
    _val = (1.0 / _Q) * _b + _phi        # dual-cone violation surrogate
    _K = _Q                              # stiffness choice from the example
    _beta = max(-_K * _val, 0.0)
    _lam = _beta * _h                    # impulse over the step
    _lcp_lam, _ = ball_floor_lcp(v=_v, m=_m, g=_g, h=_h)
    mo.md(
        f"""
```
Q          = m/h²            = {_Q:g}
b          = m v/h − m g     = {_b:g}
val        = Q⁻¹ b + φ       = {_val:g}        (negative ⇒ contact wants force)
β          = max(−K·val, 0)  = {_beta:g}       (tutorial: 10)
impulse    = β·h             = {_lam:g}
LCP impulse (section 1)      = {_lcp_lam:g}    ✓ identical
```

Same physics as the LCP, but obtained from a **closed-form expression** —
no mode enumeration, no iteration, and (after a SoftPlus swap for the max)
differentiable end-to-end. That is the entire pitch of
complementarity-free contact-implicit MPC.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        **Things to try:**

        - In §4, raise `h` to 0.02 and watch the discretization error against
          the analytic line.
        - In §2, make `G` singular (e.g. duplicate a row/column — a hyperstatic
          contact) and watch PGS's solution depend on the initial guess, then
          add `rho*I` proximal regularization and watch uniqueness return.
        - Implement the ADMM loop using `project_soc` from §3 — the tutorial's
          Algorithm box maps line-by-line onto ~15 lines of numpy.
        """
    )
    return


if __name__ == "__main__":
    app.run()
