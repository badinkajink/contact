# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "matplotlib"]
# ///
"""Companion notebook for Part II (Ch. 5): Sampling-based methods.

Verifies the MPPI weight tables (λ=1 and λ=5), the one-iteration CEM
example, demonstrates effective sample size vs. temperature, and runs
Predictive Sampling / MPPI / CEM head-to-head on the sliding block with
spline-parameterized controls.

Run locally:        marimo edit 03_sampling_mpc.py
Export for web:     marimo export html-wasm 03_sampling_mpc.py -o site/
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
        # Sampling-Based MPC: Predictive Sampling, MPPI, CEM

        Zero derivatives required — only forward rollouts. This is why these
        methods are robust through contact: they are agnostic to whether the
        simulator's dynamics are smooth.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. MPPI weights: hard vs. soft selection

        Five samples with costs $J = [10, 3, 7, 2, 8]$. Weights
        $w_i \propto e^{-J_i/\lambda}$. (Implementation note: subtract
        $\min_i J_i$ before exponentiating to avoid underflow — the weights
        are invariant to this shift.)
        """
    )
    return


@app.cell
def _(mo, np):
    def mppi_weights(J, lam):
        J = np.asarray(J, dtype=float)
        w = np.exp(-(J - J.min()) / lam)
        return w / w.sum()

    def effective_sample_size(w):
        return 1.0 / np.sum(w**2)

    J_demo = np.array([10.0, 3.0, 7.0, 2.0, 8.0])
    w1 = mppi_weights(J_demo, 1.0)
    w5 = mppi_weights(J_demo, 5.0)
    _r1 = " | ".join(f"{w:.3f}" for w in w1)
    _r5 = " | ".join(f"{w:.3f}" for w in w5)
    mo.md(
        f"""
| sample | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| J | 10 | 3 | 7 | **2** | 8 |
| w (λ=1) | {_r1} |
| w (λ=5) | {_r5} |

Matches the tutorial tables. λ=1 nearly hard-selects sample 4
(w₄ = {w1[3]:.3f}); λ=5 blends everything.

**Effective sample size** N_eff = 1/Σw²:
λ=1 → {effective_sample_size(w1):.2f} of 5 (almost hard selection);
λ=5 → {effective_sample_size(w5):.2f} of 5 (broad averaging).
Adaptive-λ schemes hold N_eff at a target fraction of N.
"""
    )
    return J_demo, effective_sample_size, mppi_weights, w1, w5


@app.cell
def _(J_demo, effective_sample_size, mppi_weights, np, plt):
    _lams = np.logspace(-1, 1.5, 60)
    _ess = [effective_sample_size(mppi_weights(J_demo, l)) for l in _lams]
    _fig, _ax = plt.subplots(figsize=(6.5, 3))
    _ax.semilogx(_lams, _ess, lw=2)
    _ax.axhline(1, color="gray", ls=":", label="hard selection limit")
    _ax.axhline(5, color="gray", ls="--", label="uniform limit (N)")
    _ax.set_xlabel("temperature λ"); _ax.set_ylabel("N_eff")
    _ax.set_title("Temperature controls the soft/hard selection dial")
    _ax.legend(); _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo, np):
    mo.md(
        r"""
        ## 2. One CEM iteration, by hand

        Ten samples from $\mathcal N(0,1)$, elite fraction 0.4. The elites
        are $\{0.9, 1.1, 0.7, 1.4\}$ (tutorial worked example).
        """
    )
    _el = np.array([0.9, 1.1, 0.7, 1.4])
    _mu, _var = _el.mean(), ((_el - _el.mean()) ** 2).mean()
    print(f"new mu = {_mu:.3f}   (tutorial: 1.025)")
    print(f"new var = {_var:.4f}  -> sigma = {np.sqrt(_var):.2f}  (tutorial: 0.0669, 0.26)")
    print("Mean moved toward the good region AND the distribution contracted 1 -> 0.26.")
    print("Contraction = refinement, but also the premature-convergence risk.")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Head-to-head on the sliding block

        Same task as notebook 02 (rest-to-rest transfer, soft terminal cost),
        controls parameterized by **P=4 zero-order-hold spline knots** over
        T=20 steps — the search space is 4-D instead of 20-D, and every
        sample is a plausible control signal. That dimensionality reduction
        is doing a lot of quiet work; try raising T with knots fixed.
        """
    )
    return


@app.cell
def _(np):
    # --- shared task definition -------------------------------------------
    T_task, h_task, P_knots = 20, 0.05, 4
    A_t = np.array([[1.0, h_task], [0.0, 1.0]])
    B_t = np.array([0.0, h_task])
    x_goal_t = np.array([1.0, 0.0])

    def knots_to_controls(theta):
        """Zero-order hold: each knot held for T/P steps."""
        reps = T_task // P_knots
        return np.repeat(theta, reps)

    def rollout_cost(theta, w=100.0, ctrl_weight=0.05):
        u = knots_to_controls(theta)
        x = np.zeros(2)
        c = 0.0
        for t in range(T_task):
            c += ctrl_weight * u[t] ** 2
            x = A_t @ x + B_t * u[t]
        c += w * float((x - x_goal_t) @ (x - x_goal_t))
        return c

    return (
        A_t, B_t, P_knots, T_task, h_task, knots_to_controls,
        rollout_cost, x_goal_t,
    )


@app.cell
def _(P_knots, np, rollout_cost):
    def run_predictive_sampling(iters=30, N=32, sigma=1.0, seed=0):
        rng = np.random.default_rng(seed)
        theta = np.zeros(P_knots)
        hist = [rollout_cost(theta)]
        for _ in range(iters):
            cands = theta + sigma * rng.standard_normal((N, P_knots))
            cands = np.vstack([theta, cands])  # always keep the nominal
            costs = np.array([rollout_cost(c) for c in cands])
            theta = cands[np.argmin(costs)]
            hist.append(costs.min())
        return theta, hist

    def run_mppi(iters=30, N=32, sigma=1.0, lam=2.0, seed=0):
        rng = np.random.default_rng(seed)
        theta = np.zeros(P_knots)
        hist = [rollout_cost(theta)]
        for _ in range(iters):
            eps = sigma * rng.standard_normal((N, P_knots))
            cands = theta + eps
            costs = np.array([rollout_cost(c) for c in cands])
            w = np.exp(-(costs - costs.min()) / lam)
            w /= w.sum()
            theta = theta + (w[:, None] * eps).sum(axis=0)
            hist.append(rollout_cost(theta))
        return theta, hist

    def run_cem(iters=30, N=32, elite_frac=0.25, sigma0=1.0,
                sigma_floor=0.05, seed=0):
        rng = np.random.default_rng(seed)
        mu = np.zeros(P_knots)
        sigma = sigma0 * np.ones(P_knots)
        hist = [rollout_cost(mu)]
        n_elite = max(2, int(elite_frac * N))
        for _ in range(iters):
            cands = mu + sigma * rng.standard_normal((N, P_knots))
            costs = np.array([rollout_cost(c) for c in cands])
            elites = cands[np.argsort(costs)[:n_elite]]
            mu = elites.mean(axis=0)
            sigma = np.maximum(elites.std(axis=0), sigma_floor)
            hist.append(rollout_cost(mu))
        return mu, hist

    ps_theta, ps_hist = run_predictive_sampling()
    mppi_theta, mppi_hist = run_mppi()
    cem_theta, cem_hist = run_cem()
    return (
        cem_hist, cem_theta, mppi_hist, mppi_theta, ps_hist, ps_theta,
        run_cem, run_mppi, run_predictive_sampling,
    )


@app.cell
def _(cem_hist, mppi_hist, plt, ps_hist):
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _ax.semilogy(ps_hist, label="Predictive Sampling (best-of-N)")
    _ax.semilogy(mppi_hist, label="MPPI (exp-weighted avg)")
    _ax.semilogy(cem_hist, label="CEM (elite refit)")
    _ax.set_xlabel("iteration"); _ax.set_ylabel("cost (log)")
    _ax.set_title("Same rollouts budget, three update rules")
    _ax.legend(); _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(cem_theta, knots_to_controls, mo, mppi_theta, np, ps_theta):
    def _final_state(theta):
        import numpy as _np
        A = _np.array([[1.0, 0.05], [0.0, 1.0]])
        B = _np.array([0.0, 0.05])
        x = _np.zeros(2)
        for u in knots_to_controls(theta):
            x = A @ x + B * u
        return x

    _rows = "\n".join(
        f"| {name} | [{', '.join(f'{k:.2f}' for k in th)}] | "
        f"({_final_state(th)[0]:.3f}, {_final_state(th)[1]:.3f}) |"
        for name, th in [
            ("Pred. Sampling", ps_theta),
            ("MPPI", mppi_theta),
            ("CEM", cem_theta),
        ]
    )
    mo.md(
        f"""
| method | knots θ | final state (goal: 1, 0) |
|---|---|---|
{_rows}

All three find the accelerate-then-brake structure. Typical signatures:
Predictive Sampling is jumpy but never regresses (elitism); MPPI is
smooth; CEM converges fastest here but would freeze without the
variance floor (`sigma_floor`) — delete it and watch.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        **Things to try:**

        - Set `P_knots = T_task` (no spline): same budget, dramatically worse —
          the curse of dimensionality, live.
        - Anneal MPPI's λ and σ across iterations (DIAL-MPC's trick).
        - Replace `rollout_cost` with a discontinuous cost (e.g. a contact-like
          `if x[0] > wall: ...`) — sampling won't care; iLQR would.

        **Next:** `04_contact_solvers.py` — LCP, PGS, cone projections, and the
        complementarity-free closed form.
        """
    )
    return


if __name__ == "__main__":
    app.run()
