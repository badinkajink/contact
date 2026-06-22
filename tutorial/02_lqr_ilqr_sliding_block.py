# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "matplotlib"]
# ///
"""Companion notebook for Part II (Ch. 3-4): LQR and iLQR on the sliding block.

Verifies: the minimum-effort block QP (u* = [4.8, 1.6, -1.6, -4.8]),
the scalar LQR Riccati table (golden-ratio gain), and the iLQR backward
pass at t=2 (Q_uu = 24.22, K_2 = [0, -2.752]) including one-iteration
convergence on this linear-quadratic problem.

Run locally:        marimo edit 02_lqr_ilqr_sliding_block.py
Export for web:     marimo export html-wasm 02_lqr_ilqr_sliding_block.py -o site/
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
        # LQR and iLQR on the Sliding Block

        The running example: a unit-mass block on a frictionless surface,

        $$x_{t+1} = \begin{bmatrix}1 & h\\ 0 & 1\end{bmatrix} x_t
        + \begin{bmatrix}0\\ h\end{bmatrix} u_t,$$

        moved from rest at $x=0$ to rest at $x=1$.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. The minimum-effort QP, solved exactly

        With $T=4$, $h=0.25$: stack the four controls, impose
        $x_4 = (1, 0)$ as two linear equations $M u = b$, and minimize
        $\|u\|^2$. The minimum-norm solution is $u = M^\top (MM^\top)^{-1} b$.
        """
    )
    return


@app.cell
def _(np):
    h_qp, T_qp = 0.25, 4
    A_qp = np.array([[1.0, h_qp], [0.0, 1.0]])
    B_qp = np.array([[0.0], [h_qp]])

    # x_T = sum_t A^(T-1-t) B u_t  (x_0 = 0)
    M_qp = np.hstack(
        [np.linalg.matrix_power(A_qp, T_qp - 1 - t) @ B_qp for t in range(T_qp)]
    )
    b_qp = np.array([1.0, 0.0])
    u_star = M_qp.T @ np.linalg.solve(M_qp @ M_qp.T, b_qp)

    xs_qp = [np.zeros(2)]
    for _t in range(T_qp):
        xs_qp.append(A_qp @ xs_qp[-1] + B_qp.flatten() * u_star[_t])
    traj_qp = np.array(xs_qp)
    return A_qp, B_qp, M_qp, T_qp, b_qp, h_qp, traj_qp, u_star, xs_qp


@app.cell
def _(mo, traj_qp, u_star):
    _u = " & ".join(f"{u:.2f}" for u in u_star)
    _x = " & ".join(f"{x:.2f}" for x in traj_qp[:, 0])
    _v = " & ".join(f"{v:.2f}" for v in traj_qp[:, 1])
    mo.md(
        f"""
**Verified solution** (matches the tutorial table):

| | t=0 | t=1 | t=2 | t=3 | t=4 |
|---|---|---|---|---|---|
| u* | {u_star[0]:.2f} | {u_star[1]:.2f} | {u_star[2]:.2f} | {u_star[3]:.2f} | — |
| x | {traj_qp[0,0]:.2f} | {traj_qp[1,0]:.2f} | {traj_qp[2,0]:.2f} | {traj_qp[3,0]:.2f} | {traj_qp[4,0]:.2f} |
| v | {traj_qp[0,1]:.2f} | {traj_qp[1,1]:.2f} | {traj_qp[2,1]:.2f} | {traj_qp[3,1]:.2f} | {traj_qp[4,1]:.2f} |

Anti-symmetric bang: accelerate, then brake. As $h \\to 0$ this converges
to the continuous solution $u^\\star(t) = 6 - 12t$.
"""
    )
    return


@app.cell
def _(plt, traj_qp, u_star):
    _t = [0, 0.25, 0.5, 0.75, 1.0]
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.plot(_t, traj_qp[:, 0], "o-", label="x(t) position")
    _ax.plot(_t, traj_qp[:, 1], "s-", label="v(t) velocity")
    _ax.plot(_t[:-1], u_star, "^-", label="u(t) force")
    _ax.legend(); _ax.grid(alpha=0.3); _ax.set_xlabel("t")
    _ax.set_title("Minimum-effort block transfer (T=4)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Scalar LQR: the Riccati recursion by hand

        $x_{t+1} = x_t + u_t$, cost $\sum \tfrac12(x_t^2 + u_t^2)$, $T=3$.
        The gains converge toward $K_\infty = 0.618$ — the golden-ratio
        conjugate, the fixed point of this Riccati equation.
        """
    )
    return


@app.cell
def _(mo, np):
    def scalar_lqr(T=3, Qc=1.0, Rc=1.0, Qf=1.0, A=1.0, B=1.0):
        P = Qf
        rows = []
        for t in reversed(range(T)):
            K = (B * P * A) / (Rc + B * P * B)
            Pn = Qc + A * P * A - (A * P * B) ** 2 / (Rc + B * P * B)
            rows.append((t, P, K, Pn))
            P = Pn
        return rows[::-1], P  # chronological, P_0

    lqr_rows, P0 = scalar_lqr()
    _tab = "\n".join(
        f"| {t} | {P:.3f} | {K:.3f} | {Pn:.3f} |" for t, P, K, Pn in lqr_rows[::-1]
    )
    # rollout from x0 = 1
    _x, _cost, _roll = 1.0, 0.0, []
    for t, _, K, _ in lqr_rows:
        u = -K * _x
        _roll.append((t, _x, u))
        _cost += 0.5 * (_x**2 + u**2)
        _x = _x + u
    _cost += 0.5 * _x**2
    mo.md(
        f"""
| t | P(t+1) | K_t | P_t |
|---|---|---|---|
{_tab}

Rollout from $x_0=1$ achieves total cost **{_cost:.3f}**, and the value
function predicts $\\tfrac12 P_0 x_0^2 = {0.5*P0:.3f}$ — exact agreement,
the LQ luxury. Gains: {", ".join(f"{K:.3f}" for _, _, K, _ in lqr_rows)} →
$K_\\infty = (\\sqrt5 - 1)/2 = {(np.sqrt(5)-1)/2:.3f}$.
"""
    )
    return P0, lqr_rows, scalar_lqr


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Full iLQR on the block

        $T=3$, $h=1/3$, running cost $u_t^2$, terminal cost
        $100\,\|x_T - (1,0)\|^2$. The backward pass at $t=2$ from the
        all-zero nominal is the tutorial's step-by-step example:

        - $Q_u = 0$ — *zero gradient despite huge position error*, because a
          force at the last step can only change final velocity, and the
          velocity error is zero. Credit gets routed backward instead.
        - $Q_{uu} = 2 + 200 h^2 = 24.22$
        - $Q_{ux} = [0,\ 200h] = [0,\ 66.7]$, so $K_2 = [0,\ -2.752]$
        """
    )
    return


@app.cell
def _(np):
    def ilqr_block(T=3, h=1.0 / 3.0, w=100.0, iters=5, x_goal=None, verbose_t=2):
        if x_goal is None:
            x_goal = np.array([1.0, 0.0])
        A = np.array([[1.0, h], [0.0, 1.0]])
        B = np.array([[0.0], [h]])

        def rollout(x0, us, ks=None, Ks=None, xs_nom=None, alpha=1.0):
            xs = [x0.copy()]
            us_new = []
            for t in range(T):
                u = us[t].copy()
                if ks is not None:
                    u = us[t] + alpha * ks[t] + Ks[t] @ (xs[-1] - xs_nom[t])
                us_new.append(u)
                xs.append(A @ xs[-1] + (B * u).flatten())
            return xs, us_new

        def total_cost(xs, us):
            return sum(float(u**2) for u in us) + w * float(
                (xs[-1] - x_goal) @ (xs[-1] - x_goal)
            )

        x0 = np.zeros(2)
        us = [np.zeros(1) for _ in range(T)]
        xs, us = rollout(x0, us)
        history = [total_cost(xs, us)]
        debug = {}

        for it in range(iters):
            # backward pass
            Vx = 2 * w * (xs[-1] - x_goal)
            Vxx = 2 * w * np.eye(2)
            ks, Ks = [None] * T, [None] * T
            for t in reversed(range(T)):
                Qx = A.T @ Vx
                Qu = 2 * us[t] + B.T @ Vx
                Qxx = A.T @ Vxx @ A
                Quu = 2 * np.eye(1) + B.T @ Vxx @ B
                Qux = B.T @ Vxx @ A
                k = -np.linalg.solve(Quu, Qu)
                K = -np.linalg.solve(Quu, Qux)
                if it == 0 and t == verbose_t:
                    debug = dict(Qu=Qu.copy(), Quu=Quu.copy(), Qux=Qux.copy(),
                                 k=k.copy(), K=K.copy())
                Vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
                Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
                ks[t], Ks[t] = k, K
            # forward pass with simple line search
            best = None
            for alpha in [1.0, 0.5, 0.25, 0.1]:
                xs_try, us_try = rollout(x0, us, ks, Ks, xs, alpha)
                c = total_cost(xs_try, us_try)
                if best is None or c < best[0]:
                    best = (c, xs_try, us_try)
            _, xs, us = best
            history.append(best[0])
        return xs, us, history, debug

    xs_il, us_il, J_hist, bp_debug = ilqr_block()
    return J_hist, bp_debug, ilqr_block, us_il, xs_il


@app.cell
def _(J_hist, bp_debug, mo, us_il, xs_il):
    mo.md(
        f"""
**Backward pass at t=2, iteration 0** (matches the tutorial step box):

- $Q_u$ = {bp_debug['Qu'].flatten()[0]:.3f}   (zero, as derived)
- $Q_{{uu}}$ = {bp_debug['Quu'].flatten()[0]:.2f}   (tutorial: 24.22)
- $Q_{{ux}}$ = [{bp_debug['Qux'][0,0]:.1f}, {bp_debug['Qux'][0,1]:.1f}]   (tutorial: [0, 66.7])
- $K_2$ = [{bp_debug['K'][0,0]:.3f}, {bp_debug['K'][0,1]:.3f}]   (tutorial: [0, −2.752])

**Convergence:** J = {" → ".join(f"{c:.2f}" for c in J_hist[:3])} — one
iteration to optimum, because the problem is linear-quadratic and the
backward pass is then *exact* dynamic programming.

**Solution found:** u = [{", ".join(f"{float(u):.2f}" for u in us_il)}],
final state = ({xs_il[-1][0]:.3f}, {xs_il[-1][1]:.3f}).
Note this differs slightly from notebook §1: the terminal constraint is
soft here (w=100), so iLQR trades a little terminal error for less effort.
"""
    )
    return


@app.cell
def _(J_hist, plt):
    _fig, _ax = plt.subplots(figsize=(6.5, 3))
    _ax.semilogy(range(len(J_hist)), J_hist, "o-")
    _ax.set_xlabel("iLQR iteration"); _ax.set_ylabel("total cost J")
    _ax.set_title("iLQR convergence (LQ problem ⇒ one step)")
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        **Things to try** (edit the cells!):

        - Increase `w` to 1e4: the soft-constrained solution approaches the
          hard-constrained QP of §1.
        - Make the dynamics nonlinear (e.g. add drag `-0.5*v*|v|*h`) and watch
          iLQR need several iterations plus the line search.
        - Set `alpha` candidates to `[1.0]` only and break the line search on
          the nonlinear variant.

        **Next:** `03_sampling_mpc.py` solves the same block with zero derivatives.
        """
    )
    return


if __name__ == "__main__":
    app.run()
