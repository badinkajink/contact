# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "matplotlib", "scipy", "mujoco==3.14.0"]
# ///
"""Companion notebook for Part VI, slide deck 21 (hybrid force-velocity control), and the lab.

Backs slides/21_hybrid_force_velocity_control.html and the lab in labs/hybrid_servoing/,
which reimplements Hou & Mason, ICRA 2019 (arXiv 1903.02715; Algorithms 1 and 2) and ICRA 2021
(arXiv 2011.04872; the closed-form optimal hybrid servoing, OCHS) and runs them in MuJoCo on
the block-tilting task. Every number on deck 21 that a reader cannot compute by hand is computed
by a function in the section after the "Deck 21" header cell; slides link to those functions
with data-code="07_hybrid_servoing.py:function_name".

mujoco (pinned to 3.14.0, the version of the WebAssembly build in slides/lib/mujoco/) is
imported inside try: so the notebook still opens in the browser (molab / Pyodide), where it is
unavailable; the MuJoCo cells then print a message instead of running.

Run locally:        uvx marimo edit --sandbox 07_hybrid_servoing.py
Run as a script:    uv run --script 07_hybrid_servoing.py
Slides:             open slides/index.html (Part VI)
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.linalg

    try:
        import mujoco
    except ImportError:  # the browser build (Pyodide) has no mujoco
        mujoco = None
    return mo, mujoco, np, plt, scipy


@app.cell
def _(mo):
    mo.md(r"""
    # Hybrid force-velocity control: numerical companion to deck 21 and the lab

    Deck 21 assembles hybrid force-velocity control from the earlier decks: natural and
    artificial constraints (Mason 1981), selection matrices (Raibert and Craig 1981), and the
    problem formulation, Algorithms 1 and 2 of Hou and Mason 2019 and the closed-form optimal
    hybrid servoing of Hou and Mason 2021. The cells after the deck header compute the numbers
    on its slides; the cells after the lab header run the lab's implementation in MuJoCo on the
    block-tilting task.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Deck 21 · Hybrid force-velocity control

    Slides: `slides/21_hybrid_force_velocity_control.html`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Lab · Hou and Mason hybrid servoing on the block-tilting task

    Code: `labs/hybrid_servoing/`
    """)
    return


@app.cell
def _(mo):
    import sys as _sys
    from pathlib import Path as _Path

    _nb = mo.notebook_dir()
    lab_dir = (_Path(_nb) if _nb else _Path.cwd()) / "labs" / "hybrid_servoing"
    if str(lab_dir) not in _sys.path:
        _sys.path.insert(0, str(lab_dir))
    try:
        import hfvc as lab_hfvc
        import sim_block_tilt as lab_sim
    except ImportError:  # the browser build (molab / Pyodide) has no labs/ directory
        lab_hfvc = lab_sim = None
    mo.md(
        f"""
    The lab code is imported from `labs/hybrid_servoing/` as `lab_hfvc` (hfvc.py, the solvers)
    and `lab_sim` (sim_block_tilt.py, the MuJoCo study). Deck 21 cells can use both names.
    Tests: `cd labs/hybrid_servoing && uv run --no-project --with pytest --with numpy --with scipy
    --with mujoco==3.14.0 pytest -q` (41 tests, about 5 s).
    """
        if lab_hfvc
        else "`labs/hybrid_servoing/` is not available here, so the lab cells below print nothing."
    )
    return lab_dir, lab_hfvc, lab_sim


@app.cell
def _(mo):
    mo.md(r"""
    ## Crashing indexes of 2021 Fig. 1 from reconstructed examples

    2021 Fig. 1 prints six crashing indexes (eq. 9) for a planar block and a point finger but
    gives no dimensions. The reconstruction in `hfvc.py` section 9 uses a square block of height
    $h$ with the finger sticking at the centre of its top face, and $v$ = (block $v_x$, $v_y$,
    $\omega$ about the block centre, finger $v_x$, $v_y$). In the top row the block slides on
    the ground on two corner contacts. In the bottom row it sticks on a pivot under its bottom
    centre. The velocity-controlled finger direction $V$ points right, diagonally down to the
    right at angle $\varphi$ below horizontal, or down.

    The top row does not depend on the geometry: ROW($J$) = span{$e_{v_y}$, $e_\omega$,
    $e_{f_y}$, $e_{f_x} - e_{b_x}$} for every block, so $V$ to the right gives
    $1 + \sqrt 2 = 2.414$ and $V$ at 45° gives $2 + \sqrt 3 = 3.732$. The printed 3.87 needs
    $\varphi = 46.76°$. The bottom row depends on $h$, because $\omega$ is in rad/s and the
    lengths enter $J$. A least-squares fit of $(\varphi, h)$ to 3.87, 7.10 and 10.48 leaves
    residuals under 0.002, and the two one-parameter fits predict 10.47 for the printed 10.48.
    """)
    return


@app.cell
def _(lab_hfvc, mo):
    def lab_fig1_markdown():
        """2021 Fig. 1: the six reconstructed crashing indexes next to the printed ones, and the
        fitted (phi, h) with its residuals (hfvc.fig1_table, hfvc.fig1_fit)."""
        _names = {"right": "right", "diag": "diagonal", "down": "down"}
        rows = lab_hfvc.fig1_table()
        fit = lab_hfvc.fig1_fit()
        lines = ["| support | $V$ | printed | reconstruction |", "|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['support']} | {_names[r['direction']]} | {r['paper']} | {r['ours']:.4f} |")
        lines.append("")
        lines.append(
            f"Fit: $\\varphi$ = {fit['phi_deg']:.2f}°, $h$ = {fit['h']:.4f}, residuals "
            f"{', '.join(f'{x:+.4f}' for x in fit['residuals'])}. One-parameter fits: "
            f"$\\varphi$ = {fit['phi_from_387']:.3f}° from 3.87 alone, $h$ = {fit['h_from_710']:.4f} "
            f"from 7.10 alone. The top-row diagonal at exactly 45° gives {fit['ground_diag_at_45']:.4f}."
        )
        return "\n".join(lines)

    mo.md(lab_fig1_markdown()) if lab_hfvc else None
    return (lab_fig1_markdown,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Block-tilting solution at 0 and 20 degrees

    2019 Sec. V: $q \in \mathbb R^{10}$ (object position, quaternion, hand position),
    $v \in \mathbb R^9$ (object body twist, hand velocity), $\dot q = \Omega(q) v$ with
    $\Omega$ from eq. 26 and $E(q)$ from eq. 27, and three sticking contacts from eq. 33 (the
    hand, and two table points at the ends of the rotation edge). `hfvc.block_tilt_jac`
    differentiates $\Phi$ by hand, and the tests check it against finite differences to 1e-8.
    Two guard sets are solved. The paper's set has eight-sided cones and a 10 N minimum normal
    force at all three contacts (eq. 35-36). The authors' `block_tilting_control.m` uses six
    sides and bounds only the hand. Forces are in N, velocities in m/s and rad/s, and $\lambda$
    is the force on the block at each contact in the world frame.
    """)
    return


@app.cell
def _(lab_hfvc, np):
    def lab_block_tilt_solution(theta_deg=0.0, guards="paper", method="ochs", seed=0):
        """The HFVC of 2019 Sec. V at tilt theta_deg: n_av, n_af, crashing index, R_a, w_av,
        eta_af, the world-frame velocity and force commands, and the three contact forces.

        guards='paper': eight-sided cones, 10 N minimum normal force at every contact (eq. 35-36);
        guards='code': six sides and the hand bound only (block_tilting_control.m)."""
        bt = lab_hfvc.BlockTilt() if guards == "paper" else lab_hfvc.BlockTilt(cone_sides=6, n_min_table=0.0)
        prob = lab_hfvc.block_tilt_problem(np.radians(theta_deg), bt)
        sol = lab_hfvc.solve_ochs(prob) if method == "ochs" else lab_hfvc.solve_hs(prob, 3, rng=seed)
        v_w, f_w = lab_hfvc.world_commands(sol.vel, sol.force, prob.n_u)
        lam = sol.force.lam.reshape(3, 3)
        return dict(method=sol.method, n_av=sol.vel.n_av, n_af=sol.vel.n_af, crash=sol.crash,
                    R_a=sol.vel.R_a, w_av=sol.vel.w_av, eta_af=sol.force.eta_af, v_world=v_w,
                    f_world=f_w, lam_hand=lam[0], lam_table1=lam[1], lam_table2=lam[2],
                    t_vel_ms=1e3 * sol.t_vel, t_force_ms=1e3 * sol.t_force)

    return (lab_block_tilt_solution,)


@app.cell
def _(lab_block_tilt_solution, lab_hfvc, mo, np):
    def _fmt(x):
        return "(" + ", ".join(f"{0.0 if abs(v) < 1e-10 else v:.4g}" for v in np.atleast_1d(x)) + ")"

    _out = None
    if lab_hfvc:
        _lines = ["| θ | guards | method | $n_{av}$ | crash | $w_{av}$ | $\\eta_{af}$ | velocity command (m/s) | force command (N) | $\\lambda_{hand}$ (N) | $\\lambda_{table}$ each (N) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _th in (0.0, 20.0):
            for _g in ("paper", "code"):
                for _m in ("ochs", "hs3"):
                    _s = lab_block_tilt_solution(_th, _g, _m)
                    _lines.append(f"| {_th:.0f}° | {_g} | {_s['method']} | {_s['n_av']} | {_s['crash']:.3f} | "
                                  f"{_fmt(_s['w_av'])} | {_fmt(_s['eta_af'])} | {_fmt(_s['v_world'])} | "
                                  f"{_fmt(_s['f_world'])} | {_fmt(_s['lam_hand'])} | {_fmt(_s['lam_table1'])} |")
        _s0 = lab_block_tilt_solution(0.0, "paper", "ochs")
        _out = mo.md("\n".join(_lines) + f"""

    $R_a$ from OCHS at θ = 0 (rows: two force-controlled axes, then the velocity-controlled axis):
    {_fmt(_s0['R_a'][0])}, {_fmt(_s0['R_a'][1])}, {_fmt(_s0['R_a'][2])}.

    Both methods give one velocity-controlled direction along the tilting arc and two
    force-controlled directions. The force command has no $y$ part and points from the hand
    toward the rotation edge, as 2019 Sec. V-E describes. With the paper's guards the 10 N
    table bound is active and the hand presses 15.1 N. With the code's guards the hand bound
    is active at 10 N. The crashing index is 26.57 at every tilt angle, because the hand's share
    of the free motion is $|r|/\\sqrt{{1 + |r|^2}}$ = 0.0752 for the $|r|$ = 75.4 mm lever: the
    index mixes metres and radians. HS3 runs the authors' settings (step $t = 10$, 50
    iterations) and stops up to 1.2° short of the OCHS direction: the top eigenvalue of $M$ is
    0.0057, so 50 steps of $(I + 20 M)$ shrink the error of the start by a factor of only 211.
    """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## OCHS and Algorithm 1 on random test problems

    The generator `hfvc.random_problems` follows the settings of 2021 Table I. The paper gives
    no distributions, so the choices are documented in `hfvc.py` section 11. Each problem gets a
    load for which a force distribution inside every guard exists, so a force-part failure is
    a failure of the method. Two goal types are drawn. `min` draws random rows over the robot's
    free motions, and `object` draws random combinations of the object's velocity only, the
    kind of goal of 2019 eq. 30. HS3 and HS10 are Algorithm 1 + 2 with 3 and 10 restarts at the
    authors' settings. An "ill-conditioned" solution has a crashing index above 100; 2021 does
    not state its threshold. Times are in ms on this workstation (numpy, one core) and are not
    comparable with the paper's MATLAB times.
    """)
    return


@app.cell
def _(lab_hfvc, np):
    def lab_table2(n_planar=100, n_3d=200, goal="min", seed=7):
        """Table II-style comparison of OCHS, OCHS(M), HS3 and HS10 on random problems.

        Returns {dim: (rows, per, extra)}: rows from hfvc.table2, the per-problem solutions,
        and extra = {'ochs_worse': number of problems where OCHS has a larger crashing index
        than HS3 or HS10, 'hs3_failures': failure messages of HS3 by cause,
        'ratio': crash(HS3) / crash(OCHS) where both solved, 'probs': the problems}."""
        out = {}
        for dim, count in ((2, n_planar), (3, n_3d)):
            probs = lab_hfvc.random_problems(count, dim=dim, seed=seed, goal=goal)
            solvers = {"OCHS": lab_hfvc.solve_ochs,
                       "OCHS(M)": lambda p: lab_hfvc.solve_ochs(p, maximal=True),
                       "HS3": lambda p: lab_hfvc.solve_hs(p, 3, rng=0),
                       "HS10": lambda p: lab_hfvc.solve_hs(p, 10, rng=0)}
            rows, per = lab_hfvc.table2(probs, solvers)
            worse, ratio, causes = 0, [], {}
            for i in range(len(probs)):
                o = per["OCHS"][i]
                for h in (per["HS3"][i], per["HS10"][i]):
                    if o.vel is not None and h.vel is not None and o.vel.n_av:
                        worse += int(o.crash > h.crash * (1 + 1e-6))
                h3 = per["HS3"][i]
                if o.vel is not None and h3.vel is not None and o.vel.n_av and np.isfinite(h3.crash):
                    ratio.append(h3.crash / o.crash)
                if not h3.ok:
                    key = ("velocity: eq. 14 (underactuated)" if "eq. 14" in h3.message else
                           "velocity: rows of C dependent" if "dependent" in h3.message else
                           "velocity: goal inclusion" if "goal inclusion" in h3.message else
                           "force: Algorithm 2 LP infeasible" if "Algorithm 2" in h3.message else h3.message)
                    causes[key] = causes.get(key, 0) + 1
            out[dim] = (rows, per, dict(ochs_worse=worse, hs3_failures=causes, ratio=np.array(ratio),
                                        probs=probs))
        return out

    return (lab_table2,)


@app.cell
def _(lab_hfvc, lab_table2, mo):
    lab_t2 = {g: lab_table2(goal=g) for g in ("min", "object")} if lab_hfvc else {}

    def _md():
        lines = ["| goal | dim | method | solved | crash mean / median | ill (> 100) | velocity ms mean / worst | force ms mean / worst |",
                 "|---|---|---|---|---|---|---|---|"]
        notes = []
        for g, res in lab_t2.items():
            for dim, (rows, _, extra) in res.items():
                for name, r in rows.items():
                    lines.append(f"| {g} | {dim}D | {name} | {r['solved']} / {r['total']} | {r['mean_crash']:.3g} / {r['median_crash']:.3g} | "
                                 f"{r['ill']} | {r['t_vel_mean']:.2f} / {r['t_vel_worst']:.2f} | "
                                 f"{r['t_force_mean']:.2f} / {r['t_force_worst']:.2f} |")
                notes.append(f"* goal `{g}`, {dim}D: OCHS worse than HS3 or HS10 on {extra['ochs_worse']} problems; "
                             f"HS3 failures: {extra['hs3_failures'] or 'none'}.")
        return "\n".join(lines) + "\n\n" + "\n".join(notes)

    mo.md(_md()) if lab_t2 else None
    return (lab_t2,)


@app.cell
def _(mo):
    mo.md(r"""
    OCHS solves every problem with goal `min` and has the smaller crashing index on every
    problem, as 2021 Sec. VI-C reports. It rejects some `object` problems at eq. 25. On each of
    them no command in the search space can meet the goal (`test_object_goals_rejected_only_when_no_valid_C_exists`).
    The mean crashing index of HS3 is dominated by a few solutions near 1e8 whose rows of $C$
    are almost in ROW($J$). Algorithm 1 fails on three causes. Its step returns dependent rows
    of $C$. It is undefined for underactuated problems (eq. 14). Algorithm 2 cannot find forces
    inside the guards. The last cause is the subject of the section after the next.

    HS3 fails far more often here than in 2021 Table II (5974 of 6000 planar and 65950 of
    72000 3D problems solved). The random problems of this lab have goals of up to
    rows($\bar U$) dimensions and loads that make the forces statically indeterminate. The
    paper's distributions are unknown, so the two tables cannot be compared number for number.
    """)
    return


@app.cell
def _(lab_hfvc, lab_t2, mo, np, plt):
    def lab_bound_gap(res):
        """Largest relative gap between the OCHS crashing index and hfvc.crash_bound over the
        problems of one lab_table2 result, and the number of problems compared."""
        gap, n = 0.0, 0
        for _, per, extra in res.values():
            for p, o in zip(extra["probs"], per["OCHS"]):
                if o.vel is not None and o.vel.n_av:
                    gap = max(gap, abs(o.crash / lab_hfvc.crash_bound(p)[0] - 1.0))
                    n += 1
        return gap, n

    _out = None
    if lab_t2:
        _gap, _n = lab_bound_gap(lab_t2["min"])
        _fig, _ax = plt.subplots(figsize=(6.4, 2.4))
        for _dim, _c in ((2, "#0000cc"), (3, "#cc0000")):
            _r = np.sort(lab_t2["min"][_dim][2]["ratio"])
            _ax.plot(np.arange(1, _r.size + 1) / _r.size, _r, color=_c, label=f"{_dim}D, goal min")
        _ax.set_yscale("log")
        _ax.set_xlabel("fraction of problems")
        _ax.set_ylabel("crash(HS3) / crash(OCHS)")
        _ax.axhline(1.0, color="k", lw=0.8)
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _out = mo.vstack([mo.md(f"""
    **OCHS attains the eigenvalue bound.** Every goal-inclusive $C$ has rows in
    $S$ = ROW([$J$; $G$]) ∩ {{$c_u = 0$}}, the span of $B_c$ in 2019 eq. 13, so Algorithm 1 and
    OCHS search the same space. For unit rows $c = B_c k$ the null-space share is
    $\\|P_{{NULL(J)}} c\\|^2 = k^\\top M k$ with $M$ the matrix of `hfvc.alg1_basis`. The best
    $n_{{av}}$-dimensional span is therefore the top-$n_{{av}}$ eigenspace of $M$, with crashing index
    $\\sqrt{{(1 + c)/(1 - c)}}$, $c^2 = 1 - \\lambda_{{n_{{av}}}}(M)$ (`hfvc.crash_bound`). $S$ splits
    orthogonally into span($\\bar U$) ∩ ROW([$J$; $G$]), where $M$ is positive definite, and
    {{$c_u = 0$}} ∩ ROW($J$), where $M$ vanishes. The OCHS rows span the first part, which is that
    eigenspace. Largest relative gap between the OCHS crashing index and the bound over {_n}
    problems: {_gap:.1e}. Random $C$ in $S$ never beat OCHS in the tests (0 of 1500 draws). The
    curve shows how far HS3 is from the optimum.
    """), _fig])
    _out
    return (lab_bound_gap,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step size of the projected gradient descent

    2019 Sec. IV-A fixes the step at $t = 10$. For two unit columns with inner product $s$
    and $M = 0$, one step of the authors' gradient maps $s$ to $s(1 - 4t) + O(s^3)$, so the
    pairwise term of eq. 15 drives the columns apart only for $t < 0.5$. At $t = 10$ the inner
    product oscillates and never reaches 0. The left plot runs this two-column case. The right
    table counts Algorithm 1 outcomes on 3D problems with goal `max`, where $n_{av}$ reaches 9.
    """)
    return


@app.cell
def _(lab_hfvc, mo, np, plt):
    def lab_pairwise_trace(t, steps=30, s0=0.1):
        """|c_1^T c_2| over projected-gradient steps for two unit columns and M = 0."""
        K = np.array([[1.0, s0], [0.0, np.sqrt(1 - s0 * s0)]])
        out = [s0]
        for _ in range(steps):
            K = K - t * lab_hfvc.alg1_grad(K, np.zeros((2, 2)), "code")
            K = K / np.linalg.norm(K, axis=0)
            out.append(abs(float(K[:, 0] @ K[:, 1])))
        return np.array(out)

    def lab_step_sweep(count=80, seed=7, settings=((10.0, 50), (1.0, 50), (0.2, 50), (0.2, 200))):
        """Algorithm 1 (Ns = 3, authors' cost) on 3D goal='max' problems for several (t, iters):
        numbers solved, failed on dependent rows, failed on eq. 14, and the median and worst of
        crash / bound over the solved ones."""
        probs = lab_hfvc.random_problems(count, dim=3, seed=seed, goal="max")
        rows = []
        for t, iters in settings:
            ok = dep = eq14 = other = 0
            ratio = []
            for i, p in enumerate(probs):
                try:
                    v = lab_hfvc.alg1_velocity(p, 3, t, iters, "code", rng=i)
                    ok += 1
                    ratio.append(lab_hfvc.crashing_index(p.J, v.C) / lab_hfvc.crash_bound(p)[0])
                except lab_hfvc.Infeasible as e:
                    msg = str(e)
                    dep += "dependent" in msg
                    eq14 += "eq. 14" in msg
                    other += not ("dependent" in msg or "eq. 14" in msg)
            ratio = np.array(ratio)
            rows.append(dict(t=t, iters=iters, solved=ok, dependent=dep, eq14=eq14, other=other,
                             median_ratio=float(np.median(ratio)) if ratio.size else np.nan,
                             worst_ratio=float(ratio.max()) if ratio.size else np.nan))
        return rows

    _out = None
    if lab_hfvc:
        _fig, _ax = plt.subplots(figsize=(3.6, 2.5))
        for _t, _c in ((10.0, "#cc0000"), (1.0, "#cc6600"), (0.4, "#008800"), (0.2, "#0000cc")):
            _ax.semilogy(np.maximum(lab_pairwise_trace(_t), 1e-17), color=_c, label=f"t = {_t:g}")
        _ax.set_xlabel("step")
        _ax.set_ylabel("|c1·c2|")
        _ax.legend(fontsize=7)
        _fig.tight_layout()
        _rows = lab_step_sweep()
        _tab = "\n".join(["| $t$ | iterations | solved | dependent rows | eq. 14 | other | crash / bound, median | worst |",
                          "|---|---|---|---|---|---|---|---|"] +
                         [f"| {r['t']:g} | {r['iters']} | {r['solved']} | {r['dependent']} | {r['eq14']} | {r['other']} | "
                          f"{r['median_ratio']:.3f} | {r['worst_ratio']:.3g} |" for r in _rows])
        _out = mo.hstack([_fig, mo.md(_tab)], widths=[1, 2])
    _out
    return lab_pairwise_trace, lab_step_sweep


@app.cell
def _(mo):
    mo.md(r"""
    ## Algorithm 2 force distribution on statically indeterminate problems

    Algorithm 2 fixes the free forces $[\lambda; \eta_u; \eta_{av}]$ at the minimum-norm solution
    of Newton's law for each force command $\eta_{af}$ (2019 eq. 21-23), then searches only over
    $\eta_{af}$. With three or more contacts on the object the contact forces are statically
    indeterminate (the four-legged table of deck 13). The minimum-norm distribution can then
    leave a friction cone although another distribution satisfies every guard. The QP of 2021
    eq. 30 searches all distributions. The count below gives Algorithm 2 and the QP the same
    $T$ from Algorithm 1, so the difference comes from the force step alone.
    """)
    return


@app.cell
def _(lab_hfvc, mo):
    def lab_alg2_vs_qp(count=100, seed=7):
        """On planar random problems: HS3 force-step failures, how many of them the eq. 30 QP
        solves with the same T, and the contact counts of the failed problems."""
        fails = qp_ok = 0
        contacts = {}
        for p in lab_hfvc.random_problems(count, dim=2, seed=seed):
            s = lab_hfvc.solve_hs(p, 3, rng=0)
            if s.vel is None or s.ok:
                continue
            fails += 1
            k = len(p.info["contacts"])
            contacts[k] = contacts.get(k, 0) + 1
            try:
                lab_hfvc.ochs_force(p, s.vel)
                qp_ok += 1
            except lab_hfvc.Infeasible:
                pass
        return dict(problems=count, alg2_failures=fails, qp_solves=qp_ok, contacts=contacts)

    _r = lab_alg2_vs_qp() if lab_hfvc else None
    mo.md(f"""
    Planar problems: {_r['problems']}; Algorithm 2 finds no forces inside the guards on
    {_r['alg2_failures']}; the eq. 30 QP with the same $T$ solves {_r['qp_solves']} of them;
    contacts on the failed problems (count: problems): {_r['contacts']}.
    """) if _r else None
    return (lab_alg2_vs_qp,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Block tilting in MuJoCo under three controllers

    `labs/hybrid_servoing/sim_block_tilt.py` runs the tilting plan (0.5 rad/s to 40°) in
    `models/block_tilt.xml`: a 75 mm wooden cube (0.295 kg) on a table and a 4 mm point finger
    driven by three force motors, stepped at 0.5 ms. The controller's model is the nominal cube
    with its table at $z = 0$, $\mu = 0.8$, and a 10 N minimum hand normal force. The real block
    size, table height and friction are perturbed. The three controllers share one plan.
    **hfvc** re-solves OCHS at 250 Hz from the measured tilt, servos the velocity along the row
    of $C$ and applies $\eta_{af}$ along the complement. **velocity** is a stiff position servo
    (10 kN/m) to the planned arc. **force** applies the model's full HFVC force with light
    damping. A trial succeeds when the block reaches 40° within 2.5 s, the table edge slides at
    most 2 mm, the finger is never off the block for more than 20 ms, and the finger normal
    force stays under 25 N, the force at which the 2019 robot stopped its runs.

    Regenerate the results (99 trials, serial, about 40 s):
    `cd labs/hybrid_servoing && uv run --script sim_block_tilt.py --fresh`.
    """)
    return


@app.cell
def _(lab_sim, mo):
    lab_rows = lab_sim.load_rows() if lab_sim else []
    lab_series = lab_sim.load_rows(lab_sim.SERIES_CSV) if lab_sim else []

    def lab_study_numbers(rows, series):
        """The numbers the study's text quotes, computed from the result CSVs."""
        def pick(sweep, ctrl, **kw):
            return [r for r in rows if r["sweep"] == sweep and r["controller"] == ctrl
                    and all(abs(r[k] - v) < 1e-9 for k, v in kw.items())]
        hs = pick("size", "hfvc")
        ht = pick("table", "hfvc")
        keys = ("peak_finger_N", "max_table_slip_mm", "t_reach_s")
        fs = [r for r in series if r["sweep"] == "nominal" and r["controller"] == "force"]

        def rate(a, b):
            return (b["theta_deg"] - a["theta_deg"]) / (b["t"] - a["t"])

        k = max(1, round(0.1 / (fs[1]["t"] - fs[0]["t"]))) if len(fs) > 1 else 1
        return dict(
            random={c: sum(int(r["success"]) for r in rows if r["sweep"] == "random" and r["controller"] == c)
                    for c in ("hfvc", "velocity", "force")},
            hfvc_size_fn=(min(r["peak_finger_N"] for r in hs), max(r["peak_finger_N"] for r in hs)),
            hfvc_size_t=(min(r["t_reach_s"] for r in hs), max(r["t_reach_s"] for r in hs)),
            hfvc_table_same="yes" if all(all(r[q] == ht[0][q] for q in keys) for r in ht) else "no",
            hfvc_mu03_slip=pick("friction", "hfvc", mu=0.3)[0]["max_table_slip_mm"],
            hfvc_mu03_fn=pick("friction", "hfvc", mu=0.3)[0]["peak_finger_N"],
            vel_fn={L: pick("size", "velocity", L_mm=L)[0]["peak_finger_N"] for L in (78.0, 81.0)},
            vel_69_reason=pick("size", "velocity", L_mm=69.0)[0]["fail_reason"] or "success",
            vel_mu03_slip=pick("friction", "velocity", mu=0.3)[0]["max_table_slip_mm"],
            force_nominal_theta=pick("nominal", "force")[0]["theta_end_deg"],
            force_rate=(rate(fs[0], fs[k]), rate(fs[-1 - k], fs[-1])) if len(fs) > k else (float("nan"),) * 2,
            force_size_ok=", ".join(f"{r['L_mm']:.0f}" for r in pick("size", "force") if r["success"]) or "none",
        )

    def lab_success_table(rows):
        """Successes / trials and peak finger normal force (N, worst trial) per sweep and controller."""
        cells = {}
        for r in rows:
            k = (r["sweep"], r["controller"])
            s, n, pk = cells.get(k, (0, 0, 0.0))
            cells[k] = (s + int(r["success"]), n + 1, max(pk, r["peak_finger_N"]))
        sweeps = list(dict.fromkeys(r["sweep"] for r in rows))
        ctrls = ("hfvc", "velocity", "force")
        lines = ["| sweep | " + " | ".join(ctrls) + " |", "|---" * (len(ctrls) + 1) + "|"]
        for sw in sweeps:
            lines.append(f"| {sw} | " + " | ".join(
                f"{cells[(sw, c)][0]}/{cells[(sw, c)][1]}, {cells[(sw, c)][2]:.1f} N" if (sw, c) in cells else ""
                for c in ctrls) + " |")
        return "\n".join(lines)

    mo.md("Successes / trials, and the largest finger normal force of any trial:\n\n" + lab_success_table(lab_rows)) \
        if lab_rows else mo.md("No results yet: run `uv run --script sim_block_tilt.py` in `labs/hybrid_servoing/`.")
    return lab_rows, lab_series, lab_study_numbers, lab_success_table


@app.cell
def _(lab_rows, lab_series, lab_study_numbers, mo, np, plt):
    _out = None
    if lab_rows:
        _col = {"hfvc": "#0000cc", "velocity": "#cc0000", "force": "#008800"}
        _fig, _axs = plt.subplots(1, 3, figsize=(9.6, 2.7))
        for _c in _col:
            _s = sorted([r for r in lab_rows if r["controller"] == _c and r["sweep"] in ("size", "nominal")],
                        key=lambda r: r["L_mm"])
            _axs[0].plot([r["L_mm"] for r in _s], [r["peak_finger_N"] for r in _s], "o-", color=_col[_c], label=_c, ms=3)
            _s = sorted([r for r in lab_rows if r["controller"] == _c and r["sweep"] in ("friction", "nominal")],
                        key=lambda r: r["mu"])
            _axs[1].semilogy([r["mu"] for r in _s], [max(r["max_table_slip_mm"], 1e-3) for r in _s], "o-", color=_col[_c], ms=3)
            _ts = [r for r in lab_series if r["controller"] == _c and r["sweep"] == "nominal"]
            _axs[2].plot([r["t"] for r in _ts], [r["theta_deg"] for r in _ts], color=_col[_c])
        _axs[0].axhline(25, color="k", lw=0.8, ls="--")
        _axs[0].set_xlabel("real block edge (mm)")
        _axs[0].set_ylabel("peak finger normal (N)")
        _axs[0].legend(fontsize=7)
        _axs[1].axhline(2, color="k", lw=0.8, ls="--")
        _axs[1].set_xlabel("real friction coefficient")
        _axs[1].set_ylabel("table edge slip (mm)")
        _axs[2].set_xlabel("time (s), nominal")
        _axs[2].set_ylabel("tilt (deg)")
        _fig.tight_layout()
        _n = lab_study_numbers(lab_rows, lab_series)
        _out = mo.vstack([_fig, mo.md(f"""
    Random trials (block edge 70-80 mm, table height ±2 mm, friction 0.4-1.0, 20 per
    controller): hfvc {_n['random']['hfvc']}, velocity {_n['random']['velocity']}, force
    {_n['random']['force']} successes.

    * **hfvc** holds the peak finger normal force between {_n['hfvc_size_fn'][0]:.1f} and
      {_n['hfvc_size_fn'][1]:.1f} N over the 69-81 mm size sweep and reaches 40° in
      {_n['hfvc_size_t'][0]:.2f}-{_n['hfvc_size_t'][1]:.2f} s. Its rows are identical for every
      table height ({_n['hfvc_table_same']}), because it uses no absolute position. At
      $\\mu$ = 0.3 (the model assumes 0.8) the table edge slides {_n['hfvc_mu03_slip']:.1f} mm and the
      peak finger force is {_n['hfvc_mu03_fn']:.1f} N.
    * **velocity** tracks the planned arc about the model's rotation edge. The peak finger force
      is {_n['vel_fn'][78.0]:.1f} N with a 78 mm block and {_n['vel_fn'][81.0]:.1f} N with an 81 mm
      block. With a 69 mm block: {_n['vel_69_reason']}. At $\\mu$ = 0.3 the stiff finger holds the
      block and the table edge slides {_n['vel_mu03_slip']:.2f} mm.
    * **force** reaches {_n['force_nominal_theta']:.1f}° at 2.5 s in the nominal case. Its tilt rate
      falls from {_n['force_rate'][0]:.1f} deg/s over the first 0.1 s to {_n['force_rate'][1]:.1f} deg/s
      over the last 0.1 s, against 28.6 deg/s planned. Beyond the model's quasi-static force its
      only drive is a 6 N s/m damper toward the planned velocity. In the size sweep it reaches
      40° only with the smaller blocks ({_n['force_size_ok']} mm), which are lighter than the model.

    The dashed lines are the 25 N force limit and the 2 mm slip limit.
    """)])
    _out
    return


if __name__ == "__main__":
    app.run()
