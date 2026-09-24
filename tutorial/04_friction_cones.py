# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "matplotlib", "scipy"]
# ///
"""Companion notebook for slide decks 11-13 (slides/11_coulomb_friction.html,
12_friction_cone_3d.html, 13_contact_wrench_cone.html): Coulomb friction, friction cones,
pyramid approximations, and many contacts. The three decks were Parts 1-3 of the stand-alone
series "Friction & Grasping from Scratch"; the section headers below cite deck and slide numbers.

Verifies every number on those slides: the will-it-slide example, friction
angles and the tilting-board test, the three equivalent cone tests, maximum
dissipation on circles / ellipses / polygons, the k-sided pyramid error table,
the sliding-puck "starburst", slide-or-tip, the wobbly table's null space, and
the floor's contact wrench cone. Also backs the tutorial's Cones (Ch. 2) and
Coulomb (Ch. 7) sections.

Run locally:        marimo edit 04_friction_cones.py
Export for web:     marimo export html-wasm 04_friction_cones.py -o site/
Slides:             open slides/index.html (decks 11-13)
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import null_space
    from scipy.optimize import nnls

    return mo, nnls, np, null_space, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Friction Cones — Numerical Companion (Decks 11–13)

    The slides build Coulomb friction from scratch: a single point contact,
    the 2D wedge, the 3D ice-cream cone, its pyramid approximations, and
    finally several contacts at once. Every number quoted there is recomputed
    below, usually two ways (closed form vs. brute force).

    Conventions: $f_n$ is the normal (pressing) part of a contact force,
    $\mathbf f_t$ the tangential part, $\mu$ the friction coefficient and
    $\alpha = \arctan\mu$ the cone's half-angle. $g = 9.81$ m/s².
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Will it slide? (deck 11, slides 15–16)

    A 10 kg box on a floor with $\mu_s = 0.5$ (static) and $\mu_k = 0.4$
    (kinetic). Static friction matches the push up to the budget $\mu_s N$;
    past it the box slides against kinetic friction $\mu_k N$.
    """)
    return


@app.cell
def _(mo, np):
    G = 9.81

    def push_box(P, m=10.0, mu_s=0.5, mu_k=0.4, g=G, down_deg=0.0):
        """A push of P newtons aimed down_deg below horizontal on a resting box."""
        side, press = P * np.cos(np.radians(down_deg)), P * np.sin(np.radians(down_deg))
        N = m * g + press
        if side <= mu_s * N:
            return dict(N=N, side=side, budget=mu_s * N, friction=side, accel=0.0, mode="sticks")
        F = mu_k * N
        return dict(N=N, side=side, budget=mu_s * N, friction=F, accel=(side - F) / m, mode="slides")

    _rows = "\n".join(
        f"| {P} | {r['budget']:.2f} | {r['friction']:.2f} | {r['accel']:.2f} | {r['mode']} |"
        for P in [30, 49, 50, 60]
        for r in [push_box(P)]
    )
    _q4 = push_box(60, down_deg=30)
    mo.md(
        f"""
| push P (N) | budget μ_s N | friction (N) | a (m/s²) | |
|---|---|---|---|---|
{_rows}

Slides quote: budget 49.05 N, 30 N sticks, 60 N slides with
F = 39.2 N and a = 2.08 m/s². ✓

**Check-yourself Q4** (60 N aimed 30° below horizontal): sideways
{_q4['side']:.1f} N, normal force {_q4['N']:.1f} N, budget {_q4['budget']:.1f} N
→ **{_q4['mode']}**. Pushing down makes it harder.
"""
    )
    return G, push_box


@app.cell
def _(G, np, plt, push_box):
    _P = np.linspace(0, 100, 501)
    _F = [push_box(p)["friction"] for p in _P]
    _fig, _ax = plt.subplots(figsize=(6.5, 3))
    _ax.plot(_P, _F, lw=2)
    _ax.axhline(0.5 * 10 * G, ls="--", c="gray", lw=1)
    _ax.axhline(0.4 * 10 * G, ls=":", c="gray", lw=1)
    _ax.set_xlabel("push P [N]"); _ax.set_ylabel("friction F [N]")
    _ax.set_title("Static ramp, then the drop to kinetic friction (slide 16)")
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The friction angle and the tilting board (deck 11, slides 24, 29–32)

    On a board tilted by $\theta$, the block sticks while
    $mg\sin\theta \le \mu\, mg\cos\theta$, i.e. $\tan\theta \le \mu$. So it
    slips at $\theta = \arctan\mu = \alpha$, whatever its mass. Below: the
    closed form next to a brute-force scan over tilt angles.
    """)
    return


@app.cell
def _(mo, np):
    def friction_angle(mu):
        return np.degrees(np.arctan(mu))

    def slip_angle_scan(mu, m=1.0, g=9.81, step=1e-3):
        """Smallest tilt (deg) where gravity along the board beats the friction budget."""
        th = np.radians(np.arange(0.0, 90.0, step))
        along, budget = m * g * np.sin(th), mu * m * g * np.cos(th)
        return np.degrees(th[np.argmax(along > budget)])

    _rows = "\n".join(
        f"| {mu} | {friction_angle(mu):.2f} | {slip_angle_scan(mu):.2f} | {slip_angle_scan(mu, m=50.0):.2f} |"
        for mu in [0.1, 0.25, 0.3, 0.4, 0.5, 1.0, 2.0]
    )
    mo.md(
        f"""
| μ | α = arctan μ (deg) | scan, 1 kg | scan, 50 kg |
|---|---|---|---|
{_rows}

Matches the slides: 5.7°, 14.0° (check-yourself Q1), 16.7°, 21.8° (phone on a
book), 26.6°, 45°, 63.4°. The 50 kg column equals the 1 kg column: mass cancels.
"""
    )
    return (friction_angle,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Three ways to say "inside the cone" (deck 11, slide 27)

    For a force $\mathbf f$ at a contact with inward normal $\hat{\mathbf n}$:

    1. $f_n \ge 0$ and $|f_t| \le \mu f_n$;
    2. the angle between $\mathbf f$ and $\hat{\mathbf n}$ is at most $\alpha$;
    3. $\mathbf f\cdot\hat{\mathbf n} \ge \|\mathbf f\|\cos\alpha$.

    They should agree on every force (away from round-off at the boundary).
    """)
    return


@app.cell
def _(mo, np):
    _rng = np.random.default_rng(0)
    _n = 200_000
    _f = _rng.normal(size=(_n, 2))
    _mu = _rng.uniform(0.0, 2.0, size=_n)
    _th = _rng.uniform(0, 2 * np.pi, size=_n)
    _nh = np.stack([np.cos(_th), np.sin(_th)], axis=1)          # random surface normals
    _th_ = np.stack([_nh[:, 1], -_nh[:, 0]], axis=1)
    _fn, _ft = np.sum(_f * _nh, 1), np.sum(_f * _th_, 1)
    _alpha = np.arctan(_mu)
    _t1 = (_fn >= 0) & (np.abs(_ft) <= _mu * _fn)
    _t2 = np.arctan2(np.abs(_ft), _fn) <= _alpha
    _t3 = _fn >= np.linalg.norm(_f, axis=1) * np.cos(_alpha)
    _near = np.abs(np.abs(_ft) - _mu * _fn) < 1e-9
    _bad = ((_t1 != _t2) | (_t1 != _t3)) & ~_near
    mo.md(
        f"""
{_n:,} random forces, normals and μ: tests 1/2/3 disagree on **{int(_bad.sum())}**
of them. {int(_t1.sum()):,} are inside the cone.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Maximum dissipation on circles and ellipses (deck 12, slides 11, 16)

    Sliding friction is the point of the unit slice that minimizes
    $\mathbf f_t\cdot\mathbf v$ (the "last touch" of a line perpendicular to
    $\mathbf v$). For an ellipse with half-widths $\mu_1, \mu_2$ the closed
    form is
    $$\mathbf f_t = -f_n\,\frac{(\mu_1^2 v_1,\ \mu_2^2 v_2)}{\sqrt{\mu_1^2 v_1^2 + \mu_2^2 v_2^2}}.$$
    We compare it against brute force over 100,000 boundary points.
    """)
    return


@app.cell
def _(mo, np):
    def mdp_ellipse(v, a, b):
        v = np.asarray(v, float)
        num = np.array([a * a * v[0], b * b * v[1]])
        return -num / np.hypot(a * v[0], b * v[1])

    def mdp_brute(boundary, v):
        return boundary[np.argmin(boundary @ np.asarray(v, float))]

    _t = np.linspace(0, 2 * np.pi, 100_000, endpoint=False)
    _v = np.array([1.0, 1.0]) / np.sqrt(2)
    _circ = np.stack([np.cos(_t), np.sin(_t)], 1)
    _ell = np.stack([0.9 * np.cos(_t), 0.3 * np.sin(_t)], 1)
    _fe, _fb = mdp_ellipse(_v, 0.9, 0.3), mdp_brute(_ell, _v)
    _off_axis = np.degrees(np.arctan2(abs(_fe[1]), abs(_fe[0])))
    _off_back = np.degrees(np.arccos(-_fe @ _v / np.linalg.norm(_fe)))
    mo.md(
        f"""
- Circle, $\\mathbf v$ at 45°: brute force gives {np.round(mdp_brute(_circ, _v), 4)},
  i.e. straight back, $-\\mathbf v$. ✓
- Skate ellipse ($\\mu_1 = 0.9$ across, $\\mu_2 = 0.3$ along), $\\mathbf v$ at 45°:
  closed form {np.round(_fe, 4)}, brute force {np.round(_fb, 4)}.
  - size $\\|\\mathbf f_t\\| / f_n$ = {np.linalg.norm(_fe):.3f}
  - angle off the $-\\hat{{\\mathbf t}}_1$ axis = **{_off_axis:.1f}°** (slide: 6.3°)
  - angle away from $-\\mathbf v$ = **{_off_back:.1f}°** (slide: 39°)
"""
    )
    return mdp_brute, mdp_ellipse


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. How wrong is a $k$-sided pyramid? (deck 12, slides 21–24)

    A regular $k$-gon **inscribed** in the unit slice (corners on the circle)
    reaches between $\mu\cos(\pi/k)$ and $\mu$; a **circumscribed** one
    (sides touching) between $\mu$ and $\mu/\cos(\pi/k)$. We measure the
    polygon's reach in 20,000 directions and compare with the formulas.
    """)
    return


@app.cell
def _(mo, np):
    def regular_polygon(k, how="inscribed", mu=1.0, phase=None):
        r = mu if how == "inscribed" else mu / np.cos(np.pi / k)
        ph = (0.0 if how == "inscribed" else np.pi / k) if phase is None else phase
        a = ph + 2 * np.pi * np.arange(k) / k
        return r * np.stack([np.cos(a), np.sin(a)], 1)

    def reach(verts, U):
        """How far a convex polygon (CCW verts, containing 0) extends along each unit row of U."""
        e = np.roll(verts, -1, axis=0) - verts
        n = np.stack([e[:, 1], -e[:, 0]], 1) / np.linalg.norm(e, axis=1, keepdims=True)
        h = np.sum(n * verts, 1)                          # side i: n_i . x <= h_i
        nu = np.atleast_2d(U) @ n.T
        with np.errstate(divide="ignore"):
            t = np.where(nu > 1e-12, h / nu, np.inf)
        out = t.min(axis=1)
        return out if np.ndim(U) == 2 else out[0]

    # 5760 directions hit every facet middle and corner exactly for k in {4, ..., 32}.
    _dirs = np.linspace(0, 2 * np.pi, 5760, endpoint=False)
    _U = np.stack([np.cos(_dirs), np.sin(_dirs)], 1)
    _rows = []
    for _k in [4, 6, 8, 12, 16, 32]:
        _ri = reach(regular_polygon(_k, "inscribed"), _U)
        _rc = reach(regular_polygon(_k, "circumscribed"), _U)
        _rows.append(
            f"| {_k} | {100*(1-_ri.min()):.2f}% | {100*(1-np.cos(np.pi/_k)):.2f}% "
            f"| {100*(_rc.max()-1):.2f}% | {100*(1/np.cos(np.pi/_k)-1):.2f}% |"
        )
    _k1 = next(k for k in range(3, 100) if 1 - np.cos(np.pi / k) < 0.01)
    mo.md(
        "| k | inscribed: worst short (measured) | formula | circumscribed: worst over (measured) | formula |\n"
        "|---|---|---|---|---|\n" + "\n".join(_rows) +
        f"\n\nFewest sides for an inscribed pyramid under 1% error: **k = {_k1}** "
        f"({100*(1-np.cos(np.pi/_k1)):.2f}%; k = {_k1-1} gives {100*(1-np.cos(np.pi/(_k1-1))):.2f}%)."
    )
    return reach, regular_polygon


@app.cell
def _(mo, np, reach, regular_polygon):
    _u = np.array([1.0, 1.0]) / np.sqrt(2)
    _budget = 0.5 * 10 * 9.81
    _dia = regular_polygon(4, "inscribed")
    _box = regular_polygon(4, "circumscribed")
    mo.md(
        f"""
### Cost 1: the budget depends on direction (deck 12, slide 25)

Reach at 45°, per unit $\\mu f_n$: circle 1.000, diamond **{reach(_dia, _u):.3f}**,
box **{reach(_box, _u):.3f}**. For the 10 kg box with $\\mu = 0.5$ pushed at 45°:
true breakaway {_budget:.1f} N, diamond model {_budget*reach(_dia, _u):.1f} N,
box model {_budget*reach(_box, _u):.1f} N. (Slide: 49.1 / 34.7 / 69.4 N.)
"""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Sliding pucks: why pyramid paths bend (deck 12, slides 26–28)

    Implicit (maximum-dissipation) time stepping: each step picks the
    friction force in $\mu g\cdot$slice that makes the next velocity as small
    as possible. That is the Euclidean projection of $-\mathbf v/h$ onto the
    slice — the same idea as the tutorial's cone projection, done here on
    circles, ellipses and polygons. The JavaScript on the slides runs the same
    loop (`slides/lib/contact.js`).
    """)
    return


@app.cell
def _(np, regular_polygon):
    def project_slice(model, z, scale):
        """Closest point to z in scale * slice."""
        z = np.asarray(z, float)
        if model["kind"] == "circle":
            r, n = model["mu"] * scale, np.linalg.norm(z)
            return z if n <= r else z * (r / n)
        if model["kind"] == "ellipse":
            a, b = model["mu"] * scale, model["mu2"] * scale
            if (z[0] / a) ** 2 + (z[1] / b) ** 2 <= 1:
                return z
            g = lambda t: (a * z[0] / (a * a + t)) ** 2 + (b * z[1] / (b * b + t)) ** 2 - 1
            lo, hi = 0.0, 1.0
            while g(hi) > 0:
                hi *= 2
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if g(mid) > 0 else (lo, mid)
            return np.array([a * a * z[0] / (a * a + hi), b * b * z[1] / (b * b + hi)])
        P = model["verts"] * scale
        cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
        inside = all(cross(P[(i + 1) % len(P)] - P[i], z - P[i]) >= 0 for i in range(len(P)))
        if inside:
            return z
        best, bd = None, np.inf
        for i in range(len(P)):
            a, e = P[i], P[(i + 1) % len(P)] - P[i]
            t = np.clip((z - a) @ e / (e @ e), 0, 1)
            q = a + t * e
            if np.linalg.norm(z - q) < bd:
                best, bd = q, np.linalg.norm(z - q)
        return best

    def slide_puck(model, v0, g=9.81, h=0.002, tmax=10.0):
        x, v = np.zeros(2), np.asarray(v0, float)
        path = [x.copy()]
        for _ in range(int(tmax / h)):
            f = project_slice(model, -v / h, g)       # friction per unit mass
            v = v + h * f
            x = x + h * v
            path.append(x.copy())
            if np.linalg.norm(v) < 1e-9:
                break
        return np.array(path)

    def friction_models(mu):
        return {
            "circle": dict(kind="circle", mu=mu),
            "diamond": dict(kind="poly", verts=regular_polygon(4, "inscribed", mu)),
            "box": dict(kind="poly", verts=regular_polygon(4, "circumscribed", mu)),
            "octagon": dict(kind="poly", verts=regular_polygon(8, "inscribed", mu)),
            "ellipse": dict(kind="ellipse", mu=mu, mu2=mu / 3),
        }

    return friction_models, project_slice, slide_puck


@app.cell
def _(friction_models, mo, np, slide_puck):
    _mu, _v0, _g = 0.3, 2.0, 9.81
    _M = friction_models(_mu)
    _d = lambda name, ang: np.linalg.norm(slide_puck(_M[name], _v0 * np.array([np.cos(ang), np.sin(ang)]))[-1])
    _rows = "\n".join(
        f"| {name} | {_d(name, 0):.3f} | {_d(name, np.pi/4):.3f} |" for name in ["circle", "diamond", "box", "octagon"]
    )
    _a = _v0**2 / (2 * _mu * _g)
    mo.md(
        f"""
Stopping distance for a puck launched at 2 m/s, $\\mu = 0.3$:

| model | along $\\hat{{\\mathbf t}}_1$ (m) | at 45° (m) |
|---|---|---|
{_rows}

Closed forms: circle $v^2/(2\\mu g)$ = {_a:.3f} m in every direction; diamond on the
diagonal decelerates at $\\mu g/\\sqrt2$ → {_a*np.sqrt(2):.3f} m; box on the diagonal at
$\\sqrt2\\,\\mu g$ → {_a/np.sqrt(2):.3f} m. (Slide 27: 0.68 / 0.96 m for the diamond.)
"""
    )
    return


@app.cell
def _(mo):
    puck_model = mo.ui.dropdown(
        options=["circle", "diamond", "box", "octagon", "ellipse"], value="diamond", label="friction model"
    )
    puck_mu = mo.ui.slider(0.05, 1.0, step=0.05, value=0.3, label="μ")
    mo.hstack([puck_model, puck_mu])
    return puck_model, puck_mu


@app.cell
def _(friction_models, np, plt, puck_model, puck_mu, slide_puck):
    _M = friction_models(puck_mu.value)[puck_model.value]
    _fig, _ax = plt.subplots(figsize=(4.8, 4.8))
    for _j in range(16):
        _a = 2 * np.pi * _j / 16
        _p = slide_puck(_M, 2.0 * np.array([np.cos(_a), np.sin(_a)]))
        _ax.plot(_p[:, 0], _p[:, 1], lw=1.5, c="C3")
        _ax.plot(*_p[-1], "k.", ms=5)
    _ax.set_aspect("equal"); _ax.grid(alpha=0.3)
    _ax.set_title(f"16 pucks at 2 m/s, {puck_model.value}, μ = {puck_mu.value:.2f}")
    _ax.set_xlabel("x [m]"); _ax.set_ylabel("y [m]")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(friction_models, mo, np, project_slice):
    # Velocity space, diamond model, launched at 30 degrees (slide 28).
    _M = friction_models(0.3)["diamond"]
    _v = 2.0 * np.array([np.cos(np.radians(30)), np.sin(np.radians(30))])
    _h, _trace = 0.002, []
    for _k in range(2000):
        _trace.append(_v.copy())
        _v = _v + _h * project_slice(_M, -_v / _h, 9.81)
        if np.linalg.norm(_v) < 1e-9:
            break
    _trace = np.array(_trace)
    _i = int(np.argmax(_trace[:, 0] <= _trace[:, 1] + 1e-6))
    mo.md(
        f"""
Diamond, launched at 30°: $v_2$ stays at {_trace[0,1]:.3f} m/s (min {_trace[:_i,1].min():.4f},
max {_trace[:_i,1].max():.4f}) while $v_1$ falls from {_trace[0,0]:.3f} to {_trace[_i,0]:.3f} m/s.
From then on $v_1 = v_2$ (largest gap {np.abs(_trace[_i:,0]-_trace[_i:,1]).max():.1e}): the puck
slides straight down the diagonal. Friction was the corner $(-\\mu g, 0)$, then the middle of
a side.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Slide or tip, and the floor's wrench cone (deck 13, slides 7–9, 22–23)

    Push a box of width $w$ with $P$ at height $h$. Torques about the front
    corner give $N_A = \tfrac12 mg - Ph/w$ and $N_B = \tfrac12 mg + Ph/w$.
    It slides at $P = \mu mg$ and tips at $P = mg\,w/(2h)$.

    The same answer, the wrench-cone way of deck 13: the floor's four edge wrenches are
    $(\pm\mu, 1, \pm w/2)$ and the needed wrench is $(-P, mg, hP)$. It is
    in the contact wrench cone iff a non-negative combination makes it,
    which we test with non-negative least squares.
    """)
    return


@app.cell
def _(mo, nnls, np):
    def slide_or_tip(P, h, w, mu, mg=1.0):
        NA, NB = mg / 2 - P * h / w, mg / 2 + P * h / w
        P_slide, P_tip = mu * mg, mg * w / (2 * h)
        return dict(NA=NA, NB=NB, P_slide=P_slide, P_tip=P_tip,
                    first="tips" if P_tip < P_slide else "slides")

    def in_floor_cone(P, h, w, mu, mg=1.0):
        E = np.array([[s * mu, 1.0, t * w / 2] for s in (1, -1) for t in (1, -1)]).T
        _, res = nnls(E, np.array([-P, mg, h * P]))
        return res < 1e-9

    _rows = []
    for _name, _w, _h, _mu in [("cabinet, pushed high", 0.4, 1.0, 0.5), ("cabinet, pushed low", 0.4, 0.3, 0.5),
                               ("bookshelf (Q1)", 0.3, 1.5, 0.4)]:
        _r = slide_or_tip(0, _h, _w, _mu)
        _rows.append(f"| {_name} | {_r['P_slide']:.3f} | {_r['P_tip']:.3f} | {_r['first']} | {_w/(2*_mu):.3f} |")

    _rng = np.random.default_rng(1)
    _bad = 0
    for _ in range(5000):
        _P, _h, _mu = _rng.uniform(0, 1), _rng.uniform(0.05, 1.5), _rng.uniform(0.1, 1)
        _analytic = (_P <= _mu) and (_h * _P <= 0.4 / 2)
        _margin = min(abs(_P - _mu), abs(_h * _P - 0.2))
        if _margin > 1e-6 and in_floor_cone(_P, _h, 0.4, _mu) != _analytic:
            _bad += 1
    mo.md(
        "| case | slide at P/mg | tip at P/mg | first | push below h = w/(2μ) to slide |\n"
        "|---|---|---|---|---|\n" + "\n".join(_rows) +
        f"\n\nWrench-cone test vs. the two formulas on 5,000 random (P, h, μ): **{_bad}** disagreements."
    )
    return in_floor_cone, slide_or_tip


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. The wobbly table (deck 13, slides 13–16)

    Four legs at $(\pm a, \pm a)$, weight $mg$ in the middle. Three equations
    (vertical force, two tipping torques), four unknowns. The null space of
    the statics matrix is the load that diagonal pairs can trade; the
    minimum-norm answer (what a lightly regularized solver tends toward) is
    the even split. Three legs: a square system with exactly one answer.
    """)
    return


@app.cell
def _(mo, np, null_space):
    _a, _mg = 0.5, 1.0
    _legs = np.array([[_a, _a], [-_a, _a], [-_a, -_a], [_a, -_a]])
    _A = np.vstack([np.ones(4), _legs[:, 0], _legs[:, 1]])
    _b = np.array([_mg, 0.0, 0.0])
    _ns = null_space(_A)[:, 0]
    _ns = _ns / _ns[0]
    _minnorm = np.linalg.pinv(_A) @ _b
    _tri = np.array([[np.cos(t), np.sin(t)] for t in np.radians([90, 210, 330])])
    _A3 = np.vstack([np.ones(3), _tri[:, 0], _tri[:, 1]])
    _N3 = np.linalg.solve(_A3, _b)
    mo.md(
        f"""
- Null space direction: {np.round(_ns, 3)} → $\\mathbf N = \\tfrac{{mg}}{{4}}(1,1,1,1) + s(1,-1,1,-1)$. ✓
- Minimum-norm split: {np.round(_minnorm, 4)} (each leg $mg/4$).
- Every $s \\in [-mg/4, mg/4]$ keeps all $N_i \\ge 0$; at the ends two diagonal legs carry
  $mg/2$ each and the table rocks.
- Three legs at 120°: rank {np.linalg.matrix_rank(_A3)}, unique answer {np.round(_N3, 4)}
  (each $mg/3$).
"""
    )
    return


@app.cell
def _(mo, np):
    _rng = np.random.default_rng(2)
    _mu, _mg = 0.4, 1.0
    _vhat = np.array([0.8, 0.6])
    _worst = 0.0
    for _ in range(1000):
        _N = _rng.dirichlet(np.ones(4)) * _mg      # a random valid load split
        _Ftot = sum(-_mu * n * _vhat for n in _N)
        _worst = max(_worst, np.linalg.norm(_Ftot - (-_mu * _mg * _vhat)))
    mo.md(
        f"""
**Sliding without turning (deck 13, slide 17).** Over 1,000 random load splits, the total
corner friction differs from $-\\mu\\, mg\\, \\hat{{\\mathbf v}}$ by at most {_worst:.1e}.
The split doesn't matter.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. The dual cone (deck 12, slide 33)

    $\mathcal K^* = \{\mathbf v : \mathbf v\cdot\mathbf f \ge 0\ \forall \mathbf f\in\mathcal K\}$.
    For the Coulomb cone it suffices to check the cone's edges. The claim:
    $\mathcal K_\mu^* = \mathcal K_{1/\mu}$.
    """)
    return


@app.cell
def _(mo, np):
    _rng = np.random.default_rng(3)
    _bad = 0
    for _ in range(20000):
        _mu = _rng.uniform(0.05, 3)
        _v = _rng.normal(size=2)                       # (v_t, v_n)
        _edges = [np.array([_mu, 1.0]), np.array([-_mu, 1.0])]
        _dual = all(_v @ e >= 0 for e in _edges)
        _k_inv = (_v[1] >= 0) and (abs(_v[0]) <= _v[1] / _mu)
        _margin = min(abs(_v @ e) for e in _edges)
        if _margin > 1e-9 and _dual != _k_inv:
            _bad += 1
    mo.md(f"20,000 random $(\\mu, \\mathbf v)$: membership in $\\mathcal K_\\mu^*$ and in $\\mathcal K_{{1/\\mu}}$ disagree **{_bad}** times.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    **Things to try:**

    - In §6, switch the dropdown to `octagon`: the bending shrinks but does
      not vanish. Then try `ellipse`: every path swings toward the
      low-friction axis, like a skate.
    - Replace `project_slice` for the polygon with a clamp of each tangent
      component separately (the "box" trick used in per-constraint solvers)
      and compare with the `box` model: same set, same answer?
    - In §7, give the floor three contacts with different $\mu$ and check
      which walls of the wrench cone move.

    **Next:** `05_contact_solvers.py` — how simulators find these forces
    every time step (LCP, PGS, cone projections). Then `06_force_closure.py`
    for decks 14–16: grasps.
    """)
    return


if __name__ == "__main__":
    app.run()
