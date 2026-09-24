# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "matplotlib", "scipy"]
# ///
"""Companion notebook for slide decks Parts 4-6 (slides/04-06) and the tutorial's
grasping chapter: force closure, Nguyen's theorem, 3D grasps, grasp quality.

Verifies: the positive-span / linear-program force-closure test, Nguyen's
theorem against that test on thousands of random grasps, the box / disk /
wedge formulas, the normal-angle shortcut (necessary; exact on spheres),
two hard vs. two soft fingers in 3D, three fingers on a ball
(tan(phi) < mu) and what pyramid approximations do to it, the three-force
theorem, Mirtich & Canny's moment formula, and the Ferrari-Canny epsilon.

Run locally:        marimo edit 06_force_closure.py
Export for web:     marimo export html-wasm 06_force_closure.py -o site/
Slides:             open slides/index.html (Parts 4-6)
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    from scipy.linalg import null_space

    return ConvexHull, linprog, mo, np, null_space, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Force Closure — Numerical Companion (Slides, Parts 4–6)

    A grasp is **force closure** when its contacts can cancel any disturbance
    wrench. With friction cones replaced by finitely many edges, that becomes
    a question about a finite set of **primitive wrenches**: do they
    positively span the whole wrench space ($\mathbb R^3$ in the plane,
    $\mathbb R^6$ in 3D)? Everything below reduces to that one test.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The test: rank plus a strictly positive squeeze (Part 6, slide 5)

    Vectors $\mathbf w_1, \dots, \mathbf w_m$ positively span $\mathbb R^d$
    iff they span it linearly **and** some combination with all weights
    $\ge 1$ (i.e. strictly positive, after scaling) sums to zero. The second
    part is a linear-program feasibility check.
    """)
    return


@app.cell
def _(linprog, np):
    def positively_spans(W):
        """W: one vector per column. True iff the columns positively span R^d."""
        W = np.asarray(W, float)
        d, m = W.shape
        if np.linalg.matrix_rank(W, tol=1e-9) < d:
            return False
        res = linprog(np.zeros(m), A_eq=W, b_eq=np.zeros(d), bounds=[(1, None)] * m, method="highs")
        return res.status == 0

    def cross2(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def unit(v):
        v = np.asarray(v, float)
        return v / np.linalg.norm(v)

    def rot(v, t):
        c, s = np.cos(t), np.sin(t)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

    def edge_wrenches_2d(contacts, mu, lam=1.0):
        """Planar primitive wrenches (f_x, f_y, tau/lam), 2 per frictional contact.
        contacts: list of (point, inward normal). Each edge has unit normal part."""
        cols = []
        a = np.arctan(mu)
        for p, n in contacts:
            n = unit(n)
            for s in (-a, a):
                f = rot(n, s) / np.cos(a)
                cols.append([f[0], f[1], cross2(p, f) / lam])
        return np.array(cols).T

    def frictionless_wrenches_2d(contacts, lam=1.0):
        return np.array([[*unit(n), cross2(p, unit(n)) / lam] for p, n in contacts]).T

    return cross2, edge_wrenches_2d, frictionless_wrenches_2d, positively_spans, rot, unit


@app.cell
def _(mo, np, positively_spans):
    _ang = lambda degs: np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in degs]).T
    mo.md(
        f"""
Warm-ups (Part 4, slides 5–6):

| vectors | positively span? |
|---|---|
| 1D: {{+1}} | {positively_spans([[1.0]])} |
| 1D: {{+1, −1}} | {positively_spans([[1.0, -1.0]])} |
| 2D: 90°, 210°, 330° | {positively_spans(_ang([90, 210, 330]))} |
| 2D: 40°, 90°, 150° (a half-plane) | {positively_spans(_ang([40, 90, 150]))} |
"""
    )
    return


@app.cell
def _(frictionless_wrenches_2d, mo, np, positively_spans):
    # Box of width 2, height 1.4 (Part 4, slides 8-9).
    _two = [(np.array([-1.0, 0.0]), [1, 0]), (np.array([1.0, 0.0]), [-1, 0])]
    _pinwheel = [(np.array([-1.0, 0.3]), [1, 0]), (np.array([1.0, -0.3]), [-1, 0]),
                 (np.array([0.5, -0.7]), [0, 1]), (np.array([-0.5, 0.7]), [0, -1])]
    _disk = [(np.array([np.cos(t), np.sin(t)]), [-np.cos(t), -np.sin(t)]) for t in np.linspace(0, 2 * np.pi, 5, endpoint=False)]
    _W2, _Wp, _Wd = (frictionless_wrenches_2d(c) for c in (_two, _pinwheel, _disk))
    mo.md(
        f"""
**Frictionless fingers.**

- Two opposite fingers on a box: rank {np.linalg.matrix_rank(_W2)}, force closure **{positively_spans(_W2)}**.
- Four fingers in a "pinwheel" on the box (left face high, right face low, bottom face right,
  top face left): force closure **{positively_spans(_Wp)}**. Four is the minimum in the plane.
- Five fingers around a disk: every normal passes through the center, so every torque is 0.
  Rank {np.linalg.matrix_rank(_Wd)}, force closure **{positively_spans(_Wd)}**.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Nguyen's theorem, tested on random grasps (Part 4, slides 12–15)

    Two frictional contacts in the plane are force closure iff the segment
    joining them lies strictly inside both friction cones. We draw random
    convex shapes (ellipses and random polygons), random contact pairs and
    random $\mu$, and compare the theorem with the exact wrench test.
    """)
    return


@app.cell
def _(ConvexHull, np, unit):
    def nguyen(p1, n1, p2, n2, mu):
        """Returns (holds, beta1, beta2) with angles in radians. n1, n2 are inward normals."""
        d = unit(p2 - p1)
        b1 = np.arccos(np.clip(d @ unit(n1), -1, 1))
        b2 = np.arccos(np.clip(-d @ unit(n2), -1, 1))
        a = np.arctan(mu)
        return (b1 < a) and (b2 < a), b1, b2

    def normal_angle(n1, n2):
        """Angle between inward normal 1 and minus inward normal 2 (0 = perfectly opposite)."""
        return np.arccos(np.clip(unit(n1) @ -unit(n2), -1, 1))

    def ellipse_point(a, b, t):
        p = np.array([a * np.cos(t), b * np.sin(t)])
        n_out = unit([b * np.cos(t), a * np.sin(t)])
        return p, -n_out

    def random_polygon(rng, n=7):
        pts = rng.normal(size=(n, 2))
        return pts[ConvexHull(pts).vertices]                # CCW

    def polygon_point(verts, s):
        """Point at arc-length fraction s in [0,1) and its inward normal."""
        e = np.roll(verts, -1, axis=0) - verts
        L = np.linalg.norm(e, axis=1)
        d = (s % 1.0) * L.sum()
        i = int(np.searchsorted(np.cumsum(L), d))
        i = min(i, len(verts) - 1)
        d0 = d - (np.cumsum(L)[i] - L[i])
        u = e[i] / L[i]
        return verts[i] + u * d0, np.array([-u[1], u[0]])  # CCW: inward = left normal

    return ellipse_point, normal_angle, nguyen, polygon_point, random_polygon


@app.cell
def _(edge_wrenches_2d, ellipse_point, mo, nguyen, normal_angle, np, polygon_point, positively_spans, random_polygon):
    _rng = np.random.default_rng(0)
    _stats = dict(trials=0, fc=0, disagree=0, nec_violations=0, angle_pass_not_fc=0)
    for _t in range(3000):
        _mu = _rng.uniform(0.05, 1.2)
        if _t % 2 == 0:
            _a, _b = _rng.uniform(0.5, 2), _rng.uniform(0.5, 2)
            (_p1, _n1), (_p2, _n2) = (ellipse_point(_a, _b, t) for t in _rng.uniform(0, 2 * np.pi, 2))
        else:
            _V = random_polygon(_rng)
            (_p1, _n1), (_p2, _n2) = (polygon_point(_V, s) for s in _rng.uniform(0, 1, 2))
        if np.linalg.norm(_p1 - _p2) < 1e-3:
            continue
        _ok, _b1, _b2 = nguyen(_p1, _n1, _p2, _n2, _mu)
        _alpha = np.arctan(_mu)
        if min(abs(_b1 - _alpha), abs(_b2 - _alpha)) < 1e-6:
            continue                                        # skip exact-boundary cases
        _exact = positively_spans(edge_wrenches_2d([(_p1, _n1), (_p2, _n2)], _mu))
        _angle_ok = normal_angle(_n1, _n2) <= 2 * _alpha
        _stats["trials"] += 1
        _stats["fc"] += _exact
        _stats["disagree"] += (_ok != _exact)
        _stats["nec_violations"] += (_exact and not _angle_ok)
        _stats["angle_pass_not_fc"] += (_angle_ok and not _exact)
    mo.md(
        f"""
{_stats['trials']:,} random grasps ({_stats['fc']:,} of them force closure):

- Nguyen's condition vs. the exact wrench test: **{_stats['disagree']}** disagreements.
- Force closure but the normal-angle test fails (would contradict *necessity*, slide 23):
  **{_stats['nec_violations']}**.
- Normal-angle test passes but not force closure (the shortcut is *not sufficient*, slide 24):
  **{_stats['angle_pass_not_fc']}**.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Three pinches you can do in your head (Part 4, slides 17–19)

    Bisect on the geometry until the exact wrench test flips, and compare
    with the closed forms: box offset $d < \mu w$, disk misplacement
    $\delta < 2\arctan\mu$, wedge half-angle $\beta < \arctan\mu$.
    """)
    return


@app.cell
def _(edge_wrenches_2d, mo, np, positively_spans):
    def bisect(pred, lo, hi, iters=50):
        """Largest x in [lo, hi] with pred(x) True, assuming pred flips once."""
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if pred(mid) else (lo, mid)
        return lo

    def box_fc(d, mu, w=6.0):
        c = [(np.array([-w / 2, -d / 2]), [1, 0]), (np.array([w / 2, d / 2]), [-1, 0])]
        return positively_spans(edge_wrenches_2d(c, mu))

    def disk_fc(delta, mu):
        p1, p2 = np.array([-1.0, 0.0]), np.array([np.cos(delta), np.sin(delta)])
        return positively_spans(edge_wrenches_2d([(p1, -p1), (p2, -p2)], mu))

    def wedge_fc(beta, mu, y=0.0):
        n1, n2 = np.array([np.cos(beta), -np.sin(beta)]), np.array([-np.cos(beta), -np.sin(beta)])
        x = 1.0
        return positively_spans(edge_wrenches_2d([(np.array([-x, y]), n1), (np.array([x, y]), n2)], mu))

    _rows = []
    for _mu in [0.1, 0.2, 0.4, 0.5, 1.0]:
        _d = bisect(lambda d: box_fc(d, _mu), 0, 20)
        _dl = np.degrees(bisect(lambda t: disk_fc(t, _mu), 0, np.pi * 0.999))
        _be = np.degrees(bisect(lambda b: wedge_fc(b, _mu), 0, np.pi / 2 * 0.999))
        _rows.append(f"| {_mu} | {_d:.3f} | {_mu*6:.3f} | {_dl:.2f} | {np.degrees(2*np.arctan(_mu)):.2f} | {_be:.2f} | {np.degrees(np.arctan(_mu)):.2f} |")
    mo.md(
        "| μ | box: max offset (w = 6 cm) | μw | disk: max misplacement (deg) | 2 arctan μ | wedge: max half-angle (deg) | arctan μ |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(_rows) +
        "\n\nSlides: $w = 6$ cm gives 3 cm at $\\mu = 0.5$ and 1.2 cm at $\\mu = 0.2$; the disk allows "
        "43.6° at $\\mu = 0.4$ (the tutorial's worked example) and 11.4° at $\\mu = 0.1$."
    )
    return (bisect,)


@app.cell
def _(mo, np):
    mo.md(
        f"""
**The shortcut on circles (slide 25).** On a circle the segment makes the same angle
$\\beta$ with both normals and $\\angle(-\\hat{{\\mathbf n}}_0, \\hat{{\\mathbf n}}_1) = 2\\beta$,
so the normal-angle test and Nguyen's condition coincide. Numerically, for fingers
$\\delta$ from opposite: normal angle = {', '.join(f'{np.degrees(np.pi - abs(np.pi - d)):.1f}°' for d in np.radians([10, 30, 43.6]))}
for $\\delta$ = 10°, 30°, 43.6°: exactly $\\delta$, i.e. $2\\beta$ with $\\beta = \\delta/2$.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Grasps in 3D: hard vs. soft fingers (Part 5, slides 6–7)

    3D primitive wrenches are $(\mathbf f, \mathbf p\times\mathbf f)\in\mathbb R^6$.
    A hard finger's cone is replaced by $k$ edges $\hat{\mathbf n} +
    \mu(\cos\theta_j\,\hat{\mathbf t}_1 + \sin\theta_j\,\hat{\mathbf t}_2)$.
    A soft finger adds two more generators: pressing with a twist of
    $\pm\gamma$ about the normal.
    """)
    return


@app.cell
def _(np):
    def contact_frame(n, up=(0.0, 0.0, 1.0)):
        """Inward normal n, t1 = the 'up' direction projected onto the tangent plane, t2 = n x t1."""
        n = np.asarray(n, float) / np.linalg.norm(n)
        t1 = np.asarray(up, float) - (np.asarray(up, float) @ n) * n
        if np.linalg.norm(t1) < 1e-9:
            t1 = np.array([1.0, 0, 0]) - n[0] * n
        t1 /= np.linalg.norm(t1)
        return n, t1, np.cross(n, t1)

    def cone_edges_3d(n, mu, k=64, how="inscribed", phase=0.0):
        n, t1, t2 = contact_frame(n)
        r = mu if how == "inscribed" else mu / np.cos(np.pi / k)
        ph = phase if how == "inscribed" else phase + np.pi / k
        th = ph + 2 * np.pi * np.arange(k) / k
        return [n + r * (np.cos(a) * t1 + np.sin(a) * t2) for a in th]

    def wrenches_3d(contacts, mu, k=64, soft=None, how="inscribed", phase=0.0, lam=1.0):
        """contacts: list of (point, inward normal). Columns are (f, p x f / lam)."""
        cols = []
        for p, n in contacts:
            p = np.asarray(p, float)
            for f in cone_edges_3d(n, mu, k, how, phase):
                cols.append(np.r_[f, np.cross(p, f) / lam])
            if soft:
                nn = np.asarray(n, float) / np.linalg.norm(n)
                for s in (1, -1):
                    cols.append(np.r_[nn, (np.cross(p, nn) + s * soft * nn) / lam])
        return np.array(cols).T

    return (wrenches_3d,)


@app.cell
def _(bisect, mo, np, positively_spans, wrenches_3d):
    _R = 1.0
    _pinch = [([_R, 0, 0], [-1, 0, 0]), ([-_R, 0, 0], [1, 0, 0])]
    _Wh = wrenches_3d(_pinch, 0.5)
    _Ws = wrenches_3d(_pinch, 0.5, soft=0.005)

    def _sphere_pinch(delta, mu, soft):
        p2 = np.array([-np.cos(delta), np.sin(delta), 0.0])
        return positively_spans(wrenches_3d([([1.0, 0, 0], [-1, 0, 0]), (p2, -p2)], mu, soft=soft))

    _crit = np.degrees(bisect(lambda t: _sphere_pinch(t, 0.5, 0.005), 0, np.pi * 0.99, iters=30))
    mo.md(
        f"""
Antipodal pinch of a unit ball, $\\mu = 0.5$:

- two **hard** fingers: rank **{np.linalg.matrix_rank(_Wh, tol=1e-9)}** of 6, force closure
  **{positively_spans(_Wh)}**. The missing direction is the twist about the grasp axis.
- two **soft** fingers ($\\gamma$ = 5 mm, MuJoCo's default torsional coefficient):
  force closure **{positively_spans(_Ws)}**.
- soft fingers, second finger slid $\\delta$ away from opposite: force closure until
  $\\delta$ = **{_crit:.2f}°** vs. $2\\arctan 0.5$ = {np.degrees(2*np.arctan(0.5)):.2f}°.
  Nguyen's condition, now in 3D.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Three fingers on a ball (Part 5, slides 14–16; Part 6, slide 11)

    Fingers 120° apart around the circle at latitude $\varphi$. The slides
    claim force closure iff $\tan\varphi < \mu$. Then: what do 4-sided
    pyramids report, depending on how they are turned about the normal?
    (Contact frame: $\hat{\mathbf t}_1$ points "up" along the meridian,
    i.e. toward the grip plane's side.)
    """)
    return


@app.cell
def _(bisect, mo, np, positively_spans, wrenches_3d):
    def ring_contacts(phi, R=1.0, angles=(0, 120, 240)):
        out = []
        for a in np.radians(angles):
            p = R * np.array([np.cos(phi) * np.cos(a), np.cos(phi) * np.sin(a), np.sin(phi)])
            out.append((p, -p))
        return out

    def ball_fc(phi, mu, **kw):
        return positively_spans(wrenches_3d(ring_contacts(phi), mu, **kw))

    _rows = []
    for _mu in [0.2, 0.5, 1.0]:
        _c = np.degrees(bisect(lambda p: ball_fc(p, _mu, k=128), 0, np.radians(85), iters=30))
        _rows.append(f"| {_mu} | {_c:.2f} | {np.degrees(np.arctan(_mu)):.2f} |")
    _mu = 0.5
    _variants = [
        ("diamond (inscribed), corner toward the grip plane", dict(k=4, how="inscribed", phase=0.0)),
        ("diamond (inscribed), side toward the grip plane", dict(k=4, how="inscribed", phase=np.pi / 4)),
        ("box (circumscribed), side toward the grip plane", dict(k=4, how="circumscribed", phase=0.0)),
        ("box (circumscribed), corner toward the grip plane", dict(k=4, how="circumscribed", phase=np.pi / 4)),
    ]
    _vrows = "\n".join(
        f"| {name} | {np.degrees(bisect(lambda p: ball_fc(p, _mu, **kw), 0, np.radians(85), iters=30)):.2f} |"
        for name, kw in _variants
    )
    _at30 = ball_fc(np.radians(30), _mu, k=4, how="circumscribed", phase=np.pi / 4)
    _true30 = ball_fc(np.radians(30), _mu, k=128)
    mo.md(
        "| μ | critical φ, 6D test (deg) | arctan μ (deg) |\n|---|---|---|\n" + "\n".join(_rows) +
        "\n\nPyramid variants at μ = 0.5 (largest latitude certified as force closure):\n\n"
        "| cone model | certifies up to (deg) |\n|---|---|\n" + _vrows +
        f"\n\nAt φ = 30°: the corner-up box says force closure = **{_at30}**; the true cone says **{_true30}**. "
        f"A false positive. (Slide: 26.6 / 19.5 / 26.6 / 35.3°; closed forms "
        f"arctan(μ/√2) = {np.degrees(np.arctan(_mu/np.sqrt(2))):.2f}°, arctan(μ√2) = {np.degrees(np.arctan(_mu*np.sqrt(2))):.2f}°.)"
    )
    return (ring_contacts,)


@app.cell
def _(bisect, mo, np, positively_spans, wrenches_3d):
    _eq = [(np.array([np.cos(t), np.sin(t), 0.0]), -np.array([np.cos(t), np.sin(t), 0.0])) for t in np.radians([0, 60, 120])]
    _fc = lambda mu: positively_spans(wrenches_3d(_eq, mu, k=64))
    _mu_min = bisect(lambda m: not _fc(m), 0.01, 5.0, iters=30)   # smallest mu that works
    mo.md(
        f"""
**Bunched fingers (Part 5, check-yourself Q4).** Three fingers on the equator at 0°, 60°, 120°:
force closure at μ = 0.5: **{_fc(0.5)}**, at μ = 2: **{_fc(2.0)}**. The smallest μ that works
is about **{_mu_min:.3f}**.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. The three-force theorem, numerically (Part 5, slides 11–12)

    Take three contacts on a random ellipsoid. The **internal forces**
    (finger forces with zero total wrench) form the null space of the
    $6\times 9$ grasp map. The theorem says every internal force lies in the
    plane of the three contacts, with the three lines of action meeting at one
    point (or parallel).
    """)
    return


@app.cell
def _(cross2, mo, np, null_space):
    _rng = np.random.default_rng(4)
    _worst_plane, _worst_concur = 0.0, 0.0
    for _ in range(200):
        _P = _rng.normal(size=(3, 3)) * np.array([1.5, 1.0, 0.7])
        _G = np.zeros((6, 9))
        for _i in range(3):
            _G[:3, 3 * _i:3 * _i + 3] = np.eye(3)
            _x, _y, _z = _P[_i]
            _G[3:, 3 * _i:3 * _i + 3] = np.array([[0, -_z, _y], [_z, 0, -_x], [-_y, _x, 0]])
        _Nsp = null_space(_G)
        _f = (_Nsp @ _rng.normal(size=_Nsp.shape[1])).reshape(3, 3)
        _nrm = np.cross(_P[1] - _P[0], _P[2] - _P[0]); _nrm /= np.linalg.norm(_nrm)
        _worst_plane = max(_worst_plane, np.abs(_f @ _nrm).max() / np.abs(_f).max())
        # in-plane coordinates, then: does line 3 pass through the crossing of lines 1 and 2?
        _u = (_P[1] - _P[0]) / np.linalg.norm(_P[1] - _P[0]); _w = np.cross(_nrm, _u)
        _p2 = [np.array([(p - _P[0]) @ _u, (p - _P[0]) @ _w]) for p in _P]
        _f2 = [np.array([f @ _u, f @ _w]) for f in _f]
        _den = cross2(_f2[0], _f2[1])
        if abs(_den) > 1e-6:
            _t = cross2(_p2[1] - _p2[0], _f2[1]) / _den
            _q = _p2[0] + _t * _f2[0]
            _worst_concur = max(_worst_concur, abs(cross2(_q - _p2[2], _f2[2])) / (np.linalg.norm(_f2[2]) * (1 + np.linalg.norm(_q))))
    mo.md(
        f"""
Over 200 random triples: null space dimension {_Nsp.shape[1]} (= 9 − 6); largest out-of-plane
force component {_worst_plane:.1e}; largest distance of line 3 from the crossing of lines 1
and 2 (relative) {_worst_concur:.1e}. Coplanar and concurrent. ✓
"""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Mirtich & Canny's optimum grasps (Part 5, slides 9, 18)

    **Two fingers, collinear normals, spacing $d$.** With total normal force
    at most 1 (each finger $1/2$), friction resists a pure twist of at most
    $\mu d / 2$.

    **Three fingers, equilateral grasp.** Each finger presses with at most
    $f$; the worst-direction twist it can resist is
    $\tfrac{\sqrt 3}{2} f(\mu s_o - s_i)$. We solve both as linear programs.
    """)
    return


@app.cell
def _(cross2, linprog, mo, np):
    def max_twist_2d(P, N, mu, sign, total=None, each=None):
        """Largest pure moment (sign * tau) the contacts can make with zero net force.
        Variables: normal c_i and tangential s_i per contact."""
        k = len(P)
        T = [np.array([-n[1], n[0]]) for n in N]
        Aeq = np.zeros((2, 2 * k)); obj = np.zeros(2 * k)
        for i in range(k):
            Aeq[:, i], Aeq[:, k + i] = N[i], T[i]
            obj[i], obj[k + i] = cross2(P[i], N[i]), cross2(P[i], T[i])
        Aub, bub = [], []
        for i in range(k):
            for s in (1, -1):
                r = np.zeros(2 * k); r[k + i] = s; r[i] = -mu; Aub.append(r); bub.append(0.0)
        if total is not None:
            r = np.zeros(2 * k); r[:k] = 1; Aub.append(r); bub.append(total)
        ub = each if each is not None else None
        res = linprog(-sign * obj, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=[0, 0],
                      bounds=[(0, ub)] * k + [(None, None)] * k, method="highs")
        return sign * obj @ res.x

    _mu, _d = 0.5, 3.0
    _P2 = [np.array([-_d / 2, 0.0]), np.array([_d / 2, 0.0])]
    _N2 = [np.array([1.0, 0.0]), np.array([-1.0, 0.0])]
    _q2 = min(max_twist_2d(_P2, _N2, _mu, s, total=1.0) for s in (1, -1))

    def equilateral(so, e):
        h = so * np.sqrt(3) / 2
        V = [np.array([-so / 2, -h / 3]), np.array([so / 2, -h / 3]), np.array([0.0, 2 * h / 3])]
        P, N = [], []
        for i in range(3):
            A, B = V[i], V[(i + 1) % 3]
            u = (B - A) / np.linalg.norm(B - A)
            P.append((A + B) / 2 + e * u); N.append(np.array([-u[1], u[0]]))
        meet = lambda p, a, q, b: p + cross2(q - p, b) / cross2(a, b) * a
        I = [meet(P[i], N[i], P[(i + 1) % 3], N[(i + 1) % 3]) for i in range(3)]
        return P, N, np.linalg.norm(I[0] - I[1])

    _rows = []
    for _so, _e, _m in [(3.2, 0.0, 0.5), (3.2, 0.3, 0.5), (2.0, 0.2, 0.8), (3.0, 0.4, 0.3)]:
        _P, _N, _si = equilateral(_so, _e)
        _lp = min(max_twist_2d(_P, _N, _m, s, each=1.0) for s in (1, -1))
        _rows.append(f"| {_so} | {_si:.3f} | {_m} | {_lp:.4f} | {max(0.0, np.sqrt(3)/2*(_m*_so - _si)):.4f} |")
    mo.md(
        f"Two fingers, $\\mu = {_mu}$, $d = {_d}$: LP gives **{_q2:.4f}**, formula $\\mu d/2$ = {_mu*_d/2:.4f}.\n\n"
        "| outer side s_o | inner side s_i | μ | LP: worst-direction twist (f = 1) | (√3/2)(μ s_o − s_i) |\n"
        "|---|---|---|---|---|\n" + "\n".join(_rows) +
        "\n\nWhen $\\mu s_o < s_i$ the grasp can't resist that twist direction at all (LP: 0)."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Ferrari–Canny quality $\varepsilon$ (Part 6, slides 6–10)

    With total finger normal force at most 1, the reachable wrenches are the
    convex hull of the primitive wrenches. $\varepsilon$ is the distance from
    the origin to the nearest hull facet (0 if the origin is outside). Qhull
    reports each facet as $\mathbf a\cdot\mathbf x + b \le 0$ with
    $\|\mathbf a\| = 1$, so the distance is $-b$.
    """)
    return


@app.cell
def _(ConvexHull, np):
    def epsilon_l1(W):
        """Ferrari-Canny epsilon (L1 version) for primitive wrenches in the columns of W."""
        try:
            hull = ConvexHull(np.asarray(W, float).T)
        except Exception:
            return 0.0                                   # flat hull: origin can't be strictly inside
        return max(0.0, float(-hull.equations[:, -1].max()))

    return (epsilon_l1,)


@app.cell
def _(edge_wrenches_2d, epsilon_l1, mo, np, plt):
    _w, _mu = 2.0, 0.5
    _eps = lambda d, mu, lam=1.0: epsilon_l1(edge_wrenches_2d(
        [(np.array([-_w / 2, -d / 2]), [1, 0]), (np.array([_w / 2, d / 2]), [-1, 0])], mu, lam))
    _d = np.linspace(0, 1.6, 161)
    _fig, _ax = plt.subplots(figsize=(6.5, 3))
    for _m in [0.3, 0.5, 0.8]:
        _ax.plot(_d, [_eps(x, _m) for x in _d], lw=2, label=f"μ = {_m}")
        _ax.axvline(_m * _w, ls=":", c="gray", lw=1)
    _ax.set_xlabel("finger offset d"); _ax.set_ylabel("ε")
    _ax.set_title("Off-center box pinch (w = 2, λ = 1): ε hits 0 at d = μw (slide 9)")
    _ax.legend(); _ax.grid(alpha=0.3); _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(
            f"At d = 0.3, μ = 0.5: ε = {_eps(0.3, 0.5):.4f} (slide 9's readout: 0.2333). "
            f"Scaling torques by λ = 0.5 / 1 / 2 gives ε = {_eps(0.3, 0.5, 0.5):.4f} / {_eps(0.3, 0.5, 1.0):.4f} / "
            f"{_eps(0.3, 0.5, 2.0):.4f}: the number depends on λ (slide 8)."
        ),
    ])
    return


@app.cell
def _(epsilon_l1, mo, ring_contacts, wrenches_3d):
    _C = ring_contacts(0.0)
    _exact = epsilon_l1(wrenches_3d(_C, 0.5, k=128))
    _rows = []
    for _k in [3, 4, 8, 16, 32]:
        _e = epsilon_l1(wrenches_3d(_C, 0.5, k=_k))
        _rows.append(f"| inscribed, k = {_k} | {_e:.3f} | {100*(_e/_exact-1):+.0f}% |")
    _ec = epsilon_l1(wrenches_3d(_C, 0.5, k=4, how="circumscribed"))
    _rows.append(f"| circumscribed, k = 4 | {_ec:.3f} | {100*(_ec/_exact-1):+.0f}% |")
    mo.md(
        f"Three fingers on the equator, μ = 0.5, λ = R (Part 6, slide 10). Reference (k = 128): "
        f"**ε = {_exact:.3f}**.\n\n| cone model | ε | error |\n|---|---|---|\n" + "\n".join(_rows)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Case study: the sphere pinch filter (Part 4, slides 22–26; Part 6, slide 13)

    The filter accepts a thumb–index pinch of a sphere when
    $\cos\angle(-\hat{\mathbf n}_0, \hat{\mathbf n}_1) \ge \cos(2\arctan\mu)$.
    Random pinches on a sphere in 3D: compare the filter with the exact 6D
    test under hard and soft contact models.
    """)
    return


@app.cell
def _(mo, np, positively_spans, wrenches_3d):
    _rng = np.random.default_rng(5)
    _agree_soft, _n, _hard_fc, _passed = 0, 0, 0, 0
    for _ in range(400):
        _mu = _rng.uniform(0.1, 1.2)
        _p0, _p1 = (v / np.linalg.norm(v) for v in _rng.normal(size=(2, 3)))
        _n0, _n1 = _p0, _p1                           # outward normals of a unit sphere
        _ang = np.arccos(np.clip(-_n0 @ _n1, -1, 1))
        if abs(_ang - 2 * np.arctan(_mu)) < 0.01:
            continue                                  # a pyramid can't resolve the exact boundary
        _filter = np.cos(_ang) >= np.cos(2 * np.arctan(_mu))
        _C = [(_p0, -_n0), (_p1, -_n1)]
        _soft = positively_spans(wrenches_3d(_C, _mu, k=64, soft=0.005))
        _hard = positively_spans(wrenches_3d(_C, _mu, k=64))
        _n += 1; _passed += _filter
        _agree_soft += (_filter == _soft); _hard_fc += _hard
    mo.md(
        f"""
{_n} random sphere pinches ({_passed} pass the filter):

- filter vs. exact test with **soft** fingertips: agree on **{_agree_soft} / {_n}**. Exact for spheres.
- exact test with **hard** point contacts: force closure in **{_hard_fc}** cases. Never: the
  twist about the thumb–index axis is free.

(Pinches within 0.01 rad of the boundary are skipped: a 64-sided inscribed pyramid is very
slightly conservative there, which is Part 2's lesson again.)
"""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    **Things to try:**

    - In §2, replace `edge_wrenches_2d` with the frictionless version and
      count how many random two-finger grasps survive (none should).
    - In §5, give each finger a different μ, or move one finger to a different
      latitude, and find the new boundary by bisection.
    - In §8, change λ and watch the ranking of two different box pinches flip.
    - Implement the $L_\infty$ quality (each finger at most 1): the grasp
      wrench space is the Minkowski sum of each finger's hull, i.e. the hull
      of all sums of one edge-or-zero per finger.

    **Previous:** `05_contact_solvers.py`. **Slides:** `slides/index.html`.
    """)
    return


if __name__ == "__main__":
    app.run()
