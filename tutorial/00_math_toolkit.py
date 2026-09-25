# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "matplotlib", "scipy"]
# ///
"""Companion notebook for Part I, the mathematical toolkit: slide decks 01-04.

Backs slides/01_vectors_and_matrices.html, 02_eigenvalues_and_least_squares.html,
03_derivatives_and_jacobians.html and 04_probability_and_sampling.html. Every number on
those slides that a reader cannot compute by hand is computed by a function in this notebook,
in the section after that deck's "Deck NN" header cell; slides link to those functions with
data-code="00_math_toolkit.py:function_name".

Run locally:        uvx marimo edit --sandbox 00_math_toolkit.py
Run as a script:    uv run --script 00_math_toolkit.py
Slides:             open slides/index.html (Part I)
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

    return mo, np, plt, scipy


@app.cell
def _(mo):
    mo.md(r"""
    # Mathematical toolkit: numerical companion to decks 01–04

    Part I builds the tools the rest of the series uses: vectors and matrices as linear maps,
    rank and null spaces, projections, least squares and the SVD, derivatives and Jacobians,
    and Gaussians and sampling. The cells after each deck's header compute the numbers on that
    deck's slides. Numbers quoted from the tex tutorial are treated as unverified until a cell
    here recomputes them.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Deck 01 · Vectors, matrices, and linear maps

    Slides: `slides/01_vectors_and_matrices.html`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Vectors (slides `vectors-arrows-lists` to `unit-balls`)

    `d01_vector_length` backs `vectors-arrows-lists`; `d01_coords` backs `linear-combinations`
    and `basis-and-coordinates`; `d01_independence_det` backs `linear-independence`;
    `d01_dot_angle` backs `dot-product` and `dot-product-angle`; `d01_pnorm` and `d01_pnorms`
    back `norms` and `unit-balls`. The running vectors are $\mathbf u = (1, 1)$ and
    $\mathbf w = (-1, 3)$.
    """)
    return


@app.cell
def _(np):
    def d01_vector_length(v):
        """Euclidean length sqrt(v1^2 + ... + vn^2) of a vector (slide vectors-arrows-lists)."""
        v = np.asarray(v, dtype=float)
        return float(np.sqrt(np.sum(v * v)))

    def d01_coords(u, w, x):
        """Coordinates (c1, c2) of x in the basis (u, w) of the plane: c1 u + c2 w = x.

        Solves the 2x2 system [u w] c = x by Cramer's rule; None when u and w are parallel.
        """
        u, w, x = (np.asarray(t, dtype=float) for t in (u, w, x))
        det = u[0] * w[1] - u[1] * w[0]
        if abs(det) <= 1e-12 * max(1.0, np.abs(np.r_[u, w]).max() ** 2):
            return None
        c1 = (x[0] * w[1] - x[1] * w[0]) / det
        c2 = (u[0] * x[1] - u[1] * x[0]) / det
        return [float(c1), float(c2)]

    def d01_independence_det(h):
        """det and rank of [v1 v2 v3] with v1 = (1,0,1), v2 = (0,1,1), v3 = (1,1,2+h).

        The determinant equals h, so the three vectors are dependent exactly when h = 0
        (then v1 + v2 - v3 = 0). Slide linear-independence.
        """
        V = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0 + h]]).T
        return {"det": float(np.linalg.det(V)), "rank": int(np.linalg.matrix_rank(V))}

    def d01_dot_angle(a, b):
        """Dot product, lengths, cosine and angle (degrees) of two vectors (slide dot-product)."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        dot = float(a @ b)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return {"dot": dot, "norm_a": na, "norm_b": nb, "cos": None, "deg": None}
        c = max(-1.0, min(1.0, dot / (na * nb)))
        return {"dot": dot, "norm_a": na, "norm_b": nb, "cos": c, "deg": float(np.degrees(np.arccos(c)))}

    def d01_pnorm(x, p):
        """The p-norm (sum |x_i|^p)^(1/p) for p >= 1; p = inf gives max |x_i|."""
        x = np.abs(np.asarray(x, dtype=float))
        if p == np.inf:
            return float(x.max())
        if p < 1:
            raise ValueError("the p-norm is a norm only for p >= 1")
        m = x.max()
        if m == 0.0:
            return 0.0
        return float(m * np.sum((x / m) ** p) ** (1.0 / p))

    def d01_pnorms(x):
        """The 1-, 2- and infinity-norms of x (slides norms and unit-balls)."""
        return [d01_pnorm(x, 1), d01_pnorm(x, 2), d01_pnorm(x, np.inf)]

    return d01_coords, d01_dot_angle, d01_independence_det, d01_pnorm, d01_pnorms, d01_vector_length


@app.cell
def _(d01_coords, d01_dot_angle, d01_independence_det, d01_pnorm, d01_pnorms, d01_vector_length, mo, np):
    _u, _w = [1, 1], [-1, 3]
    _c = d01_coords(_u, _w, [1, 5])
    _d1 = d01_dot_angle([3, 4], [4, 0])
    _d2 = d01_dot_angle([3, 4], [4, -3])
    _d3 = d01_dot_angle([1, 0, 1], [0, 1, 1])
    _n = d01_pnorms([3, -4])
    _ind = [d01_independence_det(h) for h in (0.0, 0.5, -1.0)]
    # p = 1/2 breaks the triangle inequality: |e1 + e2| = 4 > |e1| + |e2| = 2.
    _half = (np.sum(np.abs([1.0, 1.0]) ** 0.5) ** 2, 1.0 + 1.0)
    assert np.allclose(_c, [2, 1]) and np.isclose(d01_vector_length([3, 2]), np.sqrt(13))
    assert np.isclose(_d1["dot"], 12) and np.isclose(_d1["cos"], 0.6) and np.isclose(_d2["dot"], 0)
    assert np.isclose(_d3["deg"], 60) and np.allclose(_n, [7, 5, 4])
    assert [r["rank"] for r in _ind] == [2, 3, 3]
    mo.md(f"""
| slide | quantity | value |
|---|---|---|
| vectors-arrows-lists | length of (3, 2) | {d01_vector_length([3, 2]):.4f} (= √13) |
| linear-combinations, basis-and-coordinates | coordinates of (1, 5) in (u, w) | {_c} |
| basis-and-coordinates | coordinates of (0, 4), (3, 1) | {d01_coords(_u, _w, [0, 4])}, {d01_coords(_u, _w, [3, 1])} |
| linear-independence | det, rank at h = 0, 0.5, -1 | {[(round(r['det'], 12), r['rank']) for r in _ind]} |
| dot-product | (3, 4)·(4, 0), cos, angle | {_d1['dot']:.0f}, {_d1['cos']:.3f}, {_d1['deg']:.2f}° |
| dot-product | (3, 4)·(4, −3) | {_d2['dot']:.0f} |
| dot-product-angle | angle between (1, 0, 1) and (0, 1, 1) | {_d3['deg']:.2f}° |
| dot-product | angle between u and w | {d01_dot_angle(_u, _w)['deg']:.2f}° |
| norms | ‖(3, −4)‖ for p = 1, 2, ∞ | {_n} |
| norms | ‖(3, −4)‖ for p = 1.5, 3, 8 | {[round(d01_pnorm([3, -4], p), 4) for p in (1.5, 3, 8)]} |
| norms (notes) | p = 1/2: ‖e1 + e2‖ vs ‖e1‖ + ‖e2‖ | {_half[0]:.0f} > {_half[1]:.0f} |
""")
    return


@app.cell
def _(d01_pnorm, np, plt):
    # Unit balls of the 1-, 2- and infinity-norms (slide unit-balls).
    _t = np.linspace(0, 2 * np.pi, 721)
    _fig, _ax = plt.subplots(figsize=(3.4, 3.4))
    for _p, _c in [(1, "#cc6600"), (2, "#0000cc"), (np.inf, "#008800")]:
        _dirs = np.c_[np.cos(_t), np.sin(_t)]
        _pts = np.array([d / d01_pnorm(d, _p) for d in _dirs])
        _ax.plot(_pts[:, 0], _pts[:, 1], color=_c, label=f"p = {_p}")
    _ax.set_aspect("equal")
    _ax.legend(fontsize=8)
    _ax.set_title("unit balls (slide unit-balls)", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Matrices as maps (slides `matrix-vector-product` to `tall-matrix-embeds`)

    `d01_det2` backs `linear-map-grid` and `determinant-area`; `d01_inv2` backs
    `identity-and-inverse`; `d01_compose` backs `composition`; `d01_transpose_check` backs
    `transpose`; `d01_parallelogram_area` backs `determinant-derivation`; `d01_nonsquare_maps`
    backs `wide-matrix-flattens` and `tall-matrix-embeds`.
    """)
    return


@app.cell
def _(np):
    D01_A = np.array([[1.0, -1.0], [1.0, 3.0]])          # columns u = (1, 1), w = (-1, 3)
    D01_A23 = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])  # flattens R^3 onto R^2
    D01_M = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]])  # rank 2
    D01_A34 = np.array([[1.0, 2.0, 0.0, 1.0], [2.0, 4.0, 1.0, 4.0], [3.0, 6.0, 1.0, 5.0]])

    def d01_det2(A):
        """ad - bc for A = [[a, b], [c, d]]: the signed area of the image of the unit square."""
        (a, b), (c, d) = np.asarray(A, dtype=float)
        return float(a * d - b * c)

    def d01_inv2(A):
        """(1/(ad - bc)) [[d, -b], [-c, a]], or None when ad - bc = 0 (slide identity-and-inverse)."""
        (a, b), (c, d) = np.asarray(A, dtype=float)
        det = a * d - b * c
        if det == 0.0:
            return None
        return (np.array([[d, -b], [-c, a]]) / det).tolist()

    def d01_compose(B, A):
        """The matrix of 'first A, then B': (BA)[i][j] = row i of B . column j of A."""
        B, A = np.asarray(B, dtype=float), np.asarray(A, dtype=float)
        if B.shape[1] != A.shape[0]:
            raise ValueError(f"shapes {B.shape} and {A.shape} do not compose")
        return (B @ A).tolist()

    def d01_transpose_check(A, x, y):
        """(A x) . y and x . (A^T y), which are equal for every A, x, y (slide transpose)."""
        A, x, y = (np.asarray(t, dtype=float) for t in (A, x, y))
        return [float((A @ x) @ y), float(x @ (A.T @ y))]

    def d01_parallelogram_area(a, c, b, d):
        """Area of the parallelogram on columns (a, c) and (b, d) by cutting its bounding box.

        Box (a + b)(c + d) minus two triangles of area ac/2, two of area bd/2 and two
        rectangles bc. Valid for 0 <= b <= a and 0 <= c <= d (slide determinant-derivation).
        """
        box = (a + b) * (c + d)
        area = box - a * c - b * d - 2 * b * c
        return {"box": box, "ac": a * c, "bd": b * d, "2bc": 2 * b * c, "area": area, "det": a * d - b * c}

    def d01_nonsquare_maps(t=0.0, s=(1.0, 0.5)):
        """The 2x3 map A23 and the 3x2 map B32 = A23^T (slides wide-matrix-flattens, tall-matrix-embeds).

        x(t) = (2, 1, 0) + t (-1, -1, 1) maps to (2, 1) for every t; the unit cube maps onto a
        hexagon; B32 s = (s1, s2, s1 + s2) lies on the plane z = x + y, which misses (0, 0, 1).
        """
        x = np.array([2.0, 1.0, 0.0]) + t * np.array([-1.0, -1.0, 1.0])
        cube = [np.array([i, j, k], dtype=float) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        image = sorted({tuple((D01_A23 @ v).tolist()) for v in cube})
        B32 = D01_A23.T
        return {
            "x": x.tolist(), "Ax": (D01_A23 @ x).tolist(), "cube_image": image,
            "Bs": (B32 @ np.asarray(s, dtype=float)).tolist(),
            "rank_A23": int(np.linalg.matrix_rank(D01_A23)), "rank_B32": int(np.linalg.matrix_rank(B32)),
            # (0, 0, 1) is off the plane z = x + y: its z - x - y is 1, not 0.
            "e3_offset": 1.0 - 0.0 - 0.0,
        }

    return (D01_A, D01_A23, D01_A34, D01_M, d01_compose, d01_det2, d01_inv2,
            d01_nonsquare_maps, d01_parallelogram_area, d01_transpose_check)


@app.cell
def _(D01_A, D01_A23, d01_compose, d01_det2, d01_inv2, d01_nonsquare_maps, d01_parallelogram_area, d01_transpose_check, mo, np):
    _S, _R = [[1, 1], [0, 1]], [[0, -1], [1, 0]]
    _Ai = d01_inv2(D01_A)
    _tc = d01_transpose_check(D01_A23, [1, 2, 3], [1, -1])
    _pa = d01_parallelogram_area(3, 1, 1, 2)
    _nm = d01_nonsquare_maps(t=1.0, s=(1.0, 0.5))
    _presets = {"identity": [[1, 0], [0, 1]], "rotation 30°": [[np.cos(np.pi / 6), -np.sin(np.pi / 6)], [np.sin(np.pi / 6), np.cos(np.pi / 6)]],
                "shear": [[1, 1], [0, 1]], "stretch": [[2, 0], [0, 0.5]], "projection": [[1, 0], [0, 0]],
                "reflection": [[1, 0], [0, -1]], "running A": D01_A.tolist()}
    assert np.isclose(d01_det2(D01_A), 4) and np.allclose(np.array(_Ai) * 4, [[3, 1], [-1, 1]])
    assert np.allclose(np.array(_Ai) @ [1, 5], [2, 1])
    assert np.allclose(d01_compose(_R, _S), [[0, -1], [1, 1]]) and np.allclose(d01_compose(_S, _R), [[1, -1], [1, 0]])
    assert np.allclose(_tc, [-1, -1]) and _pa["area"] == _pa["det"] == 5
    assert np.allclose(_nm["x"], [1, 0, 1]) and np.allclose(_nm["Ax"], [2, 1]) and len(_nm["cube_image"]) == 7
    mo.md(f"""
| slide | quantity | value |
|---|---|---|
| matrix-vector-product | A (2, 1) | {(D01_A @ [2, 1]).tolist()} |
| determinant-area | det A | {d01_det2(D01_A):.0f} |
| identity-and-inverse | 4 A⁻¹, A⁻¹ (1, 5) | {(np.array(_Ai) * 4).tolist()}, {(np.array(_Ai) @ [1, 5]).tolist()} |
| composition | RS (shear, then rotate), SR | {d01_compose(_R, _S)}, {d01_compose(_S, _R)} |
| composition | det RS | {d01_det2(d01_compose(_R, _S)):.0f} |
| transpose | (A23 x)·y, x·(A23ᵀ y), x = (1, 2, 3), y = (1, −1) | {_tc} |
| determinant-derivation | box, ac, bd, 2bc, area for (3, 1), (1, 2) | {_pa['box']}, {_pa['ac']}, {_pa['bd']}, {_pa['2bc']}, {_pa['area']} |
| wide-matrix-flattens | x(1), A23 x(1) | {_nm['x']}, {_nm['Ax']} |
| wide-matrix-flattens | image of the unit cube (7 distinct points, hexagon hull) | {[tuple(int(v) for v in p) for p in _nm['cube_image']]} |
| tall-matrix-embeds | B32 (1, 0.5) | {_nm['Bs']} |
| linear-map-grid | det of each preset | {', '.join(f"{k}: {d01_det2(v):.3f}" for k, v in _presets.items())} |
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Systems, rank, and solution sets (slides `row-and-column-pictures` to `velocity-command-count`)

    `d01_classify_2x2` backs `row-and-column-pictures` and `three-outcomes`;
    `d01_rref_steps` backs `gaussian-elimination` and `rref-and-pivots` (it is the same
    Gauss-Jordan rule as `Num.rref` in `lib/num.js`: first nonzero pivot, clear above and
    below, log every operation); `d01_consistency` backs `consistency` and `column-space`;
    `d01_null_basis` backs `null-space` and `null-space-basis`; `d01_four_subspaces` backs
    `row-space-and-rank`; `d01_rank_nullity_table` backs `rank-nullity`; `d01_solution_set`
    backs `solution-set` and `homogeneous-systems`; `d01_table_legs` backs `four-legged-table`;
    `d01_goal_dims` backs `identity-strictest-goal`; `d01_stacked_ranks` backs
    `stacked-constraints` and `velocity-command-count`.
    """)
    return


@app.cell
def _(np):
    def d01_rank(A):
        """Numerical rank with numpy's default tolerance (the same as Num.rank)."""
        A = np.atleast_2d(np.asarray(A, dtype=float))
        return int(np.linalg.matrix_rank(A)) if A.size else 0

    def d01_classify_2x2(A, b):
        """'one' (with x), 'none', or 'infinite' (with a solution x and a null direction) for A x = b."""
        A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
        rA, rAb = d01_rank(A), d01_rank(np.column_stack([A, b]))
        if rA == 2:
            return {"status": "one", "x": np.linalg.solve(A, b).tolist(), "rankA": rA, "rankAb": rAb}
        if rA == rAb:
            return {"status": "infinite", "x": (np.linalg.pinv(A) @ b).tolist(), "rankA": rA, "rankAb": rAb}
        return {"status": "none", "x": None, "rankA": rA, "rankAb": rAb}

    def d01_rref_steps(A, tol=None):
        """Gauss-Jordan elimination with a log of every row operation.

        The rule of Num.rref(A) in lib/num.js ('first' pivots): for each column, take the first
        row at or below the current one with a nonzero entry, swap it up, scale it to make the
        pivot 1, and subtract multiples of it from every other row. Entries with |x| <= tol
        (default 1e-10 max|a_ij|) are set to 0. Each step is (op, i, j, k, text): 'swap' rows
        i, j; 'scale' R_i <- k R_i; 'add' R_i <- R_i + k R_j (rows 0-based, text 1-based).
        """
        M = np.array(A, dtype=float)
        m, n = M.shape
        tol = 1e-10 * np.abs(M).max() if tol is None else tol
        steps, pivots = [], []

        def clean():
            M[np.abs(M) <= tol] = 0.0

        clean()
        r = 0
        for c in range(n):
            if r >= m:
                break
            rows = [i for i in range(r, m) if abs(M[i, c]) > tol]
            if not rows:
                continue
            p = rows[0]
            if p != r:
                M[[r, p]] = M[[p, r]]
                steps.append(("swap", r, p, 0.0, f"R{r + 1} <-> R{p + 1}", M.copy()))
            piv = M[r, c]
            if piv != 1.0:
                M[r] = M[r] / piv
                M[r, c] = 1.0
                clean()
                steps.append(("scale", r, r, 1.0 / piv, f"R{r + 1} <- ({1.0 / piv:g}) R{r + 1}", M.copy()))
            for i in range(m):
                if i == r or M[i, c] == 0.0:
                    continue
                f = M[i, c]
                M[i] = M[i] - f * M[r]
                M[i, c] = 0.0
                clean()
                steps.append(("add", i, r, -f, f"R{i + 1} <- R{i + 1} + ({-f:g}) R{r + 1}", M.copy()))
            pivots.append(c)
            r += 1
        return {"R": M.tolist(), "pivots": pivots, "rank": len(pivots),
                "steps": [s[:5] for s in steps], "matrices": [s[5].tolist() for s in steps]}

    def d01_consistency(A, b):
        """rank A, rank [A | b], and whether A x = b has a solution (ranks equal)."""
        A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
        rA, rAb = d01_rank(A), d01_rank(np.column_stack([A, b]))
        return {"rankA": rA, "rankAb": rAb, "consistent": rA == rAb}

    def d01_null_basis(A):
        """Special solutions of A x = 0 from the RREF: one per free column (free entry 1, others 0)."""
        A = np.asarray(A, dtype=float)
        n = A.shape[1]
        rr = d01_rref_steps(A)
        R, piv = np.array(rr["R"]), rr["pivots"]
        out = []
        for f in (j for j in range(n) if j not in piv):
            x = np.zeros(n)
            x[f] = 1.0
            for i, p in enumerate(piv):
                x[p] = -R[i, f]
            out.append((x + 0.0).tolist())  # + 0.0 turns -0.0 into 0.0
        return out

    def d01_four_subspaces(A):
        """Bases and dimensions of Col(A), Row(A), Null(A) and Null(A^T) read off the RREF.

        Col: the pivot columns of A itself. Row: the nonzero rows of the RREF. Null: the special
        solutions. These bases are not orthonormal; deck 02 builds orthonormal ones.
        """
        A = np.asarray(A, dtype=float)
        m, n = A.shape
        rr = d01_rref_steps(A)
        r = rr["rank"]
        return {
            "rank": r, "pivots": rr["pivots"],
            "col": [A[:, j].tolist() for j in rr["pivots"]],
            "row": rr["R"][:r],
            "null": d01_null_basis(A),
            "left_null": d01_null_basis(A.T),
            "dims": {"col": r, "row": r, "null": n - r, "left_null": m - r},
        }

    def d01_solution_set(A, b):
        """Particular solution (free variables 0) and null-space basis of A x = b; None if inconsistent."""
        A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
        n = A.shape[1]
        rr = d01_rref_steps(np.column_stack([A, b]))
        if n in rr["pivots"]:
            return {"consistent": False, "particular": None, "null": d01_null_basis(A)}
        x = np.zeros(n)
        for i, p in enumerate(rr["pivots"]):
            x[p] = rr["R"][i][n]
        return {"consistent": True, "particular": x.tolist(), "null": d01_null_basis(A)}

    def d01_rank_nullity_table():
        """(name, m, n, rank, dim Null) for the matrices of the rank-nullity slide."""
        mats = {
            "A23 (flattens)": [[1, 0, 1], [0, 1, 1]],
            "M (columns v1, v2, v1 + v2)": [[1, 0, 1], [0, 1, 1], [1, 1, 2]],
            "A34": [[1, 2, 0, 1], [2, 4, 1, 4], [3, 6, 1, 5]],
            "B32 = A23^T (embeds)": [[1, 0], [0, 1], [1, 1]],
            "I3": np.eye(3).tolist(),
            "table E": [[1, 1, 1, 1], [0.5, 0.5, -0.5, -0.5], [0.5, -0.5, -0.5, 0.5]],
        }
        rows = []
        for name, X in mats.items():
            X = np.asarray(X, dtype=float)
            r = d01_rank(X)
            rows.append((name, X.shape[0], X.shape[1], r, X.shape[1] - r))
        return rows

    def d01_table_legs(W=100.0, t=0.0, half=0.5):
        """Leg forces of a square table with legs at (+-half, +-half) carrying W at its centre.

        Legs 1-4 at (h, h), (-h, h), (-h, -h), (h, -h). Rows of E: vertical force, moment about
        the x axis (y_i f_i), moment about the y axis (x_i f_i). E f = (W, 0, 0) has the
        solutions f = (W/4)(1, 1, 1, 1) + t (1, -1, 1, -1); all legs push when |t| <= W/4.
        """
        h = half
        E = np.array([[1, 1, 1, 1], [h, h, -h, -h], [h, -h, -h, h]], dtype=float)
        f = W / 4 * np.ones(4) + t * np.array([1.0, -1.0, 1.0, -1.0])
        return {"E": E.tolist(), "rank": d01_rank(E), "f": f.tolist(),
                "residual": (E @ f - np.array([W, 0.0, 0.0])).tolist(),
                "pushing": bool(np.all(f >= 0)), "t_range": [-W / 4, W / 4]}

    def d01_goal_dims(G):
        """Dimension n - rank(G) of the set of velocities v with G v = b_G (when consistent)."""
        G = np.atleast_2d(np.asarray(G, dtype=float))
        return int(G.shape[1] - d01_rank(G))

    def d01_stacked_ranks(N, G, bG):
        """Ranks of N, [N; G] and [N G | 0 b_G], consistency, dim Sol(N & G), and the minimum
        number of velocity commands rank([N; G]) - rank(N) (Hou & Mason 2019, eq. 8)."""
        N, G = np.atleast_2d(np.asarray(N, dtype=float)), np.atleast_2d(np.asarray(G, dtype=float))
        bG = np.atleast_1d(np.asarray(bG, dtype=float))
        n = N.shape[1]
        NG = np.vstack([N, G])
        aug = np.column_stack([NG, np.r_[np.zeros(N.shape[0]), bG]])
        rN, rNG, rAug = d01_rank(N), d01_rank(NG), d01_rank(aug)
        cons = rNG == rAug
        return {"rankN": rN, "rankNG": rNG, "rankAug": rAug, "consistent": cons,
                "dimSol": n - rNG if cons else None, "nav": rNG - rN}

    return (d01_classify_2x2, d01_consistency, d01_four_subspaces, d01_goal_dims, d01_null_basis,
            d01_rank, d01_rank_nullity_table, d01_rref_steps, d01_solution_set, d01_stacked_ranks,
            d01_table_legs)


@app.cell
def _(D01_A, D01_A23, D01_A34, D01_M, d01_classify_2x2, d01_consistency, d01_four_subspaces, d01_null_basis, d01_rref_steps, mo, np):
    _S3 = [[1, 1, 1, 6], [2, 3, 1, 11], [1, -1, 2, 5]]
    _r3 = d01_rref_steps(_S3)
    _r34 = d01_rref_steps(D01_A34)
    _fs = d01_four_subspaces(D01_A34)
    _c1, _c2 = d01_consistency(D01_M, [1, 2, 3]), d01_consistency(D01_M, [1, 2, 4])
    _wall = d01_consistency([[0, 1], [0, 1]], [0, -0.1])
    _k = [d01_classify_2x2([[1, 1], [1, k]], [2, c]) for k, c in ((3, 4), (1, 2), (1, 3))]
    assert d01_classify_2x2(D01_A, [1, 5])["x"] == [2.0, 1.0]
    assert np.allclose(np.array(_r3["R"])[:, 3], [1, 2, 3]) and len(_r3["steps"]) == 7
    assert _r34["pivots"] == [0, 2] and np.allclose(_r34["R"], [[1, 2, 0, 1], [0, 0, 1, 2], [0, 0, 0, 0]])
    assert np.allclose(d01_null_basis(D01_A34), [[-2, 1, 0, 0], [-1, 0, -2, 1]])
    assert np.allclose(D01_A34 @ np.array(d01_null_basis(D01_A34)).T, 0)
    assert _c1["consistent"] and not _c2["consistent"] and not _wall["consistent"]
    assert [r["status"] for r in _k] == ["one", "infinite", "none"]
    assert np.allclose(d01_null_basis(D01_A23), [[-1, -1, 1]]) and np.allclose(d01_null_basis(D01_M), [[-1, -1, 1]])
    mo.md(f"""
| slide | quantity | value |
|---|---|---|
| row-and-column-pictures | solution of x − y = 1, x + 3y = 5 | {d01_classify_2x2(D01_A, [1, 5])['x']} |
| three-outcomes | (k, c) = (3, 4), (1, 2), (1, 3) | {[(r['status'], r['x']) for r in _k]} |
| gaussian-elimination | row operations (Gauss-Jordan order) | {'; '.join(s[4] for s in _r3['steps'])} |
| gaussian-elimination | solution | {np.array(_r3['R'])[:, 3].tolist()} |
| rref-and-pivots | RREF of A34, pivots | {_r34['R']}, {_r34['pivots']} |
| rref-and-pivots | A34 steps | {'; '.join(s[4] for s in _r34['steps'])} |
| consistency | M with (1, 2, 3): ranks | {_c1} |
| consistency | M with (1, 2, 4): ranks | {_c2} |
| consistency | v_y = 0 and v_y = −0.1 | {_wall} |
| null-space | Null(A23), Null(M) | {d01_null_basis(D01_A23)}, {d01_null_basis(D01_M)} |
| null-space-basis | special solutions of A34 | {d01_null_basis(D01_A34)} |
| row-space-and-rank | Row, Col, left null bases of A34 | {_fs['row']}, {_fs['col']}, {_fs['left_null']} |
| row-space-and-rank | dims | {_fs['dims']} |
""")
    return


@app.cell
def _(D01_A23, D01_M, d01_goal_dims, d01_rank_nullity_table, d01_solution_set, d01_stacked_ranks, d01_table_legs, mo, np):
    _ss = d01_solution_set(D01_A23, [2, 1])
    _xmin = (np.linalg.pinv(D01_A23) @ [2, 1]).tolist()
    _M1, _M2 = d01_solution_set(D01_M, [1, 2, 3]), d01_solution_set(D01_M, [1, 2, 4])
    _tl0, _tl1 = d01_table_legs(100, 0), d01_table_legs(100, 10)
    _N = [[0, 1, 0]]
    _goals = {"I": (np.eye(3), [0.1, 0, 0]), "I, b_y = -0.05": (np.eye(3), [0.1, -0.05, 0]),
              "[1 0 0]": ([[1, 0, 0]], [0.1]), "[1 0 0; 0 0 1]": ([[1, 0, 0], [0, 0, 1]], [0.1, 0])}
    _st = {k: d01_stacked_ranks(_N, G, b) for k, (G, b) in _goals.items()}
    assert _ss["particular"] == [2.0, 1.0, 0.0] and np.allclose(_xmin, [1, 0, 1])
    assert _M1["particular"] == [1.0, 2.0, 0.0] and not _M2["consistent"]
    assert _tl0["rank"] == 3 and np.allclose(_tl1["residual"], 0) and np.allclose(_tl1["f"], [35, 15, 35, 15])
    assert [d01_goal_dims(G) for G in ([[1, 0, 0]], [[1, 0, 0], [0, 0, 1]], np.eye(3))] == [2, 1, 0]
    assert [_st[k]["nav"] for k in _goals] == [2, 2, 1, 2] and not _st["I, b_y = -0.05"]["consistent"]
    _rows = "\n".join(f"| {r[0]} | {r[1]}×{r[2]} | {r[3]} | {r[4]} |" for r in d01_rank_nullity_table())
    _strows = "\n".join(f"| {k} | {v['rankN']} | {v['rankNG']} | {v['rankAug']} | {v['consistent']} | {v['dimSol']} | {v['nav']} |" for k, v in _st.items())
    mo.md(f"""
**rank-nullity** (rank + dim Null = n):

| matrix | shape | rank | dim Null |
|---|---|---|---|
{_rows}

**solution-set**: A23 x = (2, 1) has particular solution {_ss['particular']} and null basis
{_ss['null']}; the shortest solution (deck 02, pseudoinverse) is {_xmin}.
**homogeneous-systems**: M x = (1, 2, 3): particular {_M1['particular']}, null {_M1['null']};
M x = (1, 2, 4): consistent = {_M2['consistent']}.
**four-legged-table**: rank E = {_tl0['rank']}; t = 0: f = {_tl0['f']}; t = 10: f = {_tl1['f']},
residual {_tl1['residual']}; all legs push for t in {_tl0['t_range']}.
**identity-strictest-goal**: dim Sol(G) for [1 0 0], [1 0 0; 0 0 1], I:
{[d01_goal_dims(G) for G in ([[1, 0, 0]], [[1, 0, 0], [0, 0, 1]], np.eye(3))]}.

**stacked-constraints**, **velocity-command-count** (N = [0 1 0]):

| goal G | rank N | rank [N; G] | rank [N G ∣ 0 b] | consistent | dim Sol(N & G) | commands |
|---|---|---|---|---|---|---|
{_strows}
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Tex errata found while building deck 01 (tex lines 1100-1130)

    * Line 1113-1117: the $p$-norm formula is a norm only for $p \ge 1$. For $p = 1/2$ the
      triangle inequality fails: $\|(1, 1)\|_{1/2} = 4 > \|(1, 0)\|_{1/2} + \|(0, 1)\|_{1/2} = 2$.
      The $\infty$-norm is the limit $p \to \infty$ of the formula, $\max_i |x_i|$, not an
      instance of it.
    * Line 1123-1127: $\|\mathbf v - \mathbf v^f\|_M^2$ is twice the kinetic energy of the
      velocity change; the kinetic energy is $\tfrac12\|\mathbf v - \mathbf v^f\|_M^2$, the
      objective of eq. (mujoco-primal) at line 2922. MuJoCo's documentation names the principle
      Gauss's principle of least constraint, not a least-action principle (the same wording
      recurs at line 2893).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Deck 02 · Eigenvalues, singular values, and least squares

    Slides: `slides/02_eigenvalues_and_least_squares.html`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Deck 02, part 1: projections and orthonormal bases

    Slides `projection-line`, `projection-derivation`, `orthonormal-basis`,
    `projection-subspace`, `gram-schmidt`, `orthogonal-complement`, `null-inclusion` and
    `rank-test`. Subspace bases follow Hou & Mason's convention (and `lib/num.js`): the basis
    vectors are the ROWS of the returned matrix, and they are orthonormal.

    Sign convention shared with `lib/algo/02_eigen_ls.js`: an eigenvector, singular vector or
    null vector is scaled so that its entry of largest magnitude is positive (the first such
    entry when two tie to within 1e-12), so the slide figures and these cells agree.
    """)
    return


@app.cell
def _(np):
    def _d02_sign(v):
        """Flip v so its entry of largest magnitude is positive (first one on a tie)."""
        v = np.asarray(v, dtype=float)
        k = 0
        for i in range(len(v)):
            if abs(v[i]) > abs(v[k]) + 1e-12:
                k = i
        return -v if v[k] < 0 else v

    def d02_cond(A):
        """sigma_max / sigma_min, with inf when a singular value is below the numpy rank tolerance
        (as lib/num.js cond(); numpy.linalg.cond would report about 1e16 instead)."""
        A = np.atleast_2d(np.asarray(A, dtype=float))
        s = np.linalg.svd(A, compute_uv=False)
        tol = max(A.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
        if s.size == 0 or s[-1] <= tol:
            return float("inf")
        return float(s[0] / s[-1])

    def d02_rowspace(A):
        """Orthonormal basis of ROW(A) as rows (Hou & Mason's Row(A)), signs fixed."""
        A = np.atleast_2d(np.asarray(A, dtype=float))
        _u, s, vt = np.linalg.svd(A)
        tol = max(A.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
        r = int(np.sum(s > tol))
        return np.array([_d02_sign(v) for v in vt[:r]]).reshape(r, A.shape[1])

    def d02_nullspace(A):
        """Orthonormal basis of NULL(A) as rows (Hou & Mason's Null(A)), signs fixed."""
        A = np.atleast_2d(np.asarray(A, dtype=float))
        _u, s, vt = np.linalg.svd(A)
        tol = max(A.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
        r = int(np.sum(s > tol))
        return np.array([_d02_sign(v) for v in vt[r:]]).reshape(A.shape[1] - r, A.shape[1])

    return d02_cond, d02_nullspace, d02_rowspace


@app.cell
def _(np):
    def projection_onto_line(a, b):
        """Projection of b onto the line through 0 and a (slides projection-line, -derivation).

        xhat = a.b / a.a, p = xhat a, e = b - p (perpendicular to a), P = a a^T / a^T a.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        xhat = float(a @ b / (a @ a))
        p = xhat * a
        e = b - p
        return {
            "xhat": xhat,
            "p": p.tolist(),
            "e": e.tolist(),
            "e_dot_a": float(e @ a),
            "dist": float(np.linalg.norm(e)),
            "P": (np.outer(a, a) / (a @ a)).tolist(),
        }

    def orthonormal_coordinates(x, Q):
        """Coordinates c_i = q_i . x of x in the orthonormal basis given by the rows of Q
        (slide orthonormal-basis), and the reconstruction sum_i c_i q_i."""
        Q = np.atleast_2d(np.asarray(Q, dtype=float))
        x = np.asarray(x, dtype=float)
        c = Q @ x
        return {"c": c.tolist(), "recon": (Q.T @ c).tolist(), "QQt": (Q @ Q.T).tolist()}

    def projection_onto_plane(b, Q):
        """Projection of b onto the span of the orthonormal rows of Q (slide projection-subspace):
        coordinates Q b, point p = Q^T Q b, error e = b - p, projector P = Q^T Q."""
        Q = np.atleast_2d(np.asarray(Q, dtype=float))
        b = np.asarray(b, dtype=float)
        c = Q @ b
        p = Q.T @ c
        return {"coords": c.tolist(), "p": p.tolist(), "e": (b - p).tolist(),
                "dist": float(np.linalg.norm(b - p)), "P": (Q.T @ Q).tolist()}

    return orthonormal_coordinates, projection_onto_line, projection_onto_plane


@app.cell
def _(np):
    def gram_schmidt_qr(vectors):
        """Classical Gram-Schmidt on a list of independent vectors (slide gram-schmidt).

        Returns Q (orthonormal vectors as rows), R (upper triangular, R[i][j] = q_i . a_j for
        i < j, R[j][j] = |w_j|), and w (the residual of each a_j after subtracting its
        projections on q_1 .. q_{j-1}). Then a_j = sum_i R[i][j] q_i, i.e. A = Q^T R with the
        a_j as the columns of A.
        """
        V = [np.asarray(v, dtype=float) for v in vectors]
        k = len(V)
        Q, W = [], []
        R = np.zeros((k, k))
        for j, a in enumerate(V):
            w = a.copy()
            for i, q in enumerate(Q):
                R[i, j] = q @ a
                w = w - R[i, j] * q
            R[j, j] = np.linalg.norm(w)
            W.append(w)
            Q.append(w / R[j, j])
        return {"Q": np.array(Q).tolist(), "R": R.tolist(), "w": np.array(W).tolist()}

    return (gram_schmidt_qr,)


@app.cell
def _(d02_nullspace, np):
    def complement_split(A, v):
        """Split v into its row-space and null-space parts (slide orthogonal-complement).

        v_null is the projection of v onto NULL(A), v_row = v - v_null; coef solves
        A^T coef = v_row, so v_row = sum_i coef_i (row i of A)."""
        A = np.atleast_2d(np.asarray(A, dtype=float))
        v = np.asarray(v, dtype=float)
        N = d02_nullspace(A)
        v_null = N.T @ (N @ v)
        v_row = v - v_null
        coef = np.linalg.lstsq(A.T, v_row, rcond=None)[0]
        return {"null": N.tolist(), "rows_dot_null": (A @ N.T).ravel().tolist(),
                "v_null": v_null.tolist(), "v_row": v_row.tolist(), "coef": coef.tolist()}

    def inclusion_rank_test(A, B):
        """NULL(A) inside NULL(B) iff ROW(B) inside ROW(A) iff rank [A; B] = rank A
        (slides null-inclusion and rank-test; Hou & Mason 2021 eq. 11-13)."""
        A = np.atleast_2d(np.asarray(A, dtype=float))
        B = np.atleast_2d(np.asarray(B, dtype=float))
        rA = int(np.linalg.matrix_rank(A))
        rAB = int(np.linalg.matrix_rank(np.vstack([A, B])))
        N = d02_nullspace(A)
        return {"rank_A": rA, "rank_AB": rAB, "included": rAB == rA,
                "B_null": (B @ N.T).tolist()}

    def goal_inclusion_toy(b_goal=0.1):
        """Hou & Mason 2021 eq. 11-13 on a box on a rail pushed by a hand (slide rank-test).

        v = (v_box, v_hx, v_hy). Contact: the hand touches the box's left face, so
        v_box - v_hx = 0 (J); the hand may slide up and down the face. Goal: v_box = b_goal (G).
        Two candidate velocity commands: hand x velocity (C1) or hand y velocity (C2).
        """
        J = np.array([[1.0, -1.0, 0.0]])
        G = np.array([[1.0, 0.0, 0.0]])
        rank = np.linalg.matrix_rank
        out = {"rank_J": int(rank(J)), "rank_JG": int(rank(np.vstack([J, G])))}
        out["n_av_min"] = out["rank_JG"] - out["rank_J"]
        # One special solution of J v = 0, G v = b_goal (minimum norm, Hou 2021 eq. 28).
        v_star = np.linalg.pinv(np.vstack([J, G])) @ np.array([0.0, b_goal])
        out["v_star"] = v_star.tolist()
        for name, C in [("C1", np.array([[0.0, 1.0, 0.0]])), ("C2", np.array([[0.0, 0.0, 1.0]]))]:
            rJC = int(rank(np.vstack([J, C])))
            rJCG = int(rank(np.vstack([J, C, G])))
            out[name] = {"rank_JC": rJC, "rank_JCG": rJCG, "included": rJC == rJCG,
                         "b_C": float((C @ v_star)[0])}
        return out

    return complement_split, goal_inclusion_toy, inclusion_rank_test


@app.cell
def _(
    complement_split,
    goal_inclusion_toy,
    gram_schmidt_qr,
    inclusion_rank_test,
    mo,
    np,
    orthonormal_coordinates,
    projection_onto_line,
    projection_onto_plane,
):
    _pl = projection_onto_line([2, 1], [1, 3])
    assert np.isclose(_pl["xhat"], 1) and np.allclose(_pl["e"], [-1, 2]) and abs(_pl["e_dot_a"]) < 1e-12
    _oc = orthonormal_coordinates([2, 1], [[0.6, 0.8], [-0.8, 0.6]])
    _pp = projection_onto_plane([1, 2, 3], [np.array([1, -1, 0]) / np.sqrt(2), np.array([1, 1, -2]) / np.sqrt(6)])
    assert np.allclose(_pp["p"], [-1, 0, 1]) and np.allclose(_pp["e"], [2, 2, 2])
    _gs = gram_schmidt_qr([[3, 1], [2, 2]])
    _gs3 = gram_schmidt_qr([[1, 1, 0], [1, 0, 1]])
    _cs = complement_split([[1, 1, 0], [1, 0, 1]], [1, 2, 3])
    assert np.allclose(_cs["v_row"], [7 / 3, 2 / 3, 5 / 3]) and np.allclose(_cs["coef"], [2 / 3, 5 / 3])
    _in1 = inclusion_rank_test([[1, 1, 0], [1, 0, 1]], [[2, 1, 1]])
    _in2 = inclusion_rank_test([[1, 1, 0], [1, 0, 1]], [[0, 0, 1]])
    _gt = goal_inclusion_toy()
    assert _gt["C1"]["included"] and not _gt["C2"]["included"] and _gt["n_av_min"] == 1
    _f = lambda v: "(" + ", ".join(f"{x:.4f}" for x in np.ravel(v)) + ")"
    mo.md(f"""
    | slide | quantity | value |
    |---|---|---|
    | projection-line | a = (2,1), b = (1,3): xhat, p, e, e.a | {_pl['xhat']:.4f}, {_f(_pl['p'])}, {_f(_pl['e'])}, {_pl['e_dot_a']:.1e} |
    | projection-derivation | P = a a^T / a^T a | {_f(_pl['P'])} |
    | orthonormal-basis | coordinates of (2,1) in q1 = (0.6,0.8), q2 = (-0.8,0.6) | {_f(_oc['c'])} |
    | projection-subspace | Q b, p, e for b = (1,2,3), plane x+y+z = 0 | {_f(_pp['coords'])}, {_f(_pp['p'])}, {_f(_pp['e'])} |
    | gram-schmidt | a1 = (3,1), a2 = (2,2): q1, q2 | {_f(_gs['Q'][0])}, {_f(_gs['Q'][1])} |
    | gram-schmidt | R, residual w2 | {_f(_gs['R'])}, {_f(_gs['w'][1])} |
    | gram-schmidt (notes) | a1 = (1,1,0), a2 = (1,0,1): R, q2 * sqrt 6 | {_f(_gs3['R'])}, {_f(np.array(_gs3['Q'][1]) * np.sqrt(6))} |
    | orthogonal-complement | null basis, v_null, v_row, row coefficients | {_f(_cs['null'])}, {_f(_cs['v_null'])}, {_f(_cs['v_row'])}, {_f(_cs['coef'])} |
    | null-inclusion | B = (2,1,1): rank A, rank [A;B], B Null^T | {_in1['rank_A']}, {_in1['rank_AB']}, {_f(_in1['B_null'])} |
    | null-inclusion | B = (0,0,1): rank A, rank [A;B], B Null^T | {_in2['rank_A']}, {_in2['rank_AB']}, {_f(_in2['B_null'])} |
    | rank-test | C1: rank [J;C], rank [J;C;G], b_C | {_gt['C1']['rank_JC']}, {_gt['C1']['rank_JCG']}, {_gt['C1']['b_C']:.3f} |
    | rank-test | C2: rank [J;C], rank [J;C;G] | {_gt['C2']['rank_JC']}, {_gt['C2']['rank_JCG']} |
    | rank-test | special solution v*, n_av,min | {_f(_gt['v_star'])}, {_gt['n_av_min']} |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Deck 02, part 2: eigenvalues, quadratic forms, positive definiteness, Cholesky

    Slides `eigen-definition`, `eigen-2x2`, `spectral-theorem`, `quadratic-forms`,
    `quadratic-derivation`, `positive-definite`, `gram-psd`, `cholesky`, `cholesky-exists`.
    The 2x2 formulas are written out by hand (trace, determinant, the three Cholesky entries) so
    the cells show the same arithmetic as the slides; `numpy.linalg` checks them.
    """)
    return


@app.cell
def _(np):
    def eigen_2x2(A):
        """Eigenvalues of a 2x2 matrix from lambda^2 - tr(A) lambda + det(A) = 0 (slide eigen-2x2).

        Real case: values in descending order and unit eigenvectors (largest entry positive).
        Complex case: values None and complex = [real part, |imaginary part|].
        """
        A = np.asarray(A, dtype=float)
        tr = A[0, 0] + A[1, 1]
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        disc = tr * tr / 4 - det
        out = {"tr": float(tr), "det": float(det), "disc": float(disc)}
        if disc < -1e-14 * max(1.0, tr * tr):
            out.update(real=False, values=None, vectors=None, complex=[tr / 2, float(np.sqrt(-disc))])
            return out
        s = np.sqrt(max(disc, 0.0))
        vals = [tr / 2 + s, tr / 2 - s]
        vecs = []
        for lam in vals:
            M = A - lam * np.eye(2)
            r = M[0] if np.linalg.norm(M[0]) >= np.linalg.norm(M[1]) else M[1]
            if np.linalg.norm(r) < 1e-12 * max(1.0, np.abs(A).max()):
                v = np.array([1.0, 0.0]) if not vecs else np.array([0.0, 1.0])  # A = lam I
            else:
                v = np.array([-r[1], r[0]]) / np.linalg.norm(r)
            k = 0 if abs(v[0]) >= abs(v[1]) - 1e-12 else 1
            vecs.append((v if v[k] > 0 else -v).tolist())
        out.update(real=True, values=[float(v) for v in vals], vectors=vecs, complex=None)
        return out

    def quadratic_from_eigen(l1, l2, theta_deg):
        """The symmetric A = V diag(l1, l2) V^T whose first eigenvector is at theta_deg
        (the sliders of slide quadratic-forms)."""
        t = np.radians(theta_deg)
        V = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        return (V @ np.diag([l1, l2]) @ V.T).tolist()

    def quadratic_form_axes(A, c=1.0):
        """Shape of J(z) = 1/2 z^T A z for symmetric 2x2 A (slides quadratic-forms, -derivation).

        In eigenvector coordinates y = V^T z, J = 1/2 (l1 y1^2 + l2 y2^2); the level set J = c
        has semi-axis sqrt(2c / l_i) along v_i for every positive l_i.
        """
        A = np.asarray(A, dtype=float)
        e = eigen_2x2(A)
        l1, l2 = e["values"]
        tol = 1e-9 * max(1.0, abs(l1), abs(l2))
        if l2 > tol:
            kind = "bowl"
        elif l1 < -tol:
            kind = "cap"
        elif l1 > tol and l2 < -tol:
            kind = "saddle"
        elif abs(l1) <= tol and abs(l2) <= tol:
            kind = "flat"
        else:
            kind = "valley" if l1 > tol else "ridge"
        axes = [float(np.sqrt(2 * c / l)) if l > tol else None for l in (l1, l2)]
        return {"values": [l1, l2], "vectors": e["vectors"], "kind": kind, "semi_axes": axes,
                "kappa": float(l1 / l2) if l2 > tol else None}

    def definiteness_table():
        """The three example matrices of slide positive-definite with their eigenvalues."""
        rows = []
        for M in ([[2, 1], [1, 2]], [[1, 1], [1, 1]], [[1, 2], [2, 1]]):
            q = quadratic_form_axes(M)
            rows.append({"A": M, "values": q["values"], "kind": q["kind"],
                         "det": float(np.linalg.det(np.array(M, dtype=float)))})
        z = np.array([1.0, -1.0])
        return {"rows": rows, "saddle_z": z.tolist(),
                "saddle_value": float(z @ np.array([[1.0, 2.0], [2.0, 1.0]]) @ z)}

    return definiteness_table, eigen_2x2, quadratic_form_axes, quadratic_from_eigen


@app.cell
def _(np):
    def table_delassus(legs=((1, 1), (-1, 1), (-1, -1), (1, -1)), masses=(1.0, 1.0, 1.0)):
        """Gram matrices of a four-legged table (slide gram-psd).

        The table top moves with v = (v_z, omega_x, omega_y); leg i at (x_i, y_i) has normal
        velocity v_z + omega_x y_i - omega_y x_i, so row i of the contact Jacobian is
        [1, y_i, -x_i]. M = diag(masses). G = J M^-1 J^T is the Delassus matrix.
        """
        legs = np.asarray(legs, dtype=float)
        J = np.column_stack([np.ones(len(legs)), legs[:, 1], -legs[:, 0]])
        Minv = np.diag(1.0 / np.asarray(masses, dtype=float))
        G = J @ Minv @ J.T
        w, V = np.linalg.eigh(G)
        n = V[:, 0] / np.abs(V[:, 0]).max()
        n = n if n[np.argmax(np.abs(n) > 1 - 1e-9)] > 0 else -n
        return {"J": J.tolist(), "JtJ": (J.T @ J).tolist(), "G_eigs": sorted(w.tolist(), reverse=True),
                "rank_G": int(np.linalg.matrix_rank(G)), "null": n.tolist(),
                "G_null": (G @ n).tolist()}

    def cholesky_2x2_steps(A, b=None):
        """Cholesky A = L L^T of a symmetric 2x2 matrix entry by entry (slides cholesky,
        cholesky-exists), and the two triangular solves when b is given.

        L11 = sqrt(a11), L21 = a21 / L11, L22 = sqrt(a22 - L21^2). The factorization fails
        (ok = False) when a11 <= 0 or a22 - L21^2 <= 0, which happens exactly when A is not
        positive definite.
        """
        A = np.asarray(A, dtype=float)
        out = {"L11": None, "L21": None, "L22sq": None, "L22": None, "ok": False, "L": None,
               "w": None, "z": None}
        if A[0, 0] <= 0:
            return out
        L11 = float(np.sqrt(A[0, 0]))
        L21 = float(A[1, 0] / L11)
        L22sq = float(A[1, 1] - L21 * L21)
        out.update(L11=L11, L21=L21, L22sq=L22sq)
        if L22sq <= 1e-14 * max(1.0, abs(A[1, 1])):
            return out
        L22 = float(np.sqrt(L22sq))
        out.update(L22=L22, ok=True, L=[[L11, 0.0], [L21, L22]])
        if b is not None:
            w1 = b[0] / L11
            w2 = (b[1] - L21 * w1) / L22
            z2 = w2 / L22
            z1 = (w1 - L21 * z2) / L11
            out.update(w=[float(w1), float(w2)], z=[float(z1), float(z2)])
        return out

    return cholesky_2x2_steps, table_delassus


@app.cell
def _(
    cholesky_2x2_steps,
    definiteness_table,
    eigen_2x2,
    mo,
    np,
    quadratic_form_axes,
    table_delassus,
):
    _eA = eigen_2x2([[2, 1], [1, 2]])
    _eB = eigen_2x2([[4, 1], [2, 3]])
    _eR = eigen_2x2([[0, -1], [1, 0]])
    _eS = eigen_2x2([[1, 1], [0, 1]])
    assert np.allclose(_eA["values"], [3, 1]) and np.allclose(_eB["values"], [5, 2]) and not _eR["real"]
    _q = quadratic_form_axes([[2, 1], [1, 2]])
    assert np.allclose(_q["semi_axes"], [np.sqrt(2 / 3), np.sqrt(2)])
    _dt = definiteness_table()
    _td = table_delassus()
    assert np.allclose(_td["G_eigs"], [4, 4, 4, 0]) and np.allclose(_td["null"], [1, -1, 1, -1])
    _ch = cholesky_2x2_steps([[4, 2], [2, 3]], [2, 1])
    assert np.allclose(_ch["L"], np.linalg.cholesky(np.array([[4.0, 2], [2, 3]]))) and np.allclose(_ch["z"], [0.5, 0])
    _reg = [(mu, cholesky_2x2_steps(np.array([[1.0, 2], [2, 1]]) + mu * np.eye(2))) for mu in (0, 0.5, 1, 1.5, 2, 3)]
    _f = lambda v: "(" + ", ".join(f"{x:.4f}" for x in np.ravel(v)) + ")"
    _regrows = "\n".join(
        f"    | cholesky-exists | mu = {mu}: L22^2, ok | {r['L22sq']:.4f}, {r['ok']} |" for mu, r in _reg
    )
    mo.md(f"""
    | slide | quantity | value |
    |---|---|---|
    | eigen-2x2 | [[2,1],[1,2]]: tr, det, values, vectors | {_eA['tr']}, {_eA['det']}, {_f(_eA['values'])}, {_f(_eA['vectors'])} |
    | eigen-2x2 | [[4,1],[2,3]]: values, vectors, v1.v2 | {_f(_eB['values'])}, {_f(_eB['vectors'])}, {np.dot(*_eB['vectors']):.4f} |
    | eigen-definition | shear [[1,1],[0,1]]: values, vectors | {_f(_eS['values'])}, {_f(_eS['vectors'])} |
    | eigen-definition | rotation [[0,-1],[1,0]]: complex pair | {_eR['complex'][0]} +- {_eR['complex'][1]} i |
    | quadratic-derivation | [[2,1],[1,2]]: semi-axes of J = 1, kappa | {_f(_q['semi_axes'])}, {_q['kappa']:.4f} |
    | positive-definite | eigenvalues and kind of the three examples | {'; '.join(_f(r['values']) + ' ' + r['kind'] for r in _dt['rows'])} |
    | positive-definite | z = (1,-1): z^T [[1,2],[2,1]] z | {_dt['saddle_value']} |
    | gram-psd | J^T J, eigenvalues of G = J J^T, null vector | {_f(_td['JtJ'])}, {_f(_td['G_eigs'])}, {_f(_td['null'])} |
    | cholesky | L, w, z for A = [[4,2],[2,3]], b = (2,1) | {_f(_ch['L'])}, {_f(_ch['w'])}, {_f(_ch['z'])} |
{_regrows}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Deck 02, part 3: singular values, conditioning, the drawer, row normalization

    Slides `svd-picture`, `svd-statement`, `condition-number`, `nearly-parallel-rows`,
    `fragile-vs-infeasible`, `drawer-setup`, `drawer-live`, `row-normalization`. The condition
    number is Hou & Mason 2021 eq. 8, $\operatorname{cond}(A) = \sigma_{\max}/\sigma_{\min}$; the
    crashing index is their eq. 9, the condition number of $[\hat J; \hat C]$ with $\hat J$ an
    orthonormal basis of the rows of $J$ and $\hat C$ the rows of $C$ normalized.
    """)
    return


@app.cell
def _(np):
    def svd_circle_ellipse(A):
        """SVD of a 2x2 matrix for slide svd-picture.

        sigma_1 = sqrt(largest eigenvalue of A^T A), sigma_2 = |det A| / sigma_1 (exact zero for
        a singular A). v_i: unit eigenvectors of A^T A (largest entry positive). u_1 = A v_1 /
        sigma_1 and u_2 = s * (u_1 turned 90 degrees) with s = sign(det A * det V), so that
        A = U diag(sigma) V^T also when sigma_2 = 0. cond = sigma_1 / sigma_2 (inf when singular).
        """
        A = np.asarray(A, dtype=float)
        w, V = np.linalg.eigh(A.T @ A)
        V = V[:, ::-1]
        vs = []
        for i in range(2):
            v = V[:, i]
            k = 0 if abs(v[0]) >= abs(v[1]) - 1e-12 else 1
            vs.append(v if v[k] > 0 else -v)
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        s1 = float(np.sqrt(max(w[1], 0.0)))
        s2 = abs(det) / s1 if s1 > 0 else 0.0
        u1 = A @ vs[0] / s1 if s1 > 0 else np.array([1.0, 0.0])
        detV = vs[0][0] * vs[1][1] - vs[0][1] * vs[1][0]
        sgn = -1.0 if det * detV < 0 else 1.0
        u2 = sgn * np.array([-u1[1], u1[0]])
        cond = s1 / s2 if s2 > 0 else float("inf")
        return {"S": [s1, float(s2)], "U": np.column_stack([u1, u2]).tolist(),
                "V": np.column_stack(vs).tolist(), "cond": float(cond), "det": float(det),
                "AtA_eigs": [float(w[1]), float(w[0])]}

    def svd_subspaces(A):
        """Singular values, rank, and orthonormal row-space and null-space bases (rows) from the
        SVD (slide svd-statement)."""
        A = np.atleast_2d(np.asarray(A, dtype=float))
        _u, s, vt = np.linalg.svd(A)
        tol = max(A.shape) * np.finfo(float).eps * s[0]
        r = int(np.sum(s > tol))

        def fix(v):
            k = int(np.argmax(np.abs(v) > np.abs(v).max() - 1e-12))
            return v if v[k] > 0 else -v

        return {"S": s.tolist(), "rank": r, "row": [fix(v).tolist() for v in vt[:r]],
                "null": [fix(v).tolist() for v in vt[r:]], "AAt_eigs": sorted(np.linalg.eigvalsh(A @ A.T).tolist(), reverse=True)}

    return svd_circle_ellipse, svd_subspaces


@app.cell
def _(d02_cond, np):
    def nearly_parallel_example():
        """x + y = 2, x + 1.01 y = 2.01 and the same with 2.02 (slide nearly-parallel-rows)."""
        A = np.array([[1.0, 1.0], [1.0, 1.01]])
        b1, b2 = np.array([2.0, 2.01]), np.array([2.0, 2.02])
        z1, z2 = np.linalg.solve(A, b1), np.linalg.solve(A, b2)
        rel_db = np.linalg.norm(b2 - b1) / np.linalg.norm(b1)
        rel_dz = np.linalg.norm(z2 - z1) / np.linalg.norm(z1)
        cosang = A[0] @ A[1] / (np.linalg.norm(A[0]) * np.linalg.norm(A[1]))
        return {"z1": z1.tolist(), "z2": z2.tolist(), "cond": d02_cond(A),
                "S": np.linalg.svd(A, compute_uv=False).tolist(),
                "rel_db": float(rel_db), "rel_dz": float(rel_dz), "amplification": float(rel_dz / rel_db),
                "angle_deg": float(np.degrees(np.arccos(min(1.0, cosang))))}

    def nearly_parallel_rows(phi_deg, db):
        """Two unit rows phi_deg apart, the first at 45 degrees, with right-hand side chosen so
        that z* = (1, 1) solves the system; then b_2 is changed by db (slide nearly-parallel-rows,
        live figure). cond = cot(phi / 2)."""
        a1, a2 = np.radians(45.0), np.radians(45.0 + phi_deg)
        A = np.array([[np.cos(a1), np.sin(a1)], [np.cos(a2), np.sin(a2)]])
        zs = np.array([1.0, 1.0])
        b = A @ zs + np.array([0.0, db])
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-12:
            return {"A": A.tolist(), "b": b.tolist(), "cond": float("inf"), "z": None, "shift": None}
        z = np.linalg.solve(A, b)
        return {"A": A.tolist(), "b": b.tolist(), "cond": d02_cond(A), "z": z.tolist(),
                "shift": float(np.linalg.norm(z - zs))}

    def drawer_conditioning(theta_deg, eps_deg=0.0, b=1.0):
        """The drawer of slides drawer-setup and drawer-live.

        Velocity v = (v_x, v_y) of the drawer (and of the hand holding its handle). The modeled
        rail runs along x, so the constraint is n^T v = 0 with n = (0, 1). The hand commands
        c^T v = b with c = (cos theta, sin theta), theta measured from the rail. The true rail is
        turned by eps: n_true = (-sin eps, cos eps). Returns the condition number of the modeled
        system [n; c], the modeled speed b / cos(theta), the velocity that solves the true system
        (b / cos(theta - eps) along the true rail) and whether the true system has a solution.
        """
        th, ep = np.radians(theta_deg), np.radians(eps_deg)
        c = np.array([np.cos(th), np.sin(th)])
        A_model = np.array([[0.0, 1.0], c])
        A_true = np.array([[-np.sin(ep), np.cos(ep)], c])
        out = {"cond": d02_cond(A_model), "cond_true": d02_cond(A_true),
               "cond_formula": float(np.tan(np.radians(45.0 + theta_deg / 2))),
               "speed_model": float(b / np.cos(th)) if abs(np.cos(th)) > 1e-12 else None,
               "T": [[float(np.sin(th)), float(-np.cos(th))], [float(np.cos(th)), float(np.sin(th))]]}
        ct = np.cos(th - ep)
        if abs(ct) < 1e-12:
            out.update(feasible=False, v_true=None, speed_true=None, ratio=None)
        else:
            v = np.linalg.solve(A_true, np.array([0.0, b]))
            out.update(feasible=True, v_true=v.tolist(), speed_true=float(b / ct),
                       ratio=float(np.cos(th) / ct) if out["speed_model"] is not None else None)
        return out

    def crashing_index(J, C):
        """Hou & Mason 2021 eq. 9: cond([J_hat; C_hat]) with J_hat an orthonormal basis of the rows
        of J and C_hat the rows of C scaled to unit length (slide row-normalization). Also the raw
        cond([J; C]) for comparison."""
        J = np.atleast_2d(np.asarray(J, dtype=float))
        C = np.atleast_2d(np.asarray(C, dtype=float))
        _u, s, vt = np.linalg.svd(J)
        tol = max(J.shape) * np.finfo(float).eps * s[0]
        Jh = vt[: int(np.sum(s > tol))]
        Ch = C / np.linalg.norm(C, axis=1, keepdims=True)
        return {"raw": d02_cond(np.vstack([J, C])), "index": d02_cond(np.vstack([Jh, Ch])),
                "rows_J_hat": int(Jh.shape[0])}

    return crashing_index, drawer_conditioning, nearly_parallel_example, nearly_parallel_rows


@app.cell
def _(
    crashing_index,
    d02_cond,
    drawer_conditioning,
    mo,
    nearly_parallel_example,
    np,
    svd_circle_ellipse,
    svd_subspaces,
):
    _sv = svd_circle_ellipse([[3, 0], [4, 5]])
    assert np.allclose(_sv["S"], [np.sqrt(45), np.sqrt(5)]) and np.isclose(_sv["cond"], 3)
    _ss = svd_subspaces([[1, 1, 0], [1, 0, 1]])
    _npx = nearly_parallel_example()
    assert np.allclose(_npx["z1"], [1, 1]) and np.allclose(_npx["z2"], [0, 2])
    _dr = {th: drawer_conditioning(th) for th in (0, 30, 45, 60, 75, 80, 85, 89, 89.5)}
    for _th, _r in _dr.items():
        assert np.isclose(_r["cond"], _r["cond_formula"], rtol=1e-9)
    _err = [(th, ep, drawer_conditioning(th, ep)) for th, ep in ((0, 2), (60, 2), (85, 2), (85, -2), (88, 2), (88, -2))]
    _ci = crashing_index([[0, 1]], [[10 * np.cos(np.pi / 4), 10 * np.sin(np.pi / 4)]])
    _ci0 = crashing_index([[0, 1], [0, 1]], [[1, 0]])
    _f = lambda v: "(" + ", ".join(f"{x:.4f}" for x in np.ravel(v)) + ")"
    _drrows = "\n".join(f"    | drawer-setup | theta = {th}: cond, speed / b | {r['cond']:.4f}, {r['speed_model']:.4f} |" for th, r in _dr.items())
    _errrows = "\n".join(
        f"    | drawer-live | theta = {th}, eps = {ep}: speed_true, ratio | "
        + (f"{r['speed_true']:.4f}, {r['ratio']:.4f} |" if r["feasible"] else "no solution |")
        for th, ep, r in _err
    )
    mo.md(f"""
    | slide | quantity | value |
    |---|---|---|
    | svd-picture | A = [[3,0],[4,5]]: sigma, cond, eigenvalues of A^T A | {_f(_sv['S'])}, {_sv['cond']:.4f}, {_f(_sv['AtA_eigs'])} |
    | svd-picture | V (columns), U (columns) | {_f(_sv['V'])}, {_f(_sv['U'])} |
    | svd-statement | A = [[1,1,0],[1,0,1]]: sigma, rank | {_f(_ss['S'])}, {_ss['rank']} |
    | svd-statement | Row basis, Null basis | {_f(_ss['row'])}, {_f(_ss['null'])} |
    | condition-number | cond diag(1, 0.01) | {d02_cond(np.diag([1, 0.01])):.1f} |
    | nearly-parallel-rows | z1, z2, cond, angle between rows (deg) | {_f(_npx['z1'])}, {_f(_npx['z2'])}, {_npx['cond']:.2f}, {_npx['angle_deg']:.4f} |
    | nearly-parallel-rows | relative db, relative dz, amplification | {_npx['rel_db']:.5f}, {_npx['rel_dz']:.4f}, {_npx['amplification']:.2f} |
{_drrows}
{_errrows}
    | row-normalization | cond [[0,1],[1000,0]], cond [[0,1],[0,1]] | {d02_cond([[0, 1], [1000, 0]]):.1f}, {d02_cond([[0, 1], [0, 1]])} |
    | row-normalization | drawer at 45 deg with c scaled by 10: raw cond, crashing index | {_ci['raw']:.4f}, {_ci['index']:.4f} |
    | row-normalization | redundant J = [[0,1],[0,1]], C = (1,0): raw, index | {_ci0['raw']}, {_ci0['index']:.4f} |
    """)
    return


@app.cell
def _(drawer_conditioning, np, plt):
    _th = np.linspace(0, 89, 300)
    _fig, _ax = plt.subplots(figsize=(5.5, 2.8))
    _ax.semilogy(_th, [drawer_conditioning(t)["cond"] for t in _th], lw=2)
    _ax.set_xlabel("command angle from the rail, theta (deg)")
    _ax.set_ylabel("cond [n; c]")
    _ax.set_title("Drawer: cond = tan(45 deg + theta/2) (slide drawer-setup)")
    _ax.grid(alpha=0.3, which="both")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Deck 02, part 4: least squares, minimum-norm solutions, the Schur complement

    Slides `line-fit`, `normal-equations`, `ls-projection-3d`, `min-norm`, `table-forces`,
    `pseudoinverse`, `schur-complement`, `schur-minimization`. The four-legged table picks one
    force distribution out of a line of them by minimizing $\sum f_i^2$, the rule of Hou & Mason
    2019 eq. 21 (Algorithm 2).
    """)
    return


@app.cell
def _(d02_nullspace, np):
    def line_fit_normal_equations(t, y):
        """Least-squares line y = c + d t through the points (t_i, y_i) (slide line-fit), by the
        normal equations A^T A x = A^T y with A = [1, t]."""
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        A = np.column_stack([np.ones_like(t), t])
        AtA, Aty = A.T @ A, A.T @ y
        x = np.linalg.solve(AtA, Aty)
        e = y - A @ x
        return {"AtA": AtA.tolist(), "Aty": Aty.tolist(), "c": float(x[0]), "d": float(x[1]),
                "residuals": e.tolist(), "sse": float(e @ e), "Ate": (A.T @ e).tolist()}

    def column_space_projection(A, b):
        """Least squares as projection (slide ls-projection-3d): xhat from the normal equations,
        p = A xhat, e = b - p, A^T e = 0, and the projector P = A (A^T A)^-1 A^T."""
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        xhat = np.linalg.solve(A.T @ A, A.T @ b)
        p = A @ xhat
        return {"AtA": (A.T @ A).tolist(), "Atb": (A.T @ b).tolist(), "xhat": xhat.tolist(),
                "p": p.tolist(), "e": (b - p).tolist(), "Ate": (A.T @ (b - p)).tolist(),
                "dist": float(np.linalg.norm(b - p)),
                "P": (A @ np.linalg.solve(A.T @ A, A.T)).tolist()}

    def min_norm_solution(A, b):
        """Minimum-norm solution x+ = A^T (A A^T)^-1 b of a wide system with independent rows
        (slide min-norm), checked against the pseudoinverse."""
        A = np.atleast_2d(np.asarray(A, dtype=float))
        b = np.asarray(b, dtype=float)
        x = A.T @ np.linalg.solve(A @ A.T, b)
        return {"x": x.tolist(), "norm": float(np.linalg.norm(x)),
                "pinv_x": (np.linalg.pinv(A) @ b).tolist(), "null": d02_nullspace(A).tolist()}

    def table_force_distribution(W=100.0, com=(0.2, 0.1), legs=((1, 1), (-1, 1), (-1, -1), (1, -1)), t=0.0):
        """Leg forces of a table on four legs (slide table-forces).

        Equilibrium M f = rhs: sum f_i = W, sum f_i x_i = W x_c, sum f_i y_i = W y_c. Three
        equations, four unknowns. f+ = M^T (M M^T)^-1 rhs is the minimum-norm distribution
        (Hou & Mason 2019 eq. 21); every other one is f+ + t n with n the null vector scaled so
        its largest entry is 1. t_range is the interval of t with every f_i >= 0.
        """
        legs = np.asarray(legs, dtype=float)
        M = np.vstack([np.ones(len(legs)), legs[:, 0], legs[:, 1]])
        rhs = W * np.array([1.0, com[0], com[1]])
        fp = M.T @ np.linalg.solve(M @ M.T, rhs)
        n = d02_nullspace(M)[0]
        n = n / np.abs(n).max()
        f = fp + t * n
        lo = max([-fp[i] / n[i] for i in range(len(n)) if n[i] > 1e-12] + [-np.inf])
        hi = min([-fp[i] / n[i] for i in range(len(n)) if n[i] < -1e-12] + [np.inf])
        return {"MMt": (M @ M.T).tolist(), "f_plus": fp.tolist(), "norm_plus": float(np.linalg.norm(fp)),
                "null": n.tolist(), "f": f.tolist(), "norm": float(np.linalg.norm(f)),
                "residual": (M @ f - rhs).tolist(), "t_range": [float(lo), float(hi)]}

    def pseudoinverse_rank_one(A=((1, 2), (2, 4)), b=(1, 1), mus=(1.0, 0.1, 0.01, 0.001)):
        """Pseudoinverse of a rank-one matrix from its SVD, and Tikhonov-regularized solutions
        (A^T A + mu I)^-1 A^T b approaching A+ b (slide pseudoinverse)."""
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        P = np.linalg.pinv(A)
        x = P @ b
        tik = [np.linalg.solve(A.T @ A + mu * np.eye(A.shape[1]), A.T @ b).tolist() for mu in mus]
        return {"S": np.linalg.svd(A, compute_uv=False).tolist(), "pinv": P.tolist(), "x": x.tolist(),
                "Ax": (A @ x).tolist(), "tikhonov": tik, "mus": list(mus)}

    return (
        column_space_projection,
        line_fit_normal_equations,
        min_norm_solution,
        pseudoinverse_rank_one,
        table_force_distribution,
    )


@app.cell
def _(np):
    def schur_block_solve(K, rhs, n1):
        """Solve K [x; u] = rhs by block elimination (slide schur-complement), where x holds the
        first n1 unknowns: S = A - B C^-1 B^T, S x = f - B C^-1 g, u = C^-1 (g - B^T x)."""
        K = np.asarray(K, dtype=float)
        rhs = np.asarray(rhs, dtype=float)
        A, B, C = K[:n1, :n1], K[:n1, n1:], K[n1:, n1:]
        f, g = rhs[:n1], rhs[n1:]
        S = A - B @ np.linalg.solve(C, B.T)
        x = np.linalg.solve(S, f - B @ np.linalg.solve(C, g))
        u = np.linalg.solve(C, g - B.T @ x)
        return {"S": S.tolist(), "x": x.tolist(), "u": u.tolist(),
                "check": (K @ np.concatenate([x, u]) - rhs).tolist()}

    def schur_minimize_out(a, b, c):
        """Minimize f(x, u) = 1/2 (a x^2 + 2 b x u + c u^2) over u (slide schur-minimization).

        With c > 0 the minimizer is u*(x) = -(b / c) x and f(x, u*) = 1/2 s x^2 with the Schur
        complement s = a - b^2 / c. The matrix [[a, b], [b, c]] is positive definite exactly when
        c > 0 and s > 0.
        """
        K = np.array([[a, b], [b, c]], dtype=float)
        eig = np.linalg.eigvalsh(K)
        if c <= 0:
            return {"s": None, "gain": None, "eigs": sorted(eig.tolist(), reverse=True), "pd": False, "c_pos": False}
        s = a - b * b / c
        return {"s": float(s), "gain": float(-b / c), "eigs": sorted(eig.tolist(), reverse=True),
                "pd": bool(s > 0), "c_pos": True}

    return schur_block_solve, schur_minimize_out


@app.cell
def _(
    column_space_projection,
    line_fit_normal_equations,
    min_norm_solution,
    mo,
    np,
    pseudoinverse_rank_one,
    schur_block_solve,
    schur_minimize_out,
    table_force_distribution,
):
    _lf = line_fit_normal_equations([0, 1, 2, 3], [1, 2, 2, 4])
    assert np.isclose(_lf["c"], 0.9) and np.isclose(_lf["d"], 0.9) and np.isclose(_lf["sse"], 0.7)
    _cp = column_space_projection([[1, 0], [1, 1], [1, 2]], [6, 0, 0])
    assert np.allclose(_cp["xhat"], [5, -3]) and np.allclose(_cp["e"], [1, -2, 1])
    _mn = min_norm_solution([[1, 1]], [2])
    _tf = table_force_distribution()
    assert np.allclose(_tf["f_plus"], [32.5, 22.5, 17.5, 27.5]) and np.allclose(_tf["t_range"], [-17.5, 22.5])
    _tf10 = table_force_distribution(t=10.0)
    _pi = pseudoinverse_rank_one()
    _sb = schur_block_solve([[4, 2], [2, 2]], [6, 2], 1)
    assert np.allclose(_sb["x"], [2]) and np.allclose(_sb["u"], [-1])
    _sm = {b: schur_minimize_out(4, b, 2) for b in (0, 1, 2, 2.5, 3, 4)}
    _cx = schur_minimize_out(0, 1, 1)
    _f = lambda v: "(" + ", ".join(f"{x:.4f}" for x in np.ravel(v)) + ")"
    _smrows = "\n".join(f"    | schur-minimization | a = 4, c = 2, b = {b}: s, gain, pd | {r['s']:.4f}, {r['gain']:.4f}, {r['pd']} |" for b, r in _sm.items())
    mo.md(f"""
    | slide | quantity | value |
    |---|---|---|
    | line-fit | A^T A, A^T y, (c, d), residuals, SSE | {_f(_lf['AtA'])}, {_f(_lf['Aty'])}, ({_lf['c']:.4f}, {_lf['d']:.4f}), {_f(_lf['residuals'])}, {_lf['sse']:.4f} |
    | ls-projection-3d | A^T A, A^T b, xhat, p, e, dist | {_f(_cp['AtA'])}, {_f(_cp['Atb'])}, {_f(_cp['xhat'])}, {_f(_cp['p'])}, {_f(_cp['e'])}, {_cp['dist']:.4f} |
    | ls-projection-3d | 6 P | {_f(6 * np.array(_cp['P']))} |
    | min-norm | x1 + x2 = 2: x+, norm, pinv check | {_f(_mn['x'])}, {_mn['norm']:.4f}, {_f(_mn['pinv_x'])} |
    | table-forces | M M^T, f+, norm, null vector, t range | {_f(_tf['MMt'])}, {_f(_tf['f_plus'])}, {_tf['norm_plus']:.4f}, {_f(_tf['null'])}, {_f(_tf['t_range'])} |
    | table-forces | t = 10: f, norm, residual | {_f(_tf10['f'])}, {_tf10['norm']:.4f}, {_f(_tf10['residual'])} |
    | pseudoinverse | sigma, 25 A+, x, A x | {_f(_pi['S'])}, {_f(25 * np.array(_pi['pinv']))}, {_f(_pi['x'])}, {_f(_pi['Ax'])} |
    | pseudoinverse | Tikhonov mu = 1, 0.1, 0.01, 0.001 | {'; '.join(_f(v) for v in _pi['tikhonov'])} |
    | schur-complement | [[4,2],[2,2]] [x;u] = [6;2]: S, x, u | {_f(_sb['S'])}, {_f(_sb['x'])}, {_f(_sb['u'])} |
{_smrows}
    | schur-minimization | a = 0, b = 1, c = 1: s, eigenvalues | {_cx['s']:.4f}, {_f(_cx['eigs'])} |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Deck 03 · Derivatives, gradients, and Jacobians

    Slides: `slides/03_derivatives_and_jacobians.html`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Deck 04 · Probability, Gaussians, and sampling

    Slides: `slides/04_probability_and_sampling.html`
    """)
    return


if __name__ == "__main__":
    app.run()
