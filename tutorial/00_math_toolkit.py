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
    backs `solution-set` and `homogeneous-systems`; `d01_min_norm_solution` backs `solution-set`; `d01_table_legs` backs `four-legged-table`;
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

    def d01_min_norm_solution(A, b):
        """The shortest solution pinv(A) b of A x = b (the least-squares one if A x = b has none).

        For A23 x = (2, 1) it is (1, 0, 1), the point t = (b1 + b2)/3 = 1 of the line
        (2, 1, 0) + t(-1, -1, 1) (slide solution-set). Deck 02 builds the pseudoinverse.
        """
        A = np.atleast_2d(np.asarray(A, dtype=float))
        return (np.linalg.pinv(A) @ np.asarray(b, dtype=float) + 0.0).tolist()

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

    return (d01_classify_2x2, d01_consistency, d01_four_subspaces, d01_goal_dims, d01_min_norm_solution,
            d01_null_basis, d01_rank, d01_rank_nullity_table, d01_rref_steps, d01_solution_set,
            d01_stacked_ranks, d01_table_legs)


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
def _(D01_A23, D01_M, d01_goal_dims, d01_min_norm_solution, d01_rank_nullity_table, d01_solution_set, d01_stacked_ranks, d01_table_legs, mo, np):
    _ss = d01_solution_set(D01_A23, [2, 1])
    _xmin = d01_min_norm_solution(D01_A23, [2, 1])
    _tbest = (2 + 1) / 3  # t minimizing |(2, 1, 0) + t(-1, -1, 1)|
    _M1, _M2 = d01_solution_set(D01_M, [1, 2, 3]), d01_solution_set(D01_M, [1, 2, 4])
    _tl0, _tl1 = d01_table_legs(100, 0), d01_table_legs(100, 10)
    _N = [[0, 1, 0]]
    _goals = {"I": (np.eye(3), [0.1, 0, 0]), "I, b_y = -0.05": (np.eye(3), [0.1, -0.05, 0]),
              "[1 0 0]": ([[1, 0, 0]], [0.1]), "[1 0 0; 0 0 1]": ([[1, 0, 0], [0, 0, 1]], [0.1, 0]),
              "[0 1 0], b_y = 0": ([[0, 1, 0]], [0.0]), "[0 1 0], b_y = -0.05": ([[0, 1, 0]], [-0.05])}
    _st = {k: d01_stacked_ranks(_N, G, b) for k, (G, b) in _goals.items()}
    assert _ss["particular"] == [2.0, 1.0, 0.0] and np.allclose(_xmin, [1, 0, 1])
    assert np.allclose(np.array([2, 1, 0]) + _tbest * np.array([-1, -1, 1]), _xmin)
    assert np.isclose(np.linalg.norm(_xmin), np.sqrt(2)) and np.isclose(np.linalg.norm([2, 1, 0]), np.sqrt(5))
    assert _M1["particular"] == [1.0, 2.0, 0.0] and not _M2["consistent"]
    assert _tl0["rank"] == 3 and np.allclose(_tl1["residual"], 0) and np.allclose(_tl1["f"], [35, 15, 35, 15])
    assert [d01_goal_dims(G) for G in ([[1, 0, 0]], [[1, 0, 0], [0, 0, 1]], np.eye(3))] == [2, 1, 0]
    assert [_st[k]["nav"] for k in _goals] == [2, 2, 1, 2, 0, 0] and not _st["I, b_y = -0.05"]["consistent"]
    assert _st["[0 1 0], b_y = 0"]["dimSol"] == 2 and not _st["[0 1 0], b_y = -0.05"]["consistent"]
    _rows = "\n".join(f"| {r[0]} | {r[1]}×{r[2]} | {r[3]} | {r[4]} |" for r in d01_rank_nullity_table())
    _strows = "\n".join(f"| {k} | {v['rankN']} | {v['rankNG']} | {v['rankAug']} | {v['consistent']} | {v['dimSol']} | {v['nav']} |" for k, v in _st.items())
    mo.md(f"""
**rank-nullity** (rank + dim Null = n):

| matrix | shape | rank | dim Null |
|---|---|---|---|
{_rows}

**solution-set**: A23 x = (2, 1) has particular solution {_ss['particular']} and null basis
{_ss['null']}; the shortest solution (deck 02, pseudoinverse) is {_xmin}, at t = {_tbest:g},
length {np.linalg.norm(_xmin):.4f} (against {np.linalg.norm([2, 1, 0]):.4f} at t = 0).
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
    def d02_projection_onto_line(a, b):
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

    def d02_orthonormal_coordinates(x, Q):
        """Coordinates c_i = q_i . x of x in the orthonormal basis given by the rows of Q
        (slide orthonormal-basis), and the reconstruction sum_i c_i q_i."""
        Q = np.atleast_2d(np.asarray(Q, dtype=float))
        x = np.asarray(x, dtype=float)
        c = Q @ x
        return {"c": c.tolist(), "recon": (Q.T @ c).tolist(), "QQt": (Q @ Q.T).tolist()}

    def d02_projection_onto_plane(b, Q):
        """Projection of b onto the span of the orthonormal rows of Q (slide projection-subspace):
        coordinates Q b, point p = Q^T Q b, error e = b - p, projector P = Q^T Q."""
        Q = np.atleast_2d(np.asarray(Q, dtype=float))
        b = np.asarray(b, dtype=float)
        c = Q @ b
        p = Q.T @ c
        return {"coords": c.tolist(), "p": p.tolist(), "e": (b - p).tolist(),
                "dist": float(np.linalg.norm(b - p)), "P": (Q.T @ Q).tolist()}

    return d02_orthonormal_coordinates, d02_projection_onto_line, d02_projection_onto_plane


@app.cell
def _(np):
    def d02_gram_schmidt_qr(vectors):
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

    return (d02_gram_schmidt_qr,)


@app.cell
def _(d02_nullspace, np):
    def d02_complement_split(A, v):
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

    def d02_inclusion_rank_test(A, B):
        """NULL(A) inside NULL(B) iff ROW(B) inside ROW(A) iff rank [A; B] = rank A
        (slides null-inclusion and rank-test; Hou & Mason 2021 eq. 11-13)."""
        A = np.atleast_2d(np.asarray(A, dtype=float))
        B = np.atleast_2d(np.asarray(B, dtype=float))
        rA = int(np.linalg.matrix_rank(A))
        rAB = int(np.linalg.matrix_rank(np.vstack([A, B])))
        N = d02_nullspace(A)
        return {"rank_A": rA, "rank_AB": rAB, "included": rAB == rA,
                "B_null": (B @ N.T).tolist()}

    def d02_goal_inclusion_toy(b_goal=0.1):
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

    return d02_complement_split, d02_goal_inclusion_toy, d02_inclusion_rank_test


@app.cell
def _(
    d02_complement_split,
    d02_goal_inclusion_toy,
    d02_gram_schmidt_qr,
    d02_inclusion_rank_test,
    mo,
    np,
    d02_orthonormal_coordinates,
    d02_projection_onto_line,
    d02_projection_onto_plane,
):
    _pl = d02_projection_onto_line([2, 1], [1, 3])
    assert np.isclose(_pl["xhat"], 1) and np.allclose(_pl["e"], [-1, 2]) and abs(_pl["e_dot_a"]) < 1e-12
    _oc = d02_orthonormal_coordinates([2, 1], [[0.6, 0.8], [-0.8, 0.6]])
    _pp = d02_projection_onto_plane([1, 2, 3], [np.array([1, -1, 0]) / np.sqrt(2), np.array([1, 1, -2]) / np.sqrt(6)])
    assert np.allclose(_pp["p"], [-1, 0, 1]) and np.allclose(_pp["e"], [2, 2, 2])
    _gs = d02_gram_schmidt_qr([[3, 1], [2, 2]])
    _gs3 = d02_gram_schmidt_qr([[1, 1, 0], [1, 0, 1]])
    _cs = d02_complement_split([[1, 1, 0], [1, 0, 1]], [1, 2, 3])
    assert np.allclose(_cs["v_row"], [7 / 3, 2 / 3, 5 / 3]) and np.allclose(_cs["coef"], [2 / 3, 5 / 3])
    _in1 = d02_inclusion_rank_test([[1, 1, 0], [1, 0, 1]], [[2, 1, 1]])
    _in2 = d02_inclusion_rank_test([[1, 1, 0], [1, 0, 1]], [[0, 0, 1]])
    _gt = d02_goal_inclusion_toy()
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
    def d02_eigen_2x2(A):
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

    def d02_quadratic_from_eigen(l1, l2, theta_deg):
        """The symmetric A = V diag(l1, l2) V^T whose first eigenvector is at theta_deg
        (the sliders of slide quadratic-forms)."""
        t = np.radians(theta_deg)
        V = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        return (V @ np.diag([l1, l2]) @ V.T).tolist()

    def d02_quadratic_form_axes(A, c=1.0):
        """Shape of J(z) = 1/2 z^T A z for symmetric 2x2 A (slides quadratic-forms, -derivation).

        In eigenvector coordinates y = V^T z, J = 1/2 (l1 y1^2 + l2 y2^2); the level set J = c
        has semi-axis sqrt(2c / l_i) along v_i for every positive l_i.
        """
        A = np.asarray(A, dtype=float)
        e = d02_eigen_2x2(A)
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

    def d02_definiteness_table():
        """The three example matrices of slide positive-definite with their eigenvalues."""
        rows = []
        for M in ([[2, 1], [1, 2]], [[1, 1], [1, 1]], [[1, 2], [2, 1]]):
            q = d02_quadratic_form_axes(M)
            rows.append({"A": M, "values": q["values"], "kind": q["kind"],
                         "det": float(np.linalg.det(np.array(M, dtype=float)))})
        z = np.array([1.0, -1.0])
        return {"rows": rows, "saddle_z": z.tolist(),
                "saddle_value": float(z @ np.array([[1.0, 2.0], [2.0, 1.0]]) @ z)}

    return d02_definiteness_table, d02_eigen_2x2, d02_quadratic_form_axes, d02_quadratic_from_eigen


@app.cell
def _(np):
    def d02_table_delassus(legs=((1, 1), (-1, 1), (-1, -1), (1, -1)), masses=(1.0, 1.0, 1.0)):
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

    def d02_cholesky_2x2_steps(A, b=None):
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

    return d02_cholesky_2x2_steps, d02_table_delassus


@app.cell
def _(
    d02_cholesky_2x2_steps,
    d02_definiteness_table,
    d02_eigen_2x2,
    mo,
    np,
    d02_quadratic_form_axes,
    d02_table_delassus,
):
    _eA = d02_eigen_2x2([[2, 1], [1, 2]])
    _eB = d02_eigen_2x2([[4, 1], [2, 3]])
    _eR = d02_eigen_2x2([[0, -1], [1, 0]])
    _eS = d02_eigen_2x2([[1, 1], [0, 1]])
    assert np.allclose(_eA["values"], [3, 1]) and np.allclose(_eB["values"], [5, 2]) and not _eR["real"]
    _q = d02_quadratic_form_axes([[2, 1], [1, 2]])
    assert np.allclose(_q["semi_axes"], [np.sqrt(2 / 3), np.sqrt(2)])
    _dt = d02_definiteness_table()
    _q54 = d02_quadratic_form_axes([[5, 4], [4, 5]])
    _ch125 = d02_cholesky_2x2_steps([[1, 2], [2, 5]])
    assert np.allclose(_q54["values"], [9, 1]) and np.allclose(_ch125["L"], [[1, 0], [2, 1]])
    _td = d02_table_delassus()
    assert np.allclose(_td["G_eigs"], [4, 4, 4, 0]) and np.allclose(_td["null"], [1, -1, 1, -1])
    _ch = d02_cholesky_2x2_steps([[4, 2], [2, 3]], [2, 1])
    assert np.allclose(_ch["L"], np.linalg.cholesky(np.array([[4.0, 2], [2, 3]]))) and np.allclose(_ch["z"], [0.5, 0])
    _reg = [(mu, d02_cholesky_2x2_steps(np.array([[1.0, 2], [2, 1]]) + mu * np.eye(2))) for mu in (0, 0.5, 1, 1.5, 2, 3)]
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
    | check-eigen | [[5,4],[4,5]]: eigenvalues, kappa, semi-axis ratio | {_f(_q54['values'])}, {_q54['kappa']:.4f}, {_q54['semi_axes'][1] / _q54['semi_axes'][0]:.4f} |
    | check-eigen | Cholesky factor of [[1,2],[2,5]] | {_f(_ch125['L'])} |
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
    `fragile-vs-infeasible`, `drawer-setup`, `drawer-live`, `drawer-control-axes`,
    `row-normalization`. The condition
    number is Hou & Mason 2021 eq. 8, $\operatorname{cond}(A) = \sigma_{\max}/\sigma_{\min}$; the
    crashing index is their eq. 9, the condition number of $[\hat J; \hat C]$ with $\hat J$ an
    orthonormal basis of the rows of $J$ and $\hat C$ the rows of $C$ normalized.
    """)
    return


@app.cell
def _(np):
    def d02_svd_circle_ellipse(A):
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

    def d02_svd_subspaces(A):
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

    return d02_svd_circle_ellipse, d02_svd_subspaces


@app.cell
def _(d02_cond, np):
    def d02_nearly_parallel_example():
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

    def d02_nearly_parallel_rows(phi_deg, db):
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

    def d02_drawer_conditioning(theta_deg, eps_deg=0.0, b=1.0):
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

    def d02_crashing_index(J, C):
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

    def d02_fragile_vs_infeasible(db=0.1):
        """The four panels of slide fragile-vs-infeasible, all built by d02_nearly_parallel_rows:
        rows 90 deg apart, 2 deg apart, and parallel (0 deg), with b_2 moved by db; the fourth
        panel is the parallel pair with db = 0. For each: ranks of A and [A | b], cond, the
        least-squares solution (minimum norm), its residual |A z - b|, and how far z* = (1, 1) moved.
        """
        out = []
        for name, phi, d in (("well conditioned", 90.0, db), ("ill conditioned", 2.0, db),
                             ("parallel, inconsistent", 0.0, db), ("parallel, consistent", 0.0, 0.0)):
            r = d02_nearly_parallel_rows(phi, d)
            A, b = np.array(r["A"]), np.array(r["b"])
            rA = int(np.linalg.matrix_rank(A))
            rAb = int(np.linalg.matrix_rank(np.column_stack([A, b])))
            z = np.linalg.pinv(A) @ b
            out.append({"name": name, "phi": phi, "db": d, "rank_A": rA, "rank_Ab": rAb,
                        "cond": d02_cond(A), "z": z.tolist(),
                        "residual": float(np.linalg.norm(A @ z - b)),
                        "shift": float(np.linalg.norm(z - 1.0)),
                        "solutions": "one" if rA == 2 else ("none" if rAb > rA else "a line")})
        return out

    return (
        d02_crashing_index,
        d02_drawer_conditioning,
        d02_fragile_vs_infeasible,
        d02_nearly_parallel_example,
        d02_nearly_parallel_rows,
    )


@app.cell
def _(
    d02_cond,
    d02_crashing_index,
    d02_drawer_conditioning,
    d02_fragile_vs_infeasible,
    d02_nearly_parallel_example,
    d02_svd_circle_ellipse,
    d02_svd_subspaces,
    mo,
    np,
):
    _sv = d02_svd_circle_ellipse([[3, 0], [4, 5]])
    assert np.allclose(_sv["S"], [np.sqrt(45), np.sqrt(5)]) and np.isclose(_sv["cond"], 3)
    _ss = d02_svd_subspaces([[1, 1, 0], [1, 0, 1]])
    _npx = d02_nearly_parallel_example()
    assert np.allclose(_npx["z1"], [1, 1]) and np.allclose(_npx["z2"], [0, 2])
    _dr = {th: d02_drawer_conditioning(th) for th in (0, 30, 45, 60, 75, 80, 85, 89, 89.5)}
    for _th, _r in _dr.items():
        assert np.isclose(_r["cond"], _r["cond_formula"], rtol=1e-9)
    _err = [(th, ep, d02_drawer_conditioning(th, ep)) for th, ep in ((0, 2), (60, 2), (85, 2), (85, -2), (88, 2), (88, -2))]
    _ci = d02_crashing_index([[0, 1]], [[10 * np.cos(np.pi / 4), 10 * np.sin(np.pi / 4)]])
    _ci0 = d02_crashing_index([[0, 1], [0, 1]], [[1, 0]])
    _ci1k = d02_crashing_index([[0, 1]], [[1000, 0]])
    _fr = np.array([[1.0, 1.0], [1.0, 1.001]])
    _fr_z = np.linalg.solve(_fr, [2.0, 2.1])
    assert np.allclose(_fr_z, [-98, 100])
    _fi = d02_fragile_vs_infeasible()
    assert [r["solutions"] for r in _fi] == ["one", "one", "none", "a line"]
    _f = lambda v: "(" + ", ".join(f"{x:.4f}" for x in np.ravel(v)) + ")"
    _firows = "\n".join(
        f"    | fragile-vs-infeasible | {r['name']}: rank A, rank [A b], cond, solutions, z, residual, shift | "
        f"{r['rank_A']}, {r['rank_Ab']}, {r['cond']:.4f}, {r['solutions']}, {_f(r['z'])}, {r['residual']:.5f}, {r['shift']:.4f} |"
        for r in _fi
    )
    _Trows = "\n".join(
        f"    | drawer-control-axes | theta = {th}: T, T v for v = (1/cos theta, 0) | "
        f"{_f(_dr[th]['T'])}, {_f(np.array(_dr[th]['T']) @ np.array([_dr[th]['speed_model'], 0.0]))} |"
        for th in (0, 30, 60)
    )
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
{_firows}
{_drrows}
{_errrows}
{_Trows}
    | row-normalization | cond [[0,1],[1000,0]], cond [[0,1],[0,1]] | {d02_cond([[0, 1], [1000, 0]]):.1f}, {d02_cond([[0, 1], [0, 1]])} |
    | row-normalization | drawer at 45 deg with c scaled by 10: raw cond, crashing index | {_ci['raw']:.4f}, {_ci['index']:.4f} |
    | row-normalization | redundant J = [[0,1],[0,1]], C = (1,0): raw, index | {_ci0['raw']:.4f}, {_ci0['index']:.4f} |
    | row-normalization | J = (0,1), C = (1000,0): raw, index | {_ci1k['raw']:.4f}, {_ci1k['index']:.4f} |
    | check-svd | singular values of diag(3,-2); cond | {_f(np.linalg.svd(np.diag([3.0, -2.0]), compute_uv=False))}, {d02_cond(np.diag([3.0, -2.0])):.4f} |
    | check-svd | x + y = 2, x + 1.001 y = 2.1: cond, solution | {d02_cond(_fr):.2f}, {_f(_fr_z)} |
    """)
    return


@app.cell
def _(d02_drawer_conditioning, np, plt):
    _th = np.linspace(0, 89, 300)
    _fig, _ax = plt.subplots(figsize=(5.5, 2.8))
    _ax.semilogy(_th, [d02_drawer_conditioning(t)["cond"] for t in _th], lw=2)
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
    def d02_line_fit_normal_equations(t, y):
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

    def d02_column_space_projection(A, b):
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

    def d02_min_norm_solution(A, b):
        """Minimum-norm solution x+ = A^T (A A^T)^-1 b of a wide system with independent rows
        (slide min-norm), checked against the pseudoinverse."""
        A = np.atleast_2d(np.asarray(A, dtype=float))
        b = np.asarray(b, dtype=float)
        x = A.T @ np.linalg.solve(A @ A.T, b)
        return {"x": x.tolist(), "norm": float(np.linalg.norm(x)),
                "pinv_x": (np.linalg.pinv(A) @ b).tolist(), "null": d02_nullspace(A).tolist()}

    def d02_table_force_distribution(W=100.0, com=(0.2, 0.1), legs=((1, 1), (-1, 1), (-1, -1), (1, -1)), t=0.0):
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

    def d02_pseudoinverse_rank_one(A=((1, 2), (2, 4)), b=(1, 1), mus=(1.0, 0.1, 0.01, 0.001)):
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
        d02_column_space_projection,
        d02_line_fit_normal_equations,
        d02_min_norm_solution,
        d02_pseudoinverse_rank_one,
        d02_table_force_distribution,
    )


@app.cell
def _(np):
    def d02_schur_block_solve(K, rhs, n1):
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

    def d02_schur_minimize_out(a, b, c):
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

    return d02_schur_block_solve, d02_schur_minimize_out


@app.cell
def _(
    d02_column_space_projection,
    d02_line_fit_normal_equations,
    d02_min_norm_solution,
    mo,
    np,
    d02_pseudoinverse_rank_one,
    d02_schur_block_solve,
    d02_schur_minimize_out,
    d02_table_force_distribution,
):
    _lf = d02_line_fit_normal_equations([0, 1, 2, 3], [1, 2, 2, 4])
    assert np.isclose(_lf["c"], 0.9) and np.isclose(_lf["d"], 0.9) and np.isclose(_lf["sse"], 0.7)
    _cp = d02_column_space_projection([[1, 0], [1, 1], [1, 2]], [6, 0, 0])
    assert np.allclose(_cp["xhat"], [5, -3]) and np.allclose(_cp["e"], [1, -2, 1])
    _mn = d02_min_norm_solution([[1, 1]], [2])
    _tf = d02_table_force_distribution()
    assert np.allclose(_tf["f_plus"], [32.5, 22.5, 17.5, 27.5]) and np.allclose(_tf["t_range"], [-17.5, 22.5])
    _tf10 = d02_table_force_distribution(t=10.0)
    _pi = d02_pseudoinverse_rank_one()
    _sb = d02_schur_block_solve([[4, 2], [2, 2]], [6, 2], 1)
    assert np.allclose(_sb["x"], [2]) and np.allclose(_sb["u"], [-1])
    _sm = {b: d02_schur_minimize_out(4, b, 2) for b in (0, 1, 2, 2.5, 3, 4)}
    _cx = d02_schur_minimize_out(0, 1, 1)
    _k1 = d02_line_fit_normal_equations([0, 1, 2], [0, 1, 1])
    _k2 = d02_min_norm_solution([[1, 2]], [5])
    _k3 = d02_table_force_distribution(com=(0.0, 0.0))
    _k4 = d02_schur_minimize_out(2, 1, 1)
    assert np.isclose(_k1["c"], 1 / 6) and np.isclose(_k1["d"], 0.5) and np.allclose(_k2["x"], [1, 2])
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
    | check-least-squares | line through (0,0), (1,1), (2,1): c, d | {_k1['c']:.4f}, {_k1['d']:.4f} |
    | check-least-squares | x1 + 2 x2 = 5: x+, norm | {_f(_k2['x'])}, {_k2['norm']:.4f} |
    | check-least-squares | table, load at the centre: f+ | {_f(_k3['f_plus'])} |
    | check-least-squares | [[2,1],[1,1]]: s, pd | {_k4['s']:.4f}, {_k4['pd']} |
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
    ## Deck 03, part 1: derivatives of one variable (slides `secant-and-tangent` to `taylor-error-order`)

    The running function is the quartic $J(z) = z^4 - 4z^2 + z + 4 = (z^2 - 2)^2 + z$.
    `d03_secant_slope` backs `secant-and-tangent`; `d03_secant_table` backs `derivative-as-limit`;
    `d03_quartic_derivs` backs `sum-rule-and-the-quartic`; `d03_sincos_check` and
    `d03_rotation_derivative` back `sine-cosine-derivatives`; `d03_stationary_points` and
    `d03_inflection_points` back `second-derivative-curvature`; `d03_taylor_coeffs` backs
    `taylor-polynomials`; `d03_taylor_poly` backs `taylor-models-live` (and the title figure);
    `d03_taylor_error_table` backs `taylor-error-order`. The quartic is evaluated with
    multiplications only, `z * z * z * z`, so that its JavaScript twin in
    `slides/lib/algo/03_derivatives.js` produces the same bits (the finite-difference error curve
    of part 4 depends on them).
    """)
    return


@app.cell
def _(np):
    def d03_quartic(z):
        """The running quartic J(z) = z^4 - 4 z^2 + z + 4, with multiplications only."""
        z = float(z)
        return z * z * z * z - 4.0 * z * z + z + 4.0

    def d03_quartic_derivs(z):
        """[J, J', J'', J''', J''''] of the quartic at z (slide sum-rule-and-the-quartic)."""
        z = float(z)
        return [d03_quartic(z), 4.0 * z * z * z - 8.0 * z + 1.0, 12.0 * z * z - 8.0, 24.0 * z, 24.0]

    def d03_secant_slope(z0, h):
        """Slope (J(z0 + h) - J(z0)) / h of the secant through z0 and z0 + h (slide secant-and-tangent)."""
        return (d03_quartic(z0 + h) - d03_quartic(z0)) / h

    def d03_secant_table(z0=1.0, hs=(0.5, 0.1, 0.01, 0.001)):
        """Secant slopes at z0 for shrinking h, and their distance from J'(z0) (slide derivative-as-limit).

        For the quartic at z0 = 1 the error is exactly 2h + 4h^2 + h^3.
        """
        d = d03_quartic_derivs(z0)[1]
        return [{"h": h, "slope": d03_secant_slope(z0, h), "error": d03_secant_slope(z0, h) - d} for h in hs]

    def d03_stationary_points():
        """The three roots of J'(z) = 4 z^3 - 8 z + 1 with J and J'' there (slide second-derivative-curvature).

        Newton's method on J' from -1.5, 0 and 1.5.
        """
        out = []
        for z in (-1.5, 0.0, 1.5):
            for _ in range(60):
                d = d03_quartic_derivs(z)
                step = d[1] / d[2]
                z -= step
                if abs(step) < 1e-15:
                    break
            d = d03_quartic_derivs(z)
            out.append({"z": z, "J": d[0], "d2J": d[2]})
        return out

    def d03_inflection_points():
        """Zeros of J''(z) = 12 z^2 - 8: z = -sqrt(2/3) and +sqrt(2/3)."""
        r = float(np.sqrt(2.0 / 3.0))
        return [-r, r]

    def d03_taylor_coeffs(z0):
        """c_k = J^(k)(z0) / k!, k = 0..4, of the quartic's Taylor polynomial at z0 (slide taylor-polynomials).

        The degree-4 polynomial equals J exactly: at z0 = 1, J(1 + d) = 2 - 3d + 2d^2 + 4d^3 + d^4.
        """
        d = d03_quartic_derivs(z0)
        fact = [1.0, 1.0, 2.0, 6.0, 24.0]
        return [d[k] / fact[k] for k in range(5)]

    def d03_taylor_poly(z0, delta, degree):
        """Value at z0 + delta of the degree-n Taylor polynomial of the quartic around z0.

        Degree 1 is the tangent line (linear model), degree 2 the quadratic model (slide taylor-models-live).
        """
        c = d03_taylor_coeffs(z0)
        s, p = 0.0, 1.0
        for k in range(int(degree) + 1):
            s += c[k] * p
            p *= delta
        return s

    def d03_taylor_error_table(z0=1.0, deltas=(0.2, 0.1, 0.05, 0.025)):
        """Errors of the linear and quadratic models for halving steps, with the ratios of successive
        errors, which approach 4 and 8 (slide taylor-error-order)."""
        rows = []
        for d in deltas:
            J = d03_quartic(z0 + d)
            lin, quad = d03_taylor_poly(z0, d, 1), d03_taylor_poly(z0, d, 2)
            rows.append({"delta": d, "J": J, "lin": lin, "quad": quad, "err_lin": J - lin, "err_quad": J - quad})
        for a, b in zip(rows, rows[1:]):
            b["ratio_lin"] = a["err_lin"] / b["err_lin"]
            b["ratio_quad"] = a["err_quad"] / b["err_quad"]
        return rows

    def d03_sincos_check(theta, h=1e-5):
        """Central-difference slopes of cos and sin at theta next to -sin(theta) and cos(theta)
        (slide sine-cosine-derivatives); theta in radians."""
        dc = (np.cos(theta + h) - np.cos(theta - h)) / (2 * h)
        ds = (np.sin(theta + h) - np.sin(theta - h)) / (2 * h)
        return {"dcos_fd": float(dc), "minus_sin": float(-np.sin(theta)), "dsin_fd": float(ds), "cos": float(np.cos(theta))}

    def d03_rotation_derivative(theta):
        """dR/dtheta of the planar rotation R(theta), entry by entry, and R(theta) S with the quarter
        turn S = [[0, -1], [1, 0]]; the two are equal (slide sine-cosine-derivatives)."""
        c, s = np.cos(theta), np.sin(theta)
        dR = np.array([[-s, -c], [c, -s]])
        RS = np.array([[c, -s], [s, c]]) @ np.array([[0.0, -1.0], [1.0, 0.0]])
        return {"dR": dR.tolist(), "RS": RS.tolist()}

    return (
        d03_inflection_points,
        d03_quartic,
        d03_quartic_derivs,
        d03_rotation_derivative,
        d03_secant_slope,
        d03_secant_table,
        d03_sincos_check,
        d03_stationary_points,
        d03_taylor_coeffs,
        d03_taylor_error_table,
        d03_taylor_poly,
    )


@app.cell
def _(
    d03_inflection_points,
    d03_quartic,
    d03_quartic_derivs,
    d03_rotation_derivative,
    d03_secant_slope,
    d03_secant_table,
    d03_sincos_check,
    d03_stationary_points,
    d03_taylor_coeffs,
    d03_taylor_error_table,
    d03_taylor_poly,
    mo,
    np,
):
    _dv = {z: d03_quartic_derivs(z) for z in (-1.0, 0.0, 1.0, 2.0)}
    _sec = d03_secant_table()
    _st = d03_stationary_points()
    _tc = d03_taylor_coeffs(1.0)
    _te = d03_taylor_error_table()
    _sc = d03_sincos_check(np.radians(30.0))
    _rd = d03_rotation_derivative(np.radians(30.0))
    assert _dv[1.0] == [2.0, -3.0, 4.0, 24.0, 24.0] and _dv[2.0][:3] == [6.0, 17.0, 40.0]
    assert np.isclose(d03_secant_slope(1.0, 0.5), -0.875) and np.isclose(d03_secant_slope(1.0, 0.1), -2.759)
    assert all(np.isclose(r["error"], 2 * r["h"] + 4 * r["h"] ** 2 + r["h"] ** 3) for r in _sec)
    assert np.allclose([s["z"] for s in _st], np.sort(np.roots([4, 0, -8, 1]).real))
    assert np.allclose(_tc, [2, -3, 2, 4, 1]) and np.isclose(d03_taylor_poly(1.0, 0.3, 4), d03_quartic(1.3))
    assert np.isclose(_te[1]["err_lin"], 0.0241) and np.isclose(_te[1]["err_quad"], 0.0041)
    assert np.allclose(_rd["dR"], _rd["RS"])
    # J = (z^2 - 2)^2 + z is the same polynomial (slide chain-rule-one-variable)
    assert all(np.isclose(d03_quartic(z), (z * z - 2) ** 2 + z) for z in np.linspace(-2, 2, 9))
    _f = lambda v, n=4: ", ".join(f"{x:.{n}f}" for x in v)
    mo.md(f"""
| slide | quantity | value |
|---|---|---|
| sum-rule-and-the-quartic | [J, J', J''] at z = -1, 0, 1, 2 | {"; ".join(f"z = {z:g}: " + _f(_dv[z][:3], 0) for z in _dv)} |
| secant-and-tangent, derivative-as-limit | secant slopes at z0 = 1, h = 0.5, 0.1, 0.01, 0.001 | {_f([r['slope'] for r in _sec], 6)} |
| derivative-as-limit | slope - J'(1) | {_f([r['error'] for r in _sec], 6)} |
| second-derivative-curvature | stationary points z, J, J'' | {"; ".join(_f([s['z'], s['J'], s['d2J']]) for s in _st)} |
| second-derivative-curvature | inflection points, J there | {_f(d03_inflection_points())}; {_f([d03_quartic(z) for z in d03_inflection_points()])} |
| taylor-polynomials | c_k at z0 = 1 (k = 0..4) | {_f(_tc, 0)} |
| taylor-polynomials | c_k at z0 = 0, z0 = 2 | {_f(d03_taylor_coeffs(0.0), 0)}; {_f(d03_taylor_coeffs(2.0), 0)} |
| taylor-error-order | delta, J, linear error, quadratic error | {"; ".join(_f([r['delta'], r['J'], r['err_lin'], r['err_quad']], 5) for r in _te)} |
| taylor-error-order | ratios of successive errors (linear, quadratic) | {"; ".join(_f([r['ratio_lin'], r['ratio_quad']], 2) for r in _te[1:])} |
| sine-cosine-derivatives | at 30 deg: FD of cos, -sin, FD of sin, cos | {_f([_sc['dcos_fd'], _sc['minus_sin'], _sc['dsin_fd'], _sc['cos']], 6)} |
| sine-cosine-derivatives | dR/dtheta at 30 deg | {_rd['dR']} |
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Deck 03, part 2: gradients, Hessians, matrix calculus (slides `contour-plots` to `gradient-least-squares`)

    The two-variable running function is the quartic along $z_1$ plus a valley term,
    $J(z_1, z_2) = z_1^4 - 4z_1^2 + z_1 + 4 + 2(z_2 - z_1)^2$. Its gradient is
    $(4z_1^3 - 8z_1 + 1 - 4(z_2 - z_1),\ 4(z_2 - z_1))$ and its Hessian
    $\begin{bmatrix} 12z_1^2 - 4 & -4 \\ -4 & 4 \end{bmatrix}$. `d03_valley` backs `contour-plots`;
    `d03_partial_slices` backs `partial-derivatives`; `d03_valley_grad` backs `gradient-vector`;
    `d03_directional_derivative` backs `directional-derivative`; `d03_steepest_directions` backs
    `steepest-ascent`; `d03_level_tangent` backs `gradient-perpendicular-level-set`;
    `d03_valley_hess` backs `hessian-matrix`; `d03_slice_curvature` and `d03_hessian_eigen` back
    `curvature-along-a-line`; `d03_valley_models` backs `multivariable-taylor` and
    `quadratic-model-contours`; `d03_grad_linear` and `d03_grad_quadratic` back
    `gradient-linear-quadratic-forms`; `d03_grad_least_squares` backs `gradient-least-squares`.
    `d03_valley_stationary` lists the two minima and the saddle.
    """)
    return


@app.cell
def _(d03_stationary_points, np):
    def d03_valley(z):
        """J(z1, z2) = z1^4 - 4 z1^2 + z1 + 4 + 2 (z2 - z1)^2 (slides contour-plots onward)."""
        z1, z2 = float(z[0]), float(z[1])
        e = z2 - z1
        return z1 * z1 * z1 * z1 - 4.0 * z1 * z1 + z1 + 4.0 + 2.0 * e * e

    def d03_valley_grad(z):
        """Gradient (4 z1^3 - 8 z1 + 1 - 4 (z2 - z1), 4 (z2 - z1)) of the valley function."""
        z1, z2 = float(z[0]), float(z[1])
        return [4.0 * z1 * z1 * z1 - 8.0 * z1 + 1.0 - 4.0 * (z2 - z1), 4.0 * (z2 - z1)]

    def d03_valley_hess(z):
        """Hessian [[12 z1^2 - 4, -4], [-4, 4]] of the valley function (slide hessian-matrix)."""
        z1 = float(z[0])
        return [[12.0 * z1 * z1 - 4.0, -4.0], [-4.0, 4.0]]

    def d03_valley_stationary():
        """Stationary points (r, r), r a root of the quartic's J', with J and the Hessian eigenvalues."""
        out = []
        for p in d03_stationary_points():
            ev = np.linalg.eigvalsh(np.array(d03_valley_hess([p["z"], p["z"]])))
            out.append({"z": [p["z"], p["z"]], "J": d03_valley([p["z"], p["z"]]), "eig": [float(ev[1]), float(ev[0])]})
        return out

    def d03_partial_slices(z):
        """J, the two partial derivatives, and their central-difference checks at z (slide
        partial-derivatives). dJ/dz1 is the slope of the slice t -> J(t, z2) at t = z1."""
        h = 1e-6
        z1, z2 = float(z[0]), float(z[1])
        fd1 = (d03_valley([z1 + h, z2]) - d03_valley([z1 - h, z2])) / (2 * h)
        fd2 = (d03_valley([z1, z2 + h]) - d03_valley([z1, z2 - h])) / (2 * h)
        return {"J": d03_valley(z), "grad": d03_valley_grad(z), "fd": [fd1, fd2]}

    def d03_directional_derivative(z, phi_deg):
        """grad J . d for the unit vector d = (cos phi, sin phi), next to the difference quotient
        (J(z + t d) - J(z - t d)) / (2 t), t = 1e-6 (slide directional-derivative)."""
        phi = np.radians(phi_deg)
        d = np.array([np.cos(phi), np.sin(phi)])
        g = np.array(d03_valley_grad(z))
        t = 1e-6
        z = np.asarray(z, dtype=float)
        fd = (d03_valley(z + t * d) - d03_valley(z - t * d)) / (2 * t)
        ng = float(np.linalg.norm(g))
        return {"d": d.tolist(), "D": float(g @ d), "fd": float(fd), "norm_grad": ng,
                "cos": float(g @ d / ng) if ng > 0 else None}

    def d03_steepest_directions(z):
        """Angles (deg) of steepest ascent grad J / |grad J|, steepest descent, and the two level-set
        directions (grad J . d = 0), and the steepest rate |grad J| (slide steepest-ascent)."""
        g = np.array(d03_valley_grad(z))
        a = float(np.degrees(np.arctan2(g[1], g[0])))
        wrap = lambda x: (x + 180.0) % 360.0 - 180.0
        return {"ascent": a, "descent": wrap(a + 180.0), "level": sorted([wrap(a + 90.0), wrap(a - 90.0)]),
                "rate": float(np.linalg.norm(g))}

    def d03_level_tangent(z):
        """Unit tangent of the level curve through z (grad J turned by +90 deg), grad J . tangent (= 0),
        and J at z +- 1e-3 tangent, which differs from J(z) only at second order
        (slide gradient-perpendicular-level-set)."""
        g = np.array(d03_valley_grad(z))
        t = np.array([-g[1], g[0]]) / np.linalg.norm(g)
        z = np.asarray(z, dtype=float)
        s = 1e-3
        return {"tangent": t.tolist(), "dot": float(g @ t), "J": d03_valley(z),
                "J_plus": d03_valley(z + s * t), "J_minus": d03_valley(z - s * t)}

    def d03_slice_curvature(z, phi_deg):
        """Along z + t d, d = (cos phi, sin phi): g(t) = J(z + t d) has g'(0) = grad J . d and
        g''(0) = d^T H d; a central second difference checks g''(0) (slide curvature-along-a-line)."""
        phi = np.radians(phi_deg)
        d = np.array([np.cos(phi), np.sin(phi)])
        z = np.asarray(z, dtype=float)
        H = np.array(d03_valley_hess(z))
        t = 1e-4
        fd2 = (d03_valley(z + t * d) - 2 * d03_valley(z) + d03_valley(z - t * d)) / (t * t)
        return {"g1": float(np.array(d03_valley_grad(z)) @ d), "g2": float(d @ H @ d), "fd2": float(fd2)}

    def d03_hessian_eigen(z):
        """Eigenvalues (descending) and unit eigenvectors (largest entry positive) of the valley Hessian:
        the largest and smallest curvature d^T H d over unit d and their directions."""
        w, V = np.linalg.eigh(np.array(d03_valley_hess(z)))
        vecs = []
        for k in (1, 0):
            v = V[:, k]
            if v[np.argmax(np.abs(v))] < 0:
                v = -v
            vecs.append(v.tolist())
        return {"values": [float(w[1]), float(w[0])], "vectors": vecs}

    def d03_valley_models(z0, delta):
        """J(z0 + delta) next to the linear model J + grad J . delta and the quadratic model
        (+ delta^T H delta / 2) (slides multivariable-taylor, quadratic-model-contours)."""
        z0 = np.asarray(z0, dtype=float)
        delta = np.asarray(delta, dtype=float)
        g = np.array(d03_valley_grad(z0))
        H = np.array(d03_valley_hess(z0))
        lin = d03_valley(z0) + g @ delta
        return {"J": d03_valley(z0 + delta), "lin": float(lin), "quad": float(lin + 0.5 * delta @ H @ delta)}

    def d03_fd_gradient(f, x, h=1e-6):
        """Central-difference gradient of a scalar function f at x, one coordinate at a time."""
        x = np.asarray(x, dtype=float)
        g = np.zeros_like(x)
        for k in range(len(x)):
            e = np.zeros_like(x)
            e[k] = h
            g[k] = (f(x + e) - f(x - e)) / (2 * h)
        return g

    def d03_grad_linear(b, x):
        """f(x) = b^T x has gradient b; central differences check it (slide gradient-linear-quadratic-forms)."""
        b = np.asarray(b, dtype=float)
        return {"f": float(b @ np.asarray(x, dtype=float)), "grad": b.tolist(),
                "fd": d03_fd_gradient(lambda y: b @ y, x).tolist()}

    def d03_grad_quadratic(A, x):
        """f(x) = x^T A x has gradient (A + A^T) x (2 A x for symmetric A) and Hessian A + A^T
        (slide gradient-linear-quadratic-forms)."""
        A = np.asarray(A, dtype=float)
        x = np.asarray(x, dtype=float)
        return {"f": float(x @ A @ x), "grad": ((A + A.T) @ x).tolist(), "hess": (A + A.T).tolist(),
                "fd": d03_fd_gradient(lambda y: y @ A @ y, x).tolist()}

    def d03_grad_least_squares(A, b, x):
        """f(x) = (1/2) |A x - b|^2: residual r = A x - b, gradient A^T r, Hessian A^T A, and the
        minimizer from the normal equations A^T A x = A^T b (slide gradient-least-squares)."""
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        x = np.asarray(x, dtype=float)
        r = A @ x - b
        return {"f": float(0.5 * r @ r), "r": r.tolist(), "grad": (A.T @ r).tolist(), "hess": (A.T @ A).tolist(),
                "AtB": (A.T @ b).tolist(), "xstar": np.linalg.solve(A.T @ A, A.T @ b).tolist(),
                "fd": d03_fd_gradient(lambda y: 0.5 * (A @ y - b) @ (A @ y - b), x).tolist()}

    return (
        d03_directional_derivative,
        d03_fd_gradient,
        d03_grad_least_squares,
        d03_grad_linear,
        d03_grad_quadratic,
        d03_hessian_eigen,
        d03_level_tangent,
        d03_partial_slices,
        d03_slice_curvature,
        d03_steepest_directions,
        d03_valley,
        d03_valley_grad,
        d03_valley_hess,
        d03_valley_models,
        d03_valley_stationary,
    )


@app.cell
def _(
    d03_directional_derivative,
    d03_grad_least_squares,
    d03_grad_linear,
    d03_grad_quadratic,
    d03_hessian_eigen,
    d03_level_tangent,
    d03_partial_slices,
    d03_slice_curvature,
    d03_steepest_directions,
    d03_valley,
    d03_valley_grad,
    d03_valley_hess,
    d03_valley_models,
    d03_valley_stationary,
    mo,
    np,
):
    _ps = d03_partial_slices([0.0, 1.0])
    _dd = {phi: d03_directional_derivative([0.0, 1.0], phi) for phi in (0.0, 90.0, 53.13010235415598, 126.86989764584402, 36.86989764584402)}
    _sd = d03_steepest_directions([0.0, 1.0])
    _lt = d03_level_tangent([0.0, 1.0])
    _sc = {phi: d03_slice_curvature([1.0, 0.0], phi) for phi in (0.0, 45.0, 90.0, 135.0)}
    _he = d03_hessian_eigen([1.0, 0.0])
    _he01 = d03_hessian_eigen([0.0, 1.0])
    _vm = d03_valley_models([1.0, 0.0], [0.1, 0.1])
    _gl = d03_grad_linear([2.0, -1.0], [3.0, 1.0])
    _gq = d03_grad_quadratic([[1.0, 2.0], [0.0, 3.0]], [1.0, 1.0])
    _ls = d03_grad_least_squares([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], [1.0, 2.0, 2.0], [0.0, 1.0])
    _vs = d03_valley_stationary()
    assert d03_valley([0, 1]) == 6.0 and d03_valley_grad([0, 1]) == [-3.0, 4.0] and d03_valley([1, 0]) == 4.0
    assert np.allclose(_ps["fd"], _ps["grad"], atol=1e-6) and np.isclose(_sd["rate"], 5.0)
    assert all(np.isclose(v["D"], v["fd"], atol=1e-6) for v in _dd.values())
    assert np.isclose(_dd[53.13010235415598]["D"], 1.4) and abs(_dd[36.86989764584402]["D"]) < 1e-12
    assert np.allclose(_lt["tangent"], [-0.8, -0.6]) and abs(_lt["dot"]) < 1e-12
    assert np.allclose([_sc[p]["g2"] for p in _sc], [8, 2, 4, 10]) and all(np.isclose(v["g2"], v["fd2"], atol=1e-5) for v in _sc.values())
    assert np.allclose(_he["values"], [6 + np.sqrt(20), 6 - np.sqrt(20)]) and np.allclose(_he01["values"], [np.sqrt(32), -np.sqrt(32)])
    assert np.isclose(_vm["J"], 3.7241) and np.isclose(_vm["lin"], 3.7) and np.isclose(_vm["quad"], 3.72)
    assert np.allclose(_gq["grad"], [4, 8]) and np.allclose(_ls["grad"], [1, 3]) and np.allclose(_ls["xstar"], [2 / 3, 0.5])
    _f = lambda v, n=4: ", ".join(f"{x:.{n}f}" for x in v)
    mo.md(f"""
| slide | quantity | value |
|---|---|---|
| contour-plots | J(0, 1), J(1, 0), J(1, 1) | {d03_valley([0, 1]):g}, {d03_valley([1, 0]):g}, {d03_valley([1, 1]):g} |
| partial-derivatives | grad at (0, 1), central differences | ({_f(_ps['grad'], 0)}), ({_f(_ps['fd'], 6)}) |
| gradient-vector | grad at (1, 0), (1, 1), (2, 1) | ({_f(d03_valley_grad([1, 0]), 0)}), ({_f(d03_valley_grad([1, 1]), 0)}), ({_f(d03_valley_grad([2, 1]), 0)}) |
| directional-derivative | grad . d at (0, 1) for phi = 0, 90, 53.13, 126.87, 36.87 deg | {_f([v['D'] for v in _dd.values()])} |
| steepest-ascent | ascent, descent, level directions (deg), rate | {_sd['ascent']:.2f}, {_sd['descent']:.2f}, {_f(_sd['level'], 2)}, {_sd['rate']:.1f} |
| gradient-perpendicular-level-set | tangent at (0, 1); J(z +- 0.001 t) | ({_f(_lt['tangent'], 1)}); {_lt['J_plus']:.9f}, {_lt['J_minus']:.9f} |
| hessian-matrix | H at (1, 0), (0, 1) | {d03_valley_hess([1, 0])}, {d03_valley_hess([0, 1])} |
| curvature-along-a-line | d^T H d at (1, 0), phi = 0, 45, 90, 135 deg | {_f([_sc[p]['g2'] for p in _sc])} |
| curvature-along-a-line | eigenvalues, eigenvectors at (1, 0) | {_f(_he['values'])}; {[[round(x, 4) for x in v] for v in _he['vectors']]} |
| quadratic-model-contours | eigenvalues at (0, 1) | {_f(_he01['values'])} |
| multivariable-taylor | at (1, 0), delta = (0.1, 0.1): J, linear, quadratic | {_f([_vm['J'], _vm['lin'], _vm['quad']])} |
| stationary points | z1 = z2, J, Hessian eigenvalues | {"; ".join(f"{s['z'][0]:.4f}, {s['J']:.4f}, ({_f(s['eig'], 2)})" for s in _vs)} |
| gradient-linear-quadratic-forms | b = (2, -1): grad; A = [[1, 2], [0, 3]], x = (1, 1): f, grad, Hessian | ({_f(_gl['grad'], 0)}); {_gq['f']:g}, ({_f(_gq['grad'], 0)}), {_gq['hess']} |
| gradient-least-squares | r, f, grad, A^T A, A^T b, x* | ({_f(_ls['r'], 0)}), {_ls['f']:g}, ({_f(_ls['grad'], 0)}), {_ls['hess']}, ({_f(_ls['AtB'], 0)}), ({_f(_ls['xstar'])}) |
""")
    return


@app.cell
def _(d03_valley, d03_valley_grad, np, plt):
    # Contours of the valley function with its gradient at (0, 1) (slides contour-plots, gradient-vector).
    _x = np.linspace(-2.2, 2.2, 221)
    _X, _Y = np.meshgrid(_x, _x)
    _Z = _X ** 4 - 4 * _X ** 2 + _X + 4 + 2 * (_Y - _X) ** 2
    _fig, _ax = plt.subplots(figsize=(3.6, 3.4))
    _ax.contour(_X, _Y, _Z, levels=[-1, 0, 1, 2, 3, 4, 6, 8, 11, 15, 20], colors="k", linewidths=0.6)
    _g = np.array(d03_valley_grad([0.0, 1.0]))
    _ax.annotate("", xy=(0.12 * _g[0], 1 + 0.12 * _g[1]), xytext=(0, 1), arrowprops=dict(color="#cc0000", width=1.5, headwidth=6))
    _ax.plot([0], [1], "o", color="#cc0000", ms=3)
    _ax.set_aspect("equal")
    _ax.set_title(f"J(z1, z2); J(0, 1) = {d03_valley([0, 1]):g}, grad = (-3, 4)", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Deck 03, part 3: Jacobians, linearization, constraints (slides `jacobian-matrix` to `pivoting-block-constraint`)

    `d03_arm_fk` and `d03_arm_jacobian` back `jacobian-matrix` and `arm-jacobian` (two-link arm,
    $l_1 = l_2 = 1$ m); `d03_arm_first_order` backs the first-order check on `arm-jacobian`;
    `d03_arm_distance_gradient` backs `chain-rule-jacobians`; `d03_pendulum_step` and
    `d03_pendulum_AB` back `linearizing-dynamics` (pendulum $m = 1$ kg, $l = 1$ m,
    $g = 9.81$ m/s², semi-implicit Euler, $h = 0.01$ s, the update of MuJoCo's default integrator);
    `d03_pendulum_rollouts` backs `linearization-live`; `d03_bead_constraint` backs
    `constraint-time-derivative`; `d03_pivot_constraint`, `d03_pivot_jacobian`, `d03_pivot_residual`
    and `d03_pivot_pose` back `pivoting-block-constraint` (block 0.2 m by 0.1 m pivoting on its
    lower-left corner).
    """)
    return


@app.cell
def _(d03_fd_gradient, np):
    def d03_arm_fk(theta, l=(1.0, 1.0)):
        """Fingertip of a planar two-link arm, joint angles theta (rad), link lengths l (m)."""
        t1, t2 = float(theta[0]), float(theta[1])
        return [l[0] * np.cos(t1) + l[1] * np.cos(t1 + t2), l[0] * np.sin(t1) + l[1] * np.sin(t1 + t2)]

    def d03_arm_jacobian(theta, l=(1.0, 1.0)):
        """2 x 2 Jacobian dp/dtheta of the two-link arm and its determinant l1 l2 sin(theta2)
        (slides jacobian-matrix, arm-jacobian)."""
        t1, t2 = float(theta[0]), float(theta[1])
        s1, c1 = np.sin(t1), np.cos(t1)
        s12, c12 = np.sin(t1 + t2), np.cos(t1 + t2)
        J = [[-l[0] * s1 - l[1] * s12, -l[1] * s12], [l[0] * c1 + l[1] * c12, l[1] * c12]]
        return {"J": J, "det": float(J[0][0] * J[1][1] - J[0][1] * J[1][0])}

    def d03_arm_first_order(theta, dtheta, l=(1.0, 1.0)):
        """True tip displacement p(theta + dtheta) - p(theta) next to the prediction J dtheta."""
        p0 = np.array(d03_arm_fk(theta, l))
        p1 = np.array(d03_arm_fk(np.add(theta, dtheta), l))
        J = np.array(d03_arm_jacobian(theta, l)["J"])
        return {"true": (p1 - p0).tolist(), "lin": (J @ np.asarray(dtheta, dtype=float)).tolist()}

    def d03_arm_distance_gradient(theta, target, l=(1.0, 1.0)):
        """w(theta) = (1/2) |p(theta) - p*|^2 has gradient J^T (p - p*) by the chain rule, a product
        of the 1 x 2 Jacobian (p - p*)^T and the 2 x 2 arm Jacobian (slide chain-rule-jacobians)."""
        tg = np.asarray(target, dtype=float)
        e = np.array(d03_arm_fk(theta, l)) - tg
        J = np.array(d03_arm_jacobian(theta, l)["J"])
        w = lambda th: 0.5 * float(np.sum((np.array(d03_arm_fk(th, l)) - tg) ** 2))
        return {"p": (e + tg).tolist(), "e": e.tolist(), "w": float(0.5 * e @ e), "grad": (J.T @ e).tolist(),
                "fd": d03_fd_gradient(w, theta).tolist()}

    D03_PEND = {"g": 9.81, "l": 1.0, "m": 1.0}

    def d03_pendulum_step(x, u, h=0.01, p=D03_PEND):
        """One semi-implicit Euler step of the pendulum, x = (theta, omega), torque u:
        a = -(g/l) sin(theta) + u / (m l^2); omega' = omega + h a; theta' = theta + h omega'."""
        th, om = float(x[0]), float(x[1])
        a = -(p["g"] / p["l"]) * np.sin(th) + float(u) / (p["m"] * p["l"] * p["l"])
        om1 = om + h * a
        return [th + h * om1, om1]

    def d03_pendulum_AB(xbar, ubar=0.0, h=0.01, p=D03_PEND):
        """A = df/dx and B = df/du of the step at (xbar, ubar) (slide linearizing-dynamics).

        With k = (g/l) cos(theta) and b = 1 / (m l^2): A = [[1 - h^2 k, h], [-h k, 1]],
        B = [[h^2 b], [h b]], det A = 1.
        """
        k = (p["g"] / p["l"]) * np.cos(float(xbar[0]))
        b = 1.0 / (p["m"] * p["l"] * p["l"])
        A = [[1.0 - h * h * k, h], [-h * k, 1.0]]
        B = [[h * h * b], [h * b]]
        return {"A": A, "B": B, "det": float(A[0][0] * A[1][1] - A[0][1] * A[1][0])}

    def d03_pendulum_rollouts(dtheta0_deg, theta_bar_deg=0.0, T=2.0, h=0.01):
        """Nonlinear rollout from rest at theta_bar + dtheta0 with u = 0, next to the linearized
        prediction xbar + A^k dx0 about the equilibrium (theta_bar, 0) (slide linearization-live).
        theta in degrees at every step, and the largest gap between the two."""
        tb = np.radians(theta_bar_deg)
        A = np.array(d03_pendulum_AB([tb, 0.0], 0.0, h)["A"])
        x = [tb + np.radians(dtheta0_deg), 0.0]
        dx = np.array([np.radians(dtheta0_deg), 0.0])
        th_nl, th_lin = [float(np.degrees(x[0]))], [float(np.degrees(tb + dx[0]))]
        for _ in range(int(round(T / h))):
            x = d03_pendulum_step(x, 0.0, h)
            dx = A @ dx
            th_nl.append(float(np.degrees(x[0])))
            th_lin.append(float(np.degrees(tb + dx[0])))
        gap = max(abs(a - b) for a, b in zip(th_nl, th_lin))
        return {"theta_nl": th_nl, "theta_lin": th_lin, "max_gap_deg": float(gap)}

    def d03_bead_constraint(q, qdot):
        """Bead on the unit circle: Phi = x^2 + y^2 - 1, J_Phi = [2x, 2y], and J_Phi qdot
        (slide constraint-time-derivative)."""
        x, y = float(q[0]), float(q[1])
        J = [2 * x, 2 * y]
        return {"Phi": x * x + y * y - 1.0, "J": J, "Jqdot": J[0] * float(qdot[0]) + J[1] * float(qdot[1])}

    D03_BLOCK = {"a": 0.1, "b": 0.05}

    def d03_pivot_constraint(q, block=D03_BLOCK, p0=(0.0, 0.0)):
        """Phi(q) = (x, y) + R(theta) c - p0 for q = (x, y, theta) and the lower-left corner
        c = (-a, -b) in the block frame (slide pivoting-block-constraint)."""
        x, y, th = (float(v) for v in q)
        c = np.array([-block["a"], -block["b"]])
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return (np.array([x, y]) + R @ c - np.asarray(p0, dtype=float)).tolist()

    def d03_pivot_jacobian(theta, block=D03_BLOCK):
        """J_Phi = [I | R'(theta) c] (2 x 3) of the pivot constraint, and the null vector with
        thetadot = 1: the only motions that keep the corner still are rotations about it."""
        th = float(theta)
        c = np.array([-block["a"], -block["b"]])
        col = np.array([[-np.sin(th), -np.cos(th)], [np.cos(th), -np.sin(th)]]) @ c
        return {"J": [[1.0, 0.0, float(col[0])], [0.0, 1.0, float(col[1])]], "null": [-float(col[0]), -float(col[1]), 1.0]}

    def d03_pivot_residual(theta, qdot, block=D03_BLOCK):
        """J_Phi(theta) qdot, the velocity of the corner, for qdot = (xdot, ydot, thetadot)."""
        return (np.array(d03_pivot_jacobian(theta, block)["J"]) @ np.asarray(qdot, dtype=float)).tolist()

    def d03_pivot_pose(theta, block=D03_BLOCK):
        """The configuration (x, y, theta) with Phi = 0 at angle theta: centre = -R(theta) c."""
        th = float(theta)
        c = np.array([-block["a"], -block["b"]])
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        ctr = -(R @ c)
        return [float(ctr[0]), float(ctr[1]), th]

    return (
        D03_BLOCK,
        D03_PEND,
        d03_arm_distance_gradient,
        d03_arm_first_order,
        d03_arm_fk,
        d03_arm_jacobian,
        d03_bead_constraint,
        d03_pendulum_AB,
        d03_pendulum_rollouts,
        d03_pendulum_step,
        d03_pivot_constraint,
        d03_pivot_jacobian,
        d03_pivot_pose,
        d03_pivot_residual,
    )


@app.cell
def _(
    d03_arm_distance_gradient,
    d03_arm_first_order,
    d03_arm_fk,
    d03_arm_jacobian,
    d03_bead_constraint,
    d03_pendulum_AB,
    d03_pendulum_rollouts,
    d03_pivot_constraint,
    d03_pivot_jacobian,
    d03_pivot_pose,
    d03_pivot_residual,
    mo,
    np,
):
    _th = [0.0, np.pi / 2]
    _aj = d03_arm_jacobian(_th)
    _f1 = d03_arm_first_order(_th, [0.01, 0.02])
    _f2 = d03_arm_first_order(_th, [0.1, 0.2])
    _dg = d03_arm_distance_gradient(_th, [0.5, 1.5])
    _ab0, _abpi = d03_pendulum_AB([0.0, 0.0]), d03_pendulum_AB([np.pi, 0.0])
    _gaps = {d: d03_pendulum_rollouts(d)["max_gap_deg"] for d in (5.0, 20.0, 45.0, 90.0)}
    _gtop = {d: d03_pendulum_rollouts(d, 180.0, T=1.0)["max_gap_deg"] for d in (1.0, 5.0)}
    _pj0, _pj30 = d03_pivot_jacobian(0.0), d03_pivot_jacobian(np.radians(30.0))
    assert np.allclose(d03_arm_fk(_th), [1, 1]) and np.allclose(_aj["J"], [[-1, -1], [1, 0]]) and np.isclose(_aj["det"], 1)
    assert np.allclose(_dg["grad"], [-1, -0.5]) and np.allclose(_dg["grad"], _dg["fd"], atol=1e-6)
    assert np.allclose(_ab0["A"], [[0.999019, 0.01], [-0.0981, 1]]) and np.allclose(_ab0["B"], [[1e-4], [0.01]])
    assert np.isclose(_ab0["det"], 1.0) and np.isclose(_abpi["det"], 1.0)
    assert d03_bead_constraint([0.6, 0.8], [-0.8, 0.6])["Jqdot"] == 0.0
    assert np.allclose(_pj0["J"], [[1, 0, 0.05], [0, 1, -0.1]]) and np.allclose(_pj0["null"], [-0.05, 0.1, 1])
    assert np.allclose(d03_pivot_residual(0.3, d03_pivot_jacobian(0.3)["null"]), [0, 0])
    assert np.allclose(d03_pivot_constraint(d03_pivot_pose(np.radians(30.0))), [0, 0])
    _f = lambda v, n=4: ", ".join(f"{x:.{n}f}" for x in v)
    mo.md(f"""
| slide | quantity | value |
|---|---|---|
| jacobian-matrix | p, J, det at theta = (0, 90 deg) | ({_f(d03_arm_fk(_th), 0)}), {np.round(_aj['J'], 6).tolist()}, {_aj['det']:.4f} |
| arm-jacobian | dtheta = (0.01, 0.02): true, J dtheta | ({_f(_f1['true'], 6)}), ({_f(_f1['lin'], 6)}) |
| arm-jacobian | dtheta = (0.1, 0.2): true, J dtheta | ({_f(_f2['true'])}), ({_f(_f2['lin'])}) |
| arm-jacobian | J, det at (30, 60 deg); det at theta2 = 0 | {np.round(d03_arm_jacobian(np.radians([30, 60]))['J'], 4).tolist()}, {d03_arm_jacobian(np.radians([30, 60]))['det']:.4f}; {d03_arm_jacobian([0.3, 0.0])['det']:.1f} |
| chain-rule-jacobians | p* = (0.5, 1.5): e, w, grad, central differences | ({_f(_dg['e'], 1)}), {_dg['w']:.2f}, ({_f(_dg['grad'], 2)}), ({_f(_dg['fd'], 6)}) |
| linearizing-dynamics | A, B at the bottom | {np.round(_ab0['A'], 6).tolist()}, {_ab0['B']} |
| linearizing-dynamics | A at the top | {np.round(_abpi['A'], 6).tolist()} |
| linearization-live | largest gap over 2 s (deg) from 5, 20, 45, 90 deg | {_f(list(_gaps.values()), 3)} |
| linearization-live | top, largest gap over 1 s (deg) from 1, 5 deg | {_f(list(_gtop.values()), 4)} |
| constraint-time-derivative | J_Phi at (0.6, 0.8); J qdot for (-0.8, 0.6), (1, 0) | {d03_bead_constraint([0.6, 0.8], [0, 0])['J']}; {d03_bead_constraint([0.6, 0.8], [-0.8, 0.6])['Jqdot']:g}, {d03_bead_constraint([0.6, 0.8], [1, 0])['Jqdot']:g} |
| pivoting-block-constraint | J_Phi and null vector at 0 deg | {_pj0['J']}, ({_f(_pj0['null'], 2)}) |
| pivoting-block-constraint | J_Phi at 30 deg | {np.round(_pj30['J'], 4).tolist()} |
| pivoting-block-constraint | corner velocity J qdot at 0 deg for (1, 0, 0), (0, 0, 1) | ({_f(d03_pivot_residual(0.0, [1, 0, 0]), 2)}), ({_f(d03_pivot_residual(0.0, [0, 0, 1]), 2)}) |
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Deck 03, part 4: finite differences (slides `forward-central-differences` to `finite-difference-jacobians`)

    `d03_fd_forward` and `d03_fd_central` back `forward-central-differences` and
    `floating-point-roundoff`; `d03_fd_steps` and `d03_fd_error_curve` back
    `finite-difference-error-curve` (the step sizes are parsed from decimal strings such as
    `"1.5e-7"`, so that Python and JavaScript use the same doubles and the roundoff noise in the
    curve agrees bit for bit); `d03_fd_jacobian` and `d03_fd_pendulum_AB` back
    `finite-difference-jacobians`.
    """)
    return


@app.cell
def _(d03_pendulum_AB, d03_pendulum_step, d03_quartic, d03_quartic_derivs, np):
    def d03_fd_forward(z0, h):
        """Forward difference (J(z0 + h) - J(z0)) / h of the quartic."""
        return (d03_quartic(z0 + h) - d03_quartic(z0)) / h

    def d03_fd_central(z0, h):
        """Central difference (J(z0 + h) - J(z0 - h)) / (2 h) of the quartic."""
        return (d03_quartic(z0 + h) - d03_quartic(z0 - h)) / (2.0 * h)

    def d03_fd_steps():
        """Step sizes m x 10^-e, m in 7, 5, 3, 2, 1.5, 1 and e = 0..15 (1 down to 1e-15), parsed from
        decimal strings so that Python and JavaScript use the same doubles."""
        hs = []
        for e in range(0, 16):
            for m in ("7", "5", "3", "2", "1.5", "1"):
                h = float(m + "e-" + str(e))
                if h <= 1.0:
                    hs.append(h)
        return hs

    def d03_fd_error_curve(z0=1.0, hs=None):
        """|FD - J'(z0)| for forward and central differences over the step sizes (slide
        finite-difference-error-curve): truncation error falls with h, roundoff grows like 1/h."""
        hs = d03_fd_steps() if hs is None else list(hs)
        d = d03_quartic_derivs(z0)[1]
        return {"h": hs, "forward": [abs(d03_fd_forward(z0, h) - d) for h in hs],
                "central": [abs(d03_fd_central(z0, h) - d) for h in hs]}

    def d03_fd_jacobian(f, x, h=1e-6, central=False):
        """Finite-difference Jacobian of a vector function f at x; column j perturbs x_j.
        Forward differences cost n + 1 evaluations of f, central differences 2 n."""
        x = np.asarray(x, dtype=float)
        f0 = np.asarray(f(x), dtype=float)
        cols = []
        for j in range(len(x)):
            e = np.zeros_like(x)
            e[j] = h
            if central:
                cols.append((np.asarray(f(x + e), dtype=float) - np.asarray(f(x - e), dtype=float)) / (2 * h))
            else:
                cols.append((np.asarray(f(x + e), dtype=float) - f0) / h)
        return np.array(cols).T

    def d03_fd_pendulum_AB(xbar, ubar=0.0, h=0.01, eps=1e-6):
        """A and B of the pendulum step by forward differences with step eps (3 perturbed steps plus
        the nominal one), next to the analytic Jacobians (slide finite-difference-jacobians)."""
        xu = np.r_[np.asarray(xbar, dtype=float), float(ubar)]
        JF = d03_fd_jacobian(lambda v: d03_pendulum_step(v[:2], v[2], h), xu, eps)
        an = d03_pendulum_AB(xbar, ubar, h)
        exact = np.hstack([np.array(an["A"]), np.array(an["B"])])
        return {"A_fd": JF[:, :2].tolist(), "B_fd": JF[:, 2:].tolist(), "A": an["A"], "B": an["B"],
                "max_err": float(np.max(np.abs(JF - exact)))}

    return d03_fd_central, d03_fd_error_curve, d03_fd_forward, d03_fd_jacobian, d03_fd_pendulum_AB, d03_fd_steps


@app.cell
def _(d03_fd_central, d03_fd_error_curve, d03_fd_forward, d03_fd_pendulum_AB, d03_quartic, mo, np):
    _hs = [0.1, 0.01, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    _rows = [(h, d03_fd_forward(1.0, h), d03_fd_central(1.0, h)) for h in _hs]
    _ec = d03_fd_error_curve(1.0)
    _i, _j = int(np.argmin(_ec["forward"])), int(np.argmin(_ec["central"]))
    _fp = d03_fd_pendulum_AB([0.5, 0.0], 0.0)
    _eps = float(np.finfo(float).eps)
    assert np.isclose(_rows[0][1], -2.759) and np.isclose(_rows[0][2], -2.96)
    assert 1e-9 < _ec["h"][_i] < 1e-7 and 1e-7 < _ec["h"][_j] < 1e-4 and _fp["max_err"] < 1e-7
    mo.md(f"""
| slide | quantity | value |
|---|---|---|
| forward-central-differences | J(0.9), J(1.1) | {d03_quartic(0.9):.4f}, {d03_quartic(1.1):.4f} |
| forward-central-differences | h, forward, central at z0 = 1 (J' = -3) | {"; ".join(f"{h:g}: {a:.10f}, {b:.10f}" for h, a, b in _rows)} |
| floating-point-roundoff | eps, sqrt(eps), eps^(1/3) | {_eps:.3e}, {np.sqrt(_eps):.3e}, {_eps ** (1 / 3):.3e} |
| floating-point-roundoff | (1 + 1e-12) - 1; J(1 + 1e-12) | {(1 + 1e-12) - 1:.10e}; {d03_quartic(1 + 1e-12)!r} |
| finite-difference-error-curve | best forward h, error; best central h, error | {_ec['h'][_i]:g}, {_ec['forward'][_i]:.2e}; {_ec['h'][_j]:g}, {_ec['central'][_j]:.2e} |
| finite-difference-jacobians | FD A at (0.5, 0), eps = 1e-6; largest entry error | {np.round(_fp['A_fd'], 8).tolist()}; {_fp['max_err']:.2e} |
""")
    return


@app.cell
def _(d03_fd_error_curve, plt):
    # Error of forward and central differences of J'(1) against the step size (slide finite-difference-error-curve).
    _ec = d03_fd_error_curve(1.0)
    _fig, _ax = plt.subplots(figsize=(4.2, 3.0))
    _ax.loglog(_ec["h"], [max(e, 1e-17) for e in _ec["forward"]], "o-", ms=2, lw=0.8, color="#cc6600", label="forward")
    _ax.loglog(_ec["h"], [max(e, 1e-17) for e in _ec["central"]], "o-", ms=2, lw=0.8, color="#008080", label="central")
    _ax.set_xlabel("step h", fontsize=8)
    _ax.set_ylabel("|FD - J'(1)|", fontsize=8)
    _ax.legend(fontsize=7)
    _ax.tick_params(labelsize=7)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Deck 03: tex errata

    Found while building deck 03 (tex lines 157-175, 224-236, 1303-1322 and the quartic examples it
    reuses). Values from `d03_stationary_points` and `d03_quartic`.

    | tex line | claim | correction |
    |---|---|---|
    | 257 | $J(-1.47) \approx 0.26$, $J(1.35) \approx 1.28$ | $J(-1.4730) = -1.4442$, $J(1.3470) = 1.3814$; the global minimum value is negative |
    | 270-271 | plot markers at $(-1.47, 0.26)$, $(1.35, 1.28)$, $(0.13, 3.93)$ | $(-1.473, -1.444)$, $(1.347, 1.381)$, $(0.126, 4.063)$ |
    | 334-335 | Armijo bound $6 + 10^{-4}(0.1)(-17^2) \approx 5.97$ | $6 - 0.00289 = 5.997$ (the accept decision is unchanged) |
    | 336 | $J(-1.4) \approx 0.60$ | $J(-1.4) = -1.3984$ |
    | 232-235 | steepest descent in the metric $\lVert\delta\rVert_H$ is $-H^{-1}\nabla J$, "exactly the Newton direction" | holds only for $H \succ 0$; for an indefinite $H$, $\lVert\delta\rVert_H$ is not a norm and the Newton step can point uphill (deck 05, $z_0 = 0$) |

    Lines 257-336 belong to deck 05's range; they are recorded here because deck 03 reuses the
    quartic, and left for deck 05 to correct in the tex.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Deck 04 · Probability, Gaussians, and sampling

    Slides: `slides/04_probability_and_sampling.html`
    """)
    return



@app.cell
def _(mo):
    mo.md(r"""
    ## Random numbers (slides `pseudo-random-generators`, `box-muller`, `box-muller-radius`)

    Every random number on the deck 04 slides comes from `Num.rng(seed)` in `slides/lib/num.js`:
    mulberry32 for uniforms and the Box-Muller transform for normals. `D04Rng` below is the same
    stream in numpy. mulberry32 is a counter passed through a mixing function: the $k$-th uniform
    of seed $s$ is $\mathrm{mix}(s + k \cdot \texttt{0x6D2B79F5} \bmod 2^{32}) / 2^{32}$, so a block
    of $n$ draws is one vectorized call. Uniforms match the JavaScript bit for bit; normals match to
    about $10^{-15}$. With it, every sample statistic that a slide's readout prints for a given seed
    is recomputed here for the same seed (`tools/twins/04.json` checks this).
    """)
    return


@app.cell
def _(np):
    import math as _math

    _M32 = np.uint64(0xFFFFFFFF)
    _C32 = np.uint64(0x6D2B79F5)

    class D04Rng:
        """The random stream of Num.rng(seed) in lib/num.js: mulberry32 uniforms, Box-Muller normals.

        uniforms(n) and normals(n) return the next n draws in the order that n calls of
        uniform() or normal() make in JavaScript, including the spare second normal of a
        Box-Muller pair.
        """

        def __init__(self, seed=1):
            self.seed = int(seed) & 0xFFFFFFFF
            self.k = 0              # uniforms drawn so far
            self.spare = None       # the unused sine half of the last Box-Muller pair

        def uniforms(self, n):
            k = np.arange(self.k + 1, self.k + n + 1, dtype=np.uint64)
            self.k += n
            a = (np.uint64(self.seed) + k * _C32) & _M32            # the counter
            t = ((a ^ (a >> np.uint64(15))) * (a | np.uint64(1))) & _M32
            t ^= (t + ((t ^ (t >> np.uint64(7))) * (t | np.uint64(61)))) & _M32
            return ((t ^ (t >> np.uint64(14))) & _M32).astype(float) / 4294967296.0

        def uniform(self, lo=0.0, hi=1.0):
            return lo + (hi - lo) * float(self.uniforms(1)[0])

        def normals(self, n):
            out = []
            if n > 0 and self.spare is not None:
                out.append(self.spare)
                self.spare = None
            m = n - len(out)
            if m > 0:
                pairs = (m + 1) // 2
                u = self.uniforms(2 * pairs)
                u1, u2 = 1.0 - u[0::2], u[1::2]          # u1 in (0, 1], so log(u1) is finite
                r = np.sqrt(-2.0 * np.log(u1))
                z = np.empty(2 * pairs)
                z[0::2] = r * np.cos(2 * np.pi * u2)
                z[1::2] = r * np.sin(2 * np.pi * u2)
                if m % 2:
                    self.spare = float(z[-1])
                    z = z[:-1]
                out.extend(z.tolist())
            return np.array(out, dtype=float)

    def d04_uniform_stream(seed, n):
        """The first n uniforms of Num.rng(seed) (slide pseudo-random-generators)."""
        return D04Rng(seed).uniforms(n)

    def d04_box_muller(u1, u2):
        """Box-Muller: uniforms u1 in (0, 1], u2 in [0, 1) -> two independent standard normals.

        r = sqrt(-2 ln u1) is the radius, theta = 2 pi u2 the angle (slide box-muller).
        """
        r = _math.sqrt(-2.0 * _math.log(u1))
        th = 2.0 * _math.pi * u2
        return {"r": r, "theta": th, "deg": _math.degrees(th),
                "z1": r * _math.cos(th), "z2": r * _math.sin(th)}

    def d04_radius_tail(r):
        """P(|z| > r) for a 2D standard normal z: exp(-r^2 / 2) (slide box-muller-radius)."""
        return _math.exp(-0.5 * r * r)

    def d04_radius_tail_samples(n, seed, rs=(1.0, 2.0, 3.0)):
        """Fraction of n draws (z1, z2) from Num.rng(seed) with radius above each r."""
        z = D04Rng(seed).normals(2 * n).reshape(n, 2)
        rad = np.hypot(z[:, 0], z[:, 1])
        return [float(np.mean(rad > r)) for r in rs]

    return D04Rng, d04_box_muller, d04_radius_tail, d04_radius_tail_samples, d04_uniform_stream


@app.cell
def _(mo):
    mo.md(r"""
    ## Discrete probability (slides `outcomes-events` to `law-of-large-numbers`)

    `d04_two_dice_counts` backs `two-dice` and `random-variables`; `d04_at_least_one` backs
    `independence`; `d04_pmf_moments` backs `mean`, `variance` and `variance-rules`;
    `d04_die_rolls` and `d04_sample_stats` back `histograms`; `d04_running_mean` backs
    `law-of-large-numbers`.
    """)
    return


@app.cell
def _(D04Rng, np):
    def d04_two_dice_counts():
        """How many of the 36 ordered outcomes (a, b) of two dice give each sum 2..12."""
        c = {s: 0 for s in range(2, 13)}
        for a in range(1, 7):
            for b in range(1, 7):
                c[a + b] += 1
        return c

    def d04_max_two_dice_counts():
        """How many of the 36 outcomes give each value 1..6 of max(a, b) (slide mean)."""
        c = {k: 0 for k in range(1, 7)}
        for a in range(1, 7):
            for b in range(1, 7):
                c[max(a, b)] += 1
        return c

    def d04_pmf_moments(values, probs):
        """Mean, variance, standard deviation and E[X^2] of a discrete distribution."""
        x, p = np.asarray(values, float), np.asarray(probs, float)
        m = float(p @ x)
        v = float(p @ (x - m) ** 2)
        return {"mean": m, "var": v, "std": float(np.sqrt(v)), "EX2": float(p @ x**2)}

    def d04_at_least_one(p, k):
        """P(at least one success in k independent trials of probability p) = 1 - (1 - p)^k."""
        return 1.0 - (1.0 - p) ** k

    def d04_die_rolls(n, seed, dice=1):
        """n rolls from Num.rng(seed), each the sum of `dice` dice; one die is 1 + floor(6 u)."""
        u = D04Rng(seed).uniforms(n * dice).reshape(n, dice)
        return (1 + np.floor(6 * u)).sum(axis=1).astype(int)

    def d04_sample_stats(x):
        """Sample mean, variance with 1/N, and variance with 1/(N - 1) (slide histograms)."""
        x = np.asarray(x, float)
        n, m = len(x), float(x.mean())
        ss = float(((x - m) ** 2).sum())
        return {"n": n, "mean": m, "var": ss / n, "var_unbiased": ss / (n - 1) if n > 1 else float("nan")}

    def d04_running_mean(x):
        """Mean of the first N values, for N = 1, 2, ... (slide law-of-large-numbers)."""
        x = np.asarray(x, float)
        return np.cumsum(x) / np.arange(1, len(x) + 1)

    return (d04_at_least_one, d04_die_rolls, d04_max_two_dice_counts, d04_pmf_moments,
            d04_running_mean, d04_sample_stats, d04_two_dice_counts)


@app.cell
def _(d04_at_least_one, d04_die_rolls, d04_max_two_dice_counts, d04_pmf_moments, d04_running_mean, d04_sample_stats, d04_two_dice_counts, mo, np):
    _c = d04_two_dice_counts()
    _cm = d04_max_two_dice_counts()
    _die = d04_pmf_moments(range(1, 7), [1 / 6] * 6)
    _sum = d04_pmf_moments(list(_c), [v / 36 for v in _c.values()])
    _mx = d04_pmf_moments(list(_cm), [v / 36 for v in _cm.values()])
    assert _c[7] == 6 and sum(_c.values()) == 36
    assert np.isclose(_die["var"], 35 / 12) and np.isclose(_sum["var"], 35 / 6) and np.isclose(_mx["mean"], 161 / 36)
    _hist = {n: d04_sample_stats(d04_die_rolls(n, 1, dice=2)) for n in (100, 1000, 10000, 100000)}
    _lln = {s: d04_running_mean(d04_die_rolls(10000, s))[[9, 99, 999, 9999]] for s in range(1, 6)}
    _band = 2 * _die["std"] / np.sqrt(1000)
    _inside = np.mean([abs(d04_die_rolls(1000, s).mean() - 3.5) <= _band for s in range(1, 201)])
    _hrows = "\n".join(f"| histograms | N = {n} two-dice sums, seed 1: mean, var (1/N) | {h['mean']:.4f}, {h['var']:.4f} |" for n, h in _hist.items())
    _lrows = "\n".join(f"| law-of-large-numbers | seed {s}: running mean at N = 10, 100, 1000, 10000 | {', '.join(f'{v:.4f}' for v in r)} |" for s, r in _lln.items())
    mo.md(f"""
| slide | quantity | value |
|---|---|---|
| two-dice | outcomes with sum 2..12 | {list(_c.values())} |
| independence | 1 - (5/6)^k for k = 1, 2, 3, 4, 12 | {', '.join(f'{d04_at_least_one(1 / 6, k):.4f}' for k in (1, 2, 3, 4, 12))} |
| mean | one die; sum of two; max of two | {_die['mean']:.4f}; {_sum['mean']:.4f}; {_mx['mean']:.4f} (= 161/36) |
| variance | one die: E[X^2], var, sd | {_die['EX2']:.4f}, {_die['var']:.4f}, {_die['std']:.4f} |
| variance-rules | sum of two dice: var, sd | {_sum['var']:.4f}, {_sum['std']:.4f} |
| mean (select) | max of two dice: var, sd | {_mx['var']:.4f}, {_mx['std']:.4f} |
{_hrows}
{_lrows}
| law-of-large-numbers | band 2 sigma / sqrt(1000); fraction of 200 seeds inside | {_band:.4f}; {_inside:.3f} |
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Continuous distributions (slides `uniform-density` to `shift-and-scale`)

    `d04_uniform_moments` backs `uniform-density`; `d04_gauss_pdf` backs `gaussian-pdf`;
    `d04_prob_within` and `d04_normal_cdf` back `rule-68-95-997`; `d04_sample_stats` on
    $m + \sigma\epsilon$ backs `shift-and-scale`.
    """)
    return


@app.cell
def _(np):
    import math as _math

    def d04_uniform_moments(a, b):
        """Height 1/(b - a), mean (a + b)/2, variance (b - a)^2/12 and sd of U[a, b]."""
        return {"height": 1.0 / (b - a), "mean": (a + b) / 2, "var": (b - a) ** 2 / 12,
                "std": (b - a) / _math.sqrt(12)}

    def d04_gauss_pdf(x, m=0.0, s=1.0):
        """Gaussian density exp(-(x - m)^2 / (2 s^2)) / (s sqrt(2 pi)) (slide gaussian-pdf)."""
        x = np.asarray(x, float)
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * _math.sqrt(2 * _math.pi))

    def d04_normal_cdf(x):
        """P(Z <= x) for a standard normal Z, via erfc so that both tails keep their digits."""
        return 0.5 * _math.erfc(-x / _math.sqrt(2.0))

    def d04_prob_within(k):
        """P(|X - m| <= k sigma) for any Gaussian = erf(k / sqrt 2) (slide rule-68-95-997)."""
        return _math.erf(k / _math.sqrt(2.0))

    return d04_gauss_pdf, d04_normal_cdf, d04_prob_within, d04_uniform_moments


@app.cell
def _(D04Rng, d04_box_muller, d04_gauss_pdf, d04_normal_cdf, d04_prob_within, d04_radius_tail, d04_radius_tail_samples, d04_sample_stats, d04_uniform_moments, d04_uniform_stream, mo, np):
    _u1 = d04_uniform_moments(0, 1)
    _bm = d04_box_muller(0.3, 0.6)
    _eps = D04Rng(11).normals(2000)
    _x = d04_sample_stats(3 + 0.5 * _eps)
    _grid = {n: (n + 1) * (2 * n + 1) / (6 * n * n) for n in (10, 100, 1000)}
    assert np.isclose(_bm["deg"], 216) and np.isclose(d04_prob_within(2), 0.9544997, atol=1e-7)
    mo.md(f"""
| slide | quantity | value |
|---|---|---|
| uniform-density | U[0, 1]: mean, var, sd | {_u1['mean']}, {_u1['var']:.4f}, {_u1['std']:.4f} |
| uniform-density (notes) | E[X^2] on the grid k/n, n = 10, 100, 1000 | {', '.join(f'{v:.5f}' for v in _grid.values())} |
| gaussian-pdf | peak at sigma = 1, 0.5, 2 | {', '.join(f'{float(d04_gauss_pdf(0, 0, s)):.4f}' for s in (1, 0.5, 2))} |
| rule-68-95-997 | P(within k sigma), k = 1, 2, 3 | {', '.join(f'{d04_prob_within(k):.4f}' for k in (1, 2, 3))} |
| rule-68-95-997 | one tail beyond 2 sigma; beyond 3 sigma | {1 - d04_normal_cdf(2):.5f}; {1 - d04_normal_cdf(3):.6f} |
| rule-68-95-997 | gripper error sd 2 mm: P(abs e > 4 mm); P(within 1.96 sigma) | {1 - d04_prob_within(2):.4f}; {d04_prob_within(1.96):.4f} |
| shift-and-scale | 2000 draws of 3 + 0.5 eps (seed 11): mean, sd | {_x['mean']:.4f}, {np.sqrt(_x['var']):.4f} |
| pseudo-random-generators | first 6 uniforms of seed 1 | {', '.join(f'{v:.5f}' for v in d04_uniform_stream(1, 6))} |
| pseudo-random-generators | first 6 uniforms of seed 2 | {', '.join(f'{v:.5f}' for v in d04_uniform_stream(2, 6))} |
| box-muller | u = (0.3, 0.6): r, theta (deg), z1, z2 | {_bm['r']:.4f}, {_bm['deg']:.1f}, {_bm['z1']:.4f}, {_bm['z2']:.4f} |
| box-muller-radius | exp(-r^2/2) at r = 1, 2, 3 | {', '.join(f'{d04_radius_tail(r):.4f}' for r in (1, 2, 3))} |
| box-muller-radius | fraction of 2000 draws (seed 5) beyond r = 1, 2, 3 | {d04_radius_tail_samples(2000, 5)} |
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Random vectors and Monte Carlo (slides `covariance` to `randomized-smoothing`)

    `d04_sample_cov` backs `covariance` and `cholesky-live`; `d04_directional_variance` backs
    `covariance-matrix`; `d04_mvn_pdf` backs `multivariate-gaussian`; `d04_cov_ellipse`,
    `d04_ellipse_prob` and `d04_ellipse_fractions` back `covariance-ellipse`; `d04_chol2` and
    `d04_mvn_samples` back `cholesky-sampling` and `cholesky-live`; `d04_mc_pi` and
    `d04_mc_pi_rms` back `monte-carlo` and `monte-carlo-error`; `d04_smoothed_step` and
    `d04_smoothed_step_mc` back `randomized-smoothing`. The running covariance is
    $\Sigma = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}$, the matrix of deck 02's Cholesky slide.
    """)
    return


@app.cell
def _(D04Rng, np):
    import math as _math

    def d04_sample_cov(points, unbiased=False):
        """Sample mean, covariance (1/N, or 1/(N - 1) when unbiased) and correlation of 2D points."""
        P = np.asarray(points, float)
        m = P.mean(axis=0)
        D = P - m
        S = D.T @ D / (len(P) - 1 if unbiased else len(P))
        return {"mean": m.tolist(), "cov": S.tolist(), "corr": float(S[0, 1] / _math.sqrt(S[0, 0] * S[1, 1]))}

    def d04_directional_variance(Sigma, theta_deg):
        """a^T Sigma a for the unit vector a at angle theta: the variance of a^T x."""
        a = np.array([_math.cos(_math.radians(theta_deg)), _math.sin(_math.radians(theta_deg))])
        return float(a @ np.asarray(Sigma, float) @ a)

    def d04_projected_variance(Sigma, theta_deg, n, seed):
        """a^T Sigma a and the sample variance (1/N) of a^T x_i over n draws of N(0, Sigma)."""
        X = d04_mvn_samples([0.0, 0.0], Sigma, n, seed)
        a = np.array([_math.cos(_math.radians(theta_deg)), _math.sin(_math.radians(theta_deg))])
        return {"theory": d04_directional_variance(Sigma, theta_deg), "sample": float(np.var(X @ a))}

    def d04_cov_ellipse(Sigma):
        """Principal variances s1^2 >= s2^2, their square roots, and the angle (deg) of the major axis.

        Sigma = V diag(s1^2, s2^2) V^T; the first eigenvector is signed so that its largest entry
        is positive, and the angle is reported in (-90, 90].
        """
        w, V = np.linalg.eigh(np.asarray(Sigma, float))
        w, V = w[::-1], V[:, ::-1]
        v = V[:, 0] if V[np.argmax(np.abs(V[:, 0])), 0] > 0 else -V[:, 0]
        ang = _math.degrees(_math.atan2(v[1], v[0]))
        ang = ang + 180 if ang <= -90 else ang - 180 if ang > 90 else ang
        return {"var": w.tolist(), "std": np.sqrt(np.maximum(w, 0)).tolist(), "angle": ang, "v1": v.tolist()}

    def d04_cov_from_axes(s1, s2, phi_deg):
        """Sigma = V diag(s1^2, s2^2) V^T with V the rotation by phi (the covariance-ellipse sliders)."""
        c, s = _math.cos(_math.radians(phi_deg)), _math.sin(_math.radians(phi_deg))
        V = np.array([[c, -s], [s, c]])
        return (V @ np.diag([s1 * s1, s2 * s2]) @ V.T).tolist()

    def d04_ellipse_prob(k):
        """P(a 2D Gaussian draw lies inside its k-sigma ellipse) = 1 - exp(-k^2 / 2)."""
        return 1.0 - _math.exp(-0.5 * k * k)

    def d04_ellipse_fractions(s1, s2, phi_deg, n, seed, ks=(1, 2, 3)):
        """Fractions of n draws x = V diag(s1, s2) eps inside the k-sigma ellipses of their Sigma.

        x^T Sigma^{-1} x = |eps|^2 for every draw, so the fractions do not depend on s1, s2, phi.
        """
        c, s = _math.cos(_math.radians(phi_deg)), _math.sin(_math.radians(phi_deg))
        A = np.array([[c, -s], [s, c]]) @ np.diag([s1, s2])
        E = D04Rng(seed).normals(2 * n).reshape(n, 2)
        X = E @ A.T
        S = np.asarray(d04_cov_from_axes(s1, s2, phi_deg))
        q = np.einsum("ij,jk,ik->i", X, np.linalg.inv(S), X)
        return [float(np.mean(q <= k * k)) for k in ks]

    def d04_chol2(Sigma):
        """Cholesky factor L = [[l11, 0], [l21, l22]] of a 2x2 covariance, or None when not positive definite."""
        S = np.asarray(Sigma, float)
        if S[0, 0] <= 0:
            return None
        l11 = _math.sqrt(S[0, 0])
        l21 = S[1, 0] / l11
        r = S[1, 1] - l21 * l21
        if r <= 0:
            return None
        return [[l11, 0.0], [l21, _math.sqrt(r)]]

    def d04_mvn_samples(m, Sigma, n, seed):
        """n draws x = m + L eps (eps two standard normals from Num.rng(seed), L L^T = Sigma)."""
        L = np.array(d04_chol2(Sigma))
        E = D04Rng(seed).normals(2 * n).reshape(n, 2)
        return np.asarray(m, float) + E @ L.T

    def d04_mvn_pdf(x, m, Sigma):
        """Density exp(-(x - m)^T Sigma^{-1} (x - m) / 2) / sqrt((2 pi)^n det Sigma)."""
        x, m, S = np.asarray(x, float), np.asarray(m, float), np.asarray(Sigma, float)
        d = x - m
        return float(_math.exp(-0.5 * d @ np.linalg.solve(S, d)) / _math.sqrt((2 * _math.pi) ** len(m) * np.linalg.det(S)))

    return (d04_chol2, d04_cov_ellipse, d04_cov_from_axes, d04_directional_variance, d04_ellipse_fractions,
            d04_ellipse_prob, d04_mvn_pdf, d04_mvn_samples, d04_projected_variance, d04_sample_cov)


@app.cell
def _(D04Rng, d04_normal_cdf, np):
    import math as _math

    def d04_mc_pi(n, seed):
        """pi from n uniform points in the unit square: 4 x (fraction with u1^2 + u2^2 <= 1)."""
        u = D04Rng(seed).uniforms(2 * n).reshape(n, 2)
        inside = int(((u**2).sum(axis=1) <= 1.0).sum())
        return {"n": n, "inside": inside, "estimate": 4.0 * inside / n}

    def d04_mc_pi_rms(ns, trials=200, seed0=1):
        """Root-mean-square error of the pi estimate over the seeds seed0 .. seed0 + trials - 1."""
        out = []
        for n in ns:
            e = [d04_mc_pi(n, seed0 + t)["estimate"] - _math.pi for t in range(trials)]
            out.append(float(_math.sqrt(np.mean(np.square(e)))))
        return out

    def d04_smoothed_step(z, sigma):
        """E[J(z + sigma eps)] for the step J(y) = 1 if y < 0 else 0, which is Phi(-z / sigma)."""
        return d04_normal_cdf(-z / sigma)

    def d04_smoothed_step_mc(zs, sigma, n, seed):
        """Monte Carlo estimate of the smoothed step at each z, with the same n normals at every z."""
        e = D04Rng(seed).normals(n)
        return [float(np.mean(z + sigma * e < 0)) for z in zs]

    return d04_mc_pi, d04_mc_pi_rms, d04_smoothed_step, d04_smoothed_step_mc


@app.cell
def _(d04_chol2, d04_cov_ellipse, d04_directional_variance, d04_ellipse_fractions, d04_ellipse_prob, d04_mc_pi, d04_mc_pi_rms, d04_mvn_pdf, d04_mvn_samples, d04_projected_variance, d04_sample_cov, d04_smoothed_step, d04_smoothed_step_mc, mo, np):
    _S = [[4.0, 2.0], [2.0, 3.0]]
    _pts = [(1, 2), (2, 3), (3, 5), (4, 4), (5, 6)]
    _c5 = d04_sample_cov(_pts)
    _el = d04_cov_ellipse(_S)
    _L = np.array(d04_chol2(_S))
    _x = np.array([1.0, 2.0]) + _L @ np.array([0.5, -1.0])
    _s500 = d04_sample_cov(d04_mvn_samples([1, 2], _S, 500, 3))
    _pv = d04_projected_variance(_S, 45, 1000, 3)
    _ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    _rms = d04_mc_pi_rms(_ns)
    _sf = 4 * np.sqrt(np.pi / 4 * (1 - np.pi / 4))
    assert np.allclose(_c5["cov"], [[2, 1.8], [1.8, 2]]) and np.isclose(_c5["corr"], 0.9)
    assert np.allclose(_L, [[2, 0], [1, np.sqrt(2)]]) and np.allclose(_x, [2, 2.5 - np.sqrt(2)])
    _f = lambda v: "(" + ", ".join(f"{t:.4f}" for t in np.ravel(v)) + ")"
    mo.md(f"""
| slide | quantity | value |
|---|---|---|
| covariance | 5 points: means, covariance (1/N), correlation | {_f(_c5['mean'])}, {_f(_c5['cov'])}, {_c5['corr']:.4f} |
| covariance-matrix | a^T Sigma a at 0, 90, 45, -45 deg | {', '.join(f'{d04_directional_variance(_S, t):.4f}' for t in (0, 90, 45, -45))} |
| covariance-matrix | 1000 draws (seed 3) projected at 45 deg: sample variance | {_pv['sample']:.4f} |
| multivariate-gaussian | det, inverse, peak density | {np.linalg.det(_S):.4f}, {_f(np.linalg.inv(_S))}, {d04_mvn_pdf([0, 0], [0, 0], _S):.5f} |
| covariance-ellipse | s1^2, s2^2; s1, s2; major-axis angle | {_f(_el['var'])}; {_f(_el['std'])}; {_el['angle']:.2f} deg |
| covariance-ellipse | 1 - exp(-k^2/2), k = 1, 2, 3 | {', '.join(f'{d04_ellipse_prob(k):.4f}' for k in (1, 2, 3))} |
| covariance-ellipse | 1000 draws (seed 7) inside k = 1, 2, 3 | {d04_ellipse_fractions(2.36, 1.2, 38, 1000, 7)} |
| cholesky-sampling | L; m + L eps for m = (1, 2), eps = (0.5, -1) | {_f(_L)}; {_f(_x)} |
| cholesky-live | 500 draws (seed 3): sample mean, covariance | {_f(_s500['mean'])}, {_f(_s500['cov'])} |
| monte-carlo | seed 1, N = 100, 1000, 10^4, 10^5 | {', '.join(str(d04_mc_pi(n, 1)['estimate']) for n in (100, 1000, 10000, 100000))} |
| monte-carlo-error | sigma_f = 4 sqrt(p (1 - p)) | {_sf:.4f} |
| monte-carlo-error | RMS error over 200 seeds, N = {_ns} | {', '.join(f'{v:.4f}' for v in _rms)} |
| monte-carlo-error | predicted sigma_f / sqrt(N) | {', '.join(f'{_sf / np.sqrt(n):.4f}' for n in _ns)} |
| randomized-smoothing | exact at z = -0.5, 0, 0.5 (sigma 0.3) | {', '.join(f'{d04_smoothed_step(z, 0.3):.4f}' for z in (-0.5, 0, 0.5))} |
| randomized-smoothing | N = 200 (seed 1) at the same z | {d04_smoothed_step_mc([-0.5, 0, 0.5], 0.3, 200, 1)} |
| randomized-smoothing | slope at 0 for sigma = 0.1, 0.3, 1 | {', '.join(f'{-1 / (s * np.sqrt(2 * np.pi)):.4f}' for s in (0.1, 0.3, 1.0))} |
""")
    return


@app.cell
def _(d04_mc_pi_rms, np, plt):
    # Monte Carlo error against N on log-log axes (slide monte-carlo-error).
    _ns = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
    _fig, _ax = plt.subplots(figsize=(4.2, 3.0))
    _ax.loglog(_ns, d04_mc_pi_rms(list(_ns)), "o", color="#cc0000", label="RMS error, 200 seeds")
    _ax.loglog(_ns, 4 * np.sqrt(np.pi / 4 * (1 - np.pi / 4)) / np.sqrt(_ns), color="#0000cc", label="1.642 / sqrt(N)")
    _ax.set_xlabel("N"); _ax.set_ylabel("error of the pi estimate")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Weighted samples and sampling-based search (slides `weighted-average` to `random-directions`)

    `d04_weighted_mean` backs `weighted-average`; `d04_softmax_weights` and `d04_ess` back
    `softmax-weights`, `temperature-limits` and `effective-sample-size`, with
    `d04_lambda_for_ess`; `d04_tail_estimates` and `d04_tail_relative_std` back
    `importance-sampling`; `d04_elite_fit` backs `elite-fit`; `d04_cem_run` backs
    `elite-refit-live`; `d04_quartic_stationary` and `d04_restart_success` back
    `random-restarts`; `d04_hou2021_test_count` backs `random-test-problems`; `d04_directions`
    and `d04_diagonal_fraction` back `random-directions`. The five costs $J = (10, 3, 7, 2, 8)$
    are the tex's MPPI example (lines 2273-2303) and `03_sampling_mpc.py`'s `J_demo`.
    """)
    return


@app.cell
def _(D04Rng, d04_normal_cdf, np):
    import math as _math

    def d04_weighted_mean(x, w):
        """sum w_i x_i / sum w_i (slide weighted-average)."""
        x, w = np.asarray(x, float), np.asarray(w, float)
        return float(w @ x / w.sum())

    def d04_softmax_weights(J, lam):
        """w_i = exp(-(J_i - min J) / lam) / sum_j exp(-(J_j - min J) / lam).

        Subtracting min J multiplies every exp(-J_i / lam) by the same factor, so the weights do
        not change, and the largest term is exp(0) = 1, so the sum cannot underflow to 0.
        """
        J = np.asarray(J, float)
        e = np.exp(-(J - J.min()) / lam)
        return e / e.sum()

    def d04_ess(w):
        """Effective sample size 1 / sum w_i^2 of normalized weights."""
        w = np.asarray(w, float)
        return float(1.0 / np.sum(w * w))

    def d04_lambda_for_ess(J, target, lo=1e-3, hi=1e3, iters=100):
        """Temperature at which N_eff = target, by bisection on log(lambda); N_eff grows with lambda."""
        a, b = _math.log(lo), _math.log(hi)
        for _ in range(iters):
            c = 0.5 * (a + b)
            if d04_ess(d04_softmax_weights(J, _math.exp(c))) < target:
                a = c
            else:
                b = c
        return _math.exp(0.5 * (a + b))

    def d04_tail_estimates(n, seed, shift=3.0, t=3.0):
        """P(X > t), X ~ N(0, 1): plain Monte Carlo, and importance sampling from q = N(shift, 1).

        Both use Num.rng(seed): the first n normals are the plain draws, the next n (plus shift)
        the draws from q. The importance weight is p(y)/q(y) = exp(-y^2/2 + (y - shift)^2/2).
        """
        e = D04Rng(seed).normals(2 * n)
        x, y = e[:n], shift + e[n:]
        w = np.exp(-0.5 * y**2 + 0.5 * (y - shift) ** 2)
        return {"plain": float(np.mean(x > t)), "hits_plain": int((x > t).sum()),
                "is": float(np.mean((y > t) * w)), "hits_is": int((y > t).sum()),
                "exact": 1.0 - d04_normal_cdf(t)}

    def d04_tail_relative_std(n, shift=3.0, t=3.0):
        """Standard deviation / p of both estimators of p = P(X > t), from their exact variances.

        Plain: p (1 - p) / n. Importance: (E_q[1(y > t) w^2] - p^2) / n with
        E_q[1(y > t) w^2] = exp(shift^2) P(Z > t + shift).
        """
        p = 1.0 - d04_normal_cdf(t)
        m2 = _math.exp(shift * shift) * (1.0 - d04_normal_cdf(t + shift))
        plain = _math.sqrt(p * (1 - p) / n) / p
        imp = _math.sqrt((m2 - p * p) / n) / p
        return {"p": p, "plain": plain, "is": imp, "ratio_n": (plain / imp) ** 2}

    def d04_elite_fit(z, J, frac):
        """Mean and covariance (1/K) of the K = floor(frac N) samples with the lowest costs."""
        z = np.asarray(z, float)
        z = z[:, None] if z.ndim == 1 else z
        K = int(_math.floor(frac * len(J) + 1e-9))
        idx = np.argsort(np.asarray(J, float), kind="stable")[:K]
        E = z[idx]
        m = E.mean(axis=0)
        D = E - m
        return {"K": K, "idx": idx.tolist(), "mean": m.tolist(), "cov": (D.T @ D / K).tolist()}

    return (d04_elite_fit, d04_ess, d04_lambda_for_ess, d04_softmax_weights, d04_tail_estimates,
            d04_tail_relative_std, d04_weighted_mean)


@app.cell
def _(D04Rng, d04_elite_fit, np):
    import math as _math

    _A, _B, _PHI = np.array([1.4, 0.9]), np.array([-1.4, -0.9]), _math.radians(35.0)

    def d04_cem_cost(z):
        """Two-basin cost of slide elite-refit-live: min(q_A, 1 + 0.3 |z - b|^2).

        q_A = (u / 1.2)^2 + (v / 0.3)^2 in coordinates (u, v) along and across a valley through
        a = (1.4, 0.9) at 35 degrees; J(a) = 0. The wide bowl around b = (-1.4, -0.9) has J(b) = 1.
        """
        Z = np.atleast_2d(np.asarray(z, float))
        d = Z - _A
        u = _math.cos(_PHI) * d[:, 0] + _math.sin(_PHI) * d[:, 1]
        v = -_math.sin(_PHI) * d[:, 0] + _math.cos(_PHI) * d[:, 1]
        return np.minimum((u / 1.2) ** 2 + (v / 0.3) ** 2, 1.0 + 0.3 * ((Z - _B) ** 2).sum(axis=1))

    def d04_cem_run(m0=(0.0, 2.0), s0=1.5, frac=0.2, n=40, iters=10, seed=1):
        """Repeated elite refitting: sample N(m, Sigma), keep the best floor(frac n), refit m and Sigma.

        Each iteration draws n points x = m + L eps (L = chol(Sigma + 1e-12 I), eps from
        Num.rng(seed)), then sets m, Sigma to the elite mean and covariance (1/K). Returns one
        record per iteration with the distribution it sampled from and the best cost it saw.
        """
        R = D04Rng(seed)
        m, S = np.array(m0, float), s0 * s0 * np.eye(2)
        hist = []
        for _ in range(iters):
            L = np.linalg.cholesky(S + 1e-12 * np.eye(2))
            Z = m + R.normals(2 * n).reshape(n, 2) @ L.T
            J = d04_cem_cost(Z)
            fit = d04_elite_fit(Z, J, frac)
            hist.append({"mean": m.tolist(), "cov": S.tolist(), "best": float(J.min()), "elites": fit["idx"]})
            m, S = np.array(fit["mean"]), np.array(fit["cov"])
        hist.append({"mean": m.tolist(), "cov": S.tolist(), "best": float(d04_cem_cost(m)[0]), "elites": []})
        return hist

    def d04_quartic(z):
        """The running quartic J(z) = z^4 - 4 z^2 + z + 4 of decks 05 and 04."""
        return z**4 - 4 * z**2 + z + 4

    def d04_quartic_stationary():
        """Roots of J'(z) = 4 z^3 - 8 z + 1 (ascending), with J and J'' = 12 z^2 - 8 there."""
        r = np.sort(np.roots([4.0, 0.0, -8.0, 1.0]).real)
        return {"z": r.tolist(), "J": [float(d04_quartic(t)) for t in r], "J2": [float(12 * t * t - 8) for t in r]}

    def d04_restart_success(k, lo=-2.0, hi=2.0, trials=1000, seed=1):
        """k random starts z0 ~ U[lo, hi]; rolling downhill from z0 reaches the global minimum
        exactly when z0 lies left of the local maximum. Returns p for one start, 1 - (1 - p)^k, and
        the fraction of `trials` runs (uniforms from Num.rng(seed)) with a start in that basin.
        """
        zmax = d04_quartic_stationary()["z"][1]
        p = (zmax - lo) / (hi - lo)
        z0 = lo + (hi - lo) * D04Rng(seed).uniforms(trials * k).reshape(trials, k)
        return {"zmax": zmax, "p": p, "theory": 1 - (1 - p) ** k, "empirical": float(np.mean((z0 < zmax).any(axis=1)))}

    def d04_hou2021_test_count():
        """Contact settings of Hou & Mason 2021, Table I, and the number of random test problems.

        A row is (environment contacts, contact-mode choices, contacts-per-finger choices, finger
        choices); each setting gets 1000 random sets of contact locations and normals.
        """
        planar = [(1, 2, 2, 1), (2, 1, 2, 1)]
        spatial = [(1, 2, 3, 3), (2, 3, 3, 3), (3, 3, 3, 3)]
        n_pl = sum(m * c * f for _, m, c, f in planar)
        n_3d = sum(m * c * f for _, m, c, f in spatial)
        return {"planar": n_pl, "spatial": n_3d, "problems": 1000 * (n_pl + n_3d)}

    def d04_directions(n, seed, method):
        """Angles of n random unit vectors: 'gauss' normalizes N(0, I) draws, 'square' U[-1, 1]^2 draws."""
        R = D04Rng(seed)
        P = R.normals(2 * n).reshape(n, 2) if method == "gauss" else (2 * R.uniforms(2 * n) - 1).reshape(n, 2)
        return np.arctan2(P[:, 1], P[:, 0])

    def d04_diagonal_fraction(ang, half_deg=15.0):
        """Fraction of angles within half_deg of a diagonal (45, 135, 225 or 315 degrees)."""
        a = np.degrees(np.asarray(ang, float)) % 90.0
        return float(np.mean(np.abs(a - 45.0) <= half_deg))

    return (d04_cem_cost, d04_cem_run, d04_diagonal_fraction, d04_directions, d04_hou2021_test_count,
            d04_quartic, d04_quartic_stationary, d04_restart_success)


@app.cell
def _(d04_cem_run, d04_diagonal_fraction, d04_directions, d04_elite_fit, d04_ess, d04_hou2021_test_count, d04_lambda_for_ess, d04_quartic, d04_quartic_stationary, d04_restart_success, d04_softmax_weights, d04_tail_estimates, d04_tail_relative_std, d04_weighted_mean, mo, np):
    _J = [10, 3, 7, 2, 8]
    _w1, _w5 = d04_softmax_weights(_J, 1.0), d04_softmax_weights(_J, 5.0)
    _el = d04_elite_fit([0.9, 1.1, 0.7, 1.4, -0.3, 0.1, 2.2, -1.0, 0.4, -1.7], list(range(10)), 0.4)
    _qs = d04_quartic_stationary()
    _ta = d04_tail_estimates(1000, 1)
    _tr = d04_tail_relative_std(1000)
    _h = d04_hou2021_test_count()
    _cg = d04_cem_run(s0=1.5)[-1]
    _cs = d04_cem_run(s0=0.3)[-1]
    _find = lambda s0: sum(np.linalg.norm(np.array(d04_cem_run(s0=s0, iters=15, seed=s)[-1]["mean"]) - [1.4, 0.9]) < 0.3 for s in range(1, 41))
    assert np.isclose(_el["mean"][0], 1.025) and np.isclose(_el["cov"][0][0], 0.066875) and _h["problems"] == 78000
    _f = lambda v: "(" + ", ".join(f"{t:.4f}" for t in np.ravel(v)) + ")"
    mo.md(f"""
| slide | quantity | value |
|---|---|---|
| weighted-average | x = (1, 4, 10), w = (0.5, 0.3, 0.2): mean | {d04_weighted_mean([1, 4, 10], [0.5, 0.3, 0.2]):.4f} |
| softmax-weights | lambda = 1 | {_f(_w1)} |
| softmax-weights | lambda = 5 | {_f(_w5)} |
| softmax-weights | J = (1000, 1001), lambda = 1: exp(-1000); shifted weights | {np.exp(-1000.0)}; {_f(d04_softmax_weights([1000, 1001], 1.0))} |
| temperature-limits | lambda = 0.1, 100 | {_f(d04_softmax_weights(_J, 0.1))}, {_f(d04_softmax_weights(_J, 100.0))} |
| effective-sample-size | N_eff at lambda = 1, 5, 0.1, 100 | {', '.join(f'{d04_ess(d04_softmax_weights(_J, l)):.4f}' for l in (1, 5, 0.1, 100))} |
| effective-sample-size | lambda with N_eff = 1.5, 2.5, 4 | {', '.join(f'{d04_lambda_for_ess(_J, t):.4f}' for t in (1.5, 2.5, 4.0))} |
| importance-sampling | exact P(X > 3); N = 1000, seed 1: plain (hits), IS (hits) | {_ta['exact']:.7f}; {_ta['plain']} ({_ta['hits_plain']}), {_ta['is']:.7f} ({_ta['hits_is']}) |
| importance-sampling | relative sd at N = 1000: plain, IS; sample-count ratio | {_tr['plain']:.4f}, {_tr['is']:.4f}; {_tr['ratio_n']:.1f} |
| elite-fit | tex example: K, mean, variance (1/K), sd | {_el['K']}, {_el['mean'][0]:.4f}, {_el['cov'][0][0]:.6f}, {np.sqrt(_el['cov'][0][0]):.4f} |
| elite-refit-live | sigma0 = 1.5, seed 1, after 10 iterations: mean, best | {_f(_cg['mean'])}, {_cg['best']:.4f} |
| elite-refit-live | sigma0 = 0.3, seed 1, after 10 iterations: mean, J(mean) | {_f(_cs['mean'])}, {_cs['best']:.4f} |
| elite-refit-live | seeds 1-40 reaching a within 0.3 after 15 iterations, sigma0 = 0.3, 1.5 | {_find(0.3)}, {_find(1.5)} |
| random-restarts | stationary points, J there, J'' there | {_f(_qs['z'])}, {_f(_qs['J'])}, {_f(_qs['J2'])} |
| random-restarts | p; 1 - (1 - p)^k for k = 3, 10 | {d04_restart_success(1)['p']:.4f}; {d04_restart_success(3)['theory']:.4f}, {d04_restart_success(10)['theory']:.5f} |
| random-restarts | empirical over 1000 trials (seed 1), k = 1, 3, 10 | {', '.join(str(d04_restart_success(k)['empirical']) for k in (1, 3, 10))} |
| random-test-problems | planar settings, 3D settings, problems | {_h['planar']}, {_h['spatial']}, {_h['problems']} |
| random-directions | 20000 draws (seed 1) within 15 deg of a diagonal: gauss, square; theory for square | {d04_diagonal_fraction(d04_directions(20000, 1, 'gauss')):.4f}, {d04_diagonal_fraction(d04_directions(20000, 1, 'square')):.4f}; {1 - np.tan(np.radians(30)):.4f} |
| (tex line 257) | J(-1.47), J(1.35) | {d04_quartic(-1.47):.4f}, {d04_quartic(1.35):.4f} |
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Tex errata found while building deck 04

    * Line 257: "$J(-1.47)\approx 0.26$, $J(1.35)\approx 1.28$". The quartic gives
      $J(-1.4730) = -1.4442$ and $J(1.3470) = 1.3814$ (`d04_quartic_stationary`). The conclusion
      (global minimum at $z \approx -1.47$) stands.
    * Line 2286: the normalized weight of sample 2 at $\lambda = 1$ is $0.2671$ (0.267), not 0.268
      (`d04_softmax_weights([10, 3, 7, 2, 8], 1)`).
    * Lines 2267-2270: "Predictive Sampling can be seen as MPPI with infinite temperature". As
      $\lambda \to \infty$ the weights tend to $1/N$ (slide `temperature-limits`), which averages
      all samples; predictive sampling keeps only the best one, the $\lambda \to 0$ limit stated
      in the same box.
    """)
    return


if __name__ == "__main__":
    app.run()
