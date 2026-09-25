"""Python references for the lib/num.js twin cases in tools/twins/num.json.

num.json loads this module through its "notebook" field, so every name below is visible to
the "py" expressions. The inputs come from num_data.js, the same file the JavaScript side
loads, so both sides see identical numbers.

The references are numpy / scipy where a library routine exists, and independent exact
computations where it does not: row reduction in Fractions, a QP solved by enumerating
active sets, and a port of the mulberry32 generator.
"""

import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.optimize as so

_src = (Path(__file__).with_name("num_data.js")).read_text()
D = json.loads(_src[_src.index("{"): _src.rindex("}") + 1])


def M(key):
    """Input `key` of num_data.js as a float array."""
    return np.array(D[key], dtype=float)


# ---- invariants ------------------------------------------------------------------------

def sv(A):
    return np.linalg.svd(np.asarray(A, float), compute_uv=False)


def proj_cols(Q):
    """Projector onto the span of the columns of Q (orthonormal columns)."""
    Q = np.asarray(Q, float)
    return Q @ Q.T


def null_proj(A):
    return proj_cols(sla.null_space(np.asarray(A, float)))


def qr_pos(A, mode="reduced"):
    """numpy QR with diag(R) >= 0, the convention of Num.qr."""
    Q, R = np.linalg.qr(np.asarray(A, float), mode=mode)
    k = min(R.shape)
    s = np.ones(Q.shape[1])
    s[:k] = np.where(np.diag(R)[:k] < 0, -1.0, 1.0)
    return Q * s, (R.T * s[: R.shape[0]]).T


# ---- exact row reduction (the hand method: first nonzero pivot) --------------------------

def rref_exact(A):
    """Gauss-Jordan in Fractions with the rule and step log of Num.rref(A)."""
    R = [[Fraction(x) for x in row] for row in A]
    m, n = len(R), len(R[0])
    fl = lambda: [[float(x) for x in row] for row in R]
    ops, ks, mats, pivots, r = [], [], [], [], 0
    for c in range(n):
        if r >= m:
            break
        p = next((i for i in range(r, m) if R[i][c] != 0), None)
        if p is None:
            continue
        if p != r:
            R[r], R[p] = R[p], R[r]
            ops.append(["swap", r, p]); ks.append(0.0); mats.append(fl())
        pv = R[r][c]
        if pv != 1:
            R[r] = [x / pv for x in R[r]]
            ops.append(["scale", r, -1]); ks.append(float(1 / pv)); mats.append(fl())
        for i in range(m):
            if i != r and R[i][c] != 0:
                f = R[i][c]
                R[i] = [a - f * b for a, b in zip(R[i], R[r])]
                ops.append(["add", i, r]); ks.append(float(-f)); mats.append(fl())
        pivots.append(c)
        r += 1
    return {"R": fl(), "pivots": pivots, "ops": ops, "k": ks, "mats": mats}


# ---- mulberry32 + Box-Muller, the stream of Num.rng(seed) --------------------------------

def largest_index(v):
    """Index of the entry of largest magnitude, the first among ties within 1e-12 (Num's rule)."""
    k = 0
    for i in range(1, len(v)):
        if abs(v[i]) > abs(v[k]) + 1e-12:
            k = i
    return k


class Mulberry32:
    """Port of Num.rng: uniform() matches bit for bit; normals to about 1e-15 (libm)."""

    def __init__(self, seed=1):
        self.a = seed & 0xFFFFFFFF
        self.spare = None

    def uniform(self, lo=None, hi=None):
        self.a = (self.a + 0x6D2B79F5) & 0xFFFFFFFF
        a = self.a
        t = ((a ^ (a >> 15)) * (a | 1)) & 0xFFFFFFFF
        t ^= (t + ((t ^ (t >> 7)) * (t | 61))) & 0xFFFFFFFF
        u = ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
        return u if lo is None else lo + (hi - lo) * u

    def normal(self, mu=0.0, sigma=1.0):
        if self.spare is not None:
            z, self.spare = self.spare, None
        else:
            u1, u2 = 1 - self.uniform(), self.uniform()
            r = math.sqrt(-2 * math.log(u1))
            z, self.spare = r * math.cos(2 * math.pi * u2), r * math.sin(2 * math.pi * u2)
        return mu + sigma * z

    def int(self, k):
        return math.floor(self.uniform() * k)

    def normal_vec(self, n):
        return [self.normal() for _ in range(n)]

    def mvnormal(self, mean, cov):
        cov = np.asarray(cov, float)
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Num.rng's fallback: L = V diag(sqrt(w)), eigenvalues descending, the largest
            # entry of each eigenvector positive, eigenvalues below n eps max|w| set to 0.
            w, V = np.linalg.eigh(cov)
            w, V = w[::-1], V[:, ::-1]
            for j in range(len(w)):
                if V[largest_index(V[:, j]), j] < 0:
                    V[:, j] = -V[:, j]
            cut = len(w) * np.finfo(float).eps * np.abs(w).max()
            L = V * np.where(w > cut, np.sqrt(np.maximum(w, 0)), 0.0)
        return (np.asarray(mean, float) + L @ np.array(self.normal_vec(len(mean)))).tolist()

    def shuffle(self, arr):
        out = list(arr)
        for i in range(len(out) - 1, 0, -1):
            j = self.int(i + 1)
            out[i], out[j] = out[j], out[i]
        return out


# ---- LP and QP references ----------------------------------------------------------------

_STATUS = {0: "optimal", 1: "iteration_limit", 2: "infeasible", 3: "unbounded"}


def _bounds(b, n):
    if b is None:
        return [(0, None)] * n
    if len(b) and isinstance(b[0], list) or (len(b) == n and len(b) != 2):
        return [tuple(x) if x is not None else (None, None) for x in b]
    return [tuple(b)] * n


def linprog(p):
    """scipy.optimize.linprog (HiGHS) on a num_data LP dict, in Num.lp's output shape."""
    n = len(p["c"])
    res = so.linprog(p["c"], A_ub=p.get("Aub"), b_ub=p.get("bub"), A_eq=p.get("Aeq"),
                     b_eq=p.get("beq"), bounds=_bounds(p.get("bounds"), n), method="highs")
    out = {"status": _STATUS.get(res.status, "failed")}
    if res.status == 0:
        out.update(x=res.x.tolist(), fun=float(res.fun), duals={
            "ub": res.ineqlin.marginals.tolist() if p.get("Aub") else [],
            "eq": res.eqlin.marginals.tolist() if p.get("Aeq") else [],
            "lower": res.lower.marginals.tolist(), "upper": res.upper.marginals.tolist()})
    return out


def qp_enum(p, tol=1e-9, max_active=None):
    """Exact convex-QP reference: try every subset of inequality rows as the active set,
    solve its KKT system, and keep the point that is primal and dual feasible.
    max_active limits the subset size; n suffices (Caratheodory: -(H x + g) is a conic
    combination of at most n independent active normals, modulo the equality rows)."""
    H = np.array(p["H"], float)
    g = np.array(p.get("g") or [0.0] * len(H), float)
    n = len(g)
    Aeq = np.array(p.get("Aeq") or np.zeros((0, n)), float).reshape(-1, n)
    beq = np.array(p.get("beq") or [], float)
    Aub = np.array(p.get("Aub") or np.zeros((0, n)), float).reshape(-1, n)
    bub = np.array(p.get("bub") or [], float)
    mE = len(Aeq)
    kmax = len(Aub) if max_active is None else min(len(Aub), max_active)
    for k in range(kmax + 1):
        for S in itertools.combinations(range(len(Aub)), k):
            A = np.vstack([Aeq, Aub[list(S)]])
            b = np.concatenate([beq, bub[list(S)]])
            m = len(A)
            K = np.block([[H, A.T], [A, np.zeros((m, m))]])
            rhs = np.concatenate([-g, b])
            sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
            if np.linalg.norm(K @ sol - rhs) > 1e-9 * (1 + np.linalg.norm(rhs)):
                continue
            x, mult = sol[:n], sol[n:]
            if len(Aub) and np.max(Aub @ x - bub) > tol:
                continue
            if np.any(mult[mE:] < -tol):
                continue
            lam = np.zeros(len(Aub))
            lam[list(S)] = mult[mE:]
            return {"x": x.tolist(), "fun": float(0.5 * x @ H @ x + g @ x),
                    "lamEq": mult[:mE].tolist(), "lamUb": lam.tolist()}
    return None


def qp_slsqp(p):
    """SLSQP at a tight tolerance, an independent check on qp_enum."""
    H = np.array(p["H"], float)
    g = np.array(p["g"], float)
    cons = []
    if p.get("Aub"):
        Aub, bub = np.array(p["Aub"], float), np.array(p["bub"], float)
        cons.append({"type": "ineq", "fun": lambda x: bub - Aub @ x, "jac": lambda x: -Aub})
    if p.get("Aeq"):
        Aeq, beq = np.array(p["Aeq"], float), np.array(p["beq"], float)
        cons.append({"type": "eq", "fun": lambda x: Aeq @ x - beq, "jac": lambda x: Aeq})
    res = so.minimize(lambda x: 0.5 * x @ H @ x + g @ x, np.zeros(len(g)), jac=lambda x: H @ x + g,
                      constraints=cons, method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000})
    return {"x": res.x.tolist(), "fun": float(res.fun)}


# ---- analytic derivatives for the finite-difference cases --------------------------------

def rosen(x):
    x = np.asarray(x, float)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rosen_grad(x):
    return so.rosen_der(np.asarray(x, float)).tolist()


def rosen_hess(x):
    return so.rosen_hess(np.asarray(x, float)).tolist()


def polar_jac(x):
    """Jacobian of F(r, t, s) = [r cos t, r sin t, r s^2]."""
    r, t, s = x
    return [[math.cos(t), -r * math.sin(t), 0.0],
            [math.sin(t), r * math.cos(t), 0.0],
            [s * s, 0.0, 2 * r * s]]


def kkt_solve(p):
    """x, nu from one numpy solve of [H A^T; A 0] [x; nu] = [-g; b]."""
    H, g = np.array(p["H"], float), np.array(p["g"], float)
    A, b = np.array(p["Aeq"], float), np.array(p["beq"], float)
    n, m = len(g), len(b)
    sol = np.linalg.solve(np.block([[H, A.T], [A, np.zeros((m, m))]]), np.concatenate([-g, b]))
    return {"x": sol[:n].tolist(), "nu": sol[n:].tolist()}


# ---- edge-case references ----------------------------------------------------------------

def cond2_exact(A):
    """Exact 2-norm condition number of the 2x2 matrix A as stored (its entries are binary
    fractions): sigma_max^2 / |det A| from Fractions, square roots in 60-digit Decimal."""
    from decimal import Decimal, getcontext

    getcontext().prec = 60
    a, b, c, d = (Fraction(x) for x in (A[0][0], A[0][1], A[1][0], A[1][1]))
    t, det = a * a + b * b + c * c + d * d, a * d - b * c
    if det == 0:
        return math.inf
    dec = lambda f: Decimal(f.numerator) / Decimal(f.denominator)
    smax2 = (dec(t) + dec(t * t - 4 * det * det).sqrt()) / 2
    return float(smax2 / abs(dec(det)))


def near_parallel(d):
    """[[1, 1], [1, 1 + d]]: two rows at an angle of about d / 2 radians."""
    return [[1.0, 1.0], [1.0, 1.0 + d]]


def eq_kkt(H, g, A, b):
    """x, nu of min (1/2) x^T H x + g^T x s.t. A x = b from numpy's solve of the KKT system."""
    H, g, A, b = (np.asarray(v, float) for v in (H, g, A, b))
    n, m = len(g), len(b)
    sol = np.linalg.solve(np.block([[H, A.T], [A, np.zeros((m, m))]]), np.concatenate([-g, b]))
    return sol[:n].tolist(), sol[n:].tolist()


def eq_kkt_lstsq(H, g, A, b):
    """Minimum-norm least-squares [x; nu] of the (singular) KKT system, from numpy.lstsq."""
    H, g, A, b = (np.asarray(v, float) for v in (H, g, A, b))
    n, m = len(g), len(b)
    K = np.block([[H, A.T], [A, np.zeros((m, m))]])
    sol = np.linalg.lstsq(K, np.concatenate([-g, b]), rcond=None)[0]
    return sol[:n].tolist(), sol[n:].tolist()


def sample_stats(seed, N, mean, cov):
    """Sample mean and covariance (ddof 1) of N draws of Mulberry32(seed).mvnormal."""
    g = Mulberry32(seed)
    X = np.array([g.mvnormal(mean, cov) for _ in range(N)])
    return [X.mean(axis=0).tolist(), np.cov(X.T, ddof=1).tolist()]


# ---- random degenerate problems: ports of tools/twins/num_gen.js -------------------------

def _ints(g, n, k, off):
    return [g.int(k) - off for _ in range(n)]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def gen_lp(seed):
    """NUM_GEN.lp(seed): the same draws in the same order."""
    g = Mulberry32(seed)
    n = 2 + g.int(4)
    m = 1 + g.int(7)
    v = _ints(g, n, 5, 2)
    Aub = [_ints(g, n, 7, 3) for _ in range(m)]
    bub = [_dot(a, v) + (0 if g.uniform() < 0.6 else 1 + g.int(2)) for a in Aub]
    p = {"c": _ints(g, n, 7, 3), "Aub": Aub, "bub": bub}
    if seed % 3 == 0:
        E = [_ints(g, n, 5, 2) for _ in range(1 + g.int(2))]
        if seed % 5 == 0:
            E.append([2 * x for x in E[0]])
        p["Aeq"], p["beq"] = E, [_dot(e, v) for e in E]
    kind = seed % 4
    if kind == 1:
        p["bounds"] = [None, None]
    elif kind == 2:
        p["bounds"] = [[[-3, 3], [None, 3], [-3, None]][g.int(3)] for _ in range(n)]
    elif kind == 3:
        p["bounds"] = [-4, 4]
    return p


def gen_qp(seed):
    """NUM_GEN.qp(seed): the same draws in the same order."""
    g = Mulberry32(seed)
    n = 2 + g.int(3)
    B = np.array([_ints(g, n, 5, 2) for _ in range(n)], float)
    H = B @ B.T
    if seed % 3 != 0:
        H += 0.05 * np.eye(n)
    v = _ints(g, n, 5, 2)
    Aub, bub = [], []
    for _ in range(n + 1 + g.int(3)):
        a = _ints(g, n, 7, 3)
        if not any(a):
            a[0] = 1
        Aub.append(a)
        bub.append(_dot(a, v) + (0 if g.uniform() < 0.7 else 1 + g.int(2)))
    for j in range(n):
        e = [0] * n
        e[j] = 1
        Aub += [e, [-x for x in e]]
        bub += [v[j] + 3, -v[j] + 3]
    if seed % 4 == 1:
        Aub += [list(Aub[0]), [2 * x for x in Aub[1]]]
        bub += [bub[0], 2 * bub[1]]
    p = {"H": H.tolist(), "g": _ints(g, n, 13, 6), "Aub": Aub, "bub": bub}
    if seed % 4 == 2:
        p["Aeq"], p["beq"] = [list(Aub[0]), list(Aub[0])], [bub[0], bub[0]]
    return p


def lp_summary(p):
    """[status, fun] from HiGHS (fun None unless optimal)."""
    r = linprog(p)
    return [r["status"], r.get("fun")]


def qp_summary(p):
    """[status, fun] from the enumeration (status 'infeasible' when no KKT point exists and
    the constraints are infeasible by HiGHS; the generated QPs are bounded by their box)."""
    r = qp_enum(p, max_active=len(p["H"]))
    if r is not None:
        return ["optimal", r["fun"]]
    n = len(p["H"])
    feas = linprog({"c": [0] * n, "Aub": p["Aub"], "bub": p["bub"], "Aeq": p.get("Aeq"),
                    "beq": p.get("beq"), "bounds": [None, None]})
    return ["infeasible" if feas["status"] == "infeasible" else "no KKT point found", None]
