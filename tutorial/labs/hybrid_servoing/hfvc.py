# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""Hybrid force-velocity control (HFVC) after Hou and Mason, ICRA 2019 and ICRA 2021.

Teaching reimplementation for the lab in tutorial/labs/hybrid_servoing/ and for deck 21.

    Y. Hou, M. T. Mason. Robust execution of contact-rich motion plans by hybrid force-velocity
    control. ICRA 2019, arXiv 1903.02715.  ("2019" below; Algorithms 1 and 2, Sec. V block tilting)
    Y. Hou, M. T. Mason. An efficient closed-form method for optimal hybrid force-velocity
    control. ICRA 2021, arXiv 2011.04872.  ("2021" below; crashing index eq. 9, OCHS Algorithm 1)
    Authors' MATLAB code: github.com/yifan-hou/pub-icra19-hybrid-control (solvehfvc.m and the
    block-tilting example), used here for conventions and default numbers.

Conventions (shared by both papers, notation of the curriculum):

* Generalized velocity v = [v_u; v_a] in R^n: the first n_u entries are unactuated (free
  objects), the last n_a entries are the robot's actuated velocities.
* J (2021) = N = J_Phi Omega (2019) is the n_lambda x n velocity-constraint matrix: every
  admissible velocity satisfies J v = 0. The generalized contact force is J'^T lambda, where
  J' ("J_force" below) equals J plus one row per sliding-friction force (2019 code convention).
* Newton's law, quasi-static (2019 eq. 5, 2021 eq. 3): J'^T lambda + f + F = 0, f_u = 0.
* A HFVC is (n_av, n_af, R_a, w_av, eta_af). T = diag(I_u, R_a), w = T v, eta = T f; the last
  n_av rows of T are velocity-controlled, so C = last n_av rows of T and C v = w_av (= b_C).
* Guard conditions act on [lambda; f]: Lam [lambda; f] <= b_Lam and Gam [lambda; f] = b_Gam.
* Null(A) and Row(A) return matrices whose ROWS are orthonormal bases of NULL(A) and ROW(A)
  (2021 Sec. V). 2021 eq. 23-24 silently switch to column bases; this file uses rows throughout
  (see null_rows and ochs_velocity, and errata/lab.md).

Contents: 1 linear algebra, 2 problem container and checks, 3 crashing index and goal inclusion,
4 Algorithm 1 (2019), 5 Algorithm 2 (2019), 6 OCHS (2021), 7 least-distance QP, 8 contact-system
builder, 9 planar examples of 2021 Fig. 1, 10 block tilting (2019 Sec. V), 11 random test
problems (2021 Table I), 12 solver wrappers and timing.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import brentq, least_squares, linprog, nnls

RTOL = 1e-9  # relative singular-value threshold for rank decisions


# ----------------------------------------------------------------------------------------------
# 1. Linear algebra
# ----------------------------------------------------------------------------------------------


def _as2d(A, n=None):
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        A = A.reshape(1, -1) if A.size else np.zeros((0, n or 0))
    return A


def svd_rank(A, rtol=RTOL):
    """Numerical rank: the number of singular values above rtol * sigma_max."""
    A = _as2d(A)
    if A.size == 0:
        return 0
    s = np.linalg.svd(A, compute_uv=False)
    if s[0] == 0.0:
        return 0
    return int(np.sum(s > rtol * s[0]))


def null_rows(A, n=None, rtol=RTOL):
    """Matrix whose rows are an orthonormal basis of NULL(A); shape (n - rank A, n).

    A matrix with no rows constrains nothing, so its null space is all of R^n (pass n)."""
    A = _as2d(A, n)
    n = A.shape[1] if n is None else n
    if A.shape[0] == 0:
        return np.eye(n)
    _, s, Vt = np.linalg.svd(A, full_matrices=True)
    r = 0 if s.size == 0 or s[0] == 0.0 else int(np.sum(s > rtol * s[0]))
    return Vt[r:].copy()


def row_rows(A, rtol=RTOL):
    """Matrix whose rows are an orthonormal basis of ROW(A); shape (rank A, n)."""
    A = _as2d(A)
    if A.shape[0] == 0:
        return np.zeros((0, A.shape[1]))
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    r = 0 if s[0] == 0.0 else int(np.sum(s > rtol * s[0]))
    return Vt[:r].copy()


def normalize_rows(A):
    """Each row divided by its 2-norm (2021 'C hat'); zero rows stay zero."""
    A = _as2d(A)
    nr = np.linalg.norm(A, axis=1, keepdims=True)
    return np.divide(A, nr, out=np.zeros_like(A), where=nr > 0)


def cond_rows(A, rtol=1e-10):
    """2-norm condition number sigma_1 / sigma_k over all k = min(m, n) singular values.

    2021 eq. 8 writes cond(A) = ||A|| ||A^+||, which equals this only when A has full rank; a
    rank-deficient A gets inf here, as in the 'inf' entries of 2021 Fig. 1."""
    A = _as2d(A)
    s = np.linalg.svd(A, compute_uv=False)
    if s.size == 0 or s[0] == 0.0:
        return math.inf
    if s[-1] <= rtol * s[0]:
        return math.inf
    return float(s[0] / s[-1])


def special_solution(A, b):
    """One solution of A v = b (least squares, minimum norm) and its residual norm."""
    A = _as2d(A)
    v, *_ = np.linalg.lstsq(A, np.asarray(b, float), rcond=None)
    return v, float(np.linalg.norm(A @ v - b))


# ----------------------------------------------------------------------------------------------
# 2. Problem container and physical checks
# ----------------------------------------------------------------------------------------------


@dataclass
class HFVCProblem:
    """One time step of a hybrid servoing problem.

    J        (m x n)   velocity constraints J v = 0 (2021 J; 2019 N = J_Phi Omega)
    n_u      int       number of unactuated coordinates (the first n_u entries of v)
    G, b_G             goal G v = b_G (2019 eq. 1, 2021 eq. 2)
    F        (n,)      external generalized force (gravity), Newton J'^T lam + f + F = 0
    J_force  (m' x n)  rows that carry contact forces (J plus sliding-friction rows); None = J
    Lam, b_Lam         guard inequalities Lam [lam; f] <= b_Lam (2019 eq. 6 and 18, 2021 eq. 4)
    Gam, b_Gam         guard equalities Gam [lam; f] = b_Gam (2019 eq. 6 and 19)
    """

    J: np.ndarray
    n_u: int
    G: np.ndarray
    b_G: np.ndarray
    F: Optional[np.ndarray] = None
    J_force: Optional[np.ndarray] = None
    Lam: Optional[np.ndarray] = None
    b_Lam: Optional[np.ndarray] = None
    Gam: Optional[np.ndarray] = None
    b_Gam: Optional[np.ndarray] = None
    name: str = ""
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.J = _as2d(self.J)
        n = self.J.shape[1]
        self.G = _as2d(self.G, n)
        self.b_G = np.atleast_1d(np.asarray(self.b_G, float))
        self.F = np.zeros(n) if self.F is None else np.asarray(self.F, float)
        self.J_force = self.J.copy() if self.J_force is None else _as2d(self.J_force, n)
        m = self.J_force.shape[0]
        if self.Lam is None:
            self.Lam, self.b_Lam = np.zeros((0, m + n)), np.zeros(0)
        if self.Gam is None:
            self.Gam, self.b_Gam = np.zeros((0, m + n)), np.zeros(0)
        self.Lam, self.Gam = _as2d(self.Lam, m + n), _as2d(self.Gam, m + n)
        self.b_Lam = np.atleast_1d(np.asarray(self.b_Lam, float))
        self.b_Gam = np.atleast_1d(np.asarray(self.b_Gam, float))

    @property
    def n(self):
        return self.J.shape[1]

    @property
    def n_a(self):
        return self.n - self.n_u

    @property
    def n_lam(self):
        return self.J_force.shape[0]


def newton_residual(prob: HFVCProblem, lam, f):
    """Norm of J'^T lam + f + F (quasi-static Newton law, 2019 eq. 5)."""
    return float(np.linalg.norm(prob.J_force.T @ lam + f + prob.F))


def guard_violation(prob: HFVCProblem, lam, f):
    """Largest violation of the guard conditions (0 when every row holds)."""
    x = np.concatenate([lam, f])
    viol = 0.0
    if prob.Lam.shape[0]:
        viol = max(viol, float(np.max(prob.Lam @ x - prob.b_Lam)))
    if prob.Gam.shape[0]:
        viol = max(viol, float(np.max(np.abs(prob.Gam @ x - prob.b_Gam))))
    return max(viol, 0.0)


# ----------------------------------------------------------------------------------------------
# 3. Crashing index (2021 eq. 9) and the goal-inclusion conditions (2021 eq. 10-13)
# ----------------------------------------------------------------------------------------------


def crashing_index(J, C):
    """cond([J_hat; C_hat]): J_hat = orthonormal rows spanning ROW(J), C_hat = C with unit rows.

    Large when a velocity-controlled direction is nearly a combination of the constraint rows
    (the robot would push against a natural constraint); inf when it is exactly one."""
    C = _as2d(C, _as2d(J).shape[1])
    return cond_rows(np.vstack([row_rows(J), normalize_rows(C)]))


@dataclass
class GoalInclusion:
    null_inclusion: bool  # condition 1): NULL([J;C]) subset of NULL([J;G]), via the rank test eq. 13
    common_solution: bool  # condition 2): some v with J v = 0, C v = b_C, G v = b_G
    rank_JC: int
    rank_JCG: int
    residual: float  # least-squares residual of the stacked system in condition 2)
    sampled_goal_error: float  # max |G v - b_G| over random solutions of J v = 0, C v = b_C

    @property
    def ok(self):
        return self.null_inclusion and self.common_solution


def goal_inclusion(J, G, b_G, C, b_C, samples=20, tol=1e-7, rng=0):
    """Check both goal-inclusion conditions of 2021 Sec. V for a velocity command C v = b_C."""
    J = _as2d(J)
    n = J.shape[1]
    G, C = _as2d(G, n), _as2d(C, n)
    b_G, b_C = np.atleast_1d(b_G).astype(float), np.atleast_1d(b_C).astype(float)
    JC, JCG = np.vstack([J, C]), np.vstack([J, C, G])
    r1, r2 = svd_rank(JC), svd_rank(JCG)
    A = JCG
    b = np.concatenate([np.zeros(J.shape[0]), b_C, b_G])
    _, res = special_solution(A, b)
    scale = 1.0 + np.linalg.norm(b)
    # every solution of the command system = particular solution + NULL([J;C])
    vp, res_c = special_solution(JC, np.concatenate([np.zeros(J.shape[0]), b_C]))
    Z = null_rows(JC)
    gen = np.random.default_rng(rng)
    err = 0.0
    for _ in range(samples):
        v = vp + Z.T @ gen.standard_normal(Z.shape[0]) if Z.shape[0] else vp
        err = max(err, float(np.max(np.abs(G @ v - b_G))) if G.shape[0] else 0.0)
    return GoalInclusion(r1 == r2, res <= tol * scale, r1, r2, res, err)


# ----------------------------------------------------------------------------------------------
# 4. Algorithm 1 of 2019: velocity-controlled directions by projected gradient descent
# ----------------------------------------------------------------------------------------------


@dataclass
class VelocityControl:
    """The velocity half of a HFVC: dimensions, directions (T) and magnitudes (w_av)."""

    n_av: int
    n_af: int
    C: np.ndarray  # (n_av x n) velocity-controlled rows, zero in the first n_u columns
    R_a: np.ndarray  # (n_a x n_a) [force-controlled rows; C's actuated part]
    T: np.ndarray  # diag(I_u, R_a)
    w_av: np.ndarray  # (n_av,) velocity command magnitudes, C v = w_av
    v_star: np.ndarray  # one velocity that satisfies J v = 0, G v = b_G
    method: str = ""
    info: dict = field(default_factory=dict)


class Infeasible(Exception):
    """The velocity or force problem has no solution (the message names the failed test)."""


def complete_axes(C, n_u):
    """R_a = [Null(R_C); R_C] with R_C the actuated columns of C (2019 eq. 16, 2021 eq. 27)."""
    C = _as2d(C)
    R_C = C[:, n_u:]
    n_a = C.shape[1] - n_u
    R_a = np.vstack([null_rows(R_C, n_a), R_C]) if R_C.shape[0] else np.eye(n_a)
    return R_a, block_diag(np.eye(n_u), R_a)


def alg1_cost(K, M, form="paper"):
    """Cost of 2019 eq. 15 for coefficients K (n_c x n_av), c_i = B_c k_i with B_c orthonormal.

    form='paper':   sum_{i!=j} |c_i^T c_j| - sum_i ||Null(N)^T c_i||      (eq. 15 as printed)
    form='squared': sum_{i!=j} (c_i^T c_j)^2 - sum_i ||Null(N)^T c_i||^2   (authors' MATLAB code)
    M = B_c^T Null(N)^T Null(N) B_c, so ||Null(N)^T c_i||^2 = k_i^T M k_i."""
    Gram = K.T @ K
    off = Gram - np.diag(np.diag(Gram))
    quad = np.einsum("ji,jk,ki->i", K, M, K)
    if form == "paper":
        return float(np.abs(off).sum() - np.sqrt(np.maximum(quad, 0.0)).sum())
    if form == "squared":
        return float((off**2).sum() - quad.sum())
    raise ValueError(form)


def alg1_grad(K, M, form="paper"):
    """Gradient of alg1_cost with respect to K (same shape as K).

    Each unordered pair {i, j} appears twice in sum_{i!=j}, which gives the factor 2 ('paper')
    and 4 ('squared'); the authors' code uses 2 in the squared form."""
    Gram = K.T @ K
    off = Gram - np.diag(np.diag(Gram))
    MK = M @ K
    if form == "paper":
        quad = np.maximum(np.einsum("ji,ji->i", K, MK), 1e-300)
        return 2.0 * K @ np.sign(off) - MK / np.sqrt(quad)
    if form == "squared":
        return 4.0 * K @ off - 2.0 * MK
    raise ValueError(form)


def alg1_basis(prob: HFVCProblem):
    """Steps 1-3 of 2019 Algorithm 1: ranks, n_av (eq. 10), B_c (eq. 13) and Null(N).

    Returns dict with r_N, r_NG, n_av, sigma (rows sigma_i^T), Bc (n x n_c, orthonormal columns),
    NullN (rows: orthonormal basis of null(N)) and M = B_c^T Null(N)^T Null(N) B_c."""
    N, G, n, n_u = prob.J, prob.G, prob.n, prob.n_u
    n_a = n - n_u
    NG = np.vstack([N, G])
    r_N, r_NG = svd_rank(N), svd_rank(NG)
    if r_N + n_a < n:  # eq. 14 (the authors' code asserts the strict version)
        raise Infeasible(f"eq. 14 fails: r_N + n_a = {r_N + n_a} < n = {n}")
    n_av = r_NG - r_N  # eq. 10
    sigma = null_rows(NG)  # rows sigma_i^T, i = 1..n - r_NG
    A13 = np.vstack([sigma, np.hstack([np.eye(n_u), np.zeros((n_u, n_a))])])
    Bc = null_rows(A13, n).T  # eq. 13 solution space, orthonormal columns
    NullN = null_rows(N, n)
    M = Bc.T @ NullN.T @ NullN @ Bc
    return dict(r_N=r_N, r_NG=r_NG, n_av=n_av, n_c=Bc.shape[1], sigma=sigma, Bc=Bc, NullN=NullN, M=M)


def alg1_pgd(M, n_av, K0, t=10.0, iters=50, form="paper", tol=0.0, record=False):
    """Projected gradient descent of 2019 Sec. IV-A on unit-length columns of K.

    Steps: k <- k - t grad; k_i <- k_i / ||B_c k_i|| (= ||k_i|| since B_c is orthonormal).
    Runs `iters` iterations (the authors' code and 2021 Sec. VI-B use 50) and stops early when
    the cost changes by less than tol. Returns (K, cost, history of costs if record)."""
    K = K0 / np.linalg.norm(K0, axis=0, keepdims=True)
    hist = [alg1_cost(K, M, form)] if record else None
    last = alg1_cost(K, M, form)
    for _ in range(iters):
        K = K - t * alg1_grad(K, M, form)
        K = K / np.linalg.norm(K, axis=0, keepdims=True)
        cost = alg1_cost(K, M, form)
        if record:
            hist.append(cost)
        if tol and abs(cost - last) < tol:
            last = cost
            break
        last = cost
    return K, last, hist


def alg1_velocity(prob: HFVCProblem, Ns=3, t=10.0, iters=50, form="paper", rng=None, tol=0.0):
    """2019 Algorithm 1: n_av, C (via eq. 13 and 15), R_a (eq. 16), T, v* and w_av.

    Ns random initializations (standard normal columns; the authors' code draws U[0, 1)) are
    each run through alg1_pgd; the lowest final cost wins."""
    rng = np.random.default_rng(rng)
    b = alg1_basis(prob)
    n, n_u, n_av = prob.n, prob.n_u, b["n_av"]
    n_a = n - n_u
    if b["n_c"] < n_av:
        raise Infeasible(f"n_c = {b['n_c']} < n_av = {n_av}")
    if n_av == 0:
        C = np.zeros((0, n))
        costs, best = [0.0], 0
    else:
        results = [
            alg1_pgd(b["M"], n_av, rng.standard_normal((b["n_c"], n_av)), t, iters, form, tol)
            for _ in range(Ns)
        ]
        costs = [r[1] for r in results]
        best = int(np.argmin(costs))
        C = (b["Bc"] @ results[best][0]).T
    R_a, T = complete_axes(C, n_u)
    NG = np.vstack([prob.J, prob.G])
    v_star, res = special_solution(NG, np.concatenate([np.zeros(prob.J.shape[0]), prob.b_G]))
    if res > 1e-7 * (1 + np.linalg.norm(prob.b_G)):
        raise Infeasible(f"goal inconsistent with the constraints (residual {res:.2e})")
    return VelocityControl(
        n_av, n_a - n_av, C, R_a, T, C @ v_star, v_star, f"alg1(Ns={Ns},{form})",
        dict(costs=costs, best=best, r_N=b["r_N"], r_NG=b["r_NG"], n_c=b["n_c"]),
    )


# ----------------------------------------------------------------------------------------------
# 5. Algorithm 2 of 2019: force-controlled magnitudes eta_af by the KKT system and an LP
# ----------------------------------------------------------------------------------------------


@dataclass
class ForceControl:
    eta_af: np.ndarray  # (n_af,) force command magnitudes
    lam: np.ndarray  # contact forces
    eta: np.ndarray  # transformed generalized force T f = [eta_u; eta_af; eta_av]
    f: np.ndarray  # generalized (robot) force
    method: str = ""
    info: dict = field(default_factory=dict)


def alg2_newton_system(prob: HFVCProblem, vel: VelocityControl):
    """2019 eq. 20 split into free and commanded columns (eq. 21).

    Unknowns x = [lam; eta], eta = [eta_u; eta_af; eta_av]. Rows: H T^-1 eta = 0 (f_u = 0),
    T J'^T lam + eta = -T F (eq. 17), [Gam_lam, Gam_f T^-1][lam; eta] = b_Gam (eq. 19).
    Returns M_free, M_af, b, and index arrays of the free forces [lam, eta_u, eta_av]."""
    n, n_u, m = prob.n, prob.n_u, prob.n_lam
    T = vel.T
    Tinv = np.linalg.inv(T)
    H = np.hstack([np.eye(n_u), np.zeros((n_u, n - n_u))])
    Gl, Gf = prob.Gam[:, :m], prob.Gam[:, m:]
    M = np.vstack([
        np.hstack([np.zeros((n_u, m)), H @ Tinv]),
        np.hstack([T @ prob.J_force.T, np.eye(n)]),
        np.hstack([Gl, Gf @ Tinv]),
    ])
    b = np.concatenate([np.zeros(n_u), -T @ prob.F, prob.b_Gam])
    i_af = m + n_u + np.arange(vel.n_af)
    i_free = np.setdiff1d(np.arange(m + n), i_af)
    return M[:, i_free], M[:, i_af], b, i_free, i_af, Tinv


def alg2_kkt(prob: HFVCProblem, vel: VelocityControl, eta_af):
    """Solve 2019 eq. 22 for the free forces given a force command eta_af.

    [[2I, M_free^T], [M_free, 0]] [f_free; f_free*] = [0; b - M_af eta_af], the KKT system of
    min ||f_free||^2 s.t. M_free f_free = b - M_af eta_af (eq. 21). The printed eq. 22-23 drop
    b_Gam from the right-hand side; it is kept here, as in the authors' code."""
    M_free, M_af, b, _, _, _ = alg2_newton_system(prob, vel)
    k, p = M_free.shape
    KKT = np.block([[2 * np.eye(p), M_free.T], [M_free, np.zeros((k, k))]])
    rhs = np.concatenate([np.zeros(p), b - M_af @ np.atleast_1d(eta_af)])
    sol = np.linalg.solve(KKT, rhs)
    return sol[:p], sol[p:]


def alg2_force(prob: HFVCProblem, vel: VelocityControl, objective="l1"):
    """2019 Algorithm 2: choose eta_af so that the forces fixed by eq. 23 satisfy the guards (18).

    The KKT system makes the free forces an affine function f_free = f0 + P eta_af (computed with
    the pseudo-inverse, which equals the KKT solution when the KKT matrix is nonsingular, and
    which adds the consistency rows (I - M M^+)(b - M_af eta_af) = 0 when it is singular). The
    guard rows then become linear inequalities on eta_af alone, a small LP.
    objective: 'l1' min sum |eta_af| (LP), 'feasibility' (LP, zero cost), 'l2' min ||eta_af||^2
    (the authors' code solves this QP with quadprog)."""
    M_free, M_af, b, i_free, i_af, Tinv = alg2_newton_system(prob, vel)
    m, n, n_af = prob.n_lam, prob.n, vel.n_af
    Mp = np.linalg.pinv(M_free, rcond=1e-10)
    f0, P = Mp @ b, -Mp @ M_af
    Q = np.eye(M_free.shape[0]) - M_free @ Mp  # consistency projector
    A_eq, b_eq = Q @ M_af, Q @ b
    keep = np.linalg.norm(np.hstack([A_eq, b_eq[:, None]]), axis=1) > 1e-9
    A_eq, b_eq = A_eq[keep], b_eq[keep]
    # [lam; eta] = E_free f_free + E_af eta_af
    E_free = np.zeros((m + n, len(i_free)))
    E_free[i_free, np.arange(len(i_free))] = 1
    E_af = np.zeros((m + n, n_af))
    E_af[i_af, np.arange(n_af)] = 1
    Lam_t = np.hstack([prob.Lam[:, :m], prob.Lam[:, m:] @ Tinv])  # eq. 18
    A_ub = Lam_t @ (E_free @ P + E_af)
    b_ub = prob.b_Lam - Lam_t @ (E_free @ f0)
    if n_af == 0:
        eta_af = np.zeros(0)
        if (A_ub.size and np.any(b_ub < -1e-7)) or (A_eq.size and np.any(np.abs(b_eq) > 1e-7)):
            raise Infeasible("guard conditions fail with no force-controlled direction")
    elif objective == "l2":
        eta_af, ok = least_distance(A_ub, b_ub, A_eq if A_eq.size else None, b_eq if A_eq.size else None)
        if not ok:
            raise Infeasible("Algorithm 2: guard conditions infeasible (QP)")
    else:
        c = np.concatenate([np.zeros(n_af), np.ones(n_af) if objective == "l1" else np.zeros(n_af)])
        I = np.eye(n_af)
        Aub = np.vstack([np.hstack([A_ub, np.zeros((A_ub.shape[0], n_af))]),
                         np.hstack([I, -I]), np.hstack([-I, -I])])
        bub = np.concatenate([b_ub, np.zeros(2 * n_af)])
        Aeq = np.hstack([A_eq, np.zeros((A_eq.shape[0], n_af))]) if A_eq.size else None
        res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=b_eq if A_eq.size else None,
                      bounds=[(None, None)] * n_af + [(0, None)] * n_af, method="highs")
        if res.status != 0:
            raise Infeasible(f"Algorithm 2: LP status {res.status} ({res.message})")
        eta_af = res.x[:n_af]
    x = E_free @ (f0 + P @ eta_af) + E_af @ eta_af
    lam, eta = x[:m], x[m:]
    return ForceControl(eta_af, lam, eta, Tinv @ eta, f"alg2({objective})",
                        dict(M_free_shape=M_free.shape, rank_M_free=svd_rank(M_free),
                             n_consistency_rows=int(A_eq.shape[0])))


# ----------------------------------------------------------------------------------------------
# 6. OCHS of 2021: closed-form velocity control and the force QP
# ----------------------------------------------------------------------------------------------


def ochs_velocity(prob: HFVCProblem, maximal=False):
    """2021 Algorithm 1 lines 1-13 (Optimally-Conditioned Hybrid Servoing).

    U = Null(J) (eq. 14); U_bar = Row(U S_a) (eq. 15); n_av from eq. 21 (or eq. 19 when
    maximal); K = Null(Null([J;G]) U_bar^T) (eq. 24, row-basis form); C = K U_bar (eq. 22);
    R_a from eq. 27; v* from eq. 28; w_av = C v* (eq. 29)."""
    J, G, n, n_u = prob.J, prob.G, prob.n, prob.n_u
    U = null_rows(J, n)
    US = U.copy()
    US[:, :n_u] = 0.0
    Ubar = row_rows(US)
    r_J, r_JG = svd_rank(J), svd_rank(np.vstack([J, G]))
    n_min = r_JG - r_J
    if Ubar.shape[0] < n_min:  # eq. 18
        raise Infeasible(f"eq. 18 fails: rows(U_bar) = {Ubar.shape[0]} < {n_min}")
    info = dict(U=U, Ubar=Ubar, r_J=r_J, r_JG=r_JG)
    if maximal:
        n_av, C = Ubar.shape[0], Ubar
        if svd_rank(np.vstack([J, C])) != svd_rank(np.vstack([J, C, G])):  # eq. 13
            raise Infeasible("eq. 13 fails for C = U_bar")
    else:
        n_av = n_min
        Z = null_rows(np.vstack([J, G]), n)
        K = null_rows(Z @ Ubar.T, Ubar.shape[0])  # eq. 24
        info["K"] = K
        if K.shape[0] < n_av:  # eq. 25
            raise Infeasible(f"eq. 25 fails: rows(K) = {K.shape[0]} < n_av = {n_av}")
        C = K[:n_av] @ Ubar
    R_a, T = complete_axes(C, n_u)
    v_star, res = special_solution(np.vstack([J, G]), np.concatenate([np.zeros(J.shape[0]), prob.b_G]))
    if res > 1e-7 * (1 + np.linalg.norm(prob.b_G)):
        raise Infeasible(f"eq. 28 has no solution (residual {res:.2e})")
    return VelocityControl(n_av, n - n_u - n_av, C, R_a, T, C @ v_star, v_star,
                           "ochs(M)" if maximal else "ochs", info)


def ochs_force(prob: HFVCProblem, vel: VelocityControl):
    """2021 eq. 30: min lam^T lam + eta_a^T eta_a s.t. Newton (eq. 3) and the guards (eq. 4).

    Unknowns x = [lam; eta_a] with eta_u = 0 and f = T^-1 [0; eta_a]; solved exactly as a
    least-distance problem (section 7). eta_af is the first n_af entries of eta_a."""
    m, n, n_u = prob.n_lam, prob.n, prob.n_u
    Tinv = np.linalg.inv(vel.T)
    Ta = Tinv[:, n_u:]  # f = Ta eta_a
    E_eq = np.hstack([prob.J_force.T, Ta])
    e_eq = -prob.F
    if prob.Gam.shape[0]:
        E_eq = np.vstack([E_eq, np.hstack([prob.Gam[:, :m], prob.Gam[:, m:] @ Ta])])
        e_eq = np.concatenate([e_eq, prob.b_Gam])
    A_ub = np.hstack([prob.Lam[:, :m], prob.Lam[:, m:] @ Ta])
    x, ok = least_distance(A_ub, prob.b_Lam, E_eq, e_eq)
    if not ok:
        raise Infeasible("OCHS force QP (eq. 30) infeasible")
    lam, eta_a = x[:m], x[m:]
    eta = np.concatenate([np.zeros(n_u), eta_a])
    return ForceControl(eta_a[: vel.n_af], lam, eta, Tinv @ eta, "ochs-qp")


# ----------------------------------------------------------------------------------------------
# 7. Least-distance programming: min ||x||^2 s.t. A x <= b, E x = e (exact, via NNLS)
# ----------------------------------------------------------------------------------------------


def least_distance(A_ub, b_ub, E=None, e=None, feas_tol=1e-7):
    """Minimum-norm point of a polyhedron (Lawson and Hanson, Solving Least Squares Problems,
    ch. 23, algorithm LDP). Equalities are removed first: x = x0 + Z y with x0 the minimum-norm
    solution of E x = e and Z an orthonormal basis of NULL(E), so ||x||^2 = ||x0||^2 + ||y||^2.
    Then min ||y|| s.t. Gm y >= h is solved by one NNLS: E' = [Gm^T; h^T], f = e_last,
    u = argmin ||E'u - f|| (u >= 0), r = E'u - f, y = -r[:k] / r[k]. Returns (x, feasible)."""
    A_ub = np.atleast_2d(np.asarray(A_ub, float))
    n = A_ub.shape[1] if A_ub.size else (np.atleast_2d(E).shape[1])
    b_ub = np.asarray(b_ub, float).reshape(-1)
    if E is not None and np.size(E):
        E = np.atleast_2d(np.asarray(E, float))
        e = np.asarray(e, float).reshape(-1)
        x0, res = special_solution(E, e)
        if res > feas_tol * (1 + np.linalg.norm(e)):
            return x0, False
        Z = null_rows(E, n).T
    else:
        x0, Z = np.zeros(n), np.eye(n)
    k = Z.shape[1]
    if A_ub.shape[0] == 0 or A_ub.size == 0:
        return x0, True
    Gm = -A_ub @ Z
    h = -(b_ub - A_ub @ x0)
    if k == 0:
        return x0, bool(np.all(Gm.shape[0] == 0 or h <= feas_tol))
    # scale rows so NNLS works on unit-size numbers
    s = np.maximum(np.linalg.norm(np.hstack([Gm, h[:, None]]), axis=1), 1e-300)
    Gs, hs = Gm / s[:, None], h / s
    Ep = np.vstack([Gs.T, hs[None, :]])
    fp = np.zeros(k + 1)
    fp[-1] = 1.0
    u, _ = nnls(Ep, fp, maxiter=50 * (Ep.shape[1] + 10))
    r = Ep @ u - fp
    if np.linalg.norm(r) < 1e-12 or abs(r[-1]) < 1e-12:
        return x0, False
    y = -r[:k] / r[-1]
    x = x0 + Z @ y
    ok = bool(np.all(A_ub @ x - b_ub <= feas_tol * (1 + np.abs(b_ub))))
    return x, ok


# ----------------------------------------------------------------------------------------------
# 8. Contact-system builder (planar and 3D rigid bodies and points, velocities in world frame)
# ----------------------------------------------------------------------------------------------


@dataclass
class Body:
    """A body whose velocity is (linear velocity of `ref`, angular velocity), world frame.

    kind='rigid': dim + (1 if dim == 2 else 3) coordinates; kind='point': dim coordinates."""

    ref: np.ndarray
    kind: str = "rigid"


@dataclass
class Contact:
    """A point contact at world point p with unit normal n pointing from body a (None = the
    fixed environment) into body b. lambda_n > 0 pushes b along +n and a along -n.

    mode 'stick': all frame directions are velocity constraints, friction cone guard;
    mode 'slide': only the normal is a constraint; friction mu_slide opposes the sliding
    direction slide_dir (a unit tangent) through an extra J' row and a Gam equality, as in the
    authors' flip-against-corner example. mu_slide = 0 makes the contact frictionless."""

    p: np.ndarray
    n: np.ndarray
    b: int
    a: Optional[int]
    mode: str = "stick"
    mu: float = 0.8
    n_min: float = 0.0
    mu_slide: float = 0.0
    slide_dir: Optional[np.ndarray] = None


def skew(p):
    """[p]x, the matrix with [p]x w = p x w."""
    x, y, z = p
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def body_dofs(dim, body: Body):
    return dim if body.kind == "point" else (3 if dim == 2 else 6)


def point_velocity_map(dim, body: Body, p):
    """Matrix V with (velocity of the material point of `body` at p) = V @ v_body."""
    if body.kind == "point":
        return np.eye(dim)
    r = np.asarray(p, float) - np.asarray(body.ref, float)
    if dim == 2:
        return np.array([[1.0, 0.0, -r[1]], [0.0, 1.0, r[0]]])
    return np.hstack([np.eye(3), -skew(r)])


def tangent_basis(n):
    """Two unit tangents t1, t2 with (n, t1, t2) right-handed and orthonormal."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(n, t1)


def contact_frame(dim, n):
    """Rows: the normal, then the tangent(s)."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    if dim == 2:
        return np.array([n, [-n[1], n[0]]])
    t1, t2 = tangent_basis(n)
    return np.array([n, t1, t2])


def cone_directions(sides):
    """Unit tangent directions d_i = (sin(2 pi i / sides), cos(2 pi i / sides)), i = 1..sides.

    2019 Sec. V-D uses sides = 8, d_i = [sin(pi i/4), cos(pi i/4), 0]; the constraints
    d_i^T lambda_t <= mu lambda_n define the octagon that circumscribes the circle of radius
    mu lambda_n (d_i are facet normals, not ridges), so tangential forces up to
    mu / cos(pi/8) = 1.082 mu pass along the octagon's corners."""
    i = np.arange(1, sides + 1)
    return np.stack([np.sin(2 * np.pi * i / sides), np.cos(2 * np.pi * i / sides)], axis=1)


def contact_system(dim, bodies, contacts, n_u, F, G, b_G, cone_sides=8, name="", info=None):
    """Assemble J, J', the guard rows and the problem for a list of bodies and contacts.

    Coordinates: bodies in order (body 0 first); the first n_u coordinates are unactuated.
    lambda: per contact its frame components (normal first), then one friction component per
    sliding contact with mu_slide > 0 (appended last, like the authors' J_phi 'all')."""
    offs = np.cumsum([0] + [body_dofs(dim, b) for b in bodies])
    n = int(offs[-1])
    rows, frows, lam_of, fric = [], [], [], []
    for ci, c in enumerate(contacts):
        fr = contact_frame(dim, c.n)
        dirs = fr if c.mode == "stick" else fr[:1]

        def vel_row(d, c=c):
            r = np.zeros(n)
            Vb = point_velocity_map(dim, bodies[c.b], c.p)
            r[offs[c.b]:offs[c.b + 1]] += d @ Vb
            if c.a is not None:
                Va = point_velocity_map(dim, bodies[c.a], c.p)
                r[offs[c.a]:offs[c.a + 1]] -= d @ Va
            return r

        idx = []
        for d in dirs:
            idx.append(len(rows))
            rows.append(vel_row(d))
        lam_of.append(idx)
        if c.mode == "slide" and c.mu_slide > 0:
            fric.append((ci, vel_row(np.asarray(c.slide_dir, float))))
    J = np.array(rows) if rows else np.zeros((0, n))
    m0 = J.shape[0]
    J_force = np.vstack([J] + [r[None, :] for _, r in fric]) if fric else J.copy()
    m = J_force.shape[0]
    Lam, bL, Gam, bG = [], [], [], []
    D = cone_directions(cone_sides) if dim == 3 else None
    for ci, c in enumerate(contacts):
        idx = lam_of[ci]
        row = np.zeros(m + n)
        row[idx[0]] = -1.0
        Lam.append(row)
        bL.append(-c.n_min)
        if c.mode == "stick":
            if dim == 2:
                for s in (1.0, -1.0):
                    row = np.zeros(m + n)
                    row[idx[1]], row[idx[0]] = s, -c.mu
                    Lam.append(row)
                    bL.append(0.0)
            else:
                for d in D:
                    row = np.zeros(m + n)
                    row[idx[1]], row[idx[2]], row[idx[0]] = d[0], d[1], -c.mu
                    Lam.append(row)
                    bL.append(0.0)
    for k, (ci, _) in enumerate(fric):
        row = np.zeros(m + n)
        row[m0 + k] = 1.0  # friction on b along slide_dir equals -mu_slide * lambda_n
        row[lam_of[ci][0]] = contacts[ci].mu_slide
        Gam.append(row)
        bG.append(0.0)
    return HFVCProblem(
        J, n_u, G, b_G, F, J_force,
        np.array(Lam) if Lam else None, np.array(bL) if bL else None,
        np.array(Gam) if Gam else None, np.array(bG) if bG else None,
        name, dict(info or {}, dim=dim, bodies=bodies, contacts=contacts, lam_of=lam_of),
    )


# ----------------------------------------------------------------------------------------------
# 9. Planar examples of 2021 Fig. 1 (block + point finger), reconstructed
# ----------------------------------------------------------------------------------------------
# The figure gives no geometry. v = (block vx, vy, omega at the block centre, finger vx, vy);
# the finger touches the top centre and sticks. Top row: the block slides on the ground (two
# normal-only corner contacts). Bottom row: the block sticks on a pivot under its bottom centre.
# 2.41 = 1 + sqrt(2) and both 'inf' follow from the contact structure alone. The diagonal V is
# drawn at 45 degrees, which gives 2 + sqrt(3) = 3.73; the printed 3.87 needs 46.8 degrees below
# horizontal. With the block velocity taken at its centre, the pivot-to-finger distance h enters
# the bottom row; h = 0.290 (any length unit, radians for omega) gives 7.10, and then 10.48 is
# a prediction. fig1_fit() recomputes both parameters from the printed numbers.

FIG1_PAPER = {("ground", "right"): 2.41, ("ground", "diag"): 3.87, ("ground", "down"): math.inf,
              ("pivot", "right"): 7.10, ("pivot", "diag"): 10.48, ("pivot", "down"): math.inf}
FIG1_PHI_DEG = 46.8  # angle of the diagonal V below horizontal (fitted; drawn as 45)
FIG1_H = 0.290  # pivot-to-finger distance (fitted)


def fig1_problem(support="ground", h=FIG1_H, w=None, weight=1.0, mu_finger=0.8, mu_ground=0.3,
                 n_min=0.1, speed=0.1):
    """Planar block of height h and width w (default w = h) with a point finger on its top centre.

    support='ground': the block slides right on the ground, Coulomb friction mu_ground at both
    corners (J' and Gam rows); goal: block vx = speed. support='pivot': the block sticks on a
    pivot under its bottom centre; goal: omega = -speed / h (the finger moves right at `speed`)."""
    w = h if w is None else w
    bodies = [Body(np.array([0.0, 0.0])), Body(np.array([0.0, h / 2]), "point")]
    top = Contact(np.array([0.0, h / 2]), np.array([0.0, -1.0]), b=0, a=1, mode="stick",
                  mu=mu_finger, n_min=n_min)
    if support == "ground":
        env = [Contact(np.array([s * w / 2, -h / 2]), np.array([0.0, 1.0]), b=0, a=None,
                       mode="slide", n_min=0.0, mu_slide=mu_ground, slide_dir=np.array([1.0, 0.0]))
               for s in (-1, 1)]
        G, b_G = np.array([[1.0, 0, 0, 0, 0]]), np.array([speed])
    elif support == "pivot":
        env = [Contact(np.array([0.0, -h / 2]), np.array([0.0, 1.0]), b=0, a=None, mode="stick",
                       mu=1.0, n_min=0.0)]
        G, b_G = np.array([[0.0, 0, 1, 0, 0]]), np.array([-speed / h])
    else:
        raise ValueError(support)
    F = np.array([0.0, -weight, 0.0, 0.0, 0.0])
    return contact_system(2, bodies, [top] + env, 3, F, G, b_G, name=f"fig1-{support}",
                          info=dict(h=h, w=w))


def fig1_C(direction, phi_deg=FIG1_PHI_DEG):
    """The velocity-controlled row of Fig. 1: finger velocity to the right, diagonal, or down."""
    a = math.radians(phi_deg)
    d = {"right": (1.0, 0.0), "diag": (math.cos(a), -math.sin(a)), "down": (0.0, -1.0)}[direction]
    return np.array([[0.0, 0.0, 0.0, d[0], d[1]]])


def fig1_table(phi_deg=FIG1_PHI_DEG, h=FIG1_H):
    """The six crashing indexes of 2021 Fig. 1 from the reconstruction, next to the paper's."""
    out = []
    for support in ("ground", "pivot"):
        J = fig1_problem(support, h=h).J
        for direction in ("right", "diag", "down"):
            out.append(dict(support=support, direction=direction,
                            ours=crashing_index(J, fig1_C(direction, phi_deg)),
                            paper=FIG1_PAPER[(support, direction)]))
    return out


def fig1_fit():
    """Fit (phi, h) to the printed 3.87, 7.10 and 10.48 by least squares; also return the exact
    one-parameter solutions (phi from 3.87 alone, h from 7.10 alone) and the 45-degree value."""
    def ci(support, direction, phi, h):
        return crashing_index(fig1_problem(support, h=h).J, fig1_C(direction, phi))

    phi_only = brentq(lambda p: ci("ground", "diag", p, 1.0) - 3.87, 45.0, 60.0)
    h_only = brentq(lambda hh: ci("pivot", "right", 45.0, hh) - 7.10, 0.05, 1.0)
    res = least_squares(lambda x: [ci("ground", "diag", x[0], x[1]) - 3.87,
                                   ci("pivot", "right", x[0], x[1]) - 7.10,
                                   ci("pivot", "diag", x[0], x[1]) - 10.48], x0=[46.8, 0.29])
    return dict(phi_deg=float(res.x[0]), h=float(res.x[1]), residuals=res.fun.tolist(),
                phi_from_387=phi_only, h_from_710=h_only,
                ground_diag_at_45=ci("ground", "diag", 45.0, 1.0))


# ----------------------------------------------------------------------------------------------
# 10. Block tilting, 2019 Sec. V (q in R^10 with a quaternion, v in R^9)
# ----------------------------------------------------------------------------------------------
# Frames follow the authors' block_tilting_control.m: the object frame O sits at the midpoint
# of the rotation edge (on the table), the block occupies x_O in [-L, 0], z_O in [0, L], and it
# tilts about +y (its top moves toward +x). The hand is a point on the top face at
# x_O = hand_x * L (the code's default is 0.4 L - 0.5 L = -0.1 L). Quaternions are [w, x, y, z].
# Two corrections to the printed example, both documented in errata/lab.md:
#  * eq. 33 writes the hand constraint as R p_hc + p - p_H, so lambda_hc is the force ON THE
#    OBJECT, which points along -z_O; eq. 35-36 as printed then demand a pulling hand. The guard
#    rows below use the pressing normal -z_O (the authors' code instead flips the sign of Phi_hc).
#  * eq. 34 gives the object's gravity wrench zero torque. About the frame O on the rotation edge
#    the weight has torque com_O x (R^T m g); it is included here (the code also omits it).


@dataclass
class BlockTilt:
    """Parameters of the block-tilting example. Defaults: the authors' block_tilting_control.m
    (0.5 kg block of 75 mm, 0.3 kg hand, mu = 0.8, 10 N minimum normal force, 0.5 rad/s), with
    the eight-sided cones and the normal bounds on all three contacts of the paper (eq. 35-36).
    Set n_min_table = 0 to reproduce the code, which bounds only the hand's normal force."""

    L: float = 0.075
    mass: float = 0.5
    hand_mass: float = 0.3
    g: float = 9.8
    mu_hand: float = 0.8
    mu_table: float = 0.8
    n_min_hand: float = 10.0
    n_min_table: float = 10.0
    cone_sides: int = 8
    hand_x: float = -0.1
    omega_goal: float = 0.5
    edge: Optional[np.ndarray] = None  # world position of O (rotation-edge midpoint)

    def __post_init__(self):
        if self.edge is None:
            self.edge = np.array([self.L / 2, 0.0, 0.0])
        self.edge = np.asarray(self.edge, float)

    @property
    def p_hc(self):  # hand contact in the object frame
        return np.array([self.hand_x * self.L, 0.0, self.L])

    @property
    def p_tc(self):  # the two table contacts (ends of the rotation edge) in the object frame
        return np.array([[0.0, self.L / 2, 0.0], [0.0, -self.L / 2, 0.0]])

    @property
    def com(self):  # centre of mass in the object frame
        return np.array([-self.L / 2, 0.0, self.L / 2])


def quat_to_R(q):
    """Rotation matrix of a quaternion [w, x, y, z] (homogeneous form, exact for unit q)."""
    w, x, y, z = q
    return np.array([
        [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
    ])


def quat_mul(a, b):
    """Hamilton product a b of quaternions [w, x, y, z]."""
    aw, av = a[0], np.asarray(a[1:])
    bw, bv = b[0], np.asarray(b[1:])
    return np.concatenate([[aw * bw - av @ bv], aw * bv + bw * av + np.cross(av, bv)])


def quat_integrate(q, omega_body, dt):
    """q(t + dt) = q(t) exp(omega_body dt / 2): rotation by a body angular velocity."""
    th = np.linalg.norm(omega_body) * dt
    if th == 0.0:
        return np.asarray(q, float).copy()
    axis = np.asarray(omega_body) / np.linalg.norm(omega_body)
    dq = np.concatenate([[math.cos(th / 2)], math.sin(th / 2) * axis])
    return quat_mul(q, dq)


def dRp_dq(q, p):
    """d(R(q) p)/dq, 3 x 4, from R(q) p = (w^2 - u.u) p + 2 (u.p) u + 2 w (u x p), q = [w; u]."""
    w, u = q[0], np.asarray(q[1:])
    p = np.asarray(p, float)
    d_w = 2 * w * p + 2 * np.cross(u, p)
    d_u = 2 * (u @ p) * np.eye(3) + 2 * np.outer(u, p) - 2 * np.outer(p, u) - 2 * w * skew(p)
    return np.hstack([d_w[:, None], d_u])


def E_quat(q):
    """2019 eq. 27: qdot = E(q) omega_body, 4 x 3."""
    w, x, y, z = q
    return 0.5 * np.array([[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]])


def block_tilt_omega(q):
    """2019 eq. 26: Omega(q) = diag(R, E(q), I_3), 10 x 9, so qdot = Omega v."""
    R = quat_to_R(q[3:7])
    return block_diag(R, E_quat(q[3:7]), np.eye(3))


def block_tilt_state(theta, bt: BlockTilt):
    """q = [p_O; q_O; p_H] with the block tilted by theta (rad) about +y through the edge."""
    qO = np.array([math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0])
    pH = bt.edge + quat_to_R(qO) @ bt.p_hc
    return np.concatenate([bt.edge, qO, pH])


def block_tilt_phi(q, bt: BlockTilt, p_tc_world=None):
    """2019 eq. 33: Phi(q) = [R p_hc + p_O - p_H; R p_tc,1 + p_O - W p_tc,1; ...] in R^9.

    p_tc_world (2 x 3): the fixed world points of the table contacts (default: their positions
    at theta = 0)."""
    pO, qO, pH = q[:3], q[3:7], q[7:]
    R = quat_to_R(qO)
    if p_tc_world is None:
        p_tc_world = bt.edge + bt.p_tc
    rows = [R @ bt.p_hc + pO - pH]
    rows += [R @ bt.p_tc[i] + pO - p_tc_world[i] for i in range(2)]
    return np.concatenate(rows)


def block_tilt_jac(q, bt: BlockTilt):
    """d Phi / d q, 9 x 10, derived by hand from eq. 33 (columns p_O, q_O, p_H)."""
    qO = q[3:7]
    Jp = np.zeros((9, 10))
    for k, p in enumerate([bt.p_hc, bt.p_tc[0], bt.p_tc[1]]):
        Jp[3 * k:3 * k + 3, 0:3] = np.eye(3)
        Jp[3 * k:3 * k + 3, 3:7] = dRp_dq(qO, p)
    Jp[0:3, 7:10] = -np.eye(3)
    return Jp


def block_tilt_problem(theta=0.0, bt: Optional[BlockTilt] = None, q=None):
    """The HFVC problem of 2019 Sec. V at tilt angle theta (or at configuration q).

    v = [body twist of O (v_b, omega_b); hand velocity], lambda = [lam_hc; lam_tc1; lam_tc2] in
    world frame (forces on the object), f = [body wrench on O; hand force]."""
    bt = bt or BlockTilt()
    q = block_tilt_state(theta, bt) if q is None else np.asarray(q, float)
    R = quat_to_R(q[3:7])
    pO = q[:3]
    N = block_tilt_jac(q, bt) @ block_tilt_omega(q)  # N = J_Phi Omega (2019 Sec. IV-A)
    # goal, eq. 29-30: spatial twist about the edge line, mapped to the body twist
    w_g = np.array([0.0, 1.0, 0.0])
    xi_s = np.concatenate([-np.cross(w_g, bt.edge), w_g]) * bt.omega_goal
    Ad_inv = np.block([[R.T, -R.T @ skew(pO)], [np.zeros((3, 3)), R.T]])
    G = np.hstack([np.eye(6), np.zeros((6, 3))])
    b_G = Ad_inv @ xi_s
    # external force, eq. 34 plus the gravity torque about O
    gW = np.array([0.0, 0.0, -bt.g])
    fO = R.T @ (bt.mass * gW)
    F = np.concatenate([fO, np.cross(bt.com, fO), bt.hand_mass * gW])
    # guard conditions, eq. 35-36 (hand rows use the pressing normal -z_O)
    m, n = 9, 9
    z = np.array([0.0, 0.0, 1.0])
    D = cone_directions(bt.cone_sides)
    Lam, bL = [], []
    for d in D:
        dd = np.array([d[0], d[1], 0.0])
        row = np.zeros(m + n)
        row[0:3] = (dd + bt.mu_hand * z) @ R.T  # d^T lam_O <= mu (-z^T lam_O)
        Lam.append(row)
        bL.append(0.0)
        for k in (1, 2):
            row = np.zeros(m + n)
            row[3 * k:3 * k + 3] = dd - bt.mu_table * z  # d^T lam <= mu z^T lam
            Lam.append(row)
            bL.append(0.0)
    row = np.zeros(m + n)
    row[0:3] = z @ R.T  # -z^T R^T lam_hc >= n_min_hand
    Lam.append(row)
    bL.append(-bt.n_min_hand)
    for k in (1, 2):
        if bt.n_min_table > 0:
            row = np.zeros(m + n)
            row[3 * k:3 * k + 3] = -z
            Lam.append(row)
            bL.append(-bt.n_min_table)
    return HFVCProblem(N, 6, G, b_G, F, None, np.array(Lam), np.array(bL), name="block-tilting",
                       info=dict(q=q, R=R, bt=bt))


def world_commands(vel: VelocityControl, force: Optional[ForceControl], n_u):
    """World-frame (actuated-coordinate) velocity and force commands, as the authors' code
    prints them: R_a^-1 [0; w_av] and R_a^-1 [eta_af; 0]."""
    Rinv = np.linalg.inv(vel.R_a)
    v = Rinv @ np.concatenate([np.zeros(vel.n_af), vel.w_av])
    f = Rinv @ np.concatenate([force.eta_af if force is not None else np.zeros(vel.n_af),
                               np.zeros(vel.n_av)])
    return v, f


# ----------------------------------------------------------------------------------------------
# 11. Random test problems after 2021 Table I
# ----------------------------------------------------------------------------------------------
# 2021 Sec. VI-B: one rigid object, one to three environment contacts ('f' sticking, 's'
# sliding), one to three rigid fingers with one to three sticking contacts each; 1000 samples
# of contact locations and normals per setting (6000 planar, 72000 3D) and a random goal. The
# paper does not give the distributions; the choices here are:
#   object: reference point at the origin, contact points uniform in [-0.5, 0.5]^dim;
#   environment normals uniform on the upper half-sphere (the environment pushes from below);
#   finger normals uniform on the sphere; finger reference point = mean of its contact points
#   plus a uniform offset in [-0.2, 0.2]^dim; mu_env ~ U[0.2, 1.0], mu_finger ~ U[0.5, 1.2];
#   object weight 1 along -y (2D) or -z (3D); minimum normal force 0.1 at finger contacts and 0
#   at environment contacts; sliding contacts are frictionless (2021 eq. 4 has no equality
#   rows, so it does not model sliding friction);
#   goal: n_G rows (uniform in 1..rows(U_bar) for goal='min', all of them for goal='max')
#   drawn from the projection of ROW(U_bar) onto NULL(J), plus a random ROW(J) component that
#   changes no solution set, and b_G = G v_g for a random v_g in NULL(J). Every goal is
#   therefore feasible for the velocity part, and failures come from the force part.

TABLE1 = {
    2: dict(env=("f", "s", "ss"), hand=(1, 2), fingers=(1,)),
    3: dict(env=("f", "s", "ff", "fs", "ss", "ffs", "fss", "sss"), hand=(1, 2, 3), fingers=(1, 2, 3)),
}


def table1_settings(dim):
    """All (env, contacts per finger, fingers) settings of 2021 Table I for dim = 2 or 3."""
    t = TABLE1[dim]
    return [(e, h, k) for e in t["env"] for h in t["hand"] for k in t["fingers"]]


def _unit(rng, dim, upper=False):
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    if upper and v[-1] < 0:
        v[-1] = -v[-1]
    return v


def random_problem(rng=None, dim=2, env="f", hand=1, fingers=1, goal="min", weight=1.0,
                   n_min_finger=0.1, cone_sides=8, max_tries=200):
    """One random HFVC problem of 2021 Table I (see the comment above for the distributions)."""
    rng = np.random.default_rng(rng)
    for _ in range(max_tries):
        bodies = [Body(np.zeros(dim))]
        contacts = []
        for mode in env:
            contacts.append(Contact(rng.uniform(-0.5, 0.5, dim), _unit(rng, dim, upper=True), b=0,
                                    a=None, mode="stick" if mode == "f" else "slide",
                                    mu=rng.uniform(0.2, 1.0), n_min=0.0))
        for k in range(fingers):
            pts = [rng.uniform(-0.5, 0.5, dim) for _ in range(hand)]
            bodies.append(Body(np.mean(pts, axis=0) + rng.uniform(-0.2, 0.2, dim)))
            for p in pts:
                contacts.append(Contact(p, _unit(rng, dim), b=0, a=1 + k, mode="stick",
                                        mu=rng.uniform(0.5, 1.2), n_min=n_min_finger))
        n_u = 3 if dim == 2 else 6
        base = contact_system(dim, bodies, contacts, n_u, None, np.zeros((0, 1)), np.zeros(0),
                              cone_sides)
        J, n = base.J, base.n
        U = null_rows(J, n)
        US = U.copy()
        US[:, :n_u] = 0.0
        Ubar = row_rows(US)
        if Ubar.shape[0] == 0:
            continue
        W0 = row_rows(Ubar @ U.T @ U)  # projection of ROW(U_bar) onto NULL(J)
        nG = int(rng.integers(1, W0.shape[0] + 1)) if goal == "min" else W0.shape[0]
        G = rng.standard_normal((nG, W0.shape[0])) @ W0
        if J.shape[0]:
            G = G + rng.standard_normal((nG, J.shape[0])) @ J
        v_g = U.T @ rng.standard_normal(U.shape[0])
        F = np.zeros(n)
        F[dim - 1] = -weight
        return contact_system(dim, bodies, contacts, n_u, F, G, G @ v_g, cone_sides,
                              name=f"{dim}d-{env}-{hand}x{fingers}-{goal}",
                              info=dict(env=env, hand=hand, fingers=fingers, goal=goal))
    raise RuntimeError("no controllable random problem found")


def random_problems(count, dim=2, goal="min", seed=0, settings=None):
    """`count` problems cycling through the Table I settings, reproducible from `seed`."""
    rng = np.random.default_rng(seed)
    settings = settings or table1_settings(dim)
    return [random_problem(rng, dim, *settings[i % len(settings)], goal=goal) for i in range(count)]


# ----------------------------------------------------------------------------------------------
# 12. Solver wrappers and timing
# ----------------------------------------------------------------------------------------------


@dataclass
class HFVCSolution:
    method: str
    ok: bool
    message: str = ""
    vel: Optional[VelocityControl] = None
    force: Optional[ForceControl] = None
    crash: float = math.inf
    t_vel: float = math.nan  # seconds
    t_force: float = math.nan

    def summary(self):
        return dict(method=self.method, ok=self.ok, message=self.message, crash=self.crash,
                    n_av=self.vel.n_av if self.vel else None, t_vel_ms=1e3 * self.t_vel,
                    t_force_ms=1e3 * self.t_force)


def _solve(prob, method, vel_fn, force_fn):
    t0 = time.perf_counter()
    try:
        vel = vel_fn(prob)
    except Infeasible as e:
        return HFVCSolution(method, False, f"velocity: {e}", t_vel=time.perf_counter() - t0)
    t1 = time.perf_counter()
    crash = crashing_index(prob.J, vel.C) if vel.n_av else math.nan
    try:
        force = force_fn(prob, vel)
    except Infeasible as e:
        return HFVCSolution(method, False, f"force: {e}", vel, None, crash, t1 - t0,
                            time.perf_counter() - t1)
    return HFVCSolution(method, True, "", vel, force, crash, t1 - t0, time.perf_counter() - t1)


def solve_ochs(prob: HFVCProblem, maximal=False):
    """OCHS (2021 Algorithm 1): closed-form velocity part, then the force QP of eq. 30."""
    return _solve(prob, "OCHS(M)" if maximal else "OCHS",
                  lambda p: ochs_velocity(p, maximal), ochs_force)


def solve_hs(prob: HFVCProblem, Ns=3, rng=None, form="paper", t=10.0, iters=50, objective="l1"):
    """The 2019 method (2021 calls it HS3 / HS10 by Ns): Algorithm 1, then Algorithm 2."""
    return _solve(prob, f"HS{Ns}", lambda p: alg1_velocity(p, Ns, t, iters, form, rng),
                  lambda p, v: alg2_force(p, v, objective))


@contextmanager
def stopwatch():
    """with stopwatch() as sw: ...; then sw['s'] holds the elapsed seconds."""
    sw = {}
    t0 = time.perf_counter()
    try:
        yield sw
    finally:
        sw["s"] = time.perf_counter() - t0


def time_call(fn, *args, repeat=1, **kwargs):
    """(result of the last call, fastest wall time in seconds over `repeat` calls)."""
    best, out = math.inf, None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return out, best


def table2(problems, solvers, ill=100.0):
    """Table II-style summary: for each named solver (name -> fn(prob) -> HFVCSolution) the
    number solved, mean crashing index over solved problems, the number with crashing index
    above `ill` (our threshold; 2021 does not state one), and mean / worst velocity and force
    times in ms. Also returns the per-problem solutions."""
    per = {name: [fn(p) for p in problems] for name, fn in solvers.items()}
    rows = {}
    for name, sols in per.items():
        ok = [s for s in sols if s.ok]
        cr = np.array([s.crash for s in ok if s.vel.n_av > 0])
        tv = np.array([s.t_vel for s in sols]) * 1e3
        tf = np.array([s.t_force for s in ok]) * 1e3
        rows[name] = dict(total=len(sols), solved=len(ok),
                          mean_crash=float(np.mean(cr[np.isfinite(cr)])) if cr.size else math.nan,
                          ill=int(np.sum(cr > ill)), t_vel_mean=float(np.mean(tv)),
                          t_vel_worst=float(np.max(tv)),
                          t_force_mean=float(np.mean(tf)) if tf.size else math.nan,
                          t_force_worst=float(np.max(tf)) if tf.size else math.nan)
    return rows, per
