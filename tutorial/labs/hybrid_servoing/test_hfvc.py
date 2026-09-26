"""Tests for hfvc.py (run from this directory):

    uv run --no-project --with pytest --with numpy --with scipy --with mujoco==3.14.0 pytest -q

The random-problem tests solve 300 problems with OCHS, Algorithm 1 + 2 with 3 and with 10
restarts; the whole file takes about 15 s. The MuJoCo test is skipped when mujoco is missing.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hfvc  # noqa: E402

RNG = np.random.default_rng(20260925)


# ----------------------------------------------------------------------------------------------
# Linear algebra helpers
# ----------------------------------------------------------------------------------------------


def test_null_and_row_bases_are_orthonormal_complements():
    for _ in range(20):
        m, n = RNG.integers(1, 6), RNG.integers(2, 8)
        r = RNG.integers(1, min(m, n) + 1)
        A = RNG.standard_normal((m, r)) @ RNG.standard_normal((r, n))
        Z, W = hfvc.null_rows(A), hfvc.row_rows(A)
        assert Z.shape == (n - r, n) and W.shape == (r, n)
        B = np.vstack([Z, W])
        assert np.allclose(B @ B.T, np.eye(n), atol=1e-10)
        assert np.allclose(A @ Z.T, 0, atol=1e-10)


def test_rank_of_numerically_zero_product_is_zero():
    # the case that dropped a row of K in eq. 24 before ATOL existed
    Z = hfvc.null_rows(RNG.standard_normal((3, 5)))
    W = hfvc.row_rows(RNG.standard_normal((3, 5)))
    P = Z @ W.T @ np.zeros((3, 3)) + 1e-17 * RNG.standard_normal((2, 3))
    assert hfvc.svd_rank(P) == 0
    assert hfvc.null_rows(P, 3).shape == (3, 3)


def test_least_distance_matches_slsqp():
    for _ in range(15):
        n, mi, me = 4, 5, 1
        A, E = RNG.standard_normal((mi, n)), RNG.standard_normal((me, n))
        x_feas = RNG.standard_normal(n)
        b = A @ x_feas + RNG.uniform(0.0, 1.0, mi) - 1.0 * (RNG.uniform(size=mi) < 0.5)
        b = np.maximum(b, A @ x_feas)  # x_feas stays feasible
        e = E @ x_feas
        x, ok = hfvc.least_distance(A, b, E, e)
        assert ok
        ref = minimize(lambda z: z @ z, x_feas, jac=lambda z: 2 * z, method="SLSQP",
                       constraints=[dict(type="ineq", fun=lambda z: b - A @ z, jac=lambda z: -A),
                                    dict(type="eq", fun=lambda z: E @ z - e, jac=lambda z: E)],
                       options=dict(ftol=1e-14, maxiter=500))
        assert np.all(A @ x <= b + 1e-8) and np.allclose(E @ x, e, atol=1e-9)
        assert x @ x <= ref.fun + 1e-7


def test_least_distance_detects_infeasibility():
    A = np.array([[1.0, 0.0], [-1.0, 0.0]])
    b = np.array([-1.0, -1.0])  # x <= -1 and x >= 1
    _, ok = hfvc.least_distance(A, b)
    assert not ok


# ----------------------------------------------------------------------------------------------
# Crashing index: 2021 Fig. 1, the drawer of deck 02, invariances
# ----------------------------------------------------------------------------------------------


def test_fig1_crashing_indexes_reproduced():
    """The six printed values 2.41, 3.87, inf, 7.10, 10.48, inf, to their two decimals.

    Reconstruction (hfvc.py section 9): block + point finger sticking at the top centre;
    top row slides on two frictionless-normal corner contacts, bottom row sticks on a pivot;
    V to the right, V diagonal at 46.78 deg below horizontal, V down; block height 0.2904."""
    for row in hfvc.fig1_table():
        if math.isinf(row["paper"]):
            assert math.isinf(row["ours"])
        else:
            assert round(row["ours"], 2) == pytest.approx(row["paper"], abs=1e-9), row


def test_fig1_top_row_is_geometry_free():
    for h in (0.05, 0.29, 1.0, 3.0):
        for w in (0.1, 1.0):
            J = hfvc.fig1_problem("ground", h=h, w=w).J
            assert hfvc.crashing_index(J, hfvc.fig1_C("right")) == pytest.approx(1 + math.sqrt(2))
            assert hfvc.crashing_index(J, hfvc.fig1_C("diag", 45.0)) == pytest.approx(2 + math.sqrt(3))
            assert math.isinf(hfvc.crashing_index(J, hfvc.fig1_C("down")))


def test_fig1_fit_parameters():
    fit = hfvc.fig1_fit()
    assert fit["phi_deg"] == pytest.approx(hfvc.FIG1_PHI_DEG, abs=0.01)
    assert fit["h"] == pytest.approx(hfvc.FIG1_H, abs=1e-3)
    assert max(abs(r) for r in fit["residuals"]) < 0.002
    assert fit["phi_from_387"] == pytest.approx(46.755, abs=0.01)


def test_drawer_matches_deck_02():
    for th, cond in [(0, 1.0), (45, 2.414), (60, 3.732), (80, 11.43), (85, 22.90), (89, 114.6)]:
        assert hfvc.drawer_system(th)["cond"] == pytest.approx(cond, rel=1e-3)
        assert hfvc.drawer_system(th)["cond"] == pytest.approx(math.tan(math.radians(45 + th / 2)))
    d = hfvc.drawer_system(85, 2.0)
    assert d["speed_model"] == pytest.approx(11.47, abs=0.01)
    assert d["speed_true"] == pytest.approx(8.21, abs=0.01)
    assert hfvc.drawer_system(85, -2.0)["speed_true"] == pytest.approx(19.11, abs=0.01)
    assert not hfvc.drawer_system(88, -2.0)["feasible"]


def test_drawer_crashing_index_formula():
    for th in (0.0, 30.0, 60.0, 85.0):
        c = math.sqrt((1 + math.sin(math.radians(th)) ** 2) / 2)
        assert hfvc.drawer_crashing_index(th) == pytest.approx(math.sqrt((1 + c) / (1 - c)))


def test_crashing_index_ignores_row_scaling_and_redundant_rows():
    J = hfvc.fig1_problem("pivot").J
    C = hfvc.fig1_C("diag")
    ref = hfvc.crashing_index(J, C)
    assert hfvc.crashing_index(np.vstack([J, 3 * J[:2]]), 1000 * C) == pytest.approx(ref)
    assert hfvc.crashing_index(np.diag([1, 10, 0.1, 5, 2.0]) @ J, C) == pytest.approx(ref)
    # the raw condition number is not invariant: artificial ill-conditioning
    assert hfvc.cond_rows(np.vstack([J, 1000 * C])) > 10 * ref


# ----------------------------------------------------------------------------------------------
# Block tilting (2019 Sec. V): Omega, E(q), Phi, the Jacobian, the solution
# ----------------------------------------------------------------------------------------------


def _random_q(bt):
    qO = RNG.standard_normal(4)
    qO /= np.linalg.norm(qO)
    return np.concatenate([RNG.uniform(-0.1, 0.1, 3), qO, RNG.uniform(-0.1, 0.1, 3)])


def fd_jacobian(fun, x, eps=1e-6):
    f0 = fun(x)
    Jn = np.zeros((f0.size, x.size))
    for k in range(x.size):
        dx = np.zeros_like(x)
        dx[k] = eps
        Jn[:, k] = (fun(x + dx) - fun(x - dx)) / (2 * eps)
    return Jn


@pytest.mark.parametrize("trial", range(6))
def test_phi_jacobian_matches_finite_differences(trial):
    bt = hfvc.BlockTilt()
    q = hfvc.block_tilt_state(0.3 * trial, bt) if trial < 3 else _random_q(bt)
    Jq = hfvc.block_tilt_jac(q, bt)
    Jn = fd_jacobian(lambda x: hfvc.block_tilt_phi(x, bt), q)
    assert np.max(np.abs(Jq - Jn)) < 1e-8


def test_rotation_derivative_matches_finite_differences():
    for _ in range(5):
        q = RNG.standard_normal(4)
        q /= np.linalg.norm(q)
        p = RNG.standard_normal(3)
        Jn = fd_jacobian(lambda x: hfvc.quat_to_R(x) @ p, q)
        assert np.allclose(hfvc.dRp_dq(q, p), Jn, atol=1e-8)


def test_E_maps_body_angular_velocity_to_quaternion_rate():
    for _ in range(5):
        q = RNG.standard_normal(4)
        q /= np.linalg.norm(q)
        w = RNG.standard_normal(3)
        h = 1e-6
        qdot = (hfvc.quat_integrate(q, w, h) - hfvc.quat_integrate(q, w, -h)) / (2 * h)
        assert np.allclose(hfvc.E_quat(q) @ w, qdot, atol=1e-8)


def test_N_is_the_rate_of_phi_along_a_motion():
    """N v = d/dt Phi(q(t)) when q moves with generalized velocity v (qdot = Omega v)."""
    bt = hfvc.BlockTilt()
    q = _random_q(bt)
    N = hfvc.block_tilt_jac(q, bt) @ hfvc.block_tilt_omega(q)
    v = RNG.standard_normal(9)
    h = 1e-6

    def move(q, s):
        R = hfvc.quat_to_R(q[3:7])
        return np.concatenate([q[:3] + s * R @ v[:3], hfvc.quat_integrate(q[3:7], v[3:6], s),
                               q[7:] + s * v[6:]])

    rate = (hfvc.block_tilt_phi(move(q, h), bt) - hfvc.block_tilt_phi(move(q, -h), bt)) / (2 * h)
    assert np.allclose(N @ v, rate, atol=1e-7)


@pytest.mark.parametrize("deg", [0, 15, 30, 45])
def test_block_tilting_solution(deg):
    bt = hfvc.BlockTilt()
    prob = hfvc.block_tilt_problem(math.radians(deg), bt)
    assert hfvc.svd_rank(prob.J) == 8  # one free motion: rotation about the edge
    for sol in (hfvc.solve_ochs(prob), hfvc.solve_hs(prob, 3, rng=deg)):
        assert sol.ok, sol.message
        vel, force = sol.vel, sol.force
        assert (vel.n_av, vel.n_af) == (1, 2)
        R = prob.info["R"]
        r = R @ bt.p_hc  # hand relative to the rotation edge, world frame
        v_world, f_world = hfvc.world_commands(vel, force, 6)
        # the velocity command moves the hand along the arc: omega_goal (y x r)
        assert np.allclose(v_world, bt.omega_goal * np.cross([0, 1, 0], r), atol=1e-9)
        # the force command has no y part and presses along -r (toward the rotation axis)
        assert abs(f_world[1]) < 1e-8
        assert np.linalg.norm(np.cross(f_world, r)) < 1e-6 * np.linalg.norm(f_world)
        assert f_world @ r < 0
        assert hfvc.guard_violation(prob, force.lam, force.f) < 1e-7
        assert hfvc.newton_residual(prob, force.lam, force.f) < 1e-9
    assert hfvc.solve_ochs(prob).crash == pytest.approx(hfvc.solve_hs(prob, 3, rng=1).crash, rel=1e-6)


def test_block_tilting_crashing_index_depends_on_units():
    """The crashing index mixes metres and radians: the hand's share of the free motion is
    |r| / sqrt(1 + |r|^2), so a 75 mm cube gives 26.6 and a 1 m cube gives a smaller value."""
    small = hfvc.solve_ochs(hfvc.block_tilt_problem(0.2, hfvc.BlockTilt())).crash
    big = hfvc.solve_ochs(hfvc.block_tilt_problem(0.2, hfvc.BlockTilt(L=1.0, n_min_hand=1.0,
                                                                     n_min_table=0.0))).crash
    assert small == pytest.approx(26.57, abs=0.01)
    assert big < 5


# ----------------------------------------------------------------------------------------------
# Random problems (2021 Table I): goal inclusion, guards, OCHS vs Algorithm 1
# ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def solved():
    probs = (hfvc.random_problems(150, dim=2, seed=11) + hfvc.random_problems(150, dim=3, seed=12))
    out = []
    for i, p in enumerate(probs):
        out.append((p, hfvc.solve_ochs(p), hfvc.solve_ochs(p, maximal=True),
                    hfvc.solve_hs(p, 3, rng=i), hfvc.solve_hs(p, 10, rng=i)))
    return out


def test_random_goals_are_feasible(solved):
    for p, *_ in solved:
        _, res = hfvc.special_solution(np.vstack([p.J, p.G]),
                                       np.concatenate([np.zeros(p.J.shape[0]), p.b_G]))
        assert res < 1e-9


def test_goal_inclusion_holds_for_every_returned_solution(solved):
    n = 0
    for p, *sols in solved:
        for s in sols:
            if s.vel is None:
                continue
            gi = hfvc.goal_inclusion(p.J, p.G, p.b_G, s.vel.C, s.vel.w_av)
            assert gi.ok, (p.name, s.method)
            assert gi.sampled_goal_error < 1e-7 * (1 + np.abs(p.b_G).max())
            n += 1
    assert n > 900


def test_guard_conditions_hold_for_every_returned_solution(solved):
    for p, *sols in solved:
        for s in sols:
            if not s.ok:
                continue
            assert hfvc.guard_violation(p, s.force.lam, s.force.f) < 1e-6, (p.name, s.method)
            assert hfvc.newton_residual(p, s.force.lam, s.force.f) < 1e-7, (p.name, s.method)
            f_u = s.force.f[:p.n_u]
            assert np.allclose(f_u, 0, atol=1e-8)


def test_ochs_solves_every_feasible_random_problem(solved):
    assert all(o.ok for _, o, *_ in solved)


def test_ochs_never_worse_than_algorithm_1(solved):
    compared = 0
    for p, o, m, h3, h10 in solved:
        for h in (h3, h10):
            if o.vel is None or h.vel is None or o.vel.n_av == 0:
                continue
            assert o.crash <= h.crash * (1 + 1e-6) + 1e-9, (p.name, h.method, o.crash, h.crash)
            compared += 1
    assert compared > 300


def test_minimal_and_maximal_ochs_agree_when_dimensions_match():
    probs = hfvc.random_problems(60, dim=3, seed=5, goal="max")
    for p in probs:
        o, m = hfvc.solve_ochs(p), hfvc.solve_ochs(p, maximal=True)
        assert o.vel.n_av == m.vel.n_av
        assert o.crash == pytest.approx(m.crash, rel=1e-9)


def test_algorithm1_rejects_underactuated_problems():
    """eq. 14 (r_N + n_a >= n) fails when the robot cannot fix every object DOF; OCHS still
    solves such problems when the goal allows it (2021 Sec. V-D)."""
    probs = hfvc.random_problems(40, dim=3, seed=12, settings=[("ss", 1, 1)])
    under = [p for p in probs if hfvc.svd_rank(p.J) < p.n_u]
    assert under
    for p in under:
        with pytest.raises(hfvc.Infeasible, match="eq. 14"):
            hfvc.alg1_velocity(p)
        assert hfvc.solve_ochs(p).ok


# ----------------------------------------------------------------------------------------------
# Algorithm 1 and 2 internals
# ----------------------------------------------------------------------------------------------


def test_alg1_gradient_matches_finite_differences():
    for form in ("paper", "squared"):
        n_c, n_av = 5, 3
        A = RNG.standard_normal((n_c, n_c))
        M = A @ A.T / n_c
        K = RNG.standard_normal((n_c, n_av))
        g = hfvc.alg1_grad(K, M, form)
        gn = fd_jacobian(lambda x: np.array([hfvc.alg1_cost(x.reshape(n_c, n_av), M, form)]),
                         K.ravel()).reshape(n_c, n_av)
        assert np.allclose(g, gn, atol=1e-6)


def test_alg2_kkt_equals_minimum_norm_solution():
    p = hfvc.block_tilt_problem(0.3)
    vel = hfvc.alg1_velocity(p, 3, rng=0)
    force = hfvc.alg2_force(p, vel)
    ff_kkt, _ = hfvc.alg2_kkt(p, vel, force.eta_af)
    x = np.concatenate([force.lam, force.eta])
    _, _, _, i_free, _, _ = hfvc.alg2_newton_system(p, vel)
    assert np.allclose(ff_kkt, x[i_free], atol=1e-9)


def test_cone_directions_circumscribe_the_circle():
    D = hfvc.cone_directions(8)
    assert np.allclose(np.linalg.norm(D, axis=1), 1)
    ang = np.linspace(0, 2 * np.pi, 721)
    circle = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    assert np.all(circle @ D.T <= 1 + 1e-12)  # the unit circle lies inside the octagon


# ----------------------------------------------------------------------------------------------
# MuJoCo block tilting (skipped without mujoco)
# ----------------------------------------------------------------------------------------------


def test_nominal_hfvc_trial_succeeds():
    pytest.importorskip("mujoco")
    import sim_block_tilt as sim

    row = sim.run_trial(sim.Trial("nominal", "hfvc"))
    assert row["success"] == 1, row["fail_reason"]
    assert row["peak_finger_N"] < 12.0
    assert abs(row["t_reach_s"] - sim.THETA_GOAL / sim.OMEGA) < 0.1
