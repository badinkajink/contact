# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "mujoco==3.14.0"]
# ///
"""MuJoCo study: block tilting by hybrid force-velocity control, velocity control, force control.

The block-tilting plan of Hou and Mason 2019 Sec. V runs in tutorial/models/block_tilt.xml under
three finger controllers, while the real block size, table height and friction differ from the
controller's model (75 mm cube centred at the origin, table at z = 0, mu = 0.8).

    uv run --script sim_block_tilt.py            # the full study, 99 trials, about a minute
    uv run --script sim_block_tilt.py --quick    # nominal plus three perturbations (12 trials)

Output: results/block_tilt_study.csv, one row per trial, flushed and fsynced as each trial ends
(a rerun skips the trials already in the file; --fresh starts over);
results/block_tilt_timeseries.csv, 250 Hz traces of the nominal and the L = 78 mm trials;
results/block_tilt_study.json, the parameters, thresholds and engine version of the run.

Controllers. All three read the block tilt theta (perfect orientation sensing) and the finger
position and velocity at every 0.5 ms step and write a world-frame finger force; the HFVC is
re-solved at 250 Hz, the rate of the 2019 and 2021 robot experiments. The controller's model
is hfvc.BlockTilt with the nominal geometry, the authors' mu = 0.8 and 10 N minimum hand normal
force, no table normal bound (the authors' code), and the 8-sided cones of the paper.

* hfvc: OCHS (hfvc.solve_ochs) at the measured theta gives R_a, w_av, eta_af and the force eta_av
  the model expects along the velocity axis c. Finger force
      f = R_a^-1 [eta_af; eta_av + K_v (w_av - c.v) + K_p e] - D_f (I - c c^T) v + m_f g,
  e = integral of (w_av - c.v) dt: a stiff velocity servo along c with the position-error term
  of a position-controlled robot, and force control with light damping along the complement.
* velocity: a stiff position and velocity servo on all three axes to the planned finger path,
  the arc about the model's rotation edge through the finger's measured start point, pressed
  preload / K_p = 1 mm into the block:  f = K_p (p_plan - p) + K_d (v_plan - v) + m_f g.
* force: low stiffness on all three axes: the model's full HFVC force plus a weak damper toward
  the model's finger velocity at the measured theta,  f = R_a^-1 eta_a + D_low (v_model - v) + m_f g.

A trial succeeds when theta reaches 40 degrees within 2.5 s (the plan takes 1.40 s) with the
table edge sliding at most 2 mm, the finger never off the block for more than 20 ms, and the
finger normal force never above 25 N (the force at which the 2019 robot stopped its runs).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hfvc  # noqa: E402

try:
    import mujoco
except ImportError:  # the lab and notebook 07 read the CSV without mujoco
    mujoco = None

MODEL_XML = HERE.parents[1] / "models" / "block_tilt.xml"
RESULTS = HERE / "results"
STUDY_CSV = RESULTS / "block_tilt_study.csv"
SERIES_CSV = RESULTS / "block_tilt_timeseries.csv"
META_JSON = RESULTS / "block_tilt_study.json"

# The controller's model of the task
MODEL_L = 0.075  # m, cube edge
DENSITY = 700.0  # kg/m^3, wood; the model's block mass is 0.295 kg
GRAVITY = 9.81
MU_MODEL = 0.8
HAND_X = -0.1  # finger at x_O = -0.1 L, 0.1 L behind the rotation edge (authors' code)
OMEGA = 0.5  # rad/s, planned tilt rate (authors' code)
THETA_GOAL = math.radians(40.0)
FINGER_MASS = 0.1  # kg, from block_tilt.xml; every controller cancels its weight
FINGER_R = 0.004  # m

# Timing and success thresholds
SETTLE = 0.3  # s of pressing with the preload before the controller starts
T_MAX = 2.5  # s after the start
CONTROL_DT = 0.004  # s, HFVC re-solve period (250 Hz)
FORCE_LIMIT = 25.0  # N
SLIP_LIMIT = 2e-3  # m
GAP_LIMIT = 0.020  # s


@dataclass
class Gains:
    K_v: float = 60.0  # N s/m, hfvc velocity servo along c (with K_p on the integrated error)
    D_f: float = 2.0  # N s/m, hfvc damping on the force-controlled axes
    K_p: float = 1.0e4  # N/m, velocity controller position gain
    K_d: float = 60.0  # N s/m, velocity controller velocity gain
    D_low: float = 6.0  # N s/m, force controller damping (one tenth of K_v)
    preload: float = 10.0  # N, settle force and the velocity controller's press depth * K_p


@dataclass
class Trial:
    sweep: str
    controller: str
    L: float = MODEL_L  # m, real cube edge
    dz: float = 0.0  # m, real table height (the model assumes 0)
    mu: float = MU_MODEL  # real friction on every contact
    series: bool = False  # record a 250 Hz trace

    @property
    def key(self):
        return f"{self.sweep}|{self.controller}|{self.L * 1e3:.2f}|{self.dz * 1e3:.2f}|{self.mu:.3f}"


def controller_model():
    """hfvc.BlockTilt as the controller believes it: nominal cube, table at z = 0."""
    return hfvc.BlockTilt(L=MODEL_L, mass=DENSITY * MODEL_L**3, hand_mass=0.0, g=GRAVITY,
                          mu_hand=MU_MODEL, mu_table=MU_MODEL, n_min_hand=10.0, n_min_table=0.0,
                          cone_sides=8, hand_x=HAND_X, omega_goal=OMEGA)


def build(trial: Trial):
    """Compile block_tilt.xml with the trial's block size, table height and friction."""
    spec = mujoco.MjSpec.from_file(str(MODEL_XML))
    h = trial.L / 2
    spec.geom("block").size = [h, h, h]
    spec.body("block").pos = [0.0, 0.0, trial.dz + h]
    spec.geom("table").pos = [0.0, 0.0, trial.dz]
    # the finger starts where the model puts the hand, x = (0.5 + HAND_X) L_model = 30 mm,
    # 0.5 mm above the real top face
    spec.body("finger").pos = [(0.5 + HAND_X) * MODEL_L, 0.0,
                               trial.dz + trial.L + FINGER_R + 0.0005]
    for name in ("table", "block", "finger"):
        spec.geom(name).friction = [trial.mu, 0.005, 0.0001]
    m = spec.compile()
    d = mujoco.MjData(m)
    ids = dict(block=m.body("block").id, finger_body=m.body("finger").id,
               g_block=m.geom("block").id, g_table=m.geom("table").id,
               g_finger=m.geom("finger").id,
               qf=np.array([m.joint(j).qposadr[0] for j in ("fx", "fy", "fz")]),
               vf=np.array([m.joint(j).dofadr[0] for j in ("fx", "fy", "fz")]))
    return m, d, ids


def tilt_angle(R):
    """Rotation of the block about +y (its z axis tipping toward +x)."""
    return math.atan2(R[0, 2], R[2, 2])


def contact_forces(m, d, ids, buf, vel6):
    """(finger normal, finger tangential, table normal sum, finger touching, finger sliding
    speed) from the active contacts."""
    fn = ft = tn = slide = 0.0
    touching = False
    for i in range(d.ncon):
        con = d.contact[i]
        pair = {con.geom1, con.geom2}
        if ids["g_block"] not in pair:
            continue
        mujoco.mj_contactForce(m, d, i, buf)
        if ids["g_finger"] in pair:
            touching = True
            fn += buf[0]
            ft += math.hypot(buf[1], buf[2])
            # sliding speed: tangential part of (finger velocity - block velocity at the point)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, ids["block"], vel6, 0)
            vb = vel6[3:] + np.cross(vel6[:3], con.pos - d.xpos[ids["block"]])
            vr = d.qvel[ids["vf"]] - vb
            nrm = con.frame[:3]
            slide = max(slide, float(np.linalg.norm(vr - nrm * (nrm @ vr))))
        elif ids["g_table"] in pair:
            tn += buf[0]
    return fn, ft, tn, touching, slide


class HFVCCache:
    """The OCHS solution at the latest measured theta; keeps the last good one on failure."""

    def __init__(self, bt):
        self.bt, self.sol, self.fails, self.t_solve = bt, None, 0, 0.0

    def update(self, theta):
        t0 = time.perf_counter()
        s = hfvc.solve_ochs(hfvc.block_tilt_problem(theta, self.bt))
        self.t_solve += time.perf_counter() - t0
        if s.ok:
            # the sign of C is arbitrary; flip it so that w_av >= 0 and the integral along c
            # keeps its meaning from one solve to the next
            R_a, w_av, eta_av = s.vel.R_a.copy(), float(s.vel.w_av[0]), float(s.force.eta[-1])
            if w_av < 0:
                R_a[-1], w_av, eta_av = -R_a[-1], -w_av, -eta_av
            Rinv = np.linalg.inv(R_a)
            self.sol = dict(Rinv=Rinv, c=R_a[-1], w_av=w_av, eta_af=s.force.eta_af.copy(),
                            eta_av=eta_av, f=s.force.f[6:9].copy(), v=Rinv[:, -1] * w_av)
        else:
            self.fails += 1
        return self.sol


def run_trial(trial: Trial, gains: Gains = Gains(), series_rows=None):
    """Simulate one trial; returns the result row (dict)."""
    wall0 = time.perf_counter()
    m, d, ids = build(trial)
    h = m.opt.timestep
    g_up = np.array([0.0, 0.0, FINGER_MASS * GRAVITY])
    base = m.body_pos[ids["finger_body"]].copy()
    buf, vel6 = np.zeros(6), np.zeros(6)
    bt = controller_model()
    cache = HFVCCache(bt)

    def finger_state():
        return base + d.qpos[ids["qf"]], d.qvel[ids["vf"]].copy()

    # settle: press the preload onto the top face
    for _ in range(int(round(SETTLE / h))):
        _, v = finger_state()
        d.ctrl[:] = np.array([0.0, 0.0, -gains.preload]) - 5.0 * v + g_up
        mujoco.mj_step(m, d)

    R0 = d.xmat[ids["block"]].reshape(3, 3).copy()
    pb0 = d.xpos[ids["block"]].copy()
    half = trial.L / 2
    edge0 = pb0 + R0 @ np.array([half, 0.0, -half])
    p_start, _ = finger_state()
    # velocity controller plan: arc about the MODEL edge through the pressed start point
    e_model = np.array([MODEL_L / 2, 0.0, 0.0])
    p_anchor = p_start - np.array([0.0, 0.0, gains.preload / gains.K_p])

    stats = dict(peak_fn=0.0, sum_fn=0.0, n_fn=0, peak_tn=0.0, slip=0.0, lift=0.0, fslip=0.0,
                 gap=0.0, cur_gap=0.0, peak_ctrl=0.0)
    reached, t_reach, reason = False, math.nan, ""
    steps = int(round(T_MAX / h))
    every = int(round(CONTROL_DT / h))
    theta, s_err = 0.0, 0.0
    for k in range(steps):
        t = k * h
        R = d.xmat[ids["block"]].reshape(3, 3)
        theta = tilt_angle(R)
        p, v = finger_state()
        if trial.controller in ("hfvc", "force") and k % every == 0:
            cache.update(max(theta, 0.0))
        if trial.controller == "hfvc":
            s = cache.sol
            e_v = s["w_av"] - s["c"] @ v
            s_err += e_v * h  # position error along c: the stiff inner loop of a robot
            eta_av = s["eta_av"] + gains.K_v * e_v + gains.K_p * s_err
            f = s["Rinv"] @ np.r_[s["eta_af"], eta_av]
            f = f - gains.D_f * (v - s["c"] * (s["c"] @ v)) + g_up
        elif trial.controller == "velocity":
            th_p = min(OMEGA * t, THETA_GOAL)
            c, sn = math.cos(th_p), math.sin(th_p)
            Ry = np.array([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]])
            p_plan = e_model + Ry @ (p_anchor - e_model)
            v_plan = (np.cross([0.0, OMEGA, 0.0], p_plan - e_model)
                      if OMEGA * t < THETA_GOAL else np.zeros(3))
            f = gains.K_p * (p_plan - p) + gains.K_d * (v_plan - v) + g_up
        elif trial.controller == "force":
            s = cache.sol
            f = s["f"] + gains.D_low * (s["v"] - v) + g_up
        else:
            raise ValueError(trial.controller)
        d.ctrl[:] = f
        stats["peak_ctrl"] = max(stats["peak_ctrl"], float(np.linalg.norm(np.clip(f, -200, 200))))
        mujoco.mj_step(m, d)

        # measurements after the step
        fn, ft, tn, touching, slide = contact_forces(m, d, ids, buf, vel6)
        stats["fslip"] += slide * h
        stats["peak_fn"] = max(stats["peak_fn"], fn)
        stats["peak_tn"] = max(stats["peak_tn"], tn)
        if touching:
            stats["sum_fn"] += fn
            stats["n_fn"] += 1
            stats["cur_gap"] = 0.0
        else:
            stats["cur_gap"] += h
            stats["gap"] = max(stats["gap"], stats["cur_gap"])
        R = d.xmat[ids["block"]].reshape(3, 3)
        pb = d.xpos[ids["block"]]
        edge = pb + R @ np.array([half, 0.0, -half])
        stats["slip"] = max(stats["slip"], float(np.hypot(*(edge - edge0)[:2])))
        stats["lift"] = max(stats["lift"], float(edge[2] - edge0[2]))
        theta = tilt_angle(R)
        if series_rows is not None and k % every == 0:
            series_rows.append(dict(key=trial.key, sweep=trial.sweep, controller=trial.controller,
                                    t=round(t + h, 4), theta_deg=math.degrees(theta),
                                    finger_normal_N=fn, finger_tangential_N=ft, table_normal_N=tn,
                                    table_slip_mm=1e3 * float(np.hypot(*(edge - edge0)[:2])),
                                    finger_slide_mm=1e3 * stats["fslip"]))
        if theta >= THETA_GOAL:
            reached, t_reach = True, t + h
            break
        if theta > math.radians(70):
            reason = "block fell over"
            break
        if stats["cur_gap"] > 0.25:
            reason = "finger lost the block"
            break
    fails = []
    if not reached:
        fails.append(reason or "did not reach 40 deg in 2.5 s")
    if stats["slip"] > SLIP_LIMIT:
        fails.append("table slip")
    if stats["gap"] > GAP_LIMIT:
        fails.append("finger contact lost")
    if stats["peak_fn"] > FORCE_LIMIT:
        fails.append("finger force above 25 N")
    return dict(
        key=trial.key, sweep=trial.sweep, controller=trial.controller,
        L_mm=round(trial.L * 1e3, 3), dz_mm=round(trial.dz * 1e3, 3), mu=round(trial.mu, 4),
        success=int(not fails), reached=int(reached),
        t_reach_s=round(t_reach, 4) if reached else "",
        theta_end_deg=round(math.degrees(theta), 3),
        peak_finger_N=round(stats["peak_fn"], 3),
        mean_finger_N=round(stats["sum_fn"] / max(stats["n_fn"], 1), 3),
        peak_table_N=round(stats["peak_tn"], 3),
        max_table_slip_mm=round(1e3 * stats["slip"], 4),
        max_edge_lift_mm=round(1e3 * stats["lift"], 4),
        finger_slide_mm=round(1e3 * stats["fslip"], 4),
        max_contact_gap_ms=round(1e3 * stats["gap"], 2),
        peak_command_N=round(stats["peak_ctrl"], 3),
        hfvc_solve_failures=cache.fails,
        hfvc_solve_ms=round(1e3 * cache.t_solve, 2),
        fail_reason="; ".join(fails),
        wall_s=round(time.perf_counter() - wall0, 3),
    )


CONTROLLERS = ("hfvc", "velocity", "force")


def study_trials(quick=False, n_random=20, seed=2026):
    """The trial list: nominal, one-factor sweeps, and random combinations (seeded)."""
    out = []

    def add(sweep, series=False, **kw):
        for c in CONTROLLERS:
            out.append(Trial(sweep, c, series=series, **kw))

    add("nominal", series=True)
    if quick:
        add("size", L=0.078, series=True)
        add("table", dz=0.0015)
        add("friction", mu=0.45)
        return out
    for L in (0.069, 0.072, 0.078, 0.081):
        add("size", L=L, series=(L == 0.078))
    for dz in (-0.003, -0.0015, 0.0015, 0.003):
        add("table", dz=dz)
    for mu in (0.3, 0.45, 0.6, 1.0):
        add("friction", mu=mu)
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        L, dz, mu = rng.uniform(0.070, 0.080), rng.uniform(-0.002, 0.002), rng.uniform(0.4, 1.0)
        add("random", L=float(L), dz=float(dz), mu=float(mu))
    return out


def _append_rows(path, rows, fields):
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        fh.flush()
        os.fsync(fh.fileno())


def load_rows(path=STUDY_CSV):
    """The study CSV as a list of dicts (numbers converted), or [] if it does not exist."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            for k, v in r.items():
                if k in ("key", "sweep", "controller", "fail_reason"):
                    continue
                try:
                    r[k] = float(v) if v != "" else math.nan
                except ValueError:
                    pass
            rows.append(r)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="12 trials instead of 99")
    ap.add_argument("--fresh", action="store_true", help="delete earlier results first")
    ap.add_argument("--out", type=Path, default=STUDY_CSV)
    args = ap.parse_args(argv)
    if mujoco is None:
        sys.exit("sim_block_tilt.py needs mujoco==3.14.0 (uv run --script sim_block_tilt.py)")
    out = args.out
    series_path = out.with_name(out.stem.replace("_study", "") + "_timeseries.csv") \
        if out != STUDY_CSV else SERIES_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for p in (out, series_path):
            if p.exists():
                p.unlink()
    done = {r["key"] for r in load_rows(out)}
    trials = [t for t in study_trials(args.quick) if t.key not in done]
    meta = dict(model=str(MODEL_XML.relative_to(HERE.parents[1])), mujoco=mujoco.__version__,
                model_L_m=MODEL_L, density=DENSITY, mu_model=MU_MODEL, hand_x=HAND_X,
                omega=OMEGA, theta_goal_deg=math.degrees(THETA_GOAL), settle_s=SETTLE,
                t_max_s=T_MAX, control_dt_s=CONTROL_DT, force_limit_N=FORCE_LIMIT,
                slip_limit_m=SLIP_LIMIT, gap_limit_s=GAP_LIMIT, gains=asdict(Gains()),
                bt=dict((k, v) for k, v in asdict(controller_model()).items() if k != "edge"),
                started=time.strftime("%Y-%m-%d %H:%M:%S"), quick=args.quick)
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"{len(done)} trials already in {out.name}; running {len(trials)}")
    fields = None
    t0 = time.perf_counter()
    for i, tr in enumerate(trials):
        series = [] if tr.series else None
        row = run_trial(tr, series_rows=series)
        fields = fields or list(row)
        _append_rows(out, [row], fields)
        if series:
            _append_rows(series_path, series, list(series[0]))
        print(f"[{i + 1}/{len(trials)}] {tr.key:34s} success={row['success']} "
              f"peak={row['peak_finger_N']:.1f} N slip={row['max_table_slip_mm']:.2f} mm "
              f"gap={row['max_contact_gap_ms']:.0f} ms {row['fail_reason']}", flush=True)
    print(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
