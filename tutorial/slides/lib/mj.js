/* mj.js — real MuJoCo 3.14.0 (the official WebAssembly build) inside a slide.
 *
 * Load order in a deck (mj.js inserts lib/mujoco/*.js itself on first use, resolved against its
 * own URL, so the page needs only these two tags):
 *
 *   <script src="lib/fig.js"></script>
 *   <script src="lib/mj.js"></script>
 *
 * Minimal live slide:
 *
 *   const F = Fig("#drop", { w: 560, h: 380, xlim: [-0.3, 0.3], ylim: [-0.05, 0.35] });
 *   MJ.player(document.getElementById("box-drop"), {
 *     sim: MJ.sim(MODELS.box_on_plane),            // or an XML string
 *     fig: F,                                      // loading and error text go here
 *     draw: (sim) => { F.clear(); MJ.draw2(F, sim, { contacts: true, forceScale: 0.02 }); },
 *     controls: "#drop-controls",                  // [play] [pause] [reset] [step] t = ...
 *   });
 *
 * Conventions
 *   SI units. World z is up. Matrices are row-major 3x3 as flat arrays of 9 (MuJoCo's xmat).
 *   Quaternions are [w, x, y, z]. Names or integer ids are accepted wherever a model element is
 *   named (geom, body, joint, actuator, site, sensor, key); an unknown name throws.
 *   The typed arrays sim.qpos / qvel / ctrl / act / d.* are live views into the WebAssembly heap.
 *   Read and write them freely, but re-read them after compiling another model: heap growth
 *   detaches old views (they then have length 0). The getters below always return a fresh view.
 *
 * Loading
 *   MJ.ready() -> Promise<module>       loads and starts the engine once (about 0.1-0.2 s); the
 *                                        raw bindings (mujoco.d.ts) are the module's members.
 *   MJ.sim(xml) -> Promise<Sim>         compiles MJCF text, makes MjData, runs mj_forward.
 *                                        Rejects with MuJoCo's own message on a bad model
 *                                        ("MuJoCo Error: Loading error: XML Error: ...").
 *   MJ.track(promise) -> promise        registers async work with deck.js (Deck.pending, or
 *                                        window.DECK_PENDING before deck.js has run) so the test
 *                                        tools wait for it. MJ.ready, MJ.sim and MJ.player call it;
 *                                        MJ.ready and MJ.sim register a copy that never rejects,
 *                                        so a failure is reported once, by whoever awaits it (a
 *                                        player, or the browser's unhandled-rejection error).
 *   MJ.stats                            {load: {scripts, decode, init, total, at} ms,
 *                                        players: [{slide, created, ready, firstFrame, frames,
 *                                        steps, stepMs, mjStepMs, drawMs, simTime, runMs}]}
 *                                        ready / firstFrame / at are performance.now() values
 *                                        (ms since navigation start); stepMs is the whole step
 *                                        function, mjStepMs only mj_step; runMs is wall time
 *                                        spent animating.
 *
 * Sim (all methods return plain JS numbers and arrays unless noted)
 *   sim.m, sim.d, sim.mj                raw MjModel, MjData, module
 *   sim.nq, nv, nu, na, nbody, ngeom    sizes
 *   sim.time (get/set), sim.timestep    simulation time and opt.timestep, s
 *   sim.qpos, qvel, ctrl, act, qacc, qfrc_applied, xfrc_applied   live Float64Array views
 *   sim.step(n = 1)                     n calls of mj_step
 *   sim.advance(dt, opts)               steps so that sim time keeps up with dt seconds, carrying
 *                                        the remainder to the next call; returns steps taken.
 *                                        opts: {max: 20000 steps per call, each(sim, k): called
 *                                        after every mj_step (sampling a strip chart at the
 *                                        engine's rate)}, or a number for max
 *   sim.forward()                       mj_forward (positions, contacts, forces; no time step)
 *   sim.reset(key?)                     mj_resetData (or mj_resetDataKeyframe for a key name or
 *                                        index), then mj_forward. Model edits (setOpt,
 *                                        geomParam) are kept.
 *   sim.getState() -> {time, qpos, qvel, act, ctrl, warmstart, qfrc_applied, xfrc_applied,
 *                      mocap_pos, mocap_quat}   copies (Float64Array): MuJoCo's
 *                                        mjSTATE_INTEGRATION except eq_active, userdata, plugin
 *                                        state and delay history, which the tutorial models do
 *                                        not use
 *   sim.setState(s, forward = true)     writes any of those fields back (a field of the wrong
 *                                        length throws); warmstart restores qacc_warmstart, which
 *                                        makes a restored rollout repeat bit for bit (without it,
 *                                        400 steps of the tumbling box in tools/demos/mj_demo.html
 *                                        differ by 4e-15). Pass forward = false inside rollout
 *                                        loops (mj_step runs the forward pass itself). A state
 *                                        from one Sim restores into another Sim of the same model.
 *   sim.setOpt({timestep, cone: 'pyramidal'|'elliptic', impratio, noslip_iterations,
 *               noslip_tolerance, iterations, tolerance, gravity: [x,y,z],
 *               integrator: 'euler'|'rk4'|'implicit'|'implicitfast',
 *               solver: 'pgs'|'cg'|'newton', <any other numeric mjOption field>}) -> sim
 *   sim.getOpt() -> the same fields, current values
 *   sim.geomParam(geom, {friction: mu | [slide, spin, roll], solref: [2], solimp: [<=5],
 *                 margin, gap, priority, rgba: [4], pos: [3], quat: [4]}) -> current values
 *                                        Writes take effect at the next step or forward.
 *                                        Contact friction is the larger of the two geoms'
 *                                        values unless one geom has a higher priority.
 *                                        pos / quat also clear geom_sameframe, without which
 *                                        MuJoCo ignores a geom's own pose.
 *   sim.contacts() -> [{i, pos: [3], frame: [9] (rows: normal, t1, t2), normal: [3], dist,
 *                      geom1, geom2, geom1Name, geom2Name, body1, body2, dim, mu (friction[0]),
 *                      friction: [5], active, force: [6] (contact frame: normal, t1, t2, then
 *                      torsional and rolling torques), fworld: [3]}]
 *       The normal points from geom1 to geom2 (mjContact.frame in MuJoCo's mjdata.h). force
 *       and fworld are the force that geom1 exerts on geom2, the same sign convention for
 *       every component: mj_contactForce gives force in the contact frame, for pyramidal and
 *       elliptic cones alike, and fworld = frame^T * force[0:3]. A 1 kg box resting on a plane
 *       (plane = geom1) gives four contacts of force[0] = 2.4525 N whose fworld sum to
 *       (0, 0, +9.81) N; on the 20 degree ramp of tilted_plane.xml the sum is again
 *       (0, 0, +9.81) N, the friction part pointing uphill. Checked against Python mujoco
 *       3.14.0 in tools/twins/mujoco.json. active is false for contacts that MuJoCo keeps out
 *       of the solver (efc_address < 0: inside the gap, or between bodies without degrees of
 *       freedom); their force is zero, and the renderers skip them unless opts.inactive.
 *   sim.geoms() -> [{id, type: 'plane'|'sphere'|'capsule'|'ellipsoid'|'cylinder'|'box'|'mesh'|
 *                   'hfield'|'sdf', size: [3], pos: [3], mat: [9], rgba: [4], name, body,
 *                   bodyName, group, static (true for world-body geoms)}]
 *   sim.body(b) -> {id, name, xpos, xquat, xmat, mass, com (subtree centre of mass)}
 *   sim.joint(j) -> {id, name, type: 'free'|'ball'|'slide'|'hinge', qposadr, dofadr, qpos, qvel}
 *   sim.id(kind, name) -> integer id    kind: 'body'|'joint'|'geom'|'site'|'actuator'|'sensor'|'key'
 *   sim.name(kind, id) -> string
 *   sim.sensor(s) -> [values]
 *   sim.dispose()                       frees MjData, MjModel and buffers; the Sim is dead after
 *
 * Renderers into a Fig (lib/fig.js). They append to F; call F.clear() first. Both return
 * {contacts} (the list from sim.contacts(), filtered) so a caller can print numbers without
 * querying twice.
 *   MJ.draw2(F, sim, opts)   orthographic side view. opts:
 *     plane: 'xz' (default; viewer on -y, x right, z up) | 'yz' (viewer on +x) | 'xy' (top)
 *     planes: true        draw plane geoms as hatched ground lines (clipped to the figure)
 *     geoms: true         draw solid geoms: light gray fill, black outline; boxes and meshes as
 *                         the hull of their projected vertices, spheres as circles, capsules,
 *                         cylinders and ellipsoids as projected hulls; back to front
 *     fill: '#e6e6e6'     default fill of solid geoms
 *     colors: {}          per geom name: a fill color or {fill, stroke, width, dash, opacity};
 *                         or the string 'model' to use each geom's rgba
 *     skip: []            geom names not to draw; groups: [0, 1, 2] geom groups drawn
 *     contacts: false     contact points as dots, and forces as arrows unless forces is false
 *     forces: true        true | 'split' (normal part blue, friction part green) | false
 *     forceScale: 0.02    arrow length in world units per newton
 *     forceOn: null       body name or id: only its contacts, arrows = force ON that body
 *     net: false          one arrow per body pair: the summed force at the mean contact point
 *     cones: false        2D friction wedges of half-angle arctan(mu) at each contact (with net:
 *                         one per body pair, at the mean contact point)
 *     coneLen: 0.08       wedge length, world units
 *     frames: false       contact frames: normal (blue) and tangents t1, t2 (green)
 *     frameLen: 0.05      world units
 *     bodyFrames: false   body axes x (orange), y (teal), z (purple); bodyFrameLen: 0.06
 *     filter: c => bool   keep only these contacts
 *     inactive: false     also draw contacts with active = false (zero force)
 *   MJ.draw3(F, sim, view, opts)   orthographic 3D. view is Fig.view3(az, el) (radians) or
 *     {az, el} in degrees. Box faces (back faces culled, shaded by a fixed light), sphere discs,
 *     capsule / cylinder / ellipsoid silhouettes and mesh faces are painted back to front.
 *     Planes are finite grids (their XML size, or opts.planeSize half-extent when the size is 0;
 *     opts.grid spacing). Same contact options as draw2, plus cones: true | 'circle' |
 *     'pyramid' | 'auto' (pyramid when opt.cone is pyramidal), drawn with Fig's in3.cone /
 *     in3.pyramid, and coneLen.
 *   MJ.message(F, text, color?)     centred text in a figure (used for loading and errors),
 *                                    wrapped to the figure's width, at most 8 lines
 *
 * Player: an animation that runs only while its slide can be seen.
 *   const P = MJ.player(slideEl, {
 *     sim,              Sim or Promise<Sim> (from MJ.sim); optional if step/draw need no Sim
 *     init,             optional async () => {...}, awaited before the first frame
 *     fig,              Fig for "loading" / error text (default: none; errors go to the console)
 *     draw(sim),        redraws the figure; called after every frame, reset and step
 *     step(dt, sim),    advances by dt seconds of sim time (wall time since the last frame,
 *                       clamped to maxDt, times speed); default sim.advance(dt)
 *     maxDt: 0.05,      longest wall-time interval, s, one frame may simulate
 *     onReset(sim),     runs after sim.reset(), at start and on every [reset]: put initial
 *                       conditions here
 *     fps: 60,          frame-rate cap; speed: 1 (0.25 = quarter speed)
 *     loop: null,       seconds of sim time after which the player resets and continues
 *     autoplay: true,   start when the slide is shown
 *     controls,         element or selector; [play] [pause] [reset] [step] and a "t = 0.000 s"
 *                       readout are inserted as its first child; omitted: no buttons
 *     printSteps,       print mode: a function(sim) run once, or a number of frames (each one
 *                       [step] press, 1/fps x speed seconds) before the one static frame
 *   })
 *   -> {play(), pause(), reset(), stepOnce(), draw() (redraw now, e.g. after a control changes;
 *       keeps the play state), running (animating now), playing (wants to run),
 *       ready (Promise that resolves to the player once the first frame is drawn, also after a
 *       load failure), error (the failure, or null), sim (the Sim once loaded), speed (get/set),
 *       stats}
 *   A load or draw failure is written into fig in red and logged once with console.error.
 *   Present mode: runs while slideEl is the current slide (listens to slideshown / slidehidden /
 *   deckmode and re-checks slideEl.classList.contains('current') every frame). Read mode: runs
 *   while a quarter of the slide is on screen (IntersectionObserver). Print mode (?mode=print,
 *   or print mode entered later, as Ctrl+P and the [pdf] link do through beforeprint): reset,
 *   onReset, printSteps, then one static frame; leaving print mode resumes from that state.
 *   A page without deck.js: always runs.
 *
 * Helpers: MJ.quat(axis, angle) -> [w,x,y,z]; MJ.quatMul(a, b); MJ.mat(quat) -> row-major [9];
 *   MJ.rotate(quat, v) -> [3]; MJ.deg(d) -> radians; MJ.hull(points2d) -> convex hull (CCW);
 *   MJ.GEOM_TYPES.
 *
 * Memory: d.contact returns a new vector on every read and vec.get(i) a new element handle;
 * contacts() deletes both before it returns. The 6-double force buffer is allocated once per
 * Sim and freed by dispose(). A player stepping and drawing box_on_plane.xml for 20000 frames
 * leaves the WebAssembly heap size and the address of the next allocation unchanged, and
 * 600 cycles of MJ.sim, 20 steps, contacts() and dispose() stay at the heap size of the first
 * 150 (the allocator's high-water mark).
 * Handles from struct-valued getters (m.opt, d.warning, d.solver, d.timer) point inside the
 * MjModel / MjData wrapper: never call .delete() on them (freeing an interior pointer corrupts
 * the heap or aborts the module); m.delete() and d.delete() release them.
 */
(function () {
  "use strict";

  const SELF = (document.currentScript && document.currentScript.src) || location.href;
  const LIB = new URL(".", SELF).href;

  // Same web-safe palette as fig.js (kept here so mj.js does not depend on load order).
  const PAL = {
    ink: "#000000", force: "#cc0000", normal: "#0000cc", friction: "#008800", cone: "#ffff99",
    gray: "#999999", light: "#e6e6e6", faint: "#f2f2f2", purple: "#990099", orange: "#cc6600",
    teal: "#008080",
  };
  const GEOM_TYPES = ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh", "sdf"];
  const JOINT_TYPES = ["free", "ball", "slide", "hinge"];
  const OBJ = { body: 1, joint: 3, geom: 5, site: 6, camera: 7, actuator: 19, sensor: 20, key: 24, mesh: 10 };
  const CONES = { pyramidal: 0, elliptic: 1 };
  const INTEGRATORS = { euler: 0, rk4: 1, implicit: 2, implicitfast: 3 };
  const SOLVERS = { pgs: 0, cg: 1, newton: 2 };

  const MJ = { version: "3.14.0", GEOM_TYPES, PAL, stats: { load: null, players: [] } };

  // ------------------------------------------------------------------ small vector math
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => Math.hypot(a[0], a[1], a[2]);
  const unit = (a) => { const n = norm(a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
  const col = (R, k) => [R[k], R[3 + k], R[6 + k]];                  // k-th local axis in world
  const xf = (p, R, v) => [                                           // p + R v
    p[0] + R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
    p[1] + R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
    p[2] + R[6] * v[0] + R[7] * v[1] + R[8] * v[2],
  ];

  MJ.deg = (d) => (d * Math.PI) / 180;
  MJ.quat = function (axis, angle) {
    const a = unit(axis), s = Math.sin(angle / 2);
    return [Math.cos(angle / 2), a[0] * s, a[1] * s, a[2] * s];
  };
  MJ.quatMul = function (a, b) {
    return [
      a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
      a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
      a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
      a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ];
  };
  MJ.mat = function (q) {
    const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
    const w = q[0] / n, x = q[1] / n, y = q[2] / n, z = q[3] / n;
    return [
      1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ];
  };
  MJ.rotate = (q, v) => xf([0, 0, 0], MJ.mat(q), v);

  // 2D convex hull (Andrew's monotone chain), counter-clockwise.
  function hull(pts) {
    const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (p.length < 3) return p;
    const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo = [], up = [];
    for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 1e-12) lo.pop(); lo.push(q); }
    for (let i = p.length - 1; i >= 0; i--) {
      const q = p[i];
      while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 1e-12) up.pop();
      up.push(q);
    }
    lo.pop(); up.pop();
    return lo.concat(up);
  }
  MJ.hull = hull;

  // ------------------------------------------------------------------ async bookkeeping
  MJ.track = function (p) {
    const D = window.Deck;
    if (D && typeof D.pending === "function") D.pending(p);
    else (window.DECK_PENDING = window.DECK_PENDING || []).push(p);
    return p;
  };

  function script(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.async = false;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error("mj.js: could not load " + src));
      document.head.appendChild(s);
    });
  }

  function b64bytes(s) {
    if (typeof Uint8Array.fromBase64 === "function") return Uint8Array.fromBase64(s);
    const bin = atob(s), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  let modP = null;
  async function load() {
    const t0 = performance.now();
    const need = [];
    if (typeof window.loadMujoco !== "function") need.push(script(LIB + "mujoco/mujoco.js"));
    if (!window.MUJOCO_WASM_GZ) need.push(script(LIB + "mujoco/mujoco.wasm.js"));
    await Promise.all(need);
    const t1 = performance.now();
    if (typeof DecompressionStream === "undefined")
      throw new Error("mj.js: this browser lacks DecompressionStream (needs Chrome 80, Firefox 113 or Safari 16.4)");
    if (!window.MUJOCO_WASM_GZ) throw new Error("mj.js: lib/mujoco/mujoco.wasm.js did not define MUJOCO_WASM_GZ");
    const gz = b64bytes(window.MUJOCO_WASM_GZ);
    const wasm = await new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
    const t2 = performance.now();
    const mod = await window.loadMujoco({ wasmBinary: wasm });
    const t3 = performance.now();
    try { delete window.MUJOCO_WASM_GZ; } catch (e) { window.MUJOCO_WASM_GZ = null; }
    // Enum values from the module when present (they match MuJoCo's C headers).
    try {
      for (const k in OBJ) {
        const e = mod.mjtObj["mjOBJ_" + k.toUpperCase()];
        if (e && typeof e.value === "number") OBJ[k] = e.value;
      }
    } catch (e) { /* keep the defaults above */ }
    MJ.stats.load = { scripts: t1 - t0, decode: t2 - t1, init: t3 - t2, total: t3 - t0, at: t3 };
    MJ.mod = mod;
    return mod;
  }

  // Registers p with deck.js as settled-either-way, so a rejection is not logged a second time
  // by Deck.pending; the caller still gets (and must handle) the rejecting p.
  const trackQuiet = (p) => { MJ.track(p.then(() => {}, () => {})); return p; };

  MJ.ready = function () {
    if (!modP) modP = trackQuiet(load());
    return modP;
  };

  MJ.sim = function (xml) {
    return trackQuiet(MJ.ready().then((mod) => {
      if (typeof xml !== "string" || xml.indexOf("<mujoco") < 0)
        throw new Error("mj.js: MJ.sim needs MJCF text (got " + (typeof xml === "string" ? JSON.stringify(xml.slice(0, 40)) : typeof xml) + ")");
      let m = null, d = null;
      try {
        m = mod.MjModel.from_xml_string(xml);
        d = new mod.MjData(m);
      } catch (e) {
        if (d) d.delete();
        if (m) m.delete();
        const msg = String((e && e.message) || e).replace(/\s+$/, "");
        throw new Error(msg.startsWith("MuJoCo") ? msg : "MuJoCo: " + msg);
      }
      const s = new Sim(mod, m, d, xml);
      mod.mj_forward(m, d);
      return s;
    }));
  };

  // ------------------------------------------------------------------ Sim
  function Sim(mod, m, d, xml) {
    this.mj = mod;
    this.m = m;
    this.d = d;
    this.xml = xml;
    this.opt = m.opt;                       // points inside the MjModel wrapper: never delete it
    this._f6 = new mod.DoubleBuffer(6);     // out-parameter for mj_contactForce
    this._acc = 0;
    this._names = {};
    this.nstep = 0;
    this.stepMs = 0;
    this.dead = false;
    for (const k of ["nq", "nv", "nu", "na", "nbody", "ngeom", "njnt", "nsite", "nsensor", "nkey", "nmocap"]) this[k] = m[k];
  }
  MJ.Sim = Sim;
  const S = Sim.prototype;

  const live = (name) => ({ get() { return this.d[name]; }, enumerable: true });
  Object.defineProperties(S, {
    qpos: live("qpos"), qvel: live("qvel"), ctrl: live("ctrl"), act: live("act"),
    qacc: live("qacc"), qfrc_applied: live("qfrc_applied"), xfrc_applied: live("xfrc_applied"),
    time: { get() { return this.d.time; }, set(t) { this.d.time = t; }, enumerable: true },
    timestep: { get() { return this.opt.timestep; }, enumerable: true },
    ncon: { get() { return this.d.ncon; }, enumerable: true },
  });

  S._live = function () { if (this.dead) throw new Error("mj.js: this Sim was disposed"); };

  S.step = function (n) {
    this._live();
    n = n === undefined ? 1 : n | 0;
    const t0 = performance.now(), mod = this.mj, m = this.m, d = this.d;
    for (let i = 0; i < n; i++) mod.mj_step(m, d);
    this.stepMs += performance.now() - t0;
    this.nstep += n;
    return this;
  };

  S.advance = function (dt, o) {
    o = typeof o === "number" ? { max: o } : o || {};
    const h = this.opt.timestep;
    this._acc += dt;
    let n = Math.floor(this._acc / h + 1e-9);
    const cap = o.max || 20000;
    if (n > cap) { n = cap; this._acc = 0; } else this._acc -= n * h;
    if (o.each) for (let k = 0; k < n; k++) { this.step(1); o.each(this, k); }
    else if (n > 0) this.step(n);
    return n;
  };

  S.forward = function () { this._live(); this.mj.mj_forward(this.m, this.d); return this; };

  S.reset = function (key) {
    this._live();
    if (key === undefined || key === null) this.mj.mj_resetData(this.m, this.d);
    else this.mj.mj_resetDataKeyframe(this.m, this.d, this.id("key", key));
    this._acc = 0;
    this.mj.mj_forward(this.m, this.d);
    return this;
  };

  S.getState = function () {
    this._live();
    const d = this.d;
    return {
      time: d.time,
      qpos: d.qpos.slice(), qvel: d.qvel.slice(), act: d.act.slice(), ctrl: d.ctrl.slice(),
      warmstart: d.qacc_warmstart.slice(),
      qfrc_applied: d.qfrc_applied.slice(), xfrc_applied: d.xfrc_applied.slice(),
      mocap_pos: d.mocap_pos.slice(), mocap_quat: d.mocap_quat.slice(),
    };
  };

  const STATE_FIELDS = { qpos: "qpos", qvel: "qvel", act: "act", ctrl: "ctrl", warmstart: "qacc_warmstart",
    qfrc_applied: "qfrc_applied", xfrc_applied: "xfrc_applied", mocap_pos: "mocap_pos", mocap_quat: "mocap_quat" };

  S.setState = function (s, forward) {
    this._live();
    const d = this.d;
    if (s.time !== undefined) d.time = s.time;
    for (const k in STATE_FIELDS) {
      const v = s[k];
      if (!v || !v.length) continue;
      const dst = d[STATE_FIELDS[k]];
      if (v.length !== dst.length) throw new Error("mj.js: setState " + k + " has " + v.length + " values; the model needs " + dst.length);
      dst.set(v);
    }
    this._acc = 0;
    if (forward !== false) this.mj.mj_forward(this.m, d);
    return this;
  };

  const enumOf = (table, v, what) => {
    if (typeof v === "number") return v;
    const k = String(v).toLowerCase();
    if (!(k in table)) throw new Error("mj.js: unknown " + what + " '" + v + "' (use " + Object.keys(table).join(", ") + ")");
    return table[k];
  };

  S.setOpt = function (o) {
    this._live();
    const opt = this.opt;
    for (const k in o) {
      const v = o[k];
      if (v === undefined) continue;
      if (k === "cone") opt.cone = enumOf(CONES, v, "cone");
      else if (k === "integrator") opt.integrator = enumOf(INTEGRATORS, v, "integrator");
      else if (k === "solver") opt.solver = enumOf(SOLVERS, v, "solver");
      else if (k === "gravity" || k === "wind" || k === "magnetic") opt[k].set(v);
      else if (typeof opt[k] === "number") opt[k] = v;
      else throw new Error("mj.js: setOpt does not know '" + k + "'");
    }
    return this;
  };

  S.getOpt = function () {
    const o = this.opt;
    const name = (t, v) => Object.keys(t).find((k) => t[k] === v) || v;
    return {
      timestep: o.timestep, cone: name(CONES, o.cone), impratio: o.impratio,
      noslip_iterations: o.noslip_iterations, noslip_tolerance: o.noslip_tolerance,
      iterations: o.iterations, tolerance: o.tolerance, gravity: Array.from(o.gravity),
      integrator: name(INTEGRATORS, o.integrator), solver: name(SOLVERS, o.solver),
    };
  };

  S.id = function (kind, name) {
    if (typeof name === "number") return name;
    const t = OBJ[kind];
    if (t === undefined) throw new Error("mj.js: unknown element kind '" + kind + "'");
    const i = this.mj.mj_name2id(this.m, t, String(name));
    if (i < 0) throw new Error("mj.js: the model has no " + kind + " named '" + name + "'");
    return i;
  };

  S.name = function (kind, i) {
    const cache = this._names[kind] || (this._names[kind] = {});
    if (!(i in cache)) cache[i] = this.mj.mj_id2name(this.m, OBJ[kind], i) || "";
    return cache[i];
  };

  const put = (arr, off, v, n) => {
    if (typeof v === "number") arr[off] = v;
    else for (let k = 0; k < Math.min(n, v.length); k++) arr[off + k] = v[k];
  };

  S.geomParam = function (g, p) {
    this._live();
    const m = this.m, i = this.id("geom", g);
    p = p || {};
    if (p.friction !== undefined) put(m.geom_friction, 3 * i, p.friction, 3);
    if (p.solref !== undefined) put(m.geom_solref, 2 * i, p.solref, 2);
    if (p.solimp !== undefined) put(m.geom_solimp, 5 * i, p.solimp, 5);
    if (p.margin !== undefined) m.geom_margin[i] = p.margin;
    if (p.gap !== undefined) m.geom_gap[i] = p.gap;
    if (p.priority !== undefined) m.geom_priority[i] = p.priority;
    if (p.rgba !== undefined) put(m.geom_rgba, 4 * i, p.rgba, 4);
    if (p.pos !== undefined || p.quat !== undefined) {
      if (p.pos !== undefined) put(m.geom_pos, 3 * i, p.pos, 3);
      if (p.quat !== undefined) put(m.geom_quat, 4 * i, p.quat, 4);
      m.geom_sameframe[i] = 0;              // mjSAMEFRAME_NONE: use geom_pos / geom_quat
    }
    const sl = (a, k, n) => Array.from(a.subarray(k * n, k * n + n));
    return {
      friction: sl(m.geom_friction, i, 3), solref: sl(m.geom_solref, i, 2), solimp: sl(m.geom_solimp, i, 5),
      margin: m.geom_margin[i], gap: m.geom_gap[i], priority: m.geom_priority[i],
      rgba: sl(m.geom_rgba, i, 4), pos: sl(m.geom_pos, i, 3), quat: sl(m.geom_quat, i, 4),
    };
  };

  S.contacts = function () {
    this._live();
    const mod = this.mj, m = this.m, d = this.d, n = d.ncon, out = [];
    if (!n) return out;
    const vec = d.contact, buf = this._f6, bodyid = m.geom_bodyid;
    try {
      for (let i = 0; i < n; i++) {
        const c = vec.get(i);
        try {
          const frame = Array.from(c.frame), pos = Array.from(c.pos);
          const g1 = c.geom1, g2 = c.geom2, active = c.efc_address >= 0;
          let f = [0, 0, 0, 0, 0, 0];
          if (active) { mod.mj_contactForce(m, d, i, buf); f = Array.from(buf.GetView()); }
          const nrm = frame.slice(0, 3), t1 = frame.slice(3, 6), t2 = frame.slice(6, 9);
          const fw = [0, 1, 2].map((k) => f[0] * nrm[k] + f[1] * t1[k] + f[2] * t2[k]);
          const fr = Array.from(c.friction);
          out.push({
            i, pos, frame, normal: nrm, dist: c.dist, geom1: g1, geom2: g2,
            geom1Name: this.name("geom", g1), geom2Name: this.name("geom", g2),
            body1: g1 >= 0 ? bodyid[g1] : -1, body2: g2 >= 0 ? bodyid[g2] : -1,
            dim: c.dim, mu: fr[0], friction: fr, active, force: f, fworld: fw,
          });
        } finally { c.delete(); }
      }
    } finally { vec.delete(); }
    return out;
  };

  S.geoms = function () {
    this._live();
    const m = this.m, d = this.d, out = [];
    const type = m.geom_type, size = m.geom_size, rgba = m.geom_rgba, body = m.geom_bodyid, group = m.geom_group;
    const xpos = d.geom_xpos, xmat = d.geom_xmat;
    for (let i = 0; i < this.ngeom; i++) {
      const b = body[i];
      out.push({
        id: i, type: GEOM_TYPES[type[i]] || String(type[i]),
        size: [size[3 * i], size[3 * i + 1], size[3 * i + 2]],
        pos: [xpos[3 * i], xpos[3 * i + 1], xpos[3 * i + 2]],
        mat: Array.from(xmat.subarray(9 * i, 9 * i + 9)),
        rgba: [rgba[4 * i], rgba[4 * i + 1], rgba[4 * i + 2], rgba[4 * i + 3]],
        name: this.name("geom", i), body: b, bodyName: this.name("body", b), group: group[i],
        static: b === 0, dataid: m.geom_dataid[i],
      });
    }
    return out;
  };

  S.body = function (b) {
    this._live();
    const i = this.id("body", b), d = this.d;
    return {
      id: i, name: this.name("body", i),
      xpos: Array.from(d.xpos.subarray(3 * i, 3 * i + 3)),
      xquat: Array.from(d.xquat.subarray(4 * i, 4 * i + 4)),
      xmat: Array.from(d.xmat.subarray(9 * i, 9 * i + 9)),
      mass: this.m.body_mass[i],
      com: Array.from(d.subtree_com.subarray(3 * i, 3 * i + 3)),
    };
  };

  S.joint = function (j) {
    this._live();
    const i = this.id("joint", j), m = this.m, d = this.d;
    const t = m.jnt_type[i], qa = m.jnt_qposadr[i], da = m.jnt_dofadr[i];
    const nqj = [7, 4, 1, 1][t], nvj = [6, 3, 1, 1][t];
    return {
      id: i, name: this.name("joint", i), type: JOINT_TYPES[t], qposadr: qa, dofadr: da,
      qpos: nqj === 1 ? d.qpos[qa] : Array.from(d.qpos.subarray(qa, qa + nqj)),
      qvel: nvj === 1 ? d.qvel[da] : Array.from(d.qvel.subarray(da, da + nvj)),
    };
  };

  S.sensor = function (s) {
    this._live();
    const i = this.id("sensor", s), m = this.m, a = m.sensor_adr[i], n = m.sensor_dim[i];
    return Array.from(this.d.sensordata.subarray(a, a + n));
  };

  S.dispose = function () {
    if (this.dead) return;
    this.dead = true;
    try { this._f6.delete(); } catch (e) { /* already gone */ }
    // this.opt is not deleted: it points inside the MjModel wrapper, which m.delete() frees.
    this.opt = null;
    this.d.delete();
    this.m.delete();
  };

  // ------------------------------------------------------------------ geometry for the renderers
  function meshData(sim, meshid) {
    const cache = sim._mesh || (sim._mesh = {});
    if (cache[meshid]) return cache[meshid];
    const m = sim.m;
    const va = m.mesh_vertadr[meshid], vn = m.mesh_vertnum[meshid];
    const fa = m.mesh_faceadr[meshid], fn = m.mesh_facenum[meshid];
    const verts = [], faces = [];
    const V = m.mesh_vert, Fc = m.mesh_face;
    for (let k = 0; k < vn; k++) verts.push([V[3 * (va + k)], V[3 * (va + k) + 1], V[3 * (va + k) + 2]]);
    for (let k = 0; k < fn; k++) faces.push([Fc[3 * (fa + k)], Fc[3 * (fa + k) + 1], Fc[3 * (fa + k) + 2]]);
    return (cache[meshid] = { verts, faces });
  }

  // Points (3D, world) that span a geom's outline; the projected hull of these is its silhouette.
  function outlinePoints(sim, g, k) {
    const s = g.size, p = g.pos, R = g.mat, out = [];
    k = k || 24;
    if (g.type === "box") {
      for (const a of [-1, 1]) for (const b of [-1, 1]) for (const c of [-1, 1]) out.push(xf(p, R, [a * s[0], b * s[1], c * s[2]]));
    } else if (g.type === "cylinder") {
      for (const z of [-s[1], s[1]]) for (let i = 0; i < k; i++) {
        const t = (2 * Math.PI * i) / k;
        out.push(xf(p, R, [s[0] * Math.cos(t), s[0] * Math.sin(t), z]));
      }
    } else if (g.type === "ellipsoid") {
      for (let i = 0; i <= 12; i++) for (let j = 0; j < 24; j++) {
        const u = (Math.PI * i) / 12, v = (2 * Math.PI * j) / 24;
        out.push(xf(p, R, [s[0] * Math.sin(u) * Math.cos(v), s[1] * Math.sin(u) * Math.sin(v), s[2] * Math.cos(u)]));
      }
    } else if (g.type === "mesh" && g.dataid >= 0) {
      for (const v of meshData(sim, g.dataid).verts) out.push(xf(p, R, v));
    }
    return out;
  }

  // A projected circle of radius r (world units) around a 2D point, as polygon points.
  const circ2 = (c, r, k) => Array.from({ length: k }, (_, i) => [c[0] + r * Math.cos((2 * Math.PI * i) / k), c[1] + r * Math.sin((2 * Math.PI * i) / k)]);

  function styleFor(g, o, dflt) {
    const c = o.colors;
    let st = { fill: dflt, stroke: PAL.ink, width: 1.5 };
    if (c === "model") {
      const r = g.rgba, h = (x) => Math.round(255 * Math.max(0, Math.min(1, x))).toString(16).padStart(2, "0");
      st.fill = "#" + h(r[0]) + h(r[1]) + h(r[2]);
      st.opacity = r[3] < 1 ? r[3] : undefined;
    } else if (c && c[g.name] !== undefined) {
      const v = c[g.name];
      if (typeof v === "string") st.fill = v; else st = Object.assign(st, v);
    }
    return st;
  }

  function visibleGeom(g, o) {
    if (g.rgba[3] === 0 && o.colors === "model") return false;
    if (o.skip && o.skip.indexOf(g.name) >= 0) return false;
    const groups = o.groups || [0, 1, 2];
    return groups.indexOf(g.group) >= 0;
  }

  // Keep contacts per opts; orient each force as the force on opts.forceOn when given.
  function pickContacts(sim, o) {
    let cs = sim.contacts().filter((c) => c.active || o.inactive);
    if (o.filter) cs = cs.filter(o.filter);
    if (o.forceOn !== undefined && o.forceOn !== null) {
      const b = sim.id("body", o.forceOn);
      cs = cs.filter((c) => c.body1 === b || c.body2 === b);
      cs.forEach((c) => { c.sign = c.body2 === b ? 1 : -1; });
    } else cs.forEach((c) => { c.sign = 1; });
    return cs;
  }

  function netGroups(cs, onBody) {
    const groups = {};
    cs.forEach((c) => {
      const k = Math.min(c.body1, c.body2) + ":" + Math.max(c.body1, c.body2);
      const g = groups[k] || (groups[k] = { n: 0, pos: [0, 0, 0], f: [0, 0, 0], fn: [0, 0, 0], nsum: [0, 0, 0], ref: c });
      // With forceOn, c.sign already orients each force onto that body. Otherwise orient every
      // contact of the pair as the force on the pair's first contact's body2.
      const s = onBody ? c.sign : (c.body2 === g.ref.body2 ? 1 : -1);
      g.n++;
      g.pos = add(g.pos, c.pos);
      g.f = add(g.f, mul(c.fworld, s));
      g.fn = add(g.fn, mul(c.normal, c.force[0] * s));
      g.nsum = add(g.nsum, mul(c.normal, s));
    });
    return Object.keys(groups).map((k) => {
      const g = groups[k];
      // The pair's contact frame, re-oriented so its normal matches the summed direction.
      const sg = dot(g.ref.normal, g.nsum) >= 0 ? 1 : -1;
      const fr = g.ref.frame.map((x, i) => (i < 6 ? x * sg : x));
      return { pos: mul(g.pos, 1 / g.n), f: g.f, fn: g.fn, nsum: g.nsum, n: g.n, mu: g.ref.mu, frame: fr };
    });
  }

  // Draws contact overlays with a projection P (3D -> 2D) and a 2D Fig (or in3 wrapper).
  function drawContacts(F, cs, P, o, three) {
    const scale = o.forceScale === undefined ? 0.02 : o.forceScale;
    const forces = o.forces === undefined ? true : o.forces;
    const px = (v) => Math.hypot(P(v)[0], P(v)[1]) * F.s;   // screen length of a world vector
    const arrow = (a, v, color, width) => {
      const e = add(a, mul(v, scale));
      if (px(mul(v, scale)) < 2) return;
      F.arrow(P(a), P(e), { color, width: width || 2.2, headpx: 11 });
    };
    // Cones first, so dots and arrows sit on top.
    if (o.cones) {
      const len = o.coneLen || 0.08;
      // With net, one cone per body pair at the mean contact point, about the mean normal.
      const at = o.net ? netGroups(cs, o.forceOn !== undefined && o.forceOn !== null).map((g) =>
        ({ pos: g.pos, normal: unit(g.fn.some((x) => x !== 0) ? g.fn : g.nsum), sign: 1, mu: g.mu, frame: g.frame })) : cs;
      at.forEach((c) => {
        const axis = mul(c.normal, c.sign), half = Math.atan(c.mu);
        if (three) {
          const kind = o.cones === "auto" ? (o._pyramidal ? "pyramid" : "circle") : o.cones;
          const I = F.in3(o._view);
          if (kind === "pyramid") {
            const t1 = c.frame.slice(3, 6), t2 = c.frame.slice(6, 9), r = c.mu * len;
            const top = add(c.pos, mul(axis, len));
            I.pyramid(c.pos, [add(top, mul(t1, r)), add(top, mul(t2, r)), add(top, mul(t1, -r)), add(top, mul(t2, -r))], { fillOpacity: 0.45 });
          } else I.cone(c.pos, axis, len, c.mu * len, { fillOpacity: 0.55 });
        } else {
          const a2 = P(axis), L = Math.hypot(a2[0], a2[1]);
          if (L < 0.3) return;
          F.wedge(P(c.pos), Math.atan2(a2[1], a2[0]), half, len, { fillOpacity: 0.55, flat: half > 1.45 });
        }
      });
    }
    if (o.frames) {
      const L = o.frameLen || 0.05;
      cs.forEach((c) => {
        const n = mul(c.normal, c.sign);
        F.arrow(P(c.pos), P(add(c.pos, mul(n, L))), { color: PAL.normal, width: 1.6, headpx: 8 });
        F.line(P(c.pos), P(add(c.pos, mul(c.frame.slice(3, 6), L * 0.7))), { color: PAL.friction, width: 1.4 });
        F.line(P(c.pos), P(add(c.pos, mul(c.frame.slice(6, 9), L * 0.7))), { color: PAL.friction, width: 1.4, dash: "3 3" });
      });
    }
    if (o.contacts) {
      const drawF = (pos, f, fn) => {
        if (forces === "split") {
          arrow(pos, fn, PAL.normal);
          arrow(pos, sub(f, fn), PAL.friction);
        } else if (forces) arrow(pos, f, PAL.force);
      };
      if (o.net) {
        netGroups(cs, o.forceOn !== undefined && o.forceOn !== null).forEach((g) => { F.dot(P(g.pos), { rpx: 4, fill: PAL.ink }); drawF(g.pos, g.f, g.fn); });
      } else {
        cs.forEach((c) => {
          F.dot(P(c.pos), { rpx: 3.5, fill: PAL.ink });
          drawF(c.pos, mul(c.fworld, c.sign), mul(c.normal, c.force[0] * c.sign));
        });
      }
    }
  }

  function figBounds(F) {
    const a = F.world(0, 0), b = F.world(F.w, F.h);
    return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
  }

  // ------------------------------------------------------------------ draw2
  const PLANES = {
    xz: { p: (v) => [v[0], v[2]], toward: [0, -1, 0] },
    yz: { p: (v) => [v[1], v[2]], toward: [1, 0, 0] },
    xy: { p: (v) => [v[0], v[1]], toward: [0, 0, 1] },
  };

  function plane2(F, g, V, o) {
    const n = col(g.mat, 2), n2 = V.p(n), L = Math.hypot(n2[0], n2[1]);
    const st = styleFor(g, o, PAL.light);
    if (L < 0.3) {
      // Seen face-on (a floor from above): outline its finite extent, if it has one.
      if (g.size[0] > 0 && g.size[1] > 0) {
        const c = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => V.p(xf(g.pos, g.mat, [a * g.size[0], b * g.size[1], 0])));
        F.poly(c, { fill: PAL.faint, color: PAL.gray, width: 1 });
      }
      return;
    }
    let T = unit(cross(n, V.toward));
    const t2 = [n2[1] / L, -n2[0] / L];                  // solid (the -n side) on the right
    let T2 = V.p(T);
    if (T2[0] * t2[0] + T2[1] * t2[1] < 0) { T = mul(T, -1); T2 = V.p(T); }
    let s0 = -Infinity, s1 = Infinity;
    if (g.size[0] > 0 && g.size[1] > 0) {
      const a = Math.abs(dot(T, col(g.mat, 0))), b = Math.abs(dot(T, col(g.mat, 1)));
      const half = Math.min(a > 1e-9 ? g.size[0] / a : Infinity, b > 1e-9 ? g.size[1] / b : Infinity);
      s0 = -half; s1 = half;
    }
    // Clip the line c + s T2 to the figure (slab method).
    const c2 = V.p(g.pos), B = figBounds(F);
    for (let k = 0; k < 2; k++) {
      const lo = B[k], hi = B[k + 2];
      if (Math.abs(T2[k]) < 1e-12) { if (c2[k] < lo || c2[k] > hi) return; continue; }
      let u = (lo - c2[k]) / T2[k], v = (hi - c2[k]) / T2[k];
      if (u > v) { const w = u; u = v; v = w; }
      s0 = Math.max(s0, u); s1 = Math.min(s1, v);
    }
    if (!(s1 > s0)) return;
    const a = [c2[0] + s0 * T2[0], c2[1] + s0 * T2[1]], b = [c2[0] + s1 * T2[0], c2[1] + s1 * T2[1]];
    F.surface(a, b, { color: st.stroke, width: 2 });
  }

  function solid2(F, sim, g, V, o) {
    const st = styleFor(g, o, o.fill || PAL.light);
    const sty = { fill: st.fill, color: st.stroke, width: st.width, dash: st.dash, fillOpacity: st.opacity };
    if (g.type === "sphere") return F.circle(V.p(g.pos), g.size[0], sty);
    let pts;
    if (g.type === "capsule") {
      const ax = col(g.mat, 2), e1 = add(g.pos, mul(ax, g.size[1])), e2 = add(g.pos, mul(ax, -g.size[1]));
      pts = circ2(V.p(e1), g.size[0], 32).concat(circ2(V.p(e2), g.size[0], 32));
    } else {
      pts = outlinePoints(sim, g).map(V.p);
    }
    if (pts.length < 3) return null;
    return F.poly(hull(pts), sty);
  }

  function drawBodyFrames(F, sim, P, o) {
    const L = o.bodyFrameLen || 0.06, d = sim.d;
    for (let b = 1; b < sim.nbody; b++) {
      const p = Array.from(d.xpos.subarray(3 * b, 3 * b + 3)), R = Array.from(d.xmat.subarray(9 * b, 9 * b + 9));
      [PAL.orange, PAL.teal, PAL.purple].forEach((color, k) => {
        F.arrow(P(p), P(add(p, mul(col(R, k), L))), { color, width: 1.6, headpx: 8 });
      });
    }
  }

  MJ.draw2 = function (F, sim, o) {
    o = o || {};
    const V = PLANES[o.plane || "xz"];
    if (!V) throw new Error("mj.js: draw2 plane must be 'xz', 'yz' or 'xy'");
    const gs = sim.geoms().filter((g) => visibleGeom(g, o));
    if (o.planes !== false) gs.filter((g) => g.type === "plane").forEach((g) => plane2(F, g, V, o));
    if (o.geoms !== false) {
      gs.filter((g) => g.type !== "plane" && g.type !== "hfield" && g.type !== "sdf")
        .sort((a, b) => dot(a.pos, V.toward) - dot(b.pos, V.toward))
        .forEach((g) => solid2(F, sim, g, V, o));
    }
    if (o.bodyFrames) drawBodyFrames(F, sim, V.p, o);
    const need = o.contacts || o.cones || o.frames;
    const cs = need ? pickContacts(sim, o) : [];
    if (need) drawContacts(F, cs, V.p, o, false);
    return { contacts: cs };
  };

  // ------------------------------------------------------------------ draw3
  const hex2 = (v) => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, "0");
  const gray = (v) => "#" + hex2(v) + hex2(v) + hex2(v);

  function toView(view) {
    if (typeof view === "function") return view;
    const v = view || {};
    return window.Fig.view3(MJ.deg(v.az === undefined ? 30 : v.az), MJ.deg(v.el === undefined ? 20 : v.el));
  }

  MJ.draw3 = function (F, sim, view, o) {
    o = o || {};
    const v = toView(view);
    const tw = v.toward;
    const light = unit(add(mul(tw, 0.55), [0.25, -0.35, 0.8]));
    const shade = (n) => gray(178 + 70 * Math.max(0, dot(n, light)));
    const gs = sim.geoms().filter((g) => visibleGeom(g, o));
    const items = [];   // {depth, draw()}

    if (o.planes !== false) gs.filter((g) => g.type === "plane").forEach((g) => {
      const sx = g.size[0] > 0 ? g.size[0] : (o.planeSize || 1), sy = g.size[1] > 0 ? g.size[1] : (o.planeSize || 1);
      const P = (a, b) => v(xf(g.pos, g.mat, [a, b, 0]));
      F.poly([P(-sx, -sy), P(sx, -sy), P(sx, sy), P(-sx, sy)], { fill: "#f7f7f7", color: PAL.ink, width: 1 });
      let step = o.grid || (g.size[2] > 0 ? g.size[2] : Math.max(sx, sy) / 5);
      while (Math.max(sx, sy) / step > 20) step *= 2;
      for (let a = -Math.floor(sx / step) * step; a <= sx + 1e-9; a += step) if (Math.abs(Math.abs(a) - sx) > 1e-9) F.line(P(a, -sy), P(a, sy), { color: "#bbbbbb", width: 1 });
      for (let b = -Math.floor(sy / step) * step; b <= sy + 1e-9; b += step) if (Math.abs(Math.abs(b) - sy) > 1e-9) F.line(P(-sx, b), P(sx, b), { color: "#bbbbbb", width: 1 });
    });

    if (o.geoms !== false) gs.forEach((g) => {
      const st = styleFor(g, o, null);
      const custom = st.fill !== null;
      const sty = (fill) => ({ fill: custom ? st.fill : fill, color: st.stroke, width: st.width, fillOpacity: st.opacity, dash: st.dash });
      const depth = v.depth(g.pos);
      if (g.type === "box") {
        const s = g.size;
        for (let k = 0; k < 3; k++) for (const sg of [-1, 1]) {
          const nrm = mul(col(g.mat, k), sg);
          if (dot(nrm, tw) <= 1e-9) continue;               // back face
          const i = (k + 1) % 3, j = (k + 2) % 3;
          const corner = (a, b) => { const q = [0, 0, 0]; q[k] = sg * s[k]; q[i] = a * s[i]; q[j] = b * s[j]; return xf(g.pos, g.mat, q); };
          const face = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
          const cen = mul(face.reduce(add), 0.25);
          items.push({ depth: v.depth(cen), draw: () => F.poly(face.map(v), sty(shade(nrm))) });
        }
      } else if (g.type === "sphere") {
        items.push({ depth, draw: () => {
          F.circle(v(g.pos), g.size[0], sty(shade(tw)));
          // Front half of the local equator, so rotation shows.
          const ring = Fig.ring3(g.pos, col(g.mat, 2), g.size[0] * 0.999, 48);
          let run = [];
          const flush = () => { if (run.length > 1) F.path(run.map(v), { color: PAL.gray, width: 1 }); run = []; };
          ring.forEach((q) => { if (dot(sub(q, g.pos), tw) >= 0) run.push(q); else flush(); });
          flush();
        } });
      } else if (g.type === "capsule") {
        const ax = col(g.mat, 2), e1 = add(g.pos, mul(ax, g.size[1])), e2 = add(g.pos, mul(ax, -g.size[1]));
        items.push({ depth, draw: () => F.poly(hull(circ2(v(e1), g.size[0], 32).concat(circ2(v(e2), g.size[0], 32))), sty(shade(tw))) });
      } else if (g.type === "cylinder") {
        const ax = col(g.mat, 2), s = g.size;
        items.push({ depth, draw: () => {
          F.poly(hull(outlinePoints(sim, g, 36).map(v)), sty(shade(unit(add(tw, mul(ax, -dot(ax, tw)))))));
          const sg = dot(ax, tw) >= 0 ? 1 : -1;
          if (Math.abs(dot(ax, tw)) > 0.02) {
            const cap = Array.from({ length: 36 }, (_, i) => xf(g.pos, g.mat, [s[0] * Math.cos((2 * Math.PI * i) / 36), s[0] * Math.sin((2 * Math.PI * i) / 36), sg * s[1]]));
            F.poly(cap.map(v), sty(shade(mul(ax, sg))));
          }
        } });
      } else if (g.type === "ellipsoid") {
        items.push({ depth, draw: () => F.poly(hull(outlinePoints(sim, g).map(v)), sty(shade(tw))) });
      } else if (g.type === "mesh" && g.dataid >= 0) {
        const md = meshData(sim, g.dataid);
        if (md.faces.length > 3000) {
          items.push({ depth, draw: () => F.poly(hull(outlinePoints(sim, g).map(v)), sty(shade(tw))) });
        } else {
          const W = md.verts.map((q) => xf(g.pos, g.mat, q));
          md.faces.forEach((f) => {
            const a = W[f[0]], b = W[f[1]], c = W[f[2]];
            const nrm = unit(cross(sub(b, a), sub(c, a)));
            if (dot(nrm, tw) <= 1e-9) return;
            const cen = mul(add(add(a, b), c), 1 / 3);
            items.push({ depth: v.depth(cen), draw: () => F.poly([a, b, c].map(v), Object.assign(sty(shade(nrm)), { width: 0.6 })) });
          });
        }
      }
    });
    items.sort((a, b) => a.depth - b.depth).forEach((it) => it.draw());

    if (o.bodyFrames) drawBodyFrames(F, sim, v, o);
    const need = o.contacts || o.cones || o.frames;
    const cs = need ? pickContacts(sim, o) : [];
    if (need) {
      const oo = Object.assign({}, o, { _view: v, _pyramidal: sim.opt.cone === 0 });
      drawContacts(F, cs, v, oo, true);
    }
    return { contacts: cs };
  };

  // ------------------------------------------------------------------ figure messages
  MJ.message = function (F, text, color) {
    if (!F) return;
    F.clear();
    const lines = [];
    const wrap = Math.max(16, Math.floor((F.w - 24) / 8.2));   // characters of 18 px Times per line
    String(text).split("\n").forEach((ln) => {
      while (ln.length > wrap) {
        let k = ln.lastIndexOf(" ", wrap);
        if (k < wrap / 3) k = wrap;
        lines.push(ln.slice(0, k));
        ln = ln.slice(k).trim();
      }
      if (ln.trim()) lines.push(ln);
    });
    const shown = lines.slice(0, 8);
    const c = F.world(F.w / 2, F.h / 2), lh = 22 / F.s;
    shown.forEach((ln, i) => F.text([c[0], c[1] + ((shown.length - 1) / 2 - i) * lh], ln,
      { anchor: "middle", size: 18, color: color || PAL.gray }));
  };

  // ------------------------------------------------------------------ player
  function modeNow() {
    const D = window.Deck;
    if (D && typeof D.mode === "function") return D.mode();
    const b = document.body.classList;
    return b.contains("print") ? "print" : b.contains("read") ? "read" : b.contains("present") ? "present" : "none";
  }

  MJ.player = function (slide, o) {
    if (typeof slide === "string") slide = document.querySelector(slide);
    if (!slide) throw new Error("mj.js: MJ.player needs the slide element");
    o = o || {};
    const fps = o.fps || 60, maxDt = o.maxDt || 0.05, autoplay = o.autoplay !== false;
    let speed = o.speed || 1;
    let sim = null, ready = false, playing = autoplay, raf = 0, last = null, inView = false, failed = false;
    const t0 = performance.now();
    const st = { slide: slide.id || null, created: t0, ready: null, firstFrame: null, frames: 0, steps: 0, stepMs: 0, mjStepMs: 0, drawMs: 0, simTime: 0, runMs: 0 };
    MJ.stats.players.push(st);

    // Controls: [play] [pause] [reset] [step] t = 0.000 s
    let clock = null;
    const ctl = typeof o.controls === "string" ? slide.querySelector(o.controls) || document.querySelector(o.controls) : o.controls;
    if (ctl) {
      const bar = document.createElement("span");
      bar.className = "mj-player";
      const btn = (label, fn) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = label;
        b.style.font = "inherit";
        b.addEventListener("click", (e) => { e.preventDefault(); fn(); b.blur(); });
        bar.appendChild(b);
        bar.appendChild(document.createTextNode(" "));
        return b;
      };
      btn("play", () => api.play());
      btn("pause", () => api.pause());
      btn("reset", () => api.reset());
      btn("step", () => api.stepOnce());
      clock = document.createElement("span");
      clock.style.fontFamily = '"Courier New", Courier, monospace';
      clock.textContent = " t = 0.000 s";
      bar.appendChild(clock);
      ctl.insertBefore(bar, ctl.firstChild);
    }

    const fail = (e) => {
      failed = true;
      api.error = e;
      stop();
      console.error("mj.js player" + (slide.id ? " #" + slide.id : "") + ":", e);
      const msg = String((e && e.message) || e);
      MJ.message(o.fig, /^(MuJoCo|mj\.js)/.test(msg) ? msg : "Error: " + msg, PAL.force);
    };
    const tick = () => { if (clock && sim) clock.textContent = " t = " + sim.time.toFixed(3) + " s"; };
    const drawNow = () => {
      if (!o.draw) return;
      const a = performance.now();
      o.draw(sim);
      st.drawMs += performance.now() - a;
      st.frames++;
      if (st.firstFrame === null) st.firstFrame = performance.now();
      tick();
    };
    const advance = (dt) => {
      const n0 = sim ? sim.nstep : 0, m0 = sim ? sim.stepMs : 0, a = performance.now();
      if (o.step) o.step(dt, sim); else if (sim) sim.advance(dt);
      st.stepMs += performance.now() - a;
      if (sim) { st.steps += sim.nstep - n0; st.mjStepMs += sim.stepMs - m0; st.simTime = sim.time; }
      if (o.loop && sim && sim.time >= o.loop) doReset();
    };
    const doReset = () => {
      if (sim) sim.reset();
      if (o.onReset) o.onReset(sim);
    };

    function visible() {
      const md = modeNow();
      if (md === "print") return false;
      if (md === "read") return inView;
      if (md === "present") return slide.classList.contains("current");
      return true;
    }
    function stop() { if (raf) cancelAnimationFrame(raf); raf = 0; last = null; }
    function frame(now) {
      raf = 0;
      if (!(ready && playing && visible()) || failed) { last = null; return; }
      if (last !== null && now - last < 1000 / fps - 2) { raf = requestAnimationFrame(frame); return; }
      const dt = last === null ? 0 : Math.min((now - last) / 1000, maxDt);
      if (last !== null) st.runMs += now - last;
      last = now;
      try {
        if (dt > 0) advance(dt * speed);
        drawNow();
      } catch (e) { fail(e); return; }
      raf = requestAnimationFrame(frame);
    }
    function kick() { if (!raf && ready && playing && !failed && visible()) raf = requestAnimationFrame(frame); }
    // The one static frame of print mode: from the initial state, after printSteps.
    function printFrame() {
      stop();
      doReset();
      if (typeof o.printSteps === "function") o.printSteps(sim);
      else for (let k = 0; k < (o.printSteps || 0); k++) advance(speed / fps);
      drawNow();
    }

    const api = {
      play() { playing = true; kick(); },
      pause() { playing = false; stop(); if (ready) try { drawNow(); } catch (e) { fail(e); } },
      reset() { if (!ready) return; try { doReset(); drawNow(); } catch (e) { fail(e); } },
      stepOnce() { if (!ready) return; playing = false; stop(); try { advance(speed / fps); drawNow(); } catch (e) { fail(e); } },
      draw() { if (ready && !failed) try { drawNow(); } catch (e) { fail(e); } },
      get running() { return raf !== 0; },
      get playing() { return playing; },
      get sim() { return sim; },
      get speed() { return speed; },
      set speed(v) { speed = v; },
      stats: st,
      ready: null,
      error: null,
    };

    slide.addEventListener("slideshown", kick);
    slide.addEventListener("slidehidden", stop);
    document.addEventListener("deckmode", () => {
      stop();
      if (ready && !failed && modeNow() === "print") { try { printFrame(); } catch (e) { fail(e); } }
      else kick();
    });
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((es) => { es.forEach((e) => { inView = e.isIntersecting; }); if (inView) kick(); else if (modeNow() === "read") stop(); },
        { threshold: 0.25 }).observe(slide);
    }

    if (o.fig) MJ.message(o.fig, "loading MuJoCo ...");
    api.ready = MJ.track((async () => {
      try {
        if (o.sim) sim = await o.sim;
        if (o.init) await o.init(sim);
        if (document.readyState === "loading") await new Promise((r) => document.addEventListener("DOMContentLoaded", r, { once: true }));
        st.ready = performance.now();
        if (o.fig) o.fig.clear();
        ready = true;
        if (modeNow() === "print") printFrame();
        else { doReset(); drawNow(); kick(); }
      } catch (e) {
        fail(e);                          // drawn in the figure and logged once; ready still resolves
      }
      return api;
    })());
    return api;
  };

  window.MJ = MJ;
})();
