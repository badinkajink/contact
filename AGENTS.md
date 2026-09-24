# Agent guide: optimization, contact, and control tutorial

This file is the handoff for any agent or contributor working on this repository. It describes
what the tutorial is for, how it is laid out, the conventions every file follows, how to check
work, and what is finished versus planned. Read it before editing anything under `tutorial/`.
Keep the status tables current when you finish a piece of work.

## 1. Purpose

`tutorial/` is a curriculum that takes a strong high-school senior or first-year undergraduate
from "what is a vector?" to reading contact-rich robotics papers without a walkthrough. It is
visual (every concept has a figure), interactive (sliders, draggable handles, real MuJoCo running
in the slide), and mathematical (every derivation a first-year student cannot do in their head is
shown one step at a time).

The concrete test of the curriculum is two papers:

* Y. Hou and M. T. Mason, *Robust Execution of Contact-Rich Motion Plans by Hybrid Force-Velocity
  Control*, ICRA 2019, [arXiv 1903.02715](https://arxiv.org/abs/1903.02715).
* Y. Hou and M. T. Mason, *An Efficient Closed-Form Method for Optimal Hybrid Force-Velocity
  Control*, ICRA 2021, [arXiv 2011.04872](https://arxiv.org/abs/2011.04872).

A student who finishes the series should be able to follow every equation in both, and in papers
like Posa et al. 2014 (contact-implicit trajectory optimization) and the MuJoCo MPC paper. The
lab in `tutorial/labs/hybrid_servoing/` reimplements the two Hou & Mason algorithms.

## 2. Repository layout

```
AGENTS.md                      this guide (CLAUDE.md imports it)
README.md                      one-line project description
pyproject.toml, uv.lock        root uv project (marimo); notebooks carry their own PEP 723 deps
tutorial/
  trajopt_contact_tutorial_v5.tex   long-form LaTeX tutorial (3977 lines), content source for most decks
  PLAN.md                           scope of every unwritten deck, the lab, and the 11-16 integration
  00_math_toolkit.py ...            marimo companion notebooks (section 5)
  models/                           MuJoCo MJCF models shared by notebooks and slides
  labs/hybrid_servoing/             Hou & Mason lab (planned)
  mjview.py                         mjviser-in-marimo helper (planned)
  slides/
    NN_name.html                    one self-contained HTML slide deck per topic
    index.html                      series landing page
    template.html                   every layout, as a copyable example
    README.md                       presenting and exporting
    export.py                       PDF / PNG / PPTX export (Playwright)
    lib/                            engine, theme, figure and numerics libraries (section 6)
    tools/                          checking, screenshots, twin tests, sync (section 8)
ddp/                            earlier standalone DDP / iLQR experiments (pendulum, cart-pole)
linear_algebra/                 COMS 3251 lab notebooks; VMLS-Companions submodule (not initialized)
video_to_gif/                   unrelated local video-to-GIF tool
```

## 3. The curriculum

Twenty-two decks in six parts. File names are final; other files link to them.

| # | file (`tutorial/slides/`) | title | notebook | status |
|---|---|---|---|---|
| **I** | | **Mathematical toolkit** | | |
| 01 | `01_vectors_and_matrices.html` | Vectors, matrices, and linear maps | `00_math_toolkit.py` | planned |
| 02 | `02_eigenvalues_and_least_squares.html` | Eigenvalues, singular values, and least squares | `00_math_toolkit.py` | planned |
| 03 | `03_derivatives_and_jacobians.html` | Derivatives, gradients, and Jacobians | `00_math_toolkit.py` | planned |
| 04 | `04_probability_and_sampling.html` | Probability, Gaussians, and sampling | `00_math_toolkit.py` | planned |
| **II** | | **Optimization** | | |
| 05 | `05_unconstrained_optimization.html` | Unconstrained optimization: gradient descent and Newton's method | `01_optimization_fundamentals.py` | planned |
| 06 | `06_constrained_optimization.html` | Constrained optimization: Lagrange multipliers and the KKT conditions | `01_optimization_fundamentals.py` | planned |
| 07 | `07_convexity_and_duality.html` | Convexity, duality, and complementarity | `01_optimization_fundamentals.py` | planned |
| **III** | | **Trajectory optimization and control** | | |
| 08 | `08_optimal_control_and_lqr.html` | Optimal control, dynamic programming, and LQR | `02_lqr_ilqr_sliding_block.py` | planned |
| 09 | `09_ddp_and_ilqr.html` | Differential dynamic programming and iLQR | `02_lqr_ilqr_sliding_block.py` | planned |
| 10 | `10_sampling_mpc.html` | Sampling-based model predictive control and MuJoCo MPC | `03_sampling_mpc.py` | planned |
| **IV** | | **Contact statics and grasping** | | |
| 11 | `11_coulomb_friction.html` | Forces and Coulomb friction | `04_friction_cones.py` | exists (was `01_pushing`) |
| 12 | `12_friction_cone_3d.html` | The friction cone in 3D | `04_friction_cones.py` | exists (was `02_the_cone`) |
| 13 | `13_contact_wrench_cone.html` | Multiple contacts and the contact wrench cone | `04_friction_cones.py` | exists (was `03_many_contacts`) |
| 14 | `14_planar_force_closure.html` | Planar force closure with two fingers | `06_force_closure.py` | exists (was `04_two_fingers`) |
| 15 | `15_grasps_in_3d.html` | Grasps in 3D | `06_force_closure.py` | exists (was `05_grasps_in_3d`) |
| 16 | `16_grasp_quality.html` | Grasp quality metrics | `06_force_closure.py` | exists (was `06_grasp_quality`) |
| **V** | | **Contact dynamics and simulation** | | |
| 17 | `17_rigid_body_motion.html` | Rigid-body motion: rotations, quaternions, twists, and wrenches | `05_contact_solvers.py` | planned |
| 18 | `18_contact_dynamics.html` | Rigid-body dynamics with contact | `05_contact_solvers.py` | planned |
| 19 | `19_contact_solvers.html` | Contact solvers: LCP, CCP, and MuJoCo's soft contact | `05_contact_solvers.py` | planned |
| 20 | `20_complementarity_free_contact.html` | Complementarity-free contact | `05_contact_solvers.py` | planned |
| **VI** | | **Synthesis** | | |
| 21 | `21_hybrid_force_velocity_control.html` | Hybrid force-velocity control | `07_hybrid_servoing.py` | planned |
| 22 | `22_contact_trajopt.html` | Trajectory optimization through contact | `08_contact_trajopt.py` | planned |

Decks 11-16 were written first as a stand-alone six-part series ("Friction & Grasping from
Scratch") and renumbered into this one. They still carry the old series-overview slide, old
titles, and no slide ids; integrating them (retitling, ids, notes, cross-links to the new decks,
MuJoCo demos) is planned work.

### Tex chapters behind each deck

| deck | tex lines |
|---|---|
| 01 | 1100-1130 (plus new material: span, basis, rank, null space, solution sets) |
| 02 | 1130-1303 |
| 03 | 157-175, 224-236, 1303-1322 |
| 04 | new; serves 2144-2437 |
| 05 | 97-468 |
| 06 | 469-788 |
| 07 | 789-1099, 1322-1428 |
| 08 | 1429-1758 |
| 09 | 1759-2143 |
| 10 | 2144-2584 |
| 11-13 | 2734-2759, 1322-1428 |
| 14-16 | 3605-3844 |
| 17 | new; 2613-2632 |
| 18 | 2585-2781 |
| 19 | 2782-3178, 3560-3604 |
| 20 | 3179-3559 |
| 21 | none; the Hou & Mason papers, Mason 1981, Raibert & Craig 1981, Modern Robotics 11.6 |
| 22 | 3845-3977 |

The tex states results without building them (KKT, Riccati, the Q-function expansion, Cholesky,
the Schur complement, matrix calculus). The decks build them. At least one tex table is wrong:
in the gradient-descent table at $z_0 = 2$, $\alpha = 0.05$ (lines 294-297) every row after
$k = 0$ was off (the tex gave $J(1.15) = 0.948$, $J'(1.15) = -2.908$, $z_2 = 1.295$; the correct
values are $1.609$, $-2.117$, $1.256$, then $1.312$, $1.335$). That table is now corrected; the
rest of the tex is unchecked. The tex does not build on the owner's workstation as installed:
`algorithm2e.sty` is missing (Ubuntu package `texlive-science`). Treat every tex
number as unverified until a notebook recomputes it, and record tex errors you find in the deck's
notebook section and in the tex itself.

### Hou & Mason coverage map

Each concept the two papers use has one owning deck, which must teach it with a worked example.

| concept | deck |
|---|---|
| vectors, linear combinations, span, basis; matrices as maps, non-square matrices | 01 |
| column / row / null space, rank, rank-nullity; solution set = particular solution + null space; "the identity is the strictest goal" ($G = I$ pins every velocity) | 01 |
| projection, orthonormal bases (Gram-Schmidt, QR); row space is the orthogonal complement of the null space; null-space inclusion as reversed row-space inclusion and its rank test (Hou 2021 eq. 11-13) | 02 |
| least squares, normal equations, minimum-norm solutions and the pseudoinverse (how Hou 2019 Algorithm 2 picks one force distribution) | 02 |
| SVD, condition number, nearly parallel rows, fragility (ill-conditioned) vs infeasibility (inconsistent), the drawer example, row normalization | 02 |
| partial derivatives, chain rule, Jacobian; $\Phi(q) = 0 \Rightarrow J_\Phi \dot q = 0$ | 03 |
| random restarts; random test-problem generation | 04 |
| gradient descent, local minima, projected gradient descent on the unit sphere (Hou 2019 eq. 15) | 05 |
| equality-constrained QP solved by one KKT linear system (Hou 2019 eq. 21-23) | 06 |
| linear inequalities, LP feasibility, guard conditions, the 8-sided polyhedral friction cone | 07, 12 |
| free-body diagrams, equilibrium, statical indeterminacy (four-legged table), contact modes, Coulomb friction, friction cones | 11, 12, 13 |
| configuration space, DOF, rotation matrices, SO(3), gimbal lock, quaternions and the unit-norm constraint, why $\dot q$ is not a velocity, $\dot q = \Omega(q) v$, body vs spatial twists, the adjoint, wrenches, virtual work $\tau = J^\top\lambda$, $f = \Omega^\top\tau$ | 17 |
| holonomic constraints, contact Jacobians, sticking and sliding contacts as velocity constraints, quasi-static Newton law | 18 |
| natural vs artificial constraints, selection matrices, hybrid motion-force control; Hou 2019 formulation and Algorithms 1-2; Hou 2021 crashing index, goal-inclusion conditions, closed-form OCHS | 21 |

A student's six-week study plan for these papers (kept outside the repository) maps onto the
decks as: week 1 -> 01; week 2 -> 02; week 3 -> 11, 12, 13; week 4 -> 03, 17; week 5 -> 05, 06,
07; week 6 -> 17, 21; then the lab. `slides/index.html` carries this reading path.

## 4. Writing rules

The owner reviews every sentence against these rules and flags violations.

* Plain declarative sentences, the technical noun in the subject position, numbers with units,
  the result before the story. The model is an IROS paper, scaled down for a first-year reader.
* Titles of decks and slides are descriptive noun phrases naming the subject: "Gradient descent
  on an ill-conditioned bowl", not "Why gradient descent zig-zags" or "The valley problem". No
  cute halves after a colon, no "The X and the Y" openers, no puns, no question titles.
* Delete on sight: "It is not X, it is Y" inversions; triads of parallel clauses; one-line
  paragraph endings that announce significance; restating a result as a moral; *worth noting,
  importantly, crucially, notably, it turns out, the key insight, fundamentally, elegant,
  powerful, robust* (as praise), *leverage* (as a verb); more than one em-dash per paragraph;
  bold on whole sentences (bold a term or a number); hedging stacks; recap sections that repeat
  the document.
* A deck ends with a slide titled "Results used in later decks": the specific facts, each with
  the deck number that uses it, then the link to the next deck. No summary slide.
* A `.box` holds one concrete, checkable statement, never a moral.
* Negative results and failure modes are stated flatly, in the same voice as positive ones.
* Report files and pages outside the curriculum are date-prefixed (`20260924-name.html`).
  Curriculum decks keep their `NN_` prefix because their order is the reading order.

## 5. Pedagogy

For each concept, in order: a concrete case with numbers the reader can check by hand; a
picture, live when a parameter or motion helps; the general statement with every symbol defined;
the derivation, one algebraic move per build step; a worked example with every intermediate
number; two to five "check yourself" questions with `<details>` answers; a link to the code that
computes the slide's numbers.

Reading-view notes (`<aside class="notes">`, shown by `?mode=read`) are the textbook: 60-200 words
per content slide carrying the full argument, proofs that do not fit, and pointers to the tex. A
student must be able to learn from the reading view alone.

When a deck uses a tool from an earlier deck, it recaps it in one slide or box and links to the
slide that built it (`02_eigenvalues_and_least_squares.html#svd`).

### Notation

| symbol | meaning |
|---|---|
| $a$, $\alpha$ | scalars |
| $\va$, $\vf$ | vectors (bold; KaTeX macros `\va \vf \vv ...` in `lib/deck.js`) |
| $A$, $M$ | matrices |
| $z$ | decision variable (Part II) |
| $J(z)$ | cost (Parts II-III) |
| $x$, $u$ | state and control (Part III) |
| $q$, $v$ | configuration and generalized velocity, $\dot q = \Omega(q) v$ (Parts V-VI) |
| $J_\Phi$, $J_c$ | constraint and contact Jacobians, always subscripted to avoid the cost $J$ |
| $\mu$ | friction coefficient, everywhere |
| $\lambda$ | multiplier of an inequality or contact constraint; contact force (eigenvalues in deck 02 only) |
| $\nu$ | multiplier of an equality constraint |
| $\alpha$ | step size in Part II; friction angle $\arctan\mu$ in Part IV |
| $\sigma_i$ | singular values; $\sigma$ is also a sampling standard deviation in decks 04 and 10 |
| $h$ | time step |
| $\nhat$, $\that$ | contact normal and tangent |

## 6. Slide decks

A deck is one HTML file with no build step. Read `tutorial/slides/README.md`, `template.html`,
`lib/retro.css`, and `11_coulomb_friction.html` before writing one.

* Canvas 1280x720, body text 27 px, `h2` 37 px, `overflow: hidden`. Content past the footer is
  invisible, and `tools/check.py` reports it as an error. Split slides rather than shrink text;
  never go below 19 px.
* Every `<section class="slide">` gets a stable kebab-case `id`; cross-links use `deck.html#id`.
* Build steps: `class="step"`, grouped or reordered with `data-step="n"`.
* Math is KaTeX (`$...$`, `$$...$$`), vendored in `lib/katex/`. Deck-local macros go in
  `window.DECK_MACROS` before `deck.js` loads.
* Figures are inline SVG drawn with `lib/fig.js` (`Fig`). Palette: red `#cc0000` forces, blue
  `#0000cc` normals, green `#008800` friction, pale yellow cones, purple, orange, teal, grays.
  Times New Roman everywhere, 1 px black rules, default link blue: the look of a 1998 professor's
  home page.
* Controls are plain `<input type="range">`, checkboxes and `<select>` in `.controls`, with a
  monospace `.readout` for live numbers.
* No network resources. Every deck works offline from `file://`.

### Libraries in `tutorial/slides/lib/`

| file | role | status |
|---|---|---|
| `deck.js` | navigation, build steps, present / read / print modes, KaTeX macros (`\T \rank \Null \Row \Col \cond \diag \tr \sign \proj \E \Var \SO \se \Ad \dd`, more bold vectors); `#id` hashes; `slideshown` / `slidehidden` on slides and `deckmode` on document; `Deck.pending(promise)` (or `window.DECK_PENDING.push` before `deck.js` runs) and `<html data-settled>`; auto-loads `series.js`, `codemap.js`, `code.js`; prev / next decks from `SERIES`; keys C (code) and M (series menu) | exists |
| `retro.css` | theme and layouts (`.cols`, `.box`, `.example`, `.warn`, ...) | exists |
| `fig.js` | SVG drawing in world coordinates, 2D and simple 3D | exists |
| `contact.js` | friction-cone and planar-grasp mechanics for decks 11-16 | exists |
| `katex/` | KaTeX, MIT license | vendored |
| `mujoco/` | official MuJoCo 3.14.0 WebAssembly build (DeepMind `@mujoco/mujoco`, Apache-2.0) repackaged as classic scripts: `mujoco.js` defines `window.loadMujoco`; `mujoco.wasm.js` holds the gzipped, base64 wasm (2.5 MB) | vendored by `tools/vendor_mujoco.py` |
| `mj.js` | MuJoCo wrapper: `MJ.ready()`, `MJ.sim(xml)`, contacts with forces, `MJ.draw2` / `MJ.draw3` renderers into a `Fig`, `MJ.player` that runs only while its slide is shown | exists, not yet reviewed; `tools/demos/mj_demo.html` passes `check.py`; missing: the Python-vs-WASM parity twins (`tools/twins/mujoco.json`) |
| `num.js` | small dense linear algebra (LU, QR, Cholesky, symmetric eigen, SVD, null / row space bases with rows orthonormal as in Hou & Mason, pinv, cond, RREF with a step log), seeded RNG, LP (simplex), QP, KKT solve, NNLS, finite differences | exists, not yet reviewed; all 113 twin cases in `tools/twins/num.json` pass |
| `plot.js` | axes, curves, contours, heat maps, feasible regions, quiver, linear-map grids, 3D surfaces, histograms, strip charts | exists, not yet reviewed; `tools/demos/plot_demo.html` passes `check.py` (7 warnings) |
| `code.js`, `codemap.js` | slide-to-code overlay (section 7); `codemap.js` is generated by `tools/sync.py` | exists |
| `series.js` | the deck list for navigation and the series menu; set a deck's `status` from `"planned"` to `"exists"` when its file lands (footers and the menu link only existing decks) | exists |
| `models.js` | `tutorial/models/*.xml` as `window.MODELS`; generated by `tools/sync.py` | exists |
| `algo/NN_name.js` | the algorithms behind one deck's live figures, one file per deck | planned |

Decks treat the shared libraries as read-only. Deck-specific algorithms go in the deck's own
`lib/algo/NN_*.js`.

### MuJoCo in the slides

The WebAssembly build runs the real engine inside a slide: about 0.2 s to start and about 10 us
per step for small models, measured in headless Chrome from `file://`. Use it where the real
engine teaches something a toy cannot: soft-contact penetration against `solref` / `solimp`,
pyramidal vs elliptic cones, creep on a tilted plane, a grasp slipping, sampling MPC with real
rollouts, hybrid force-velocity control tilting a block. Models live in `tutorial/models/` and
are loaded by both the notebook (Python `mujoco==3.14.0`, pinned to the WASM version) and the
slide, so both report numbers from the same model on the same engine version.

mjviser (Kevin Zakka's viser-based MuJoCo viewer) needs a running Python server, so it cannot live
in a static deck. It can live in a marimo notebook: `mjviser.scene.ViserMujocoScene(server,
model)` with `scene.update_from_mjdata(data)` does not block, and the planned `tutorial/mjview.py`
wraps it into an iframe (not written yet).

## 7. Code coupling

The slide text, the live figures, and the notebooks must agree number for number.

* Every number on a slide that a reader cannot compute by hand comes from a function in the
  companion notebook, in the section headed with the deck number.
* A slide names that function with `data-code="NOTEBOOK.py:function_name"` on the `<section>` or
  on an inline element. `lib/code.js` turns it into a `[code]` link that opens the
  function's source (extracted into `lib/codemap.js` by `tools/sync.py`) with a GitHub link to
  the exact lines and the command to open the notebook. The reading view shows the source under
  the notes.
* Each JavaScript algorithm with a Python counterpart has cases in `tools/twins/NN.json`, run by
  `tools/twins.py`: the Python function (from the notebook) and the JS function (in headless
  Chrome) get the same inputs and must match to a tolerance.

### Notebooks

marimo notebooks in `tutorial/`, each with a PEP 723 header so `uvx marimo edit --sandbox FILE`
and `uv run --script FILE` work without setup. Notebooks keep their tex-chapter numbers:

| notebook | backs decks | status |
|---|---|---|
| `00_math_toolkit.py` | 01-04 | skeleton: imports, intro, one header cell per deck |
| `01_optimization_fundamentals.py` | 05-07 (and tex ch. 1) | exists |
| `02_lqr_ilqr_sliding_block.py` | 08-09 (tex ch. 3-4) | exists |
| `03_sampling_mpc.py` | 10 (tex ch. 5) | exists |
| `04_friction_cones.py` | 11-13 | exists |
| `05_contact_solvers.py` | 17-20 (tex ch. 7-8) | exists |
| `06_force_closure.py` | 14-16 | exists |
| `07_hybrid_servoing.py` | 21 and the lab | skeleton: imports (mujoco in `try:`), intro, deck 21 and lab header cells |
| `08_contact_trajopt.py` | 22 | skeleton: imports (mujoco in `try:`), intro, deck 22 header cell |

Rules: each deck's cells go after a markdown header cell "Deck NN · Title" (marimo drops bare
comments when it re-saves a file, so anchors must be cells). Public names must be unique within a
notebook. Import `mujoco` inside `try:` and degrade to a message, because notebooks also run in
the browser (molab / Pyodide), where `mujoco` is unavailable. Run a notebook top to bottom with
`tutorial/slides/tools/run_notebook.sh FILE` before calling it done.

## 8. Checking work

All tools live in `tutorial/slides/tools/`; run them from `tutorial/slides/` with `uv run`.
They start the system Chrome through Playwright and hold a file lock
(`~/.cache/contact-slides/*.lock`) so only one browser runs at a time.

| command | what it does |
|---|---|
| `uv run tools/check.py 05` | renders every slide of a deck (number, file name or path) in print mode; errors for content past the footer or right edge, KaTeX errors, undrawn figures, JS errors, broken links and slide ids, duplicate ids, unknown `data-code`; warnings for missing slide ids, missing notes, clipped SVG labels, text under 17 px |
| `uv run tools/check.py 05 --shots DIR` | also one PNG per slide; look at them |
| `uv run tools/shoot.py 05 newton-step --steps 3 --eval "set('#mu', 0.8)" -o a.png` | present-mode screenshot after build steps, control changes, drags (`--drag SEL DX DY`), or several frames of an animation (`--frames N --interval MS`) |
| `uv run tools/twins.py 05` | Python-vs-JS twin cases in `tools/twins/05.json` |
| `tools/run_notebook.sh ../01_optimization_fundamentals.py` | runs a notebook top to bottom, one notebook at a time |
| `uv run tools/vendor_mujoco.py` | re-vendors the pinned MuJoCo WASM build and checks its sha256 |
| `uv run tools/sync.py [--check] [--list FILE]` | regenerates `lib/codemap.js` and `lib/models.js`; `--check` fails on stale files or on a `data-code` in `NN_*.html`, `template.html` or `tools/demos/*.html` that does not resolve; `--list FILE` prints the references a notebook offers |
| `uv run export.py [deck]` | PDF, PNG or PPTX copies in `slides/out/` |

A deck is done when: `check.py` reports zero errors and no missing ids or notes; every slide PNG
has been looked at; every live control has been exercised with `shoot.py`; its twin cases pass;
its notebook runs top to bottom; `sync.py --check` passes; and the text follows section 4.

## 9. Machine etiquette

The owner's workstation also runs long simulation batches and a desktop session, and it has
frozen when the sum of memory use outran RAM. On that machine:

* Start browsers only through the tools above (they serialize on a lock).
* Before anything that runs longer than a minute or uses more than one core, check free memory
  and memory pressure (`~/.claude/bin/resguard.sh status` there; `free -g` and
  `/proc/pressure/memory` elsewhere). Run heavy jobs through
  `~/.claude/bin/resguard.sh run --mem <GB> --cpu <pct> -- cmd` when it exists.
* Parallel sweeps are serial queues by default. Ask before multi-hour grids.

## 10. Status and next steps

As of 2026-09-24, end of the first working session.

Done and checked:

* Decks 11-16 (renumbered from the old 01-06; content unchanged apart from links).
* Test tools: `check.py`, `shoot.py`, `twins.py`, `run_notebook.sh`, `sync.py`, `vendor_mujoco.py`.
* The vendored MuJoCo 3.14.0 WASM build; it steps from `file://` at about 9 us per step.
* The `deck.js` upgrade, `code.js`, `series.js`, and the generated `codemap.js` / `models.js`.
* Notebook skeletons `00`, `07`, `08`, deck header cells in `01`, `02`, `03`, `05`, and the
  renumbered references in `04` and `06`. All nine notebooks run top to bottom.
* Starter models `tutorial/models/box_on_plane.xml`, `tilted_plane.xml`, `two_finger_pinch.xml`.

Checks at the time of the last commit: `check.py` reports 0 errors on decks 11-16, `template.html`,
`tools/demos/plot_demo.html` and `tools/demos/mj_demo.html`; `sync.py --check` reports 0 problems
(59 definitions, 3 models). The warnings on decks 11-16 are the missing slide ids and notes that
their integration will add.

Written but interrupted before their authors' final self-test (the agents building them were
stopped to save usage): `lib/num.js`, `lib/plot.js`, `lib/mj.js`, the two demo decks. Review them
before building decks on them. Known gaps:

* `tutorial/mjview.py` (mjviser in marimo) and `tools/twins/mujoco.json` do not exist yet.

Also done: `slides/index.html` rewritten for the 22-deck series; all 113 `num.js` twin cases
pass after three cases were corrected (a hand-written rank, and two LPs with non-unique optima
now compared by objective); the gradient-descent table in the tex (section 3), corrected but not rebuilt, since
the tex needs `algorithm2e.sty`.

Not started, in order: decks 01-10 and 17-22, the lab, and the integration of decks 11-16 (scope
of each in [`tutorial/PLAN.md`](tutorial/PLAN.md), which also holds the concept-ownership table
and the assignment of MuJoCo demos to decks); corrections to the rest of the tex and cross-references from it to the decks. (`slides/README.md` and `slides/index.html` already lists the 22-deck series, links the six
existing decks, and index.html carries the Hou & Mason reading path; link each new deck in both as it lands.)

Update this section and the status columns above whenever a piece lands.
