# Slides

The slide decks of *Optimization, Contact, and Control from Scratch*: 22 decks in six parts,
from vectors to hybrid force-velocity control, plus the retro theme and the libraries they are
built on. Open `index.html` in any browser. Decks 11-16 are written; the other sixteen are
planned (scope in `../PLAN.md`; conventions and status in `../../AGENTS.md`).

| decks | part | companion notebook |
|---|---|---|
| 01-04 | mathematical toolkit | `../00_math_toolkit.py` |
| 05-07 | optimization | `../01_optimization_fundamentals.py` |
| 08-10 | trajectory optimization and control | `../02_lqr_ilqr_sliding_block.py`, `../03_sampling_mpc.py` |
| 11-13 | Coulomb friction, friction cones, contact wrench cones (written) | `../04_friction_cones.py` |
| 14-16 | force closure, 3D grasps, grasp quality (written) | `../06_force_closure.py` |
| 17-20 | rigid-body motion, contact dynamics, contact solvers, complementarity-free contact | `../05_contact_solvers.py` |
| 21-22 | hybrid force-velocity control, trajectory optimization through contact | `../07_hybrid_servoing.py`, `../08_contact_trajopt.py` |

## Why HTML

- Plain text, diffable, no build step. A deck is one `.html` file.
- Math is KaTeX, vendored in `lib/katex/` (MIT license), so decks work offline.
- Figures are inline SVG drawn by small scripts, so they're exact and many are
  live (sliders, draggable handles).
- The look matches the project page: Times, black on white, 1px rules, default
  link blue.

## Presenting

Keys: `→`/`space` next, `←` back, `R` reading view (all slides on one scrolling
page, with notes), `F` fullscreen, `12 Enter` jumps to slide 12, `?` help.
Every slide has a URL: `12_friction_cone_3d.html#17`. `?mode=read` opens the
reading view directly.

## Exporting

```sh
uv run export.py                    # every deck -> out/pdf/<deck>.pdf
uv run export.py 14_planar_force_closure.html --png --pptx
```

`export.py` drives headless Chromium through Playwright (downloaded on first
run). `--png` writes one 2x image per slide; `--pptx` wraps those images in a
PowerPoint file with speaker notes, for venues that insist on PowerPoint.
Printing from Chrome also works: the page size is already 1280×720. `out/` is
ignored by git; the HTML is the source.

## Writing a new deck

Copy `template.html`; it demonstrates every layout. The pieces:

- `lib/retro.css`: the theme. Layouts: `.cols` (halves), `.cols l2`/`r2`
  (3:2), `.cols three`; boxes: `.box` (key idea), `.example`, `.warn`,
  `.quote`; `.slide.title` and `.slide.section`.
- `lib/deck.js`: navigation, scaling, build steps, reading/print modes.
  Add `class="step"` to reveal an element on the next key press;
  `data-step="2"` groups or reorders steps (SVG groups work too).
  `<aside class="notes">` shows only in the reading view.
- `lib/fig.js`: `Fig("#svg-id", {w, h, xlim, ylim})` gives a drawing surface
  in world coordinates (y up) with `arrow`, `wedge` (2D friction cone),
  `ground`, `surface`, `rect`, `circle`, `arc`, `label` (`f_n`, `f_{t1}`
  sub/superscripts), `handle` (draggable), and `in3(view)` for simple 3D
  (`cone`, `pyramid`, `ring`).
- `lib/contact.js`: the mechanics behind the live figures (cone tests, maximum
  dissipation, puck simulation, Nguyen, exact planar force-closure and
  epsilon). Each function has a Python twin in the companion notebooks.
- `lib/plot.js`, `lib/num.js`, `lib/mj.js`: plotting (axes, contours, heat maps,
  regions, strip charts), small dense numerics (SVD, null spaces, LP, QP), and
  real MuJoCo 3.14.0 in the slide (WebAssembly in `lib/mujoco/`). Each file's
  header comment is its manual.
- `data-code="NOTEBOOK.py:function"` on a slide links it to the notebook
  function that computes its numbers (key `C`, or the `[code]` footer link);
  `tools/sync.py` regenerates `lib/codemap.js` and `lib/models.js`.
- Checking: `uv run tools/check.py NN [--shots DIR]` (layout, math, links,
  figures), `uv run tools/shoot.py` (screenshots after interactions),
  `uv run tools/twins.py NN` (JavaScript vs the notebook's Python). See
  `AGENTS.md` section 8 for the definition of a finished deck.
