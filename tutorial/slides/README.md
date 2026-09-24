# Slides

Six HTML slide decks, *Friction & Grasping from Scratch*, plus the retro
theme they're built on. Open `index.html` in any browser.

| deck | topic | companion code |
|---|---|---|
| `01_pushing.html` | forces, Coulomb's law, the 2D friction cone | `../04_friction_cones.py` |
| `02_the_cone.html` | 3D cones, maximum dissipation, elliptic cones, pyramids | `../04_friction_cones.py` |
| `03_many_contacts.html` | torque, slide-or-tip, indeterminacy, wrench cones | `../04_friction_cones.py` |
| `04_two_fingers.html` | force closure, Nguyen's theorem, the normal-angle test | `../06_force_closure.py` |
| `05_grasps_in_3d.html` | 6D wrenches, soft fingers, three-finger grasps | `../06_force_closure.py` |
| `06_grasp_quality.html` | LP / convex-hull tests, Ferrari–Canny epsilon | `../06_force_closure.py` |

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
Every slide has a URL: `02_the_cone.html#17`. `?mode=read` opens the
reading view directly.

## Exporting

```sh
uv run export.py                    # every deck -> out/pdf/<deck>.pdf
uv run export.py 04_two_fingers.html --png --pptx
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
