# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright"]
# ///
"""Screenshot one slide in present mode, after build steps and interactions.

    uv run tools/shoot.py 05 newton-derivation -o /tmp/a.png            # slide by id
    uv run tools/shoot.py 05 12 --steps 3 -o /tmp/a.png                 # slide 12, 3 build steps
    uv run tools/shoot.py 19 solref-live --eval "set('#tc', 0.05)" --wait 1500 -o /tmp/b.png
    uv run tools/shoot.py 19 solref-live --frames 5 --interval 400 -o /tmp/anim.png  # anim-1.png ...

--eval runs JavaScript in the page after the slide is shown. Helpers available there:
  set(selector, value)   sets an <input>/<select> and fires input + change events
  click(selector)        clicks an element
  drag(selector, dx, dy) drags an SVG handle by (dx, dy) screen pixels
Console errors are printed, so a broken interaction shows up as text too.
"""

import argparse
import sys
from pathlib import Path

from _browser import launch, locked, open_page, resolve_deck

HELPERS = r"""
window.set = (sel, v) => { const e = document.querySelector(sel); if (!e) throw new Error('no ' + sel);
  if (e.type === 'checkbox') e.checked = !!v; else e.value = v;
  e.dispatchEvent(new Event('input', {bubbles: true})); e.dispatchEvent(new Event('change', {bubbles: true})); };
window.click = (sel) => { const e = document.querySelector(sel); if (!e) throw new Error('no ' + sel); e.click(); };
0;  // a script whose value is a function would be called by page.evaluate; end on a number
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("deck")
    ap.add_argument("slide", help="slide id or 1-based number")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--steps", type=int, default=-1, help="build steps to reveal (-1 = all)")
    ap.add_argument("--eval", action="append", default=[], help="JS to run after showing the slide")
    ap.add_argument("--drag", nargs=3, action="append", default=[], metavar=("SEL", "DX", "DY"))
    ap.add_argument("--wait", type=int, default=600, help="ms to wait before the (first) shot")
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--interval", type=int, default=500)
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--mode", default="present", choices=["present", "read", "print"])
    args = ap.parse_args()

    deck = resolve_deck(args.deck)
    from playwright.sync_api import sync_playwright

    with locked("chrome"), sync_playwright() as p:
        browser = launch(p)
        url = deck.as_uri() + "#" + args.slide
        page, msgs = open_page(browser, url, mode=args.mode, scale=args.scale)
        page.evaluate(HELPERS)
        if args.mode == "present":
            n = page.evaluate("""(sl) => {
                const all = Array.from(document.querySelectorAll('main.deck > section.slide'));
                let i = /^[0-9]+$/.test(sl) ? parseInt(sl, 10) - 1 : all.findIndex(s => s.id === sl);
                if (i < 0) throw new Error('no slide ' + sl);
                window.Deck.show(i + 1);
                return i;
            }""", args.slide)
            steps = args.steps
            if steps < 0:
                steps = page.evaluate("""(i) => {
                    const s = document.querySelectorAll('main.deck > section.slide')[i];
                    const els = Array.from(s.querySelectorAll('.step'));
                    const keys = new Set(els.map((e, j) => e.dataset.step !== undefined ? e.dataset.step : 'k' + j));
                    return keys.size; }""", n)
            for _ in range(steps):
                page.evaluate("window.Deck.next()")
        for js in args.eval:
            try:
                page.evaluate(js)
            except Exception as e:  # noqa: BLE001
                print(f"eval error: {e}", file=sys.stderr)
        for sel, dx, dy in args.drag:
            h = page.locator(f"section.slide.current {sel}" if args.mode == "present" else sel).first
            b = h.bounding_box()
            if not b:
                print(f"drag: {sel} not visible", file=sys.stderr)
                continue
            x, y = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + float(dx), y + float(dy), steps=8)
            page.mouse.up()
        page.wait_for_timeout(args.wait)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        target = page.locator("section.slide.current").first if args.mode == "present" else page
        for k in range(args.frames):
            path = out if args.frames == 1 else out.with_name(f"{out.stem}-{k + 1}{out.suffix}")
            target.screenshot(path=str(path))
            print(f"wrote {path}")
            if k + 1 < args.frames:
                page.wait_for_timeout(args.interval)
        for kind, text in msgs:
            print(f"{kind}: {text}", file=sys.stderr)
        browser.close()


if __name__ == "__main__":
    main()
