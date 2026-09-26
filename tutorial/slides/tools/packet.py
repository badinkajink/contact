# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""Build a review packet for the owner: contact sheets of every slide plus a slide ledger.

    uv run tools/packet.py 01            # -> out/review/01_vectors_and_matrices/
    uv run tools/packet.py 01 02 --cols 5

The packet holds
  sheet-1.png, sheet-2.png, ...   20 slide thumbnails per sheet, labelled with number and id
  REVIEW.md                       check.py result, then one row per slide: id, title, build steps,
                                  figures, words of notes, and the notebook functions (data-code)
                                  that compute its numbers, followed by what to verify by hand
  shots/NN-id.png                 the full-size slides the sheets are made from

out/ is ignored by git; rebuild the packet after every change to the deck.
"""

import argparse
import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SLIDES = HERE.parent
OUT = SLIDES / "out" / "review"


def resolve(arg):
    p = SLIDES / arg
    if p.exists():
        return p
    hits = sorted(SLIDES.glob(f"{arg}_*.html"))
    if len(hits) == 1:
        return hits[0]
    sys.exit(f"no deck matches {arg!r}")


class Ledger(HTMLParser):
    """One record per <section class="slide">: id, title, steps, svgs, notes words, data-code."""

    def __init__(self):
        super().__init__()
        self.rows, self.cur, self.depth, self.grab = [], None, 0, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if tag == "section" and "slide" in cls:
            self.cur = {"id": a.get("id") or "", "title": "", "steps": 0, "svgs": 0, "notes": 0,
                        "code": [a["data-code"]] if a.get("data-code") else [], "kind": " ".join(c for c in cls if c != "slide")}
            self.depth = 1
            return
        if not self.cur:
            return
        if tag == "br" and self.grab == "title":
            self.cur["title"] += " "
        if tag == "section":
            self.depth += 1
        if "step" in cls:
            self.cur["steps"] += 1
        if tag == "svg":
            self.cur["svgs"] += 1
        if a.get("data-code") and tag != "section":
            self.cur["code"].append(a["data-code"])
        if tag in ("h1", "h2") and not self.cur["title"]:
            self.grab = "title"
        if tag == "aside" and "notes" in cls:
            self.grab = "notes"

    def handle_endtag(self, tag):
        if not self.cur:
            return
        if tag in ("h1", "h2") and self.grab == "title":
            self.grab = None
        if tag == "aside" and self.grab == "notes":
            self.grab = None
        if tag == "section":
            self.depth -= 1
            if self.depth == 0:
                self.rows.append(self.cur)
                self.cur = None

    def handle_data(self, data):
        if not self.cur:
            return
        if self.grab == "title":
            self.cur["title"] += data
        elif self.grab == "notes":
            self.cur["notes"] += len(data.split())


def font(size):
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sheets(shots, dest, cols, per):
    tw, th, pad, lab = 320, 180, 10, 22
    f = font(15)
    out = []
    for k in range(0, len(shots), per):
        chunk = shots[k:k + per]
        rows = (len(chunk) + cols - 1) // cols
        img = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + lab + pad) + pad), "white")
        d = ImageDraw.Draw(img)
        for i, p in enumerate(chunk):
            x = pad + (i % cols) * (tw + pad)
            y = pad + (i // cols) * (th + lab + pad)
            im = Image.open(p).convert("RGB").resize((tw, th), Image.LANCZOS)
            img.paste(im, (x, y + lab))
            d.rectangle([x - 1, y + lab - 1, x + tw, y + lab + th], outline="black")
            d.text((x, y + 3), p.stem[:44], fill="black", font=f)
        path = dest / f"sheet-{k // per + 1}.png"
        img.save(path)
        out.append(path)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decks", nargs="+")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--per", type=int, default=20, help="thumbnails per sheet")
    args = ap.parse_args()
    for arg in args.decks:
        deck = resolve(arg)
        dest = OUT / deck.stem
        shots = dest / "shots"
        shots.mkdir(parents=True, exist_ok=True)
        for old in shots.glob("*.png"):
            old.unlink()
        rep = dest / "check.json"
        r = subprocess.run(["uv", "run", "--quiet", str(HERE / "check.py"), str(deck), "--shots", str(dest / "tmp"),
                            "--json", str(rep), "--quiet"], capture_output=True, text=True, cwd=SLIDES)
        summary = [ln for ln in r.stdout.splitlines() if ln.startswith(("==", "total")) or "ERROR" in ln]
        tmp = dest / "tmp" / deck.stem
        for p in sorted(tmp.glob("*.png")):
            p.replace(shots / p.name)
        for d in (tmp, dest / "tmp"):
            if d.exists():
                d.rmdir()
        pngs = sorted(shots.glob("*.png"))
        made = sheets(pngs, dest, args.cols, args.per)

        led = Ledger()
        led.feed(deck.read_text(encoding="utf-8"))
        lines = [f"# Review packet: {deck.name}", "",
                 f"{len(led.rows)} slides. Contact sheets: " + ", ".join(p.name for p in made) + ".", "",
                 "## check.py", "", "```", *summary, "```", "",
                 "## Slides", "",
                 "| # | id | title | steps | figures | notes (words) | code |",
                 "|---|---|---|---|---|---|---|"]
        for i, s in enumerate(led.rows, 1):
            title = re.sub(r"\s+", " ", s["title"]).strip().replace("|", "\\|")
            code = ", ".join(f"`{c}`" for c in s["code"]) or ""
            lines.append(f"| {i} | {s['id']} | {title} | {s['steps']} | {s['svgs']} | {s['notes']} | {code} |")
        thin = [s["id"] or str(i) for i, s in enumerate(led.rows, 1) if s["notes"] < 60 and "section" not in s["kind"] and "title" not in s["kind"]]
        lines += ["", "## What to verify by hand", "",
                  "1. Skim the contact sheets for layout: cramped slides, overlapping labels, empty figures.",
                  "2. Open the deck, press R for the reading view, and read three slides closely: every symbol "
                  "defined, no skipped algebra, notes that teach.",
                  "3. On two slides with numbers, press C and check the numbers against the notebook function "
                  "shown (or run the notebook).",
                  "4. Drag every slider on one live figure to both ends; the figure should stay sensible.",
                  "5. Titles are descriptive noun phrases; no banned words (AGENTS.md section 4).", ""]
        if thin:
            lines += [f"Content slides with fewer than 60 words of notes: {', '.join(thin)}.", ""]
        (dest / "REVIEW.md").write_text("\n".join(lines))
        print(f"{deck.name}: {len(pngs)} slides, {len(made)} sheet(s) in {dest.relative_to(SLIDES)}/")
        for ln in summary:
            print("  " + ln)


if __name__ == "__main__":
    main()
