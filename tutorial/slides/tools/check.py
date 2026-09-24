# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright"]
# ///
"""Render every slide of one or more decks in headless Chrome and report what is broken.

    uv run tools/check.py 05                      # one deck (by number, file name, or path)
    uv run tools/check.py                         # every NN_*.html deck
    uv run tools/check.py 05 --shots /tmp/shots   # also one PNG per slide, for looking at
    uv run tools/check.py 05 --slides 3,7-9 --shots DIR
    uv run tools/check.py 05 --strict             # warnings count as failures

Errors (exit 1): content running past the footer or the right edge, KaTeX parse errors,
figures whose SVG never got drawn, JavaScript errors, links to files or slide ids that do not
exist, duplicate ids, data-code references missing from lib/codemap.js.
Warnings: slides without an id, SVG labels clipped at the figure border, text under 17 px,
notes missing on content slides.
"""

import argparse
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

from _browser import SLIDES, all_decks, launch, locked, open_page, resolve_deck

PROBE_JS = r"""
() => {
  const out = [];
  const slides = Array.from(document.querySelectorAll('main.deck > section.slide, .slot > section.slide'));
  const codemap = window.CODEMAP || null;
  const ids = {};
  document.querySelectorAll('[id]').forEach(e => { ids[e.id] = (ids[e.id] || 0) + 1; });
  const dup = Object.keys(ids).filter(k => ids[k] > 1);
  const desc = (e) => {
    let s = e.tagName.toLowerCase();
    if (e.id) s += '#' + e.id;
    if (e.className && typeof e.className === 'string') s += '.' + e.className.trim().split(/\s+/).join('.');
    const t = (e.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 50);
    return t ? s + ' "' + t + '"' : s;
  };
  slides.forEach((s, i) => {
    const r = s.getBoundingClientRect();
    const foot = s.querySelector(':scope > .foot');
    const bottom = foot ? foot.getBoundingClientRect().top : r.bottom - 44;
    const rec = { n: i + 1, id: s.id || null, title: (s.querySelector('h1,h2') || {}).textContent || '',
                  errors: [], warnings: [], section: s.classList.contains('section') || s.classList.contains('title') };
    // Overflow: every rendered element must end above the footer and inside the right margin.
    const seen = new Set();
    s.querySelectorAll('*').forEach(e => {
      if (e.closest('.foot') || e.closest('aside.notes') || e.closest('script,style') || e.closest('.katex-mathml')) return;
      if (e.closest('svg') && e.tagName.toLowerCase() !== 'svg') return;
      const cs = getComputedStyle(e);
      if (cs.display === 'none' || cs.visibility === 'hidden' || cs.position === 'fixed') return;
      const b = e.getBoundingClientRect();
      if (b.width === 0 && b.height === 0) return;
      let msg = null;
      if (b.bottom > bottom + 1.5) msg = 'runs ' + Math.round(b.bottom - bottom) + 'px past the footer';
      else if (b.right > r.right - 12) msg = 'runs ' + Math.round(b.right - (r.right - 12)) + 'px past the right edge';
      if (msg) {
        // Report the outermost offender only.
        let p = e.parentElement, inner = false;
        while (p && p !== s) { if (seen.has(p)) { inner = true; break; } p = p.parentElement; }
        if (!inner) { seen.add(e); rec.errors.push('overflow: ' + desc(e) + ' ' + msg); }
      }
    });
    s.querySelectorAll('.katex-error').forEach(e => rec.errors.push('katex: ' + (e.getAttribute('title') || e.textContent).slice(0, 160)));
    s.querySelectorAll('figure svg, svg.fig').forEach(svg => {
      const n = svg.querySelectorAll('path,line,circle,rect,polygon,polyline,text,image,ellipse').length;
      const canvas = svg.parentElement && svg.parentElement.querySelector('canvas');
      if (n === 0 && !canvas) rec.errors.push('figure never drawn: svg' + (svg.id ? '#' + svg.id : ''));
      const sb = svg.getBoundingClientRect();
      let clipped = 0;
      svg.querySelectorAll('text').forEach(t => {
        const tb = t.getBoundingClientRect();
        if (tb.width && (tb.left < sb.left - 2 || tb.right > sb.right + 2 || tb.top < sb.top - 2 || tb.bottom > sb.bottom + 2)) clipped++;
      });
      if (clipped) rec.warnings.push('svg' + (svg.id ? '#' + svg.id : '') + ': ' + clipped + ' label(s) clipped at the figure border');
    });
    s.querySelectorAll('[data-code]').forEach(e => {
      e.dataset.code.split(/[,\s]+/).filter(Boolean).forEach(ref => {
        if (!codemap) rec.warnings.push('data-code ' + ref + ': lib/codemap.js not loaded');
        else if (!codemap[ref]) rec.errors.push('data-code ' + ref + ' is not in lib/codemap.js (run tools/sync.py)');
      });
    });
    if (s.dataset.code) {
      s.dataset.code.split(/[,\s]+/).filter(Boolean).forEach(ref => {
        if (codemap && !codemap[ref]) rec.errors.push('data-code ' + ref + ' is not in lib/codemap.js (run tools/sync.py)');
      });
    }
    let tiny = 0;
    s.querySelectorAll('p,li,td,th,span,div,figcaption,label').forEach(e => {
      if (e.closest('.foot') || e.closest('aside.notes') || e.closest('.katex')) return;
      if (!(e.textContent || '').trim()) return;
      if (parseFloat(getComputedStyle(e).fontSize) < 16.5) tiny++;
    });
    if (tiny) rec.warnings.push(tiny + ' element(s) with text under 17px');
    if (!s.id) rec.warnings.push('slide has no id');
    if (!rec.section && !s.querySelector('aside.notes')) rec.warnings.push('no reader notes');
    rec.links = Array.from(s.querySelectorAll('a[href]')).map(a => a.getAttribute('href'));
    out.push(rec);
  });
  return { slides: out, dup, title: document.title };
}
"""


class IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.nslides = set(), 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        if tag == "section" and "slide" in (a.get("class") or "").split():
            self.nslides += 1


_id_cache = {}


def target_ids(path: Path):
    if path not in _id_cache:
        c = IdCollector()
        c.feed(path.read_text(encoding="utf-8", errors="replace"))
        _id_cache[path] = c
    return _id_cache[path]


def check_link(deck: Path, href: str):
    if re.match(r"^[a-z]+:", href) or href.startswith("//"):
        return None  # external (http, mailto, javascript:)
    if href in ("#", ""):
        return None
    path_part, _, frag = href.partition("#")
    path_part = path_part.split("?")[0]
    target = deck if not path_part else (deck.parent / path_part).resolve()
    if not target.exists():
        return f"link {href}: {target.name} does not exist"
    if frag and target.suffix == ".html":
        info = target_ids(target)
        if frag.isdigit():
            if int(frag) > max(info.nslides, 1) and info.nslides:
                return f"link {href}: {target.name} has only {info.nslides} slides"
        elif frag not in info.ids:
            return f"link {href}: no id '{frag}' in {target.name}"
    return None


def parse_sel(spec, n):
    if not spec:
        return None
    keep = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            keep.update(range(int(a), int(b) + 1))
        elif part.isdigit():
            keep.add(int(part))
        else:
            keep.add(part)  # an id
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decks", nargs="*")
    ap.add_argument("--shots", help="directory for one PNG per slide")
    ap.add_argument("--slides", help="restrict shots to these slides: 3,5-7,newton-step")
    ap.add_argument("--strict", action="store_true", help="warnings are failures")
    ap.add_argument("--json", help="write the full report as JSON here")
    ap.add_argument("--quiet", action="store_true", help="only print problems")
    args = ap.parse_args()

    decks = [resolve_deck(d) for d in args.decks] or all_decks()
    report, nerr, nwarn = {}, 0, 0
    from playwright.sync_api import sync_playwright

    with locked("chrome"), sync_playwright() as p:
        browser = launch(p)
        for deck in decks:
            page, msgs = open_page(browser, deck.as_uri(), mode="print")
            page.wait_for_timeout(300)
            res = page.evaluate(PROBE_JS)
            deck_err = [f"js {kind}: {text}" for kind, text in msgs if kind != "warning"]
            deck_err += [f"duplicate id '{d}'" for d in res["dup"]]
            for s in res["slides"]:
                for href in s.pop("links"):
                    bad = check_link(deck, href)
                    if bad:
                        s["errors"].append(bad)
            if args.shots:
                keep = parse_sel(args.slides, len(res["slides"]))
                folder = Path(args.shots) / deck.stem
                folder.mkdir(parents=True, exist_ok=True)
                loc = page.locator("section.slide")
                for i, s in enumerate(res["slides"]):
                    if keep and (i + 1) not in keep and s["id"] not in keep:
                        continue
                    name = f"{i + 1:02d}" + (f"-{s['id']}" if s["id"] else "") + ".png"
                    loc.nth(i).screenshot(path=str(folder / name))
            page.close()
            report[deck.name] = {"errors": deck_err, "slides": res["slides"]}
            e = len(deck_err) + sum(len(s["errors"]) for s in res["slides"])
            w = sum(len(s["warnings"]) for s in res["slides"])
            nerr += e
            nwarn += w
            print(f"== {deck.name}: {len(res['slides'])} slides, {e} errors, {w} warnings")
            for m in deck_err:
                print(f"   ERROR  {m}")
            for s in res["slides"]:
                tag = f"#{s['n']}" + (f" ({s['id']})" if s["id"] else "")
                for m in s["errors"]:
                    print(f"   ERROR  {tag}: {m}")
                if not args.quiet or args.strict:
                    for m in s["warnings"]:
                        print(f"   warn   {tag}: {m}")
            if args.shots:
                print(f"   shots in {Path(args.shots) / deck.stem}/")
        browser.close()
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    print(f"total: {nerr} errors, {nwarn} warnings")
    sys.exit(1 if nerr or (args.strict and nwarn) else 0)


if __name__ == "__main__":
    main()
