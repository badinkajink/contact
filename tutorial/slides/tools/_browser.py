"""Shared helpers for the slide tools: one headless Chrome at a time, deck loading.

The workstation runs other jobs, so every tool that starts a browser takes an exclusive
file lock first. Tools use the system Chrome (Playwright channel "chrome") and fall back to
Playwright's bundled Chromium only if Chrome is missing.
"""

import contextlib
import fcntl
import sys
import time
from pathlib import Path

SLIDES = Path(__file__).resolve().parent.parent
TUTORIAL = SLIDES.parent
REPO = TUTORIAL.parent
LOCKDIR = Path.home() / ".cache" / "contact-slides"
W, H = 1280, 720


@contextlib.contextmanager
def locked(name="chrome"):
    """Hold an exclusive lock so only one browser (or notebook run) is alive at a time."""
    LOCKDIR.mkdir(parents=True, exist_ok=True)
    with open(LOCKDIR / f"{name}.lock", "w") as fh:
        t0 = time.time()
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[waiting for the {name} lock held by another tool]", file=sys.stderr)
            fcntl.flock(fh, fcntl.LOCK_EX)
            print(f"[got the {name} lock after {time.time() - t0:.0f} s]", file=sys.stderr)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def launch(p):
    from playwright.sync_api import Error as PlaywrightError

    try:
        return p.chromium.launch(channel="chrome", headless=True)
    except PlaywrightError:
        return p.chromium.launch(headless=True)


def resolve_deck(arg: str) -> Path:
    """Accept 'NN', 'NN_name.html', or a path."""
    p = Path(arg)
    if p.exists():
        return p.resolve()
    if (SLIDES / arg).exists():
        return (SLIDES / arg).resolve()
    hits = sorted(SLIDES.glob(f"{arg}_*.html")) if arg.isdigit() or len(arg) == 2 else []
    if len(hits) == 1:
        return hits[0]
    sys.exit(f"no deck matches {arg!r}")


def all_decks():
    return sorted(SLIDES.glob("[0-9][0-9]_*.html"))


# Resolves once fonts are ready (deck.js sets data-ready) and every promise registered with
# Deck.pending(...) has settled (deck.js sets data-settled). Older decks only set data-ready.
WAIT_JS = """
() => {
  const d = document.documentElement.dataset;
  if (d.ready !== '1') return false;
  if (window.Deck && typeof window.Deck.pending === 'function') return d.settled === '1';
  return true;
}
"""


def open_page(browser, url: str, mode="print", scale=1, timeout=45_000, log=None):
    page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=scale)
    msgs = []

    def on_console(m):
        if m.type in ("error", "warning"):
            msgs.append((m.type, m.text))

    page.on("console", on_console)
    page.on("pageerror", lambda e: msgs.append(("pageerror", str(e))))
    sep = "&" if "?" in url else "?"
    page.goto(url if mode is None else f"{url}{sep}mode={mode}")
    try:
        page.wait_for_function(WAIT_JS, timeout=timeout)
    except Exception as e:  # noqa: BLE001 - report and continue with what rendered
        msgs.append(("timeout", f"deck never became ready: {e}"))
    return page, msgs
