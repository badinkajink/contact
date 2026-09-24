# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright", "marimo", "numpy", "scipy", "matplotlib", "mujoco==3.14.0"]
# ///
"""Check that each JavaScript algorithm behind a live figure matches its Python twin.

    uv run tools/twins.py            # every tools/twins/*.json
    uv run tools/twins.py 05 num     # only 05.json and num.json

A twin file looks like

    {
      "notebook": "01_optimization_fundamentals.py",       # optional: run it, use its definitions
      "py_setup": "import numpy as np\\nA = np.eye(2)",     # optional: Python run after the notebook
      "scripts": ["lib/num.js", "lib/algo/05_unconstrained.js"],   # relative to slides/
      "js_setup": "const A = [[1,0],[0,1]];",              # optional
      "cases": [
        {"name": "Newton from z0 = 2", "py": "run_newton(2.0, 6)[-1][1]",
         "js": "Unconstrained.newton(2.0, 6).at(-1).z", "tol": 1e-9}
      ]
    }

"py" is a Python expression evaluated in the notebook's namespace (numpy is `np`). "js" is a
JavaScript expression (it may use await) evaluated after the scripts load, in the same scope as
js_setup. Numbers match when |js - py| <= tol * (1 + |py|); lists, tuples, arrays and dicts are
compared element by element. "notebook" may also be a path relative to tutorial/, e.g.
"labs/hybrid_servoing/hfvc.py" for a plain module.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from _browser import SLIDES, TUTORIAL, launch, locked  # noqa: E402

TWINS = SLIDES / "tools" / "twins"
_ns_cache = {}


def notebook_namespace(rel):
    """Definitions of a marimo notebook (via app.run()) or of a plain module."""
    if rel in _ns_cache:
        return _ns_cache[rel]
    path = (TUTORIAL / rel).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"twin_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ns = {}
    if hasattr(mod, "app") and hasattr(mod.app, "run"):
        _, defs = mod.app.run()
        ns.update(defs)
    ns.update({k: v for k, v in vars(mod).items() if not k.startswith("__")})
    _ns_cache[rel] = ns
    return ns


def plain(x):
    """numpy / tuples -> JSON-like Python values."""
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.bool_):
            return bool(x)
    except ImportError:
        pass
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    return x


def compare(a, b, tol, path=""):
    """Return a list of mismatch descriptions (empty when a ~ b)."""
    if isinstance(b, bool) or isinstance(a, bool):
        return [] if bool(a) == bool(b) else [f"{path}: js {a} vs py {b}"]
    if isinstance(b, (int, float)) and isinstance(a, (int, float)):
        if b is None or a is None:
            return [f"{path}: js {a} vs py {b}"]
        if math.isnan(b) and math.isnan(a):
            return []
        if math.isinf(b) or math.isinf(a):
            return [] if a == b else [f"{path}: js {a} vs py {b}"]
        return [] if abs(a - b) <= tol * (1 + abs(b)) else [f"{path}: js {a!r} vs py {b!r} (diff {abs(a - b):.3g})"]
    if isinstance(b, list) and isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: js length {len(a)} vs py length {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += compare(x, y, tol, f"{path}[{i}]")
        return out
    if isinstance(b, dict) and isinstance(a, dict):
        out = []
        for k in b:
            if k not in a:
                out.append(f"{path}.{k}: missing in js")
            else:
                out += compare(a[k], b[k], tol, f"{path}.{k}")
        return out
    return [] if a == b else [f"{path}: js {a!r} vs py {b!r}"]


HOST = SLIDES / "tools" / "twin_host.html"


def run_file(page, spec_path: Path):
    spec = json.loads(spec_path.read_text())
    ns = {}
    if spec.get("notebook"):
        ns.update(notebook_namespace(spec["notebook"]))
    import numpy as np

    ns.setdefault("np", np)
    if spec.get("py_setup"):
        exec(spec["py_setup"], ns)
    page.goto(HOST.as_uri())
    for s in spec.get("scripts", []):
        page.add_script_tag(url=(SLIDES / s).resolve().as_uri())
    cases = spec["cases"]
    body = (spec.get("js_setup") or "") + "\nconst __out = [];\n"
    for i, c in enumerate(cases):
        body += (f"try {{ __out.push({{ok: true, v: await (async () => ({c['js']}))()}}); }}"
                 f" catch (e) {{ __out.push({{ok: false, v: String(e && e.stack || e)}}); }}\n")
    body += "return JSON.parse(JSON.stringify(__out, (k, v) => (v instanceof Float64Array || v instanceof Float32Array || v instanceof Int32Array) ? Array.from(v) : (typeof v === 'number' && !isFinite(v) ? String(v) : v)));"
    js_vals = page.evaluate(f"async () => {{ {body} }}")
    fails = 0
    for c, jv in zip(cases, js_vals):
        tol = c.get("tol", 1e-9)
        try:
            pv = plain(eval(c["py"], ns))
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {c['name']}: python raised {e!r}")
            fails += 1
            continue
        if not jv["ok"]:
            print(f"  FAIL {c['name']}: javascript raised {jv['v'][:300]}")
            fails += 1
            continue
        v = jv["v"]
        v = json.loads(json.dumps(v).replace('"Infinity"', "Infinity").replace('"-Infinity"', "-Infinity").replace('"NaN"', "NaN"))
        bad = compare(v, pv, tol)
        if bad:
            fails += 1
            print(f"  FAIL {c['name']}: " + "; ".join(bad[:4]) + (" ..." if len(bad) > 4 else ""))
        else:
            print(f"  ok   {c['name']}")
    return len(cases), fails


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", help="twin file stems (default: all)")
    args = ap.parse_args()
    files = sorted(TWINS.glob("*.json"))
    if args.which:
        files = [f for f in files if f.stem in args.which or f.stem.split("_")[0] in args.which]
    if not files:
        sys.exit("no twin files found")
    from playwright.sync_api import sync_playwright

    total = failed = 0
    with locked("chrome"), sync_playwright() as p:
        browser = launch(p)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        for f in files:
            print(f"== {f.name}")
            try:
                n, k = run_file(page, f)
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL the whole file: {e!r}")
                n, k = 1, 1
            total += n
            failed += k
            for e in errors:
                print(f"  page error: {e}")
            errors.clear()
        browser.close()
    print(f"total: {total - failed}/{total} twin cases agree")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
