# /// script
# requires-python = ">=3.10"
# ///
"""Vendor the official MuJoCo WebAssembly build into lib/mujoco/ as classic scripts.

    uv run tools/vendor_mujoco.py                       # pinned 3.14.0, verify sha256, write lib/mujoco/
    uv run tools/vendor_mujoco.py --version 3.15.0 --no-verify   # print new hashes, then pin them below

Source: DeepMind's npm package @mujoco/mujoco (Apache-2.0), fetched from jsdelivr, and the
LICENSE file from the MuJoCo GitHub tag of the same version. The package is an ES module that
fetches mujoco.wasm next to itself; neither works from file://. The script therefore

  1. rewrites mujoco.js into a classic script: `export default loadMujoco;` becomes
     `window.loadMujoco = loadMujoco;`, and `import.meta.url` becomes the script URL captured
     from document.currentScript while the script runs;
  2. gzips mujoco.wasm (level 9, mtime 0, so the output is byte-for-byte reproducible) and
     base64-encodes it into mujoco.wasm.js as `window.MUJOCO_WASM_GZ`. lib/mj.js decodes it
     with DecompressionStream('gzip') and passes it to loadMujoco({wasmBinary}).

Downloads are cached in ~/.cache/contact-slides/mujoco-VERSION/ so a rerun is offline.
"""

import argparse
import base64
import gzip
import hashlib
import sys
import urllib.request
from pathlib import Path

SLIDES = Path(__file__).resolve().parent.parent
OUT = SLIDES / "lib" / "mujoco"
CACHE = Path.home() / ".cache" / "contact-slides"

PINNED = "3.14.0"
# sha256 of the upstream files for the pinned version (jsdelivr lists the same hashes in base64
# at https://data.jsdelivr.com/v1/packages/npm/@mujoco/mujoco@3.14.0).
SHA256 = {
    "3.14.0": {
        "mujoco.js": "bc6980e4bc5ee5f809a6cabde0743798637e113d333f9e85124714d42be1ccae",
        "mujoco.wasm": "2d380f1070a556c3003b4a7c7c10c98a5f61db4e91815de1075ec478e9487e76",
        "mujoco.d.ts": "e68c12b284f4675a8bb598d0e04af84fd2f3d434101f1877764a806aaf8faa19",
        "package.json": "678ac9788eaf5802b64a3d03ff3294631008704729706e914c910bcb2467303f",
        "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    },
}


def urls(version):
    npm = f"https://cdn.jsdelivr.net/npm/@mujoco/mujoco@{version}/"
    return {
        "mujoco.js": npm + "mujoco.js",
        "mujoco.wasm": npm + "mujoco.wasm",
        "mujoco.d.ts": npm + "mujoco.d.ts",
        "package.json": npm + "package.json",
        "LICENSE": f"https://raw.githubusercontent.com/google-deepmind/mujoco/{version}/LICENSE",
    }


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(version):
    folder = CACHE / f"mujoco-{version}"
    folder.mkdir(parents=True, exist_ok=True)
    got = {}
    for name, url in urls(version).items():
        path = folder / name
        if not path.exists():
            print(f"download {url}")
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            path.write_bytes(data)
        got[name] = path.read_bytes()
    return got


HEAD = """/* MuJoCo {version} WebAssembly bindings (Google DeepMind, Apache-2.0), repackaged as a classic
 * script by tools/vendor_mujoco.py so it loads from file://. Upstream: npm @mujoco/mujoco@{version},
 * mujoco.js sha256 {sha}. Defines window.loadMujoco(moduleArg) -> Promise<module>.
 * Do not edit; rerun the vendoring script instead. */
(function () {{
var __mjScriptURL = (document.currentScript && document.currentScript.src) || location.href;
"""
TAIL = """
window.loadMujoco = loadMujoco;
})();
"""


def convert_js(src: str, version: str, digest: str) -> str:
    n_export = src.count("export default loadMujoco;")
    if n_export != 1:
        sys.exit(f"expected one 'export default loadMujoco;' in mujoco.js, found {n_export}")
    body = src.replace("export default loadMujoco;", "")
    n_meta = body.count("import.meta.url")
    body = body.replace("import.meta.url", "__mjScriptURL")
    if "import.meta" in body:
        sys.exit("mujoco.js still contains import.meta after the rewrite; inspect it by hand")
    print(f"mujoco.js: rewrote 1 export and {n_meta} import.meta.url")
    return HEAD.format(version=version, sha=digest) + body + TAIL


def wasm_script(wasm: bytes, version: str) -> tuple[str, int]:
    gz = gzip.compress(wasm, compresslevel=9, mtime=0)
    b64 = base64.b64encode(gz).decode("ascii")
    info = f'{{version: "{version}", bytes: {len(wasm)}, gzBytes: {len(gz)}, sha256: "{sha(wasm)}"}}'
    js = (f"/* MuJoCo {version} mujoco.wasm, gzip level 9 + base64 (tools/vendor_mujoco.py). */\n"
          f"window.MUJOCO_WASM_INFO = {info};\n"
          f'window.MUJOCO_WASM_GZ = "{b64}";\n')
    return js, len(gz)


README = """# MuJoCo {version} WebAssembly, vendored

Official MuJoCo WebAssembly bindings from Google DeepMind (npm package
[`@mujoco/mujoco@{version}`](https://www.npmjs.com/package/@mujoco/mujoco), source in
[google-deepmind/mujoco/wasm](https://github.com/google-deepmind/mujoco/tree/{version}/wasm)),
licensed Apache-2.0 (full text in `LICENSE`, taken from the MuJoCo repository at tag `{version}`).
The files here are generated by `tools/vendor_mujoco.py`; do not edit them by hand.

Decks do not load these files directly. They load `lib/mj.js`, which inserts both scripts on
first use and wraps the bindings (see the header of `lib/mj.js` for the API).

| file | size | what it is |
|---|---:|---|
| `mujoco.js` | {js_kb} KB | upstream `mujoco.js` (single-threaded build) rewritten as a classic script; defines `window.loadMujoco` |
| `mujoco.wasm.js` | {wjs_kb} KB | upstream `mujoco.wasm` ({wasm_mb} MB), gzip level 9 ({gz_mb} MB), base64; defines `window.MUJOCO_WASM_GZ` and `window.MUJOCO_WASM_INFO` |
| `mujoco.d.ts` | {dts_kb} KB | upstream TypeScript declarations, unchanged; the reference for every binding name and signature |
| `LICENSE` | {lic_kb} KB | Apache License 2.0 |

## Provenance (sha256 of the upstream files)

| upstream file | sha256 |
|---|---|
{sha_rows}

jsdelivr publishes the same hashes (base64) at
`https://data.jsdelivr.com/v1/packages/npm/@mujoco/mujoco@{version}`.

## Changes made to the upstream files

1. `mujoco.js`: the ES-module line `export default loadMujoco;` is replaced by
   `window.loadMujoco = loadMujoco;`, every `import.meta.url` is replaced by the script URL
   captured from `document.currentScript` while the script runs, and the whole file is wrapped
   in a function so its top-level names stay local. ES modules and `fetch()` of a sibling
   `.wasm` both fail from `file://`; classic scripts do not.
2. `mujoco.wasm`: stored gzipped and base64-encoded inside a script. `lib/mj.js` decodes it with
   `DecompressionStream('gzip')` and passes the bytes as `loadMujoco({{wasmBinary}})`. Decoding
   takes about 130 ms and module start-up about 30 ms in desktop Chrome.

The engine code is unchanged, so a model compiled here and in Python `mujoco=={version}` steps
to the same numbers (checked by `tools/twins/mujoco.json`).

## Updating to a new MuJoCo release

```sh
cd tutorial/slides
uv run tools/vendor_mujoco.py --version X.Y.Z --no-verify   # downloads, prints the new hashes
# paste the printed hashes into SHA256 in tools/vendor_mujoco.py and set PINNED = "X.Y.Z"
uv run tools/vendor_mujoco.py                               # verified rebuild
uv run tools/twins.py mujoco                                # Python mujoco==X.Y.Z vs WASM
```

Also bump `mujoco==` in `tools/twins.py`, the notebooks, and `tutorial/mjview.py`, so the
slides and the notebooks keep running the same engine version. The bindings are young
(upstream calls them work in progress); read the upstream `README.md` diff for renamed
functions before updating.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=PINNED)
    ap.add_argument("--no-verify", action="store_true", help="skip the sha256 check (for a new version)")
    args = ap.parse_args()
    v = args.version

    files = fetch(v)
    pins = SHA256.get(v)
    print(f"MuJoCo {v} upstream files:")
    bad = []
    for name, data in files.items():
        h = sha(data)
        ok = "" if not pins else ("ok" if pins.get(name) == h else "MISMATCH")
        print(f"  {name:13s} {len(data):>10d} B  sha256 {h}  {ok}")
        if pins and pins.get(name) != h:
            bad.append(name)
    if not args.no_verify:
        if not pins:
            sys.exit(f"no pinned hashes for {v}; rerun with --no-verify and pin the hashes above")
        if bad:
            sys.exit("sha256 mismatch for " + ", ".join(bad) + "; refusing to vendor")

    OUT.mkdir(parents=True, exist_ok=True)
    js = convert_js(files["mujoco.js"].decode("utf-8"), v, sha(files["mujoco.js"]))
    (OUT / "mujoco.js").write_text(js, encoding="utf-8")
    wjs, gz_len = wasm_script(files["mujoco.wasm"], v)
    (OUT / "mujoco.wasm.js").write_text(wjs, encoding="ascii")
    (OUT / "mujoco.d.ts").write_bytes(files["mujoco.d.ts"])
    (OUT / "LICENSE").write_bytes(files["LICENSE"])
    rows = "\n".join(f"| `{n}` | `{sha(d)}` |" for n, d in files.items())
    kb = lambda n: f"{n / 1024:.0f}"  # noqa: E731
    (OUT / "README.md").write_text(README.format(
        version=v, sha_rows=rows,
        js_kb=kb(len(js.encode())), wjs_kb=kb(len(wjs)), dts_kb=kb(len(files["mujoco.d.ts"])),
        lic_kb=f"{len(files['LICENSE']) / 1024:.1f}",
        wasm_mb=f"{len(files['mujoco.wasm']) / 1e6:.1f}", gz_mb=f"{gz_len / 1e6:.1f}",
    ), encoding="utf-8")
    print(f"wrote {OUT.relative_to(SLIDES)}/:")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:15s} {p.stat().st_size:>10d} B  sha256 {sha(p.read_bytes())}")


if __name__ == "__main__":
    main()
