#!/usr/bin/env python3
"""video_to_gif -- a local, no-upload video -> GIF studio.

    python video_to_gif/video_to_gif.py [video]

Opens http://127.0.0.1:7842 in your browser. Crop in space, trim in time,
downsample resolution/fps, hit render. ffmpeg does the work; nothing is
uploaded anywhere -- the server binds to loopback and reads your files in
place, so there is no size limit but your disk.

Requires: ffmpeg + ffprobe on PATH. Everything else is Python stdlib.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
GIFSICLE = shutil.which("gifsicle")

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".gif", ".wmv",
              ".flv", ".mpg", ".mpeg", ".ts", ".m2ts", ".mts", ".3gp", ".ogv"}

FORMATS = {
    "gif": {"ext": ".gif", "mime": "image/gif"},
    "mp4": {"ext": ".mp4", "mime": "video/mp4"},
    "webm": {"ext": ".webm", "mime": "video/webm"},
    "apng": {"ext": ".png", "mime": "image/apng"},
}
DITHERS = {"none", "bayer", "floyd_steinberg", "sierra2", "sierra2_4a"}
STATS_MODES = {"full", "diff", "single"}

MEDIA: dict[str, dict] = {}       # token -> {path, meta}
OVERLAYS: dict[str, Path] = {}    # sha1 -> text-layer png on disk
THUMBS: dict[str, bytes] = {}     # token:n -> jpeg mosaic
JOBS: dict[str, dict] = {}        # job id -> state
LOCK = threading.Lock()
SCRATCH = Path(tempfile.mkdtemp(prefix="video_to_gif-"))
PRELOAD: dict | None = None

atexit.register(lambda: shutil.rmtree(SCRATCH, ignore_errors=True))


# ---------------------------------------------------------------- ffprobe ----

def _rate(value: str | None) -> float:
    if not value or value in ("0/0", "N/A"):
        return 0.0
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe(path: Path) -> dict:
    """Pull the handful of facts the UI needs out of ffprobe."""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0 or not out.stdout.strip():
        detail = (out.stderr or "").strip().splitlines()
        raise ValueError(f"ffprobe cannot read {path.name}"
                         + (f": {detail[-1]}" if detail else ""))
    info = json.loads(out.stdout)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("no video stream in this file")

    fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate")) or 25.0
    duration = float(video.get("duration") or info.get("format", {}).get("duration") or 0.0)

    rotation = 0
    for side in video.get("side_data_list", []) or []:
        if "rotation" in side:
            rotation = int(round(float(side["rotation"]))) % 360
    if not rotation:
        try:
            rotation = int(video.get("tags", {}).get("rotate", 0)) % 360
        except (TypeError, ValueError):
            rotation = 0

    width, height = int(video["width"]), int(video["height"])
    # ffmpeg and browsers both auto-apply the display matrix, so report the
    # dimensions as seen, and crop in that same space.
    if rotation in (90, 270):
        width, height = height, width

    nb_frames = int(video.get("nb_frames") or 0) or int(round(duration * fps))
    return {
        "path": str(path),
        "name": path.name,
        "dir": str(path.parent),
        "size": path.stat().st_size,
        "width": width,
        "height": height,
        "fps": round(fps, 6),
        "duration": round(duration, 6),
        "frames": nb_frames,
        "codec": video.get("codec_name", "?"),
        "rotation": rotation,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }


def register(path: Path, temporary: bool = False) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    meta = probe(path)
    meta["temporary"] = temporary
    meta["default_out_dir"] = default_out_dir(path, temporary)
    token = uuid.uuid4().hex
    with LOCK:
        MEDIA[token] = {"path": path, "meta": meta}
    return {"token": token, "meta": meta}


def default_out_dir(path: Path, temporary: bool) -> str:
    if not temporary and os.access(path.parent, os.W_OK):
        return str(path.parent)
    for candidate in (Path.home() / "Downloads", Path.home() / "Desktop", Path.home()):
        if candidate.is_dir() and os.access(candidate, os.W_OK):
            return str(candidate)
    return str(Path.cwd())


# ------------------------------------------------------------- thumbnails ----

def thumb_strip(token: str, count: int = 48) -> bytes:
    """One horizontal mosaic of `count` frames, used as the timeline backdrop."""
    key = f"{token}:{count}"
    cached = THUMBS.get(key)
    if cached:
        return cached
    entry = MEDIA[token]
    path, meta = entry["path"], entry["meta"]
    duration = max(meta["duration"], 0.04)
    rate = max(count / duration, 0.000001)
    tile_h = 72
    out = SCRATCH / f"strip-{key.replace(':', '-')}.jpg"
    cmd = [
        FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-y",
        "-i", str(path),
        "-vf", (f"fps={rate:.6f},scale=-1:{tile_h}:flags=fast_bilinear,"
                f"tile={count}x1:padding=0:color=black"),
        "-frames:v", "1", "-q:v", "6", str(out),
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=180)
    data = out.read_bytes()
    with LOCK:
        THUMBS[key] = data
    return data


# ----------------------------------------------------------------- render ----

def _num(params: dict, key: str, default: float) -> float:
    try:
        value = params.get(key)
        return default if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return default


def plan(token: str, params: dict) -> dict:
    """Turn UI params into concrete ffmpeg invocations + an output path."""
    entry = MEDIA[token]
    path, meta = entry["path"], entry["meta"]
    fmt = params.get("format", "gif")
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}")

    duration_src = max(meta["duration"], 0.04)
    start = max(0.0, min(_num(params, "start", 0.0), duration_src - 0.01))
    end = max(start + 0.02, min(_num(params, "end", duration_src), duration_src))
    span = end - start

    crop = params.get("crop") or {}
    cx = int(round(max(0, _num(crop, "x", 0))))
    cy = int(round(max(0, _num(crop, "y", 0))))
    cw = int(round(_num(crop, "w", meta["width"])))
    ch = int(round(_num(crop, "h", meta["height"])))
    cw = max(2, min(cw, meta["width"] - cx))
    ch = max(2, min(ch, meta["height"] - cy))
    cropped = (cx, cy, cw, ch) != (0, 0, meta["width"], meta["height"])

    fps = max(0.5, min(_num(params, "fps", min(meta["fps"], 15)), 120))
    speed = max(0.05, min(_num(params, "speed", 1.0), 20))
    out_w = int(round(_num(params, "width", cw)))
    out_w = max(8, min(out_w, 8192))
    out_h = max(2, int(round(ch * out_w / cw)))
    if fmt in ("mp4", "webm"):
        out_w -= out_w % 2
        out_h -= out_h % 2

    overlay = OVERLAYS.get(str(params.get("overlay") or ""))
    if overlay is not None and not overlay.is_file():
        overlay = None
    window = params.get("text_window") or {}
    enable = ""
    if overlay is not None and (window.get("from") or window.get("to")):
        seen_from = max(0.0, _num(window, "from", 0.0))
        seen_to = _num(window, "to", span / max(_num(params, "speed", 1.0), 0.05))
        # commas inside a filter option have to be escaped, not quoted
        enable = f":enable=between(t\\,{seen_from:.3f}\\,{seen_to:.3f})"

    reverse = bool(params.get("reverse"))
    boomerang = bool(params.get("boomerang"))
    keep_audio = (bool(params.get("keep_audio")) and meta["has_audio"]
                  and fmt in ("mp4", "webm") and not (reverse or boomerang))

    chain = []
    if cropped:
        chain.append(f"crop={cw}:{ch}:{cx}:{cy}")
    if abs(speed - 1.0) > 1e-6:
        chain.append(f"setpts=PTS/{speed:.6f}")
    chain.append(f"fps={fps:g}")
    if (out_w, out_h) != (cw, ch):
        chain.append(f"scale={out_w}:{out_h}:flags=lanczos")
    base = ",".join(chain)

    out_frames = max(1, int(round(span / speed * fps)))
    if boomerang:
        out_frames *= 2

    in_args = ["-ss", f"{start:.6f}", "-t", f"{span:.6f}", "-i", str(path)]
    ov_index = None
    if overlay is not None:
        ov_index = 1
        in_args += ["-i", str(overlay)]
    out_path = resolve_out_path(params, meta, fmt)
    passes: list[dict] = []

    def picture(label: str, extras: bool) -> str:
        """Graph from source to [label]: crop, retime, scale, text, then time tricks.

        The text layer goes in before the palette is measured, so its white and
        black end up in the palette instead of being dithered into mush.
        """
        segs = [f"[0:v]{base}[pic0]"]
        last = "pic0"
        if ov_index is not None:
            segs.append(f"[{ov_index}:v]scale={out_w}:{out_h}:flags=bicubic[txt]")
            segs.append(f"[{last}][txt]overlay=0:0:format=auto{enable}[pic1]")
            last = "pic1"
        if extras:
            if reverse and not boomerang:
                segs.append(f"[{last}]reverse[pic2]")
                last = "pic2"
            if boomerang:
                segs.append(f"[{last}]split[ba][bb];[bb]reverse[bb2];"
                            f"[ba][bb2]concat=n=2:v=1:a=0[pic3]")
                last = "pic3"
        segs.append(f"[{last}]null[{label}]")
        return ";".join(segs) + ";"

    if fmt in ("gif", "apng"):
        colors = int(max(2, min(_num(params, "colors", 128), 256)))
        dither = params.get("dither", "sierra2_4a")
        dither = dither if dither in DITHERS else "sierra2_4a"
        stats = params.get("stats_mode", "diff")
        stats = stats if stats in STATS_MODES else "diff"
        loop_forever = params.get("loop", "forever") != "once"

    if fmt == "gif":
        use = [f"dither={dither}"]
        if dither == "bayer":
            use.append(f"bayer_scale={int(max(0, min(_num(params, 'bayer_scale', 3), 5)))}")
        if stats == "diff":
            use.append("diff_mode=rectangle")
        gif_out = ["-map", "[out]", "-an", "-loop", "0" if loop_forever else "-1", str(out_path)]

        if stats == "single":
            # A palette per frame: has to stay one pass, since the palettes are a
            # stream rather than a file.
            use.append("new=1")
            graph = (picture("seq", True) + f"[seq]split[s1][s2];"
                     f"[s2]palettegen=max_colors={colors}:stats_mode=single[pal];"
                     f"[s1][pal]paletteuse={':'.join(use)}[out]")
            passes.append({
                "label": "writing gif", "weight": 1.0, "frames": out_frames,
                "cmd": [FFMPEG, "-hide_banner", "-nostdin", "-y", "-v", "error",
                        "-progress", "pipe:1", "-nostats", *in_args,
                        "-filter_complex", graph, *gif_out],
            })
        else:
            # Two passes: one palette for the whole clip, then map onto it.
            palette = SCRATCH / f"palette-{uuid.uuid4().hex}.png"
            passes.append({
                "label": "analysing colours",
                "weight": 0.35,
                "frames": max(1, int(round(span / speed * fps))),
                "cmd": [FFMPEG, "-hide_banner", "-nostdin", "-y", "-v", "error",
                        "-progress", "pipe:1", "-nostats", *in_args, "-an",
                        "-filter_complex", picture("pic", False)
                        + f"[pic]palettegen=max_colors={colors}:stats_mode={stats}[out]",
                        "-map", "[out]", "-update", "1", str(palette)],
            })
            passes.append({
                "label": "writing gif",
                "weight": 0.65,
                "frames": out_frames,
                "cmd": [FFMPEG, "-hide_banner", "-nostdin", "-y", "-v", "error",
                        "-progress", "pipe:1", "-nostats", *in_args, "-i", str(palette),
                        "-filter_complex", picture("seq", True)
                        + f"[seq][{(ov_index or 0) + 1}:v]paletteuse={':'.join(use)}[out]",
                        *gif_out],
            })
    else:
        cmd = [FFMPEG, "-hide_banner", "-nostdin", "-y", "-v", "error",
               "-progress", "pipe:1", "-nostats", *in_args,
               "-filter_complex", picture("out", True), "-map", "[out]"]
        if fmt == "mp4":
            crf = int(max(0, min(_num(params, "crf", 23), 51)))
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        elif fmt == "webm":
            crf = int(max(0, min(_num(params, "crf", 32), 63)))
            cmd += ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0",
                    "-row-mt", "1", "-pix_fmt", "yuv420p"]
        else:  # apng -- needs the muxer named, or the .png extension picks image2
            cmd += ["-c:v", "apng", "-f", "apng", "-plays", "0" if loop_forever else "1",
                    "-pix_fmt", "rgb24"]
        if keep_audio:
            cmd += ["-map", "0:a:0", "-c:a", "aac" if fmt == "mp4" else "libopus",
                    "-b:a", "128k", "-shortest"]
        else:
            cmd += ["-an"]
        cmd.append(str(out_path))
        passes.append({"label": f"encoding {fmt}", "weight": 1.0,
                       "frames": out_frames, "cmd": cmd})

    return {
        "passes": passes,
        "out_path": out_path,
        "format": fmt,
        "summary": {
            "width": out_w, "height": out_h, "fps": round(fps, 3),
            "frames": out_frames, "duration": round(span / speed, 3),
            "start": round(start, 3), "end": round(end, 3),
            "crop": {"x": cx, "y": cy, "w": cw, "h": ch},
        },
    }


def resolve_out_path(params: dict, meta: dict, fmt: str) -> Path:
    ext = FORMATS[fmt]["ext"]
    out_dir = Path(str(params.get("out_dir") or meta["default_out_dir"])).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(params.get("out_name") or "").strip()
    if stem.lower().endswith(ext):
        stem = stem[: -len(ext)]
    stem = re.sub(r"[/\\\0]", "_", stem) or Path(meta["name"]).stem
    candidate = out_dir / f"{stem}{ext}"
    n = 1
    while candidate.exists() and not params.get("overwrite"):
        candidate = out_dir / f"{stem}-{n}{ext}"
        n += 1
    return candidate


def run_job(job: dict, steps: list[dict]) -> None:
    total_weight = sum(s["weight"] for s in steps) or 1.0
    done_weight = 0.0
    try:
        for step in steps:
            if job["status"] == "cancelled":
                return
            job["stage"] = step["label"]
            proc = subprocess.Popen(
                step["cmd"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )
            job["proc"] = proc
            errors = deque(maxlen=60)
            err_thread = threading.Thread(
                target=lambda: [errors.append(line.rstrip()) for line in proc.stderr],
                daemon=True)
            err_thread.start()

            frames = max(1, step["frames"])
            for line in proc.stdout:
                key, _, value = line.strip().partition("=")
                if key == "frame":
                    try:
                        share = min(1.0, int(value) / frames)
                    except ValueError:
                        continue
                    job["progress"] = round((done_weight + share * step["weight"]) / total_weight, 4)
            code = proc.wait()
            err_thread.join(timeout=1)
            job["log"] = [l for l in errors if l]
            if job["status"] == "cancelled":
                return
            if code != 0:
                job["status"] = "error"
                job["error"] = "\n".join(job["log"]) or f"ffmpeg exited with {code}"
                return
            done_weight += step["weight"]
            job["progress"] = round(done_weight / total_weight, 4)

        out = Path(job["out_path"])
        if not out.exists():
            job["status"] = "error"
            job["error"] = "ffmpeg reported success but wrote no file"
            return
        job["size"] = out.stat().st_size
        job["progress"] = 1.0
        job["status"] = "done"
        job["elapsed"] = round(time.time() - job["started"], 2)
    except Exception as exc:                                  # noqa: BLE001
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        job["proc"] = None


def start_job(token: str, params: dict) -> dict:
    built = plan(token, params)
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id, "status": "running", "progress": 0.0, "stage": "starting",
        "out_path": str(built["out_path"]), "out_name": built["out_path"].name,
        "format": built["format"], "summary": built["summary"],
        "started": time.time(), "log": [], "error": None, "size": None,
        "proc": None, "elapsed": None,
    }
    with LOCK:
        JOBS[job_id] = job
    threading.Thread(target=run_job, args=(job, built["passes"]), daemon=True).start()
    return job


def cancel_job(job: dict) -> None:
    """SIGTERM, then SIGKILL if ffmpeg dawdles, then bin the half-written file."""
    job["status"] = "cancelled"
    job["stage"] = "cancelled"
    proc = job.get("proc")

    def reaper():
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        partial = Path(job["out_path"])
        if partial.exists():
            partial.unlink(missing_ok=True)

    threading.Thread(target=reaper, daemon=True).start()


def job_view(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in ("proc", "started")}


# ------------------------------------------------------- native file dialog --

def native_pick(kind: str = "file") -> str | None:
    if sys.platform == "darwin":
        prompt = "Choose a video" if kind == "file" else "Choose an output folder"
        verb = "choose file" if kind == "file" else "choose folder"
        script = f'POSIX path of ({verb} with prompt "{prompt}")'
        res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        return res.stdout.strip() or None
    zenity = shutil.which("zenity")
    if zenity:
        cmd = [zenity, "--file-selection"]
        if kind != "file":
            cmd.append("--directory")
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.stdout.strip() or None
    return None


def reveal(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)], check=False)
    elif sys.platform.startswith("linux") and shutil.which("xdg-open"):
        subprocess.run(["xdg-open", str(path.parent)], check=False)
    elif os.name == "nt":
        subprocess.run(["explorer", f"/select,{path}"], check=False)


# ------------------------------------------------------------------ server ---

class Handler(BaseHTTPRequestHandler):
    server_version = "video_to_gif"
    protocol_version = "HTTP/1.1"

    # --- plumbing -----------------------------------------------------------
    def log_message(self, fmt, *args):            # quieter console
        if os.environ.get("V2G_DEBUG"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, body=b"", ctype="application/octet-stream", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD" and body:
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _fail(self, message, code=400):
        self._json({"error": str(message)}, code)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length) or b"{}")

    def _serve_file(self, path: Path, mime: str | None = None, download: str | None = None):
        """Static/media serving with Range support so <video> can seek."""
        if not path.is_file():
            return self._fail("not found", 404)
        mime = mime or guess_mime(path)
        size = path.stat().st_size
        extra = {"Accept-Ranges": "bytes"}
        if download:
            extra["Content-Disposition"] = f'attachment; filename="{download}"'
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng:
            match = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if match:
                first, last = match.groups()
                if first:
                    start = min(int(first), max(size - 1, 0))
                    end = int(last) if last else size - 1
                else:                                  # suffix range
                    start = max(0, size - int(last or 0))
                end = min(end, size - 1)
                code = 206
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        length = max(0, end - start + 1)
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(length))
        for key, value in extra.items():
            self.send_header(key, value)
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(1 << 20, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    # --- routes -------------------------------------------------------------
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urlparse(self.path)
        route = unquote(url.path)
        query = parse_qs(url.query)
        try:
            if route in ("/", "/index.html"):
                return self._serve_file(STATIC / "index.html", "text/html; charset=utf-8")
            if route.startswith("/static/"):
                target = (STATIC / route[len("/static/"):]).resolve()
                if STATIC.resolve() not in target.parents:
                    return self._fail("nope", 403)
                return self._serve_file(target)
            if route == "/api/state":
                return self._json({
                    "preload": PRELOAD,
                    "gifsicle": bool(GIFSICLE),
                    "platform": sys.platform,
                    "cwd": str(Path.cwd()),
                    "home": str(Path.home()),
                })
            if route == "/api/pick":
                kind = (query.get("kind") or ["file"])[0]
                picked = native_pick(kind)
                if not picked:
                    return self._json({"cancelled": True})
                if kind == "file":
                    return self._json(register(Path(picked)))
                return self._json({"path": picked})
            if route.startswith("/api/media/"):
                token = route.rsplit("/", 1)[-1]
                entry = MEDIA.get(token)
                if not entry:
                    return self._fail("unknown media token", 404)
                return self._serve_file(entry["path"])
            if route.startswith("/api/thumbs/"):
                token = route.rsplit("/", 1)[-1]
                if token not in MEDIA:
                    return self._fail("unknown media token", 404)
                count = int((query.get("n") or [48])[0])
                return self._send(200, thumb_strip(token, max(8, min(count, 200))), "image/jpeg")
            if route.startswith("/api/job/"):
                job = JOBS.get(route.rsplit("/", 1)[-1])
                if not job:
                    return self._fail("unknown job", 404)
                return self._json(job_view(job))
            if route.startswith("/api/result/"):
                job = JOBS.get(route.rsplit("/", 1)[-1])
                if not job or job["status"] != "done":
                    return self._fail("result not ready", 404)
                name = (query.get("download") or [None])[0]
                return self._serve_file(Path(job["out_path"]),
                                        FORMATS[job["format"]]["mime"],
                                        download=job["out_name"] if name else None)
            if route == "/api/browse":
                return self._json(browse_dir((query.get("path") or [str(Path.home())])[0]))
            return self._fail("not found", 404)
        except (ValueError, FileNotFoundError, IsADirectoryError, PermissionError) as exc:
            return self._fail(exc, 400)
        except Exception as exc:                                  # noqa: BLE001
            return self._fail(f"{type(exc).__name__}: {exc}", 500)

    def do_POST(self):
        route = unquote(urlparse(self.path).path)
        try:
            if route == "/api/open":
                body = self._read_json()
                return self._json(register(Path(body["path"])))
            if route == "/api/upload":
                name = self.headers.get("X-Filename") or "dropped.mp4"
                name = re.sub(r"[/\\\0]", "_", unquote(name))
                length = int(self.headers.get("Content-Length") or 0)
                staging = SCRATCH / f"drop-{uuid.uuid4().hex[:8]}"
                staging.mkdir(parents=True, exist_ok=True)
                dest = staging / name
                with dest.open("wb") as fh:
                    remaining = length
                    while remaining > 0:
                        chunk = self.rfile.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        fh.write(chunk)
                        remaining -= len(chunk)
                return self._json(register(dest, temporary=True))
            if route == "/api/overlay":
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 < length <= 48 << 20:
                    return self._fail("text layer missing or too large")
                data = self.rfile.read(length)
                if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                    return self._fail("text layer must be a png")
                key = hashlib.sha1(data).hexdigest()
                dest = SCRATCH / f"text-{key}.png"
                if not dest.exists():
                    dest.write_bytes(data)
                with LOCK:
                    OVERLAYS[key] = dest
                return self._json({"overlay": key, "bytes": len(data)})
            if route == "/api/render":
                body = self._read_json()
                token = body.get("token")
                if token not in MEDIA:
                    return self._fail("unknown media token", 404)
                return self._json(job_view(start_job(token, body.get("params") or {})))
            if route == "/api/plan":
                body = self._read_json()
                token = body.get("token")
                if token not in MEDIA:
                    return self._fail("unknown media token", 404)
                built = plan(token, body.get("params") or {})
                return self._json({
                    "summary": built["summary"],
                    "out_path": str(built["out_path"]),
                    "commands": [" ".join(p["cmd"]) for p in built["passes"]],
                })
            if route.startswith("/api/cancel/"):
                job = JOBS.get(route.rsplit("/", 1)[-1])
                if not job:
                    return self._fail("unknown job", 404)
                cancel_job(job)
                return self._json(job_view(job))
            if route == "/api/reveal":
                body = self._read_json()
                target = Path(str(body.get("path", ""))).expanduser()
                if target.exists():
                    reveal(target)
                    return self._json({"ok": True})
                return self._fail("no such path", 404)
            return self._fail("not found", 404)
        except (ValueError, FileNotFoundError, IsADirectoryError, PermissionError) as exc:
            return self._fail(exc, 400)
        except Exception as exc:                                  # noqa: BLE001
            return self._fail(f"{type(exc).__name__}: {exc}", 500)


def browse_dir(raw: str) -> dict:
    """Tiny server-side directory listing, for picking files without a dialog."""
    base = Path(raw).expanduser()
    if not base.is_dir():
        base = Path.home()
    base = base.resolve()
    dirs, files = [], []
    try:
        for child in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith("."):
                continue
            try:
                if child.is_dir():
                    dirs.append({"name": child.name, "path": str(child)})
                elif child.suffix.lower() in VIDEO_EXTS:
                    files.append({"name": child.name, "path": str(child),
                                  "size": child.stat().st_size})
            except OSError:
                continue
    except PermissionError:
        pass
    return {"path": str(base), "parent": str(base.parent), "dirs": dirs, "files": files}


def guess_mime(path: Path) -> str:
    return {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".json": "application/json",
        ".svg": "image/svg+xml",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".gif": "image/gif", ".webp": "image/webp",
        ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
        ".webm": "video/webm", ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
        ".ogv": "video/ogg",
    }.get(path.suffix.lower(), "application/octet-stream")


def serve(host: str, port: int, open_browser: bool) -> None:
    last_error = None
    for candidate in range(port, port + 25):
        try:
            httpd = ThreadingHTTPServer((host, candidate), Handler)
            break
        except OSError as exc:
            last_error = exc
    else:
        raise SystemExit(f"could not bind {host}:{port}-{port + 24}: {last_error}")

    httpd.daemon_threads = True
    url = f"http://{host}:{httpd.server_address[1]}"
    print(f"\n  video_to_gif  ->  {url}")
    print(f"  ffmpeg: {FFMPEG}")
    print("  local only; nothing is uploaded. ctrl-c to stop.\n", flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  bye")
    finally:
        httpd.server_close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", nargs="?", help="video to open on startup")
    parser.add_argument("--port", type=int, default=7842)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    missing = [name for name, binary in (("ffmpeg", FFMPEG), ("ffprobe", FFPROBE)) if not binary]
    if missing:
        raise SystemExit(f"missing {' and '.join(missing)} on PATH -- try: brew install ffmpeg")

    global PRELOAD
    if args.video:
        try:
            PRELOAD = register(Path(args.video))
        except Exception as exc:                                  # noqa: BLE001
            print(f"  could not open {args.video}: {exc}")
    serve(args.host, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
