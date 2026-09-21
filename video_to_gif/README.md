# video_to_gif

A local video → GIF studio. Crop in space, trim in time, downsample resolution and
fps, hit render. No upload, no queue, no 50 MB ceiling — the page talks to a Python
server on loopback that shells out to `ffmpeg` and reads your files where they sit.

Runs on macOS and Linux. Python 3.8+ and ffmpeg 4.3+, no pip packages at all.

## Install

**macOS** — needs [Homebrew](https://brew.sh):

```bash
brew install ffmpeg && curl -L https://github.com/badinkajink/contact/archive/refs/heads/main.tar.gz | tar xz --strip-components=1 contact-main/video_to_gif && python3 video_to_gif/video_to_gif.py
```

**Ubuntu 22.04** (and other Debian-based distros):

```bash
sudo apt update && sudo apt install -y ffmpeg python3 curl zenity fonts-liberation && curl -L https://github.com/badinkajink/contact/archive/refs/heads/main.tar.gz | tar xz --strip-components=1 contact-main/video_to_gif && python3 video_to_gif/video_to_gif.py
```

Either line downloads just this folder into the current directory and opens the tool at
<http://127.0.0.1:7842>. To start it again later:

```bash
python3 video_to_gif/video_to_gif.py            # opens the browser for you
python3 video_to_gif/video_to_gif.py clip.mov   # ...with a file already loaded
python3 video_to_gif/video_to_gif.py --port 9000 --no-browser
```

If you'd rather have the whole repo and `git pull` for updates:

```bash
git clone https://github.com/badinkajink/contact.git && python3 contact/video_to_gif/video_to_gif.py
```

### What those packages are for

| Package | Why |
| --- | --- |
| `ffmpeg` | every decode and encode; also provides `ffprobe`. Ubuntu 22.04 ships 4.4, which is plenty |
| `python3` | runs the server — standard library only, nothing to `pip install` |
| `zenity` | the native "choose a video" dialog on Linux. Skip it and the built-in file browser takes over |
| `fonts-liberation` | Arial/Times lookalikes, so text overlays land close to what macOS renders. Optional |

On macOS `python3` comes with the Xcode command line tools — run `xcode-select --install`
if the shell says it can't find it.

## Getting a video in

- **Open video…** — native file dialog; the file is read in place.
- **Browse** — built-in file browser, for when the dialog is in the way.
- **Drag and drop** — the only path that copies bytes (into a temp dir), because the
  browser hands over file contents, not a path.
- Paste a path into the box and hit enter.

## The controls that matter

Output size is roughly `frames × width × height`, so the two knobs with real leverage
are **fps** and **width** — not colour depth. Halving fps halves the file.

| Knob | What it does |
| --- | --- |
| crop box | drag to move, handles to resize, `shift`-drag a corner to keep the current aspect |
| aspect chips | lock the box to 1:1, 16:9, 9:16… |
| in/out handles | trim on the thumbnail timeline; the selection loops while you play |
| width | output width; height follows the crop's aspect (forced even for mp4/webm) |
| fps | frame rate of the output, independent of the source |
| speed | `1.5` = 1.5× faster, and fewer frames to store |
| colours | GIF palette size, 2–256 |
| dither | `sierra2_4a` looks best; `bayer` is noticeably smaller; `none` is flat and sharp |
| palette | `diff` optimises for a moving subject, `full` for the whole clip, `single` recomputes per frame (best colour, biggest file) |
| ping-pong | plays forward then backward — doubles the frames |
| text | captions and speed badges, drawn over the crop — see below |

**Presets**: tiny (320px/10fps/64c), balanced (480px/15fps/128c), crisp (720px/24fps/256c).

Formats: GIF, MP4 (h264), WebM (VP9), APNG. MP4 can keep the audio track when the clip
isn't reversed or sped up.

## Text

Set the speed, then hit **+ speed badge (2×)** — you get bold white text with a black
outline tucked into the bottom-right corner, and it relabels itself if you change the
speed again. Edit the text by hand and it stops following the speed.

**+ text** adds a free one. Drag it anywhere on the frame, or use the 3×3 grid to snap it
to an edge. Per text: font (sans / serif / mono / impact), size as a % of output height,
fill and outline colours, outline weight, bold, left/centre/right alignment for multiple
lines, and an optional translucent backdrop box. Newlines in the box work. So do emoji.

`show from` / `to` hides the text outside that window, in output seconds — blank means the
whole clip. It applies to every text at once; render twice if you need separate timings.

Text scales with whatever output size you pick, because the size is a fraction of the
frame rather than a pixel count, and it's drawn *after* the downscale so it stays crisp.

The **poster** font is Impact on macOS and Ubuntu Condensed on Ubuntu; the panel prints
which font it actually resolved to, so you're never guessing.

The text is rendered in the browser on a canvas at exactly the output resolution and
handed to ffmpeg as a PNG for the `overlay` filter — so the preview is literally the
render, and any font on your Mac is available. (ffmpeg's own `drawtext` would need a
freetype build; homebrew's ffmpeg 9 doesn't have one.) For GIFs the text is composited
before the palette is measured, so a white caption stays white instead of dithering.

## Keyboard

`space` play/pause · `←`/`→` step one frame (`shift` = 10) · `i`/`o` set in/out ·
`l` toggle loop · `t` add text · `home`/`end` jump to in/out · `enter` render ·
`esc` close overlays

## How it renders

GIFs go through ffmpeg's two-pass palette workflow — `palettegen` over the trimmed,
cropped, scaled stream, then `paletteuse` — which is what separates a decent GIF from a
muddy one. Per-frame palettes (`single`) can't be a separate file, so that mode runs as
a single pass. **show ffmpeg command** in the sidebar prints the exact invocations if
you want to run one in a terminal or stick it in a script.

Rotated phone video is handled: ffprobe's rotation metadata is applied before you see
the frame, so the crop box is in the same coordinate space ffmpeg will crop in.

## Notes

- The server binds `127.0.0.1` only, and each opened file gets a random token — the
  page can't read paths that weren't opened through it.
- If the preview stays black but the timeline thumbnails are fine, the browser can't
  decode that codec (common for h265/mkv). Cropping numerically and rendering still work.
- Output lands next to the source by default; change the folder and name under **Save to**.
- Cancelling a render kills ffmpeg and deletes the half-written file.
- On Linux, **Show in file manager** asks the desktop over D-Bus to select the file and
  falls back to opening the folder with `xdg-open`.
- Over SSH or on WSL no browser opens; the URL is printed instead, so open it yourself
  (`--host 0.0.0.0` if the browser is on another machine — it's an open door on that
  network, so only do it on one you trust).
