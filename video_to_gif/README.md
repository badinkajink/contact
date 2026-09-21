# video_to_gif

A local video → GIF studio. Crop in space, trim in time, downsample resolution and
fps, hit render. No upload, no queue, no 50 MB ceiling — the page talks to a Python
server on loopback that shells out to `ffmpeg` and reads your files where they sit.

```bash
python video_to_gif/video_to_gif.py            # opens http://127.0.0.1:7842
python video_to_gif/video_to_gif.py clip.mov   # ...with a file already loaded
python video_to_gif/video_to_gif.py --port 9000 --no-browser
```

Needs `ffmpeg` and `ffprobe` on PATH (`brew install ffmpeg`). Nothing else — no pip
install, stdlib only.

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
