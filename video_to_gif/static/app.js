/* video_to_gif -- browser side. Crop in source pixels, trim in seconds,
   let the local server shell out to ffmpeg. */

const $ = (sel) => document.querySelector(sel);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const MIN_CROP = 8;

const S = {
  token: null, meta: null, dur: 0, fps: 25,
  crop: { x: 0, y: 0, w: 0, h: 0 },
  tin: 0, tout: 0, aspect: 'free', scale: 1,
  texts: [], activeText: null, layout: [],
  job: null, timer: null, dragging: null,
};

const el = {};
for (const id of [
  'srcInfo', 'btnPick', 'btnPick2', 'btnBrowse', 'drop', 'pathInput', 'btnPath', 'upbar', 'upbarFill',
  'viewport', 'frame', 'video', 'cropBox', 'cropDim', 'transport', 'btnPlay', 'btnPrevF', 'btnNextF',
  'tcNow', 'tcDur', 'tcFrame', 'loopSel', 'btnMute', 'timelineWrap', 'timeline', 'thumbs',
  'tlBefore', 'tlAfter', 'tlSel', 'tlBadge', 'tlIn', 'tlOut', 'playhead', 'tlLabels',
  'btnSetIn', 'btnSetOut', 'btnCropReset', 'cx', 'cy', 'cw', 'ch', 'btnTrimAll', 'tIn', 'tOut', 'tLen',
  'fmt', 'outW', 'outDims', 'fps', 'speed', 'colors', 'dither', 'bayerScale', 'statsMode', 'crf',
  'keepAudio', 'boomerang', 'reverse', 'loopForever', 'outDir', 'btnOutDir', 'outName', 'outExt',
  'textLayer', 'textSel', 'btnAddText', 'btnSpeedBadge', 'txList', 'txEdit', 'txText',
  'txFont', 'txSize', 'txColor', 'txOutlineColor', 'txOutline', 'txAlign', 'txBold',
  'txBackdrop', 'txAnchors', 'btnDelText', 'txFrom', 'txTo',
  'estimate', 'btnRender', 'btnCmd', 'cmdBox', 'resultCard', 'resultMedia', 'resultMeta',
  'btnReveal', 'btnDownload', 'btnCloseResult', 'progress', 'progStage', 'progFill', 'progPct',
  'btnCancel', 'browser', 'bwPath', 'bwList', 'bwClose', 'toast',
]) el[id] = document.getElementById(id) || $('#' + id);

/* ------------------------------------------------------------- helpers --- */
const fmtTime = (t) => {
  if (!isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60);
  return `${m}:${(t - m * 60).toFixed(2).padStart(5, '0')}`;
};
const fmtBytes = (n) =>
  n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1048576).toFixed(2)} MB`;

let toastTimer;
function toast(msg, ms = 6000) {
  el.toast.textContent = msg;
  el.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.toast.hidden = true; }, ms);
}

async function api(path, body) {
  const res = await fetch(path, body ? {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  } : undefined);
  const data = await res.json().catch(() => ({ error: `${res.status} ${res.statusText}` }));
  if (data.error) throw new Error(data.error);
  return data;
}

/* --------------------------------------------------------- load a video --- */
function loadMedia(data) {
  S.token = data.token;
  S.meta = data.meta;
  S.dur = data.meta.duration || 0;
  S.fps = data.meta.fps || 25;
  S.crop = { x: 0, y: 0, w: data.meta.width, h: data.meta.height };
  S.tin = 0;
  S.tout = S.dur;
  S.aspect = 'free';
  document.querySelectorAll('.chip.asp').forEach((c) =>
    c.classList.toggle('on', c.dataset.aspect === 'free'));

  el.video.src = `/api/media/${data.token}`;
  el.thumbs.hidden = false;
  el.thumbs.src = `/api/thumbs/${data.token}?n=60`;
  el.drop.hidden = true;
  el.viewport.hidden = el.transport.hidden = el.timelineWrap.hidden = false;
  el.resultCard.hidden = true;
  el.upbar.hidden = true;

  const m = data.meta;
  el.srcInfo.textContent = '';
  const nameEl = document.createElement('b');
  nameEl.textContent = m.name;
  el.srcInfo.append(nameEl, `  ${m.width}×${m.height} · ${m.fps.toFixed(2)} fps · `
    + `${fmtTime(m.duration)} · ${m.codec} · ${fmtBytes(m.size)}`
    + (m.temporary ? ' · copied into temp' : ''));
  el.tcDur.textContent = fmtTime(S.dur);

  el.fps.value = Math.min(15, Math.round(S.fps * 10) / 10);
  el.outW.value = Math.min(m.width, 480);
  el.outDir.value = m.default_out_dir;
  el.outName.value = m.name.replace(/\.[^.]+$/, '');
  el.keepAudio.disabled = !m.has_audio;

  fitFrame();
  drawCrop();
  drawTimeline();
  syncTextPanel();
  syncFormat();
}

async function pick() {
  try {
    const res = await api('/api/pick');
    if (!res.cancelled) loadMedia(res);
  } catch (e) { toast(`could not open: ${e.message}`); }
}

function uploadFile(file) {
  el.upbar.hidden = false;
  el.upbarFill.style.width = '0%';
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload');
  xhr.setRequestHeader('X-Filename', encodeURIComponent(file.name));
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) el.upbarFill.style.width = `${(e.loaded / e.total) * 100}%`;
  };
  xhr.onload = () => {
    let data;
    try { data = JSON.parse(xhr.responseText); } catch { data = { error: 'bad response' }; }
    if (data.error) { el.upbar.hidden = true; toast(data.error); return; }
    loadMedia(data);
  };
  xhr.onerror = () => { el.upbar.hidden = true; toast('copy failed'); };
  xhr.send(file);
}

/* ----------------------------------------------------- frame + crop box --- */
function fitFrame() {
  if (!S.meta) return;
  const box = el.viewport.getBoundingClientRect();
  const avail = { w: Math.max(80, box.width), h: Math.max(80, box.height) };
  const k = Math.min(avail.w / S.meta.width, avail.h / S.meta.height);
  const w = Math.max(40, Math.floor(S.meta.width * k));
  const h = Math.max(40, Math.floor(S.meta.height * k));
  el.frame.style.width = `${w}px`;
  el.frame.style.height = `${h}px`;
  S.scale = w / S.meta.width;
}

function setCrop(rect, fromInputs = false) {
  const W = S.meta.width, H = S.meta.height;
  let w = clamp(Math.round(rect.w), MIN_CROP, W);
  let h = clamp(Math.round(rect.h), MIN_CROP, H);
  const x = clamp(Math.round(rect.x), 0, W - w);
  const y = clamp(Math.round(rect.y), 0, H - h);
  S.crop = { x, y, w, h };
  drawCrop(fromInputs);
}

function drawCrop(skipInputs = false) {
  const { x, y, w, h } = S.crop;
  const s = S.scale;
  Object.assign(el.cropBox.style, {
    left: `${x * s}px`, top: `${y * s}px`, width: `${w * s}px`, height: `${h * s}px`,
  });
  el.cropDim.textContent = `${w}×${h}  @ ${x},${y}`;
  if (!skipInputs) {
    el.cx.value = x; el.cy.value = y; el.cw.value = w; el.ch.value = h;
  }
  updateEstimate();
}

function ratioFor(name) {
  if (name === 'free') return null;
  if (name === 'source') return S.meta.width / S.meta.height;
  const [a, b] = name.split(':').map(Number);
  return a / b;
}

/** Resize keeping `ratio`, anchored on whichever edges are not being dragged. */
function lockAspect(start, moved, dir, ratio, W, H) {
  const hasW = dir.includes('w'), hasE = dir.includes('e');
  const hasN = dir.includes('n'), hasS = dir.includes('s');
  const vertical = dir === 'n' || dir === 's';
  let w = vertical ? moved.h * ratio : moved.w;
  let h = w / ratio;

  const ax = hasW ? start.x + start.w : hasE ? start.x : start.x + start.w / 2;
  const ay = hasN ? start.y + start.h : hasS ? start.y : start.y + start.h / 2;
  const maxW = hasW ? ax : hasE ? W - ax : 2 * Math.min(ax, W - ax);
  const maxH = hasN ? ay : hasS ? H - ay : 2 * Math.min(ay, H - ay);

  const k = Math.min(1, maxW / w, maxH / h);
  w = Math.max(MIN_CROP, w * k);
  h = Math.max(MIN_CROP, w / ratio);
  const x = hasW ? ax - w : hasE ? ax : ax - w / 2;
  const y = hasN ? ay - h : hasS ? ay : ay - h / 2;
  return { x, y, w, h };
}

el.cropBox.addEventListener('pointerdown', (ev) => {
  if (!S.meta) return;
  ev.preventDefault();
  const dir = ev.target.dataset.dir || 'move';
  const start = { ...S.crop };
  const origin = { x: ev.clientX, y: ev.clientY };
  const W = S.meta.width, H = S.meta.height;

  const onMove = (e) => {
    const dx = (e.clientX - origin.x) / S.scale;
    const dy = (e.clientY - origin.y) / S.scale;
    if (dir === 'move') {
      setCrop({ x: start.x + dx, y: start.y + dy, w: start.w, h: start.h });
      return;
    }
    let r = { ...start };
    if (dir.includes('w')) { r.x = start.x + dx; r.w = start.w - dx; }
    if (dir.includes('e')) { r.w = start.w + dx; }
    if (dir.includes('n')) { r.y = start.y + dy; r.h = start.h - dy; }
    if (dir.includes('s')) { r.h = start.h + dy; }
    if (dir.includes('w')) { r.x = clamp(r.x, 0, start.x + start.w - MIN_CROP); r.w = start.x + start.w - r.x; }
    if (dir.includes('n')) { r.y = clamp(r.y, 0, start.y + start.h - MIN_CROP); r.h = start.y + start.h - r.y; }
    r.w = clamp(r.w, MIN_CROP, W - r.x);
    r.h = clamp(r.h, MIN_CROP, H - r.y);

    const ratio = ratioFor(S.aspect) || (e.shiftKey ? start.w / start.h : null);
    if (ratio) r = lockAspect(start, r, dir, ratio, W, H);
    setCrop(r);
  };
  const onUp = () => {
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
  };
  window.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', onUp);
});

for (const [input, key] of [[el.cx, 'x'], [el.cy, 'y'], [el.cw, 'w'], [el.ch, 'h']]) {
  input.addEventListener('input', () => {
    if (!S.meta) return;
    setCrop({ ...S.crop, [key]: Number(input.value) || 0 }, true);
  });
}

document.querySelectorAll('.chip.asp').forEach((chip) => chip.addEventListener('click', () => {
  if (!S.meta) return;
  document.querySelectorAll('.chip.asp').forEach((c) => c.classList.toggle('on', c === chip));
  S.aspect = chip.dataset.aspect;
  const ratio = ratioFor(S.aspect);
  if (ratio) setCrop(lockAspect(S.crop, S.crop, '', ratio, S.meta.width, S.meta.height));
}));

el.btnCropReset.addEventListener('click', () => {
  if (!S.meta) return;
  S.aspect = 'free';
  document.querySelectorAll('.chip.asp').forEach((c) => c.classList.toggle('on', c.dataset.aspect === 'free'));
  setCrop({ x: 0, y: 0, w: S.meta.width, h: S.meta.height });
});

/* ------------------------------------------------------------ text layer --
   ffmpeg's drawtext needs a freetype build, which homebrew's ffmpeg no longer
   has, so the text is drawn here on a canvas at output resolution and handed to
   ffmpeg as a PNG for `overlay`. Upshot: the preview *is* the render, and any
   font on the machine (emoji included) just works. */
const FONT_STACKS = {
  sans: '"Helvetica Neue", Helvetica, Arial, sans-serif',
  serif: '"Times New Roman", Georgia, serif',
  mono: 'Menlo, "SF Mono", ui-monospace, monospace',
  impact: 'Impact, Haettenschweiler, "Arial Narrow Bold", sans-serif',
};

const speedLabel = () => `${String(Math.round((Number(el.speed.value) || 1) * 100) / 100)}×`;
const activeText = () => (S.activeText === null ? null : S.texts[S.activeText]);

function newText(over = {}) {
  return {
    text: '2×', font: 'sans', bold: true, size: 10, color: '#ffffff',
    outlineColor: '#000000', outline: 8, align: 'center', backdrop: false,
    nx: 0.5, ny: 0.86, auto: false, ...over,
  };
}

function addText(over) {
  S.texts.push(newText(over));
  S.activeText = S.texts.length - 1;
  syncTextPanel();
  drawTextLayer();
  return activeText();
}

function removeText(i) {
  S.texts.splice(i, 1);
  S.activeText = S.texts.length ? Math.min(i, S.texts.length - 1) : null;
  syncTextPanel();
  drawTextLayer();
}

function roundRect(ctx, x, y, w, h, r) {
  const k = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + k, y);
  ctx.arcTo(x + w, y, x + w, y + h, k);
  ctx.arcTo(x + w, y + h, x, y + h, k);
  ctx.arcTo(x, y + h, x, y, k);
  ctx.arcTo(x, y, x + w, y, k);
  ctx.closePath();
}

/** Draw every text at output resolution; the canvas is CSS-scaled to the crop box. */
function drawTextLayer() {
  if (!S.meta || !el.textLayer.getContext) return;
  const canvas = el.textLayer;
  const { w: W, h: H } = outputDims();
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  S.layout = [];

  S.texts.forEach((t, i) => {
    const px = Math.max(4, (t.size / 100) * H);
    ctx.font = `${t.bold ? '700' : '400'} ${px}px ${FONT_STACKS[t.font] || FONT_STACKS.sans}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    const lines = String(t.text).split('\n');
    const lineH = px * 1.16;
    const widths = lines.map((l) => ctx.measureText(l).width);
    const boxW = Math.max(1, ...widths);
    const boxH = lineH * lines.length;
    // keep it on frame whatever the size change was; too big to fit -> centre
    const halfW = boxW / 2 / W, halfH = boxH / 2 / H;
    t.nx = halfW >= 0.5 ? 0.5 : clamp(t.nx, halfW, 1 - halfW);
    t.ny = halfH >= 0.5 ? 0.5 : clamp(t.ny, halfH, 1 - halfH);
    const left = t.nx * W - boxW / 2;
    const top = t.ny * H - boxH / 2;
    S.layout[i] = { x: left, y: top, w: boxW, h: boxH };
    if (!String(t.text).trim()) return;

    if (t.backdrop) {
      const pad = px * 0.24;
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      roundRect(ctx, left - pad, top - pad * 0.55, boxW + pad * 2, boxH + pad * 1.1, pad * 0.7);
      ctx.fill();
    }
    ctx.lineWidth = px * (t.outline / 100) * 2;   // canvas strokes centred, ffmpeg's outward
    ctx.strokeStyle = t.outlineColor;
    ctx.fillStyle = t.color;
    ctx.lineJoin = 'round';
    ctx.miterLimit = 2;
    lines.forEach((line, k) => {
      const baseline = top + lineH * k + px * 0.84;
      const x = t.align === 'center' ? left + (boxW - widths[k]) / 2
        : t.align === 'right' ? left + boxW - widths[k] : left;
      if (ctx.lineWidth > 0.2) ctx.strokeText(line, x, baseline);
      ctx.fillText(line, x, baseline);
    });
  });

  canvas.style.pointerEvents = S.texts.length ? 'auto' : 'none';
  drawTextSel();
}

function drawTextSel() {
  const rect = S.activeText === null ? null : S.layout[S.activeText];
  if (!rect) { el.textSel.hidden = true; return; }
  const W = el.textLayer.width || 1, H = el.textLayer.height || 1;
  const pad = Math.max(3, H * 0.008);
  Object.assign(el.textSel.style, {
    left: `${((rect.x - pad) / W) * 100}%`,
    top: `${((rect.y - pad) / H) * 100}%`,
    width: `${((rect.w + pad * 2) / W) * 100}%`,
    height: `${((rect.h + pad * 2) / H) * 100}%`,
  });
  el.textSel.hidden = false;
}

function syncTextList() {
  el.txList.textContent = '';
  S.texts.forEach((t, i) => {
    const row = document.createElement('div');
    row.className = `txrow${i === S.activeText ? ' on' : ''}`;
    const label = document.createElement('span');
    label.textContent = String(t.text).replace(/\n/g, ' / ').trim() || '(empty)';
    const del = document.createElement('button');
    del.textContent = '×';
    del.title = 'delete';
    del.addEventListener('click', (e) => { e.stopPropagation(); removeText(i); });
    row.append(label, del);
    row.addEventListener('click', () => {
      S.activeText = i;
      syncTextPanel();
      drawTextSel();
    });
    el.txList.append(row);
  });
  el.btnSpeedBadge.textContent = `+ speed badge (${speedLabel()})`;
}

function syncTextPanel() {
  syncTextList();
  const t = activeText();
  el.txEdit.hidden = !t;
  if (!t) return;
  el.txText.value = t.text;
  el.txFont.value = t.font;
  el.txSize.value = t.size;
  el.txColor.value = t.color;
  el.txOutlineColor.value = t.outlineColor;
  el.txOutline.value = t.outline;
  el.txAlign.value = t.align;
  el.txBold.checked = t.bold;
  el.txBackdrop.checked = t.backdrop;
}

function refreshAutoBadges() {
  let touched = false;
  S.texts.forEach((t) => {
    if (t.auto && t.text !== speedLabel()) { t.text = speedLabel(); touched = true; }
  });
  if (touched) syncTextList();
}

// editor fields -> active text (the list label updates, the fields are left alone
// so the caret doesn't jump while typing)
[[el.txText, 'text', (i) => i.value], [el.txFont, 'font', (i) => i.value],
 [el.txSize, 'size', (i) => Number(i.value) || 1], [el.txColor, 'color', (i) => i.value],
 [el.txOutlineColor, 'outlineColor', (i) => i.value],
 [el.txOutline, 'outline', (i) => Math.max(0, Number(i.value) || 0)],
 [el.txAlign, 'align', (i) => i.value], [el.txBold, 'bold', (i) => i.checked],
 [el.txBackdrop, 'backdrop', (i) => i.checked],
].forEach(([input, key, read]) => input.addEventListener('input', () => {
  const t = activeText();
  if (!t) return;
  t[key] = read(input);
  if (key === 'text') { t.auto = false; syncTextList(); }
  drawTextLayer();
}));

el.btnAddText.addEventListener('click', () => {
  if (!S.meta) { toast('open a video first'); return; }
  addText({ text: '' });
  el.txText.focus();
});

el.btnSpeedBadge.addEventListener('click', () => {
  if (!S.meta) { toast('open a video first'); return; }
  const t = addText({ text: speedLabel(), auto: true, size: 9 });
  const rect = S.layout[S.activeText];          // now that it has been measured,
  if (rect) {                                   // tuck it into the bottom-right
    t.nx = 1 - 0.04 - rect.w / 2 / el.textLayer.width;
    t.ny = 1 - 0.04 - rect.h / 2 / el.textLayer.height;
    drawTextLayer();
  }
});

el.btnDelText.addEventListener('click', () => {
  if (S.activeText !== null) removeText(S.activeText);
});

el.txAnchors.addEventListener('click', (ev) => {
  const btn = ev.target.closest('button');
  const t = activeText();
  if (!btn || !t) return;
  const rect = S.layout[S.activeText] || { w: 0, h: 0 };
  const W = el.textLayer.width || 1, H = el.textLayer.height || 1;
  const m = 0.035, halfW = rect.w / 2 / W, halfH = rect.h / 2 / H;
  const col = Number(btn.dataset.col), row = Number(btn.dataset.row);
  t.nx = halfW >= 0.5 ? 0.5 : col === 0 ? m + halfW : col === 2 ? 1 - m - halfW : 0.5;
  t.ny = halfH >= 0.5 ? 0.5 : row === 0 ? m + halfH : row === 2 ? 1 - m - halfH : 0.5;
  drawTextLayer();
});

[el.txFrom, el.txTo].forEach((i) => i.addEventListener('input', updateEstimate));

// drag text around the frame; a miss falls through to the crop box underneath
el.textLayer.addEventListener('pointerdown', (ev) => {
  const canvas = el.textLayer;
  const box = canvas.getBoundingClientRect();
  const W = canvas.width, H = canvas.height;
  const toCanvas = (cx, cy) => ({
    x: ((cx - box.left) / box.width) * W, y: ((cy - box.top) / box.height) * H,
  });
  const from = toCanvas(ev.clientX, ev.clientY);
  let hit = null;
  for (let i = S.layout.length - 1; i >= 0; i--) {
    const r = S.layout[i];
    const slop = Math.max(6, H * 0.01);
    if (r && from.x >= r.x - slop && from.x <= r.x + r.w + slop
          && from.y >= r.y - slop && from.y <= r.y + r.h + slop) { hit = i; break; }
  }
  if (hit === null) {
    if (S.activeText !== null) { S.activeText = null; syncTextPanel(); drawTextSel(); }
    return;
  }
  ev.preventDefault();
  ev.stopPropagation();
  S.activeText = hit;
  syncTextPanel();
  const t = S.texts[hit];
  const grab = { nx: t.nx, ny: t.ny, x: from.x, y: from.y };
  const onMove = (e) => {
    const to = toCanvas(e.clientX, e.clientY);
    const r = S.layout[hit] || { w: 0, h: 0 };
    // text bigger than the frame centres instead of pinning half off-screen
    const halfW = r.w / 2 / W, halfH = r.h / 2 / H;
    t.nx = halfW >= 0.5 ? 0.5 : clamp(grab.nx + (to.x - grab.x) / W, halfW, 1 - halfW);
    t.ny = halfH >= 0.5 ? 0.5 : clamp(grab.ny + (to.y - grab.y) / H, halfH, 1 - halfH);
    drawTextLayer();
  };
  const onUp = () => {
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
  };
  window.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', onUp);
});

/** Hand the drawn layer to the server; returns an id to pass with the render. */
async function overlayToken() {
  if (!S.texts.some((t) => String(t.text).trim())) return null;
  drawTextLayer();
  const blob = await new Promise((res) => el.textLayer.toBlob(res, 'image/png'));
  if (!blob) return null;
  const res = await fetch('/api/overlay', { method: 'POST', body: blob });
  const data = await res.json().catch(() => ({ error: 'could not post the text layer' }));
  if (data.error) throw new Error(data.error);
  return data.overlay;
}

/* -------------------------------------------------------------- timeline -- */
function drawTimeline() {
  if (!S.dur) return;
  const a = (S.tin / S.dur) * 100, b = (S.tout / S.dur) * 100;
  el.tlBefore.style.left = '0'; el.tlBefore.style.width = `${a}%`;
  el.tlAfter.style.left = `${b}%`; el.tlAfter.style.width = `${100 - b}%`;
  el.tlSel.style.left = `${a}%`; el.tlSel.style.width = `${Math.max(0, b - a)}%`;
  el.tlIn.style.left = `${a}%`;
  el.tlOut.style.left = `${b}%`;
  el.tlBadge.textContent = `${(S.tout - S.tin).toFixed(2)}s`;
  el.tlLabels.innerHTML = `<span>in ${fmtTime(S.tin)}</span><span>out ${fmtTime(S.tout)}</span>`;
  el.tIn.value = S.tin.toFixed(2);
  el.tOut.value = S.tout.toFixed(2);
  el.tLen.value = (S.tout - S.tin).toFixed(2);
  updateEstimate();
}

const timeAt = (clientX) => {
  const r = el.timeline.getBoundingClientRect();
  return clamp((clientX - r.left) / r.width, 0, 1) * S.dur;
};

function dragTimeline(ev, mode) {
  if (!S.dur) return;
  ev.preventDefault();
  const frame = 1 / S.fps;
  const onMove = (e) => {
    const t = timeAt(e.clientX);
    if (mode === 'in') S.tin = Math.min(t, S.tout - frame);
    else if (mode === 'out') S.tout = Math.max(t, S.tin + frame);
    else { el.video.currentTime = t; }
    if (mode !== 'scrub') { S.tin = Math.max(0, S.tin); S.tout = Math.min(S.dur, S.tout); }
    drawTimeline();
    if (mode !== 'scrub') el.video.currentTime = mode === 'in' ? S.tin : S.tout;
    else drawPlayhead();
  };
  const onUp = () => {
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
  };
  window.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', onUp);
  onMove(ev);
}

el.tlIn.addEventListener('pointerdown', (e) => { e.stopPropagation(); dragTimeline(e, 'in'); });
el.tlOut.addEventListener('pointerdown', (e) => { e.stopPropagation(); dragTimeline(e, 'out'); });
el.timeline.addEventListener('pointerdown', (e) => dragTimeline(e, 'scrub'));
el.thumbs.addEventListener('error', () => { el.thumbs.hidden = true; });

el.btnSetIn.addEventListener('click', () => setIn(el.video.currentTime));
el.btnSetOut.addEventListener('click', () => setOut(el.video.currentTime));
el.btnTrimAll.addEventListener('click', () => { S.tin = 0; S.tout = S.dur; drawTimeline(); });

function setIn(t) { S.tin = clamp(t, 0, S.tout - 1 / S.fps); drawTimeline(); }
function setOut(t) { S.tout = clamp(t, S.tin + 1 / S.fps, S.dur); drawTimeline(); }

el.tIn.addEventListener('input', () => setIn(Number(el.tIn.value) || 0));
el.tOut.addEventListener('input', () => setOut(Number(el.tOut.value) || 0));
el.tLen.addEventListener('input', () => setOut(S.tin + (Number(el.tLen.value) || 0)));

/* ------------------------------------------------------------- transport -- */
function drawPlayhead() {
  if (!S.dur) return;
  const t = el.video.currentTime;
  el.playhead.style.left = `${(t / S.dur) * 100}%`;
  el.tcNow.textContent = fmtTime(t);
  el.tcFrame.textContent = `f${Math.round(t * S.fps)}`;
}

function tick() {
  if (!el.video.paused && !el.video.ended) {
    if (el.video.currentTime >= S.tout - 0.004) {
      if (el.loopSel.checked) el.video.currentTime = S.tin;
      else el.video.pause();
    }
    drawPlayhead();
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

function togglePlay() {
  if (!S.meta) return;
  if (el.video.paused) {
    const t = el.video.currentTime;
    if (t < S.tin - 0.01 || t > S.tout - 0.01) el.video.currentTime = S.tin;
    el.video.play().catch(() => {});
  } else el.video.pause();
}
const stepFrames = (n) => {
  if (!S.meta) return;
  el.video.pause();
  el.video.currentTime = clamp(el.video.currentTime + n / S.fps, 0, Math.max(0, S.dur - 1e-4));
};

el.btnPlay.addEventListener('click', togglePlay);
el.btnPrevF.addEventListener('click', () => stepFrames(-1));
el.btnNextF.addEventListener('click', () => stepFrames(1));
el.video.addEventListener('play', () => { el.btnPlay.textContent = '❚❚'; });
el.video.addEventListener('pause', () => { el.btnPlay.textContent = '▶'; });
el.video.addEventListener('timeupdate', drawPlayhead);
el.video.addEventListener('seeked', drawPlayhead);
el.video.addEventListener('loadedmetadata', () => {
  if (!S.dur || !isFinite(S.dur)) { S.dur = el.video.duration; S.tout = S.dur; }
  fitFrame(); drawCrop(); drawTimeline();
});
el.video.addEventListener('error', () => {
  if (S.token) toast('your browser cannot decode this codec for preview — the timeline '
    + 'thumbnails and numeric crop still work, and rendering is unaffected.', 9000);
});
el.btnMute.addEventListener('click', () => {
  el.video.muted = !el.video.muted;
  el.btnMute.textContent = el.video.muted ? '🔇' : '🔊';
});

/* -------------------------------------------------------------- settings -- */
function outputDims() {
  if (!S.meta) return { w: 0, h: 0 };
  let w = clamp(Math.round(Number(el.outW.value) || S.crop.w), 8, 8192);
  let h = Math.max(2, Math.round(S.crop.h * w / S.crop.w));
  if (el.fmt.value === 'mp4' || el.fmt.value === 'webm') { w -= w % 2; h -= h % 2; }
  return { w, h };
}

function syncFormat() {
  const fmt = el.fmt.value;
  const isGif = fmt === 'gif';
  const isVid = fmt === 'mp4' || fmt === 'webm';
  document.querySelectorAll('.gifonly').forEach((n) => { n.hidden = !isGif; });
  document.querySelectorAll('.videoonly').forEach((n) => { n.hidden = !isVid; });
  document.querySelectorAll('.loopRow').forEach((n) => { n.hidden = isVid; });
  document.querySelectorAll('.bayeronly').forEach((n) => { n.hidden = !(isGif && el.dither.value === 'bayer'); });
  el.outExt.textContent = fmt === 'apng' ? '.png' : `.${fmt}`;
  el.keepAudio.disabled = !S.meta?.has_audio || el.reverse.checked || el.boomerang.checked;
  refreshAutoBadges();
  updateEstimate();
}

function updateEstimate() {
  if (!S.meta) return;
  const { w, h } = outputDims();
  const speed = Math.max(0.05, Number(el.speed.value) || 1);
  const fps = Math.max(0.5, Number(el.fps.value) || 10);
  const span = Math.max(0.02, S.tout - S.tin) / speed;
  let frames = Math.max(1, Math.round(span * fps));
  if (el.boomerang.checked) frames *= 2;
  const lines = [
    `<b>${w}×${h}</b> · <b>${fps}</b> fps · <b>${frames}</b> frames · <b>${span.toFixed(2)}s</b>`,
  ];
  if (el.fmt.value === 'gif' || el.fmt.value === 'apng') {
    const perPixel = el.fmt.value === 'apng' ? 0.55 : 0.085 * (el.dither.value === 'none' ? 0.72 : 1);
    const est = frames * w * h * perPixel;
    lines.push(`rough size ≈ <b>${fmtBytes(Math.round(est))}</b>`
      + (est > 8 * 1048576 ? ' <span class="warn">(big — drop fps/width)</span>' : ''));
  }
  const notes = [];
  if (w > S.crop.w) notes.push('upscaling');
  if (fps > S.meta.fps + 0.01) notes.push(`above source ${S.meta.fps.toFixed(2)} fps`);
  if (notes.length) lines.push(`<span class="warn">${notes.join(' · ')}</span>`);
  el.estimate.innerHTML = lines.join('<br>');
  el.outDims.textContent = `→ ${w}×${h}`;
  drawTextLayer();
}

document.querySelectorAll('#scaleChips .chip').forEach((chip) => chip.addEventListener('click', () => {
  if (!S.meta) return;
  el.outW.value = Math.max(8, Math.round(S.crop.w * Number(chip.dataset.scale)));
  updateEstimate();
}));
document.querySelectorAll('#fpsChips .chip').forEach((chip) => chip.addEventListener('click', () => {
  el.fps.value = chip.dataset.fps;
  updateEstimate();
}));
document.querySelectorAll('[data-preset]').forEach((chip) => chip.addEventListener('click', () => {
  if (!S.meta) return;
  const p = {
    tiny: { w: 320, fps: 10, colors: 64, dither: 'bayer' },
    balanced: { w: 480, fps: 15, colors: 128, dither: 'sierra2_4a' },
    crisp: { w: 720, fps: 24, colors: 256, dither: 'sierra2_4a' },
    source: { w: S.crop.w, fps: Math.round(S.meta.fps * 100) / 100, colors: 256, dither: 'sierra2_4a' },
  }[chip.dataset.preset];
  el.outW.value = Math.min(p.w, Math.max(8, S.crop.w));
  el.fps.value = p.fps;
  el.colors.value = p.colors;
  el.dither.value = p.dither;
  syncFormat();
}));

['fmt', 'outW', 'fps', 'speed', 'colors', 'dither', 'statsMode', 'crf', 'bayerScale',
 'boomerang', 'reverse', 'loopForever', 'keepAudio'].forEach((id) =>
  el[id].addEventListener('input', syncFormat));

// Only reset quality when the container changes -- otherwise typing in the field
// would fight the default on every keystroke.
el.fmt.addEventListener('change', () => {
  el.crf.value = el.fmt.value === 'webm' ? 32 : 23;
  syncFormat();
});

function params(overlay) {
  const window_ = {};
  if (el.txFrom.value !== '') window_.from = Number(el.txFrom.value);
  if (el.txTo.value !== '') window_.to = Number(el.txTo.value);
  return {
    format: el.fmt.value,
    overlay: overlay || null, text_window: window_,
    crop: { ...S.crop },
    start: S.tin, end: S.tout,
    fps: Number(el.fps.value), width: Number(el.outW.value), speed: Number(el.speed.value),
    colors: Number(el.colors.value), dither: el.dither.value,
    bayer_scale: Number(el.bayerScale.value), stats_mode: el.statsMode.value,
    crf: Number(el.crf.value),
    reverse: el.reverse.checked, boomerang: el.boomerang.checked,
    keep_audio: el.keepAudio.checked && !el.keepAudio.disabled,
    loop: el.loopForever.checked ? 'forever' : 'once',
    out_dir: el.outDir.value, out_name: el.outName.value,
  };
}

/* ---------------------------------------------------------------- render -- */
async function render() {
  if (!S.token) { toast('open a video first'); return; }
  el.btnRender.disabled = true;
  try {
    const overlay = await overlayToken();
    const job = await api('/api/render', { token: S.token, params: params(overlay) });
    S.job = job;
    el.progress.hidden = false;
    el.progFill.style.width = '0%';
    el.progPct.textContent = '0%';
    el.progStage.textContent = job.stage || 'starting…';
    poll();
  } catch (e) {
    el.btnRender.disabled = false;
    toast(e.message);
  }
}

function poll() {
  clearInterval(S.timer);
  S.timer = setInterval(async () => {
    try {
      const job = await api(`/api/job/${S.job.id}`);
      S.job = job;
      el.progFill.style.width = `${Math.round(job.progress * 100)}%`;
      el.progPct.textContent = `${Math.round(job.progress * 100)}%`;
      el.progStage.textContent = job.stage;
      if (job.status === 'running') return;
      clearInterval(S.timer);
      el.progress.hidden = true;
      el.btnRender.disabled = false;
      if (job.status === 'done') showResult(job);
      else if (job.status === 'error') toast(job.error || 'render failed', 14000);
    } catch (e) {
      clearInterval(S.timer);
      el.progress.hidden = true;
      el.btnRender.disabled = false;
      toast(e.message);
    }
  }, 220);
}

function showResult(job) {
  const src = `/api/result/${job.id}?t=${Date.now()}`;
  const s = job.summary;
  el.resultMedia.innerHTML = (job.format === 'mp4' || job.format === 'webm')
    ? `<video src="${src}" autoplay loop muted playsinline></video>`
    : `<img src="${src}" alt="result">`;
  el.resultMeta.textContent = '';
  const size = document.createElement('b');
  size.textContent = fmtBytes(job.size);
  el.resultMeta.append(size,
    ` · ${s.width}×${s.height} · ${s.fps} fps · ${s.frames} frames · ${s.duration}s`
    + ` · ${job.elapsed}s to render`, document.createElement('br'), job.out_path);
  el.btnDownload.href = `/api/result/${job.id}?download=1`;
  el.btnDownload.setAttribute('download', job.out_name);
  el.resultCard.hidden = false;
  el.resultCard.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

el.btnRender.addEventListener('click', render);
el.btnCancel.addEventListener('click', () => S.job && api(`/api/cancel/${S.job.id}`, {}).catch(() => {}));
el.btnCloseResult.addEventListener('click', () => { el.resultCard.hidden = true; });
el.btnReveal.addEventListener('click', () =>
  api('/api/reveal', { path: S.job?.out_path }).catch((e) => toast(e.message)));
el.btnCmd.addEventListener('click', async () => {
  if (!S.token) return;
  if (!el.cmdBox.hidden) { el.cmdBox.hidden = true; return; }
  try {
    const res = await api('/api/plan', { token: S.token, params: params(await overlayToken()) });
    el.cmdBox.textContent = res.commands.join('\n\n');
    el.cmdBox.hidden = false;
  } catch (e) { toast(e.message); }
});

/* ------------------------------------------------------- opening a video -- */
el.btnPick.addEventListener('click', pick);
el.btnPick2.addEventListener('click', pick);
el.btnPath.addEventListener('click', async () => {
  const path = el.pathInput.value.trim();
  if (!path) return;
  try { loadMedia(await api('/api/open', { path })); } catch (e) { toast(e.message); }
});
el.pathInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') el.btnPath.click(); });
el.btnOutDir.addEventListener('click', async () => {
  try {
    const res = await api('/api/pick?kind=dir');
    if (res.path) el.outDir.value = res.path;
  } catch (e) { toast(e.message); }
});

['dragenter', 'dragover'].forEach((type) => document.addEventListener(type, (e) => {
  e.preventDefault();
  el.drop.classList.add('over');
}));
document.addEventListener('dragleave', (e) => {
  if (e.relatedTarget === null) el.drop.classList.remove('over');
});
document.addEventListener('drop', (e) => {
  e.preventDefault();
  el.drop.classList.remove('over');
  const file = e.dataTransfer?.files?.[0];
  if (!file) return;
  el.drop.hidden = false;
  el.viewport.hidden = el.transport.hidden = el.timelineWrap.hidden = true;
  uploadFile(file);
});

/* built-in browser, for when the native dialog is unavailable */
async function browseTo(path) {
  try {
    const res = await api(`/api/browse?path=${encodeURIComponent(path)}`);
    el.bwPath.textContent = res.path;
    el.bwList.innerHTML = '';
    const add = (label, cls, onClick, size) => {
      const row = document.createElement('div');
      row.className = `bw-item ${cls}`;
      const text = document.createElement('span');
      text.textContent = label;
      row.append(text);
      if (size) {
        const sz = document.createElement('span');
        sz.className = 'sz';
        sz.textContent = fmtBytes(size);
        row.append(sz);
      }
      row.addEventListener('click', onClick);
      el.bwList.appendChild(row);
    };
    if (res.path !== res.parent) add('📁 ..', 'dir', () => browseTo(res.parent));
    res.dirs.forEach((d) => add(`📁 ${d.name}`, 'dir', () => browseTo(d.path)));
    res.files.forEach((f) => add(`🎬 ${f.name}`, '', async () => {
      try {
        loadMedia(await api('/api/open', { path: f.path }));
        el.browser.hidden = true;
      } catch (e) { toast(e.message); }
    }, f.size));
    el.browser.hidden = false;
  } catch (e) { toast(e.message); }
}
el.btnBrowse.addEventListener('click', () => browseTo(el.outDir.value || S.meta?.dir || ''));
el.bwClose.addEventListener('click', () => { el.browser.hidden = true; });

/* ----------------------------------------------------------- keyboard ----- */
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    el.browser.hidden = true;
    el.toast.hidden = true;
    return;
  }
  const t = e.target.tagName;
  if (t === 'INPUT' || t === 'SELECT' || t === 'TEXTAREA') {
    if (e.key === 'Enter') e.target.blur();
    return;
  }
  const big = e.shiftKey ? 10 : 1;
  switch (e.key) {
    case ' ': e.preventDefault(); togglePlay(); break;
    case 'ArrowLeft': e.preventDefault(); stepFrames(-big); break;
    case 'ArrowRight': e.preventDefault(); stepFrames(big); break;
    case 'i': setIn(el.video.currentTime); break;
    case 'o': setOut(el.video.currentTime); break;
    case 'l': el.loopSel.checked = !el.loopSel.checked; break;
    case 't': if (S.meta) { addText({ text: '' }); el.txText.focus(); } break;
    case 'Home': el.video.currentTime = S.tin; break;
    case 'End': el.video.currentTime = Math.max(0, S.tout - 1 / S.fps); break;
    case 'Enter': e.preventDefault(); render(); break;
    default: break;
  }
});

new ResizeObserver(() => { if (S.meta) { fitFrame(); drawCrop(); } }).observe(el.viewport);

/* ------------------------------------------------------------------ boot -- */
(async () => {
  try {
    const state = await api('/api/state');
    syncTextPanel();
    if (state.preload) loadMedia(state.preload);
    else el.pathInput.placeholder = `${state.home}/Movies/clip.mov`;
  } catch (e) { toast(`server not reachable: ${e.message}`); }
})();
