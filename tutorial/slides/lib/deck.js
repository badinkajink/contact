/* deck.js — a small, dependency-free slide engine.
 *
 * A deck is a plain HTML file:
 *
 *   <body data-short="12 · Friction cone in 3D" data-index="index.html"
 *         data-prev="11_coulomb_friction.html" data-next="13_contact_wrench_cone.html">
 *     <main class="deck">
 *       <section class="slide" id="cone-slices" data-code="04_friction_cones.py:in_cone"> ... </section>
 *       ...
 *     </main>
 *
 * data-short, data-prev and data-next are optional for decks listed in lib/series.js: the
 * footer name and the neighbouring decks then come from window.SERIES. An empty attribute
 * (data-next="") turns a link off.
 *
 * Modes (pick with ?mode=present|read|print, or press R to toggle):
 *   present  one slide at a time, scaled to fit the window (default)
 *   read     all slides stacked like a web page, reader notes and linked code shown
 *   print    one slide per 1280x720 page, for PDF export
 *
 * Links: deck.html#some-id opens the slide with that id (or the slide holding the element with
 * that id); deck.html#12 opens slide 12. The address bar shows the slide's id when it has one.
 *
 * Keys: right/space/PgDn next, left/PgUp prev, Home/End, digits+Enter jump, R read mode,
 * F fullscreen, C code behind the current slide, M menu of all decks, Esc close, ? help.
 * Elements with class "step" appear one at a time (data-step="2" groups or reorders them).
 * <aside class="notes"> is shown only in read mode.
 *
 * Events (present mode), dispatched on the slide and bubbling to document:
 *   slidehidden  on the slide being left (also when leaving present mode); detail {index, id, to}
 *   slideshown   on the slide being shown; detail {index, id, from}
 * On document: "deckmode" {mode} after every mode change, "deckready" after init,
 * "decksettled" each time data-settled becomes "1".
 *
 * Async work: Deck.pending(promise) registers work (a MuJoCo model compiling, a solver run).
 * <html data-settled="1"> is set once fonts are ready and every registered promise has settled
 * (15 s cap, then it settles anyway with a console.warn). The test tools wait for it. A deck's
 * inline script runs before this deferred file, so there it registers with
 *   (window.DECK_PENDING = window.DECK_PENDING || []).push(promise);
 * which works both before and after deck.js has started.
 *
 * deck.js loads lib/series.js, lib/codemap.js and lib/code.js itself (next to this file), so a
 * deck needs no extra script tags for the series menu or the [code] links.
 */
(function () {
  "use strict";

  const W = 1280, H = 720;
  // Resolve sibling libraries against this file, not the page (tools/demos/ pages use it too).
  const SELF = (document.currentScript && document.currentScript.src) || "";
  const LIB = SELF ? new URL(".", SELF).href : "lib/";

  const params = new URLSearchParams(location.search);
  // Tolerate a query typed after the hash ("deck.html#12?mode=read").
  const qh = location.hash.indexOf("?");
  if (qh >= 0) new URLSearchParams(location.hash.slice(qh + 1)).forEach((v, k) => { if (!params.has(k)) params.set(k, v); });
  let mode = params.get("mode") || (params.has("print") ? "print" : "present");
  if (!["present", "read", "print"].includes(mode)) mode = "present";

  let slides = [];
  let cur = 0;        // current slide index
  let step = 0;       // number of revealed steps on current slide
  let typed = "";     // digits typed for jump
  let shownIdx = -1;  // slide that last received "slideshown" and no "slidehidden" since
  let inited = false;

  const body = document.body;
  const deck = document.querySelector(".deck");
  const root = document.documentElement;

  // Shared KaTeX macros; a deck may add more via window.DECK_MACROS.
  // \argmin, \argmax, \det, \dim, \ker, \Pr, \Re, \Im, \N, \Z are KaTeX built-ins and stay theirs.
  const MACROS = Object.assign({
    "\\R": "\\mathbb{R}",
    "\\vf": "\\mathbf{f}",
    "\\vw": "\\mathbf{w}",
    "\\vp": "\\mathbf{p}",
    "\\vv": "\\mathbf{v}",
    "\\vd": "\\mathbf{d}",
    "\\vr": "\\mathbf{r}",
    "\\vq": "\\mathbf{q}",
    "\\vu": "\\mathbf{u}",
    "\\va": "\\mathbf{a}",
    "\\vb": "\\mathbf{b}",
    "\\vg": "\\mathbf{g}",
    "\\vtau": "\\boldsymbol{\\tau}",
    "\\nhat": "\\hat{\\mathbf{n}}",
    "\\that": "\\hat{\\mathbf{t}}",
    "\\K": "\\mathcal{K}",
    "\\norm": "\\left\\lVert #1 \\right\\rVert",
    "\\abs": "\\left\\lvert #1 \\right\\rvert",
    // More bold vectors.
    "\\vc": "\\mathbf{c}",
    "\\ve": "\\mathbf{e}",
    "\\vh": "\\mathbf{h}",
    "\\vm": "\\mathbf{m}",
    "\\vn": "\\mathbf{n}",
    "\\vs": "\\mathbf{s}",
    "\\vt": "\\mathbf{t}",
    "\\vx": "\\mathbf{x}",
    "\\vy": "\\mathbf{y}",
    "\\vz": "\\mathbf{z}",
    "\\vzero": "\\mathbf{0}",
    "\\vone": "\\mathbf{1}",
    "\\vlambda": "\\boldsymbol{\\lambda}",
    "\\vnu": "\\boldsymbol{\\nu}",
    "\\vomega": "\\boldsymbol{\\omega}",
    "\\vxi": "\\boldsymbol{\\xi}",
    // Linear algebra and optimization operators. Write A^\T for the transpose.
    "\\T": "{\\top}",
    "\\rank": "\\operatorname{rank}",
    "\\nullity": "\\operatorname{nullity}",
    "\\cond": "\\operatorname{cond}",
    "\\diag": "\\operatorname{diag}",
    "\\tr": "\\operatorname{tr}",
    "\\sign": "\\operatorname{sign}",
    "\\sgn": "\\operatorname{sgn}",
    "\\proj": "\\operatorname{proj}",
    "\\Null": "\\operatorname{Null}",
    "\\Row": "\\operatorname{Row}",
    "\\Col": "\\operatorname{Col}",
    "\\Span": "\\operatorname{span}",
    "\\range": "\\operatorname{range}",
    "\\dom": "\\operatorname{dom}",
    "\\conv": "\\operatorname{conv}",
    "\\cone": "\\operatorname{cone}",
    "\\dist": "\\operatorname{dist}",
    "\\prox": "\\operatorname{prox}",
    "\\relint": "\\operatorname{relint}",
    // Probability.
    "\\E": "\\mathbb{E}",
    "\\Var": "\\operatorname{Var}",
    "\\Cov": "\\operatorname{Cov}",
    // Rigid-body motion.
    "\\SO": "\\mathrm{SO}",
    "\\SE": "\\mathrm{SE}",
    "\\so": "\\mathfrak{so}",
    "\\se": "\\mathfrak{se}",
    "\\Ad": "\\operatorname{Ad}",
    "\\ad": "\\operatorname{ad}",
    // Upright differential: \frac{\dd J}{\dd z}.
    "\\dd": "\\mathrm{d}",
  }, window.DECK_MACROS || {});

  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const clamp = (i) => Math.max(0, Math.min(slides.length - 1, i));
  const fire = (el, type, detail) => el.dispatchEvent(new CustomEvent(type, { bubbles: true, detail }));

  // Build steps of a slide, as groups revealed together. An element's key is its
  // data-step number if given (so several elements can share a step, or a figure
  // part can appear after later text), otherwise its position in the document.
  function groups(s) {
    const els = Array.from(s.querySelectorAll(".step"));
    const key = (e, i) => (e.dataset.step !== undefined ? parseFloat(e.dataset.step) : i + 1);
    const keys = Array.from(new Set(els.map(key))).sort((a, b) => a - b);
    return keys.map((k) => els.filter((e, i) => key(e, i) === k));
  }
  function reveal(s, n) {
    groups(s).forEach((g, j) => g.forEach((el) => el.classList.toggle("shown", j < n)));
  }

  function renderMath() {
    if (!window.renderMathInElement) return;
    window.renderMathInElement(deck, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
      ],
      macros: MACROS,
      throwOnError: false,
      ignoredClasses: ["nomath"],
    });
  }

  // ---------------------------------------------------------------- hashes
  function fragment(h) {
    let f = (h || "").replace(/^#/, "");
    const q = f.indexOf("?");
    if (q >= 0) f = f.slice(0, q);
    try { f = decodeURIComponent(f); } catch (e) { /* keep raw */ }
    return f;
  }
  // Slide index for "12" (1-based) or an id (a slide's, or any element's inside a slide); -1 if none.
  function slideIndex(f) {
    if (f === undefined || f === null || f === "") return -1;
    if (typeof f === "number" || /^[0-9]+$/.test(f)) return clamp(parseInt(f, 10) - 1);
    const el = document.getElementById(f);
    const s = el && el.closest("section.slide");
    return s ? slides.indexOf(s) : -1;
  }
  const hashFor = (i) => "#" + (slides[i].id || String(i + 1));

  // ---------------------------------------------------------------- series
  function seriesIndex() {
    const S = window.SERIES;
    if (!Array.isArray(S)) return -1;
    let file = location.pathname.split("/").pop();
    try { file = decodeURIComponent(file); } catch (e) { /* keep raw */ }
    return S.findIndex((d) => d.file === file);
  }
  // Neighbouring deck: the body attribute wins (empty = none); otherwise the adjacent deck in
  // SERIES, linked only when its status is not "planned".
  function deckLink(which) {
    const attr = "data-" + which;
    if (body.hasAttribute(attr)) return body.getAttribute(attr) || null;
    const i = seriesIndex();
    if (i < 0) return null;
    const d = window.SERIES[which === "prev" ? i - 1 : i + 1];
    return d && d.status !== "planned" ? d.file : null;
  }
  function deckName() {
    if (body.dataset.short) return body.dataset.short;
    const i = seriesIndex();
    if (i >= 0) {
      const d = window.SERIES[i];
      return String(d.n).padStart(2, "0") + " · " + d.short;
    }
    return document.title;
  }

  // ---------------------------------------------------------------- footers
  function buildFooters() {
    slides.forEach((s, i) => {
      if (s.hasAttribute("data-nofoot")) return;
      const f = document.createElement("div");
      f.className = "foot";
      f.innerHTML =
        '<span class="deckname"></span>' +
        '<span><span class="nav"><span class="navlinks"></span></span>' +
        '<span class="num">' + (i + 1) + " / " + slides.length + "</span></span>";
      s.appendChild(f);
    });
    refreshFooters();
    // In-deck prev/next links should step through builds, not jump.
    deck.addEventListener("click", (e) => {
      const a = e.target.closest("a[data-go]");
      if (!a || mode !== "present") return;
      const href = a.getAttribute("href");
      if (!href.startsWith("#")) return;
      e.preventDefault();
      a.dataset.go === "next" ? next() : prev();
    });
  }
  // Fill in the deck name and the prev / index / next links (again once series.js has loaded).
  function refreshFooters() {
    const index = body.dataset.index || "index.html";
    const name = deckName();
    const dprev = deckLink("prev"), dnext = deckLink("next");
    slides.forEach((s, i) => {
      const f = s.querySelector(":scope > .foot");
      if (!f) return;
      const prev = i > 0 ? hashFor(i - 1) : dprev;
      const next = i < slides.length - 1 ? hashFor(i + 1) : dnext;
      f.querySelector(".deckname").textContent = name;
      f.querySelector(".navlinks").innerHTML =
        (prev ? '[<a href="' + esc(prev) + '" data-go="prev">&lt;&lt; prev</a>] ' : "[&lt;&lt; prev] ") +
        '[<a href="' + esc(index) + '">index</a>] ' +
        (next ? '[<a href="' + esc(next) + '" data-go="next">next &gt;&gt;</a>]' : "[next &gt;&gt;]");
    });
  }

  // ---------------------------------------------------------------- present
  function fit() {
    if (mode === "present") {
      const s = Math.min(innerWidth / W, innerHeight / H);
      const x = (innerWidth - W * s) / 2, y = (innerHeight - H * s) / 2;
      deck.style.transform = "translate(" + x + "px," + y + "px) scale(" + s + ")";
    } else if (mode === "read") {
      const s = Math.min(1, (body.clientWidth - 34) / W);
      document.querySelectorAll(".slot").forEach((slot) => {
        slot.style.width = W * s + "px";
        slot.style.height = H * s + "px";
        slot.firstElementChild.style.transform = "scale(" + s + ")";
      });
    }
  }

  function show(i, st) {
    cur = clamp(i);
    const s = slides[cur];
    const n = groups(s).length;
    step = st === "end" ? n : Math.max(0, Math.min(n, st || 0));
    slides.forEach((x, j) => x.classList.toggle("current", j === cur));
    reveal(s, step);
    const h = hashFor(cur);
    if (location.hash !== h) history.replaceState(null, "", h);
    if (shownIdx !== cur) {
      const from = shownIdx;
      if (from >= 0) fire(slides[from], "slidehidden", { index: from, id: slides[from].id || null, to: cur });
      shownIdx = cur;
      fire(s, "slideshown", { index: cur, id: s.id || null, from });
    }
  }

  function next() {
    if (step < groups(slides[cur]).length) {
      step++;
      reveal(slides[cur], step);
    } else if (cur < slides.length - 1) {
      show(cur + 1, 0);
    } else {
      const d = deckLink("next");
      if (d) location.href = d;
    }
  }

  function prev() {
    if (step > 0) {
      step--;
      reveal(slides[cur], step);
    } else if (cur > 0) {
      show(cur - 1, "end");
    }
  }

  // ---------------------------------------------------------------- read mode helpers
  const slotOf = (s) => (s.parentElement && s.parentElement.classList.contains("slot") ? s.parentElement : s);
  // The slide at the top of the window in read mode (the one being read).
  function visibleIndex() {
    for (let i = 0; i < slides.length; i++) {
      if (slotOf(slides[i]).getBoundingClientRect().bottom > 40) return i;
    }
    return slides.length - 1;
  }
  let scrolledTo = null;
  function scrollToSlide(i) {
    if (i <= 0) scrollTo(0, 0);
    else slotOf(slides[i]).scrollIntoView({ block: "start" });
    scrolledTo = scrollY;
  }

  // ---------------------------------------------------------------- modes
  function setMode(m) {
    if (mode === "present" && m !== "present" && shownIdx >= 0) {
      const from = shownIdx;
      shownIdx = -1;
      fire(slides[from], "slidehidden", { index: from, id: slides[from].id || null, to: -1 });
    }
    if (mode === "read" && m !== "read" && document.querySelector(".slot")) cur = visibleIndex();
    // Undo read-mode wrappers before switching.
    document.querySelectorAll(".slot").forEach((slot) => {
      const s = slot.firstElementChild;
      s.style.transform = "";
      slot.replaceWith(s);
    });
    document.querySelectorAll(".notes-out, .readhead").forEach((x) => x.remove());
    deck.style.transform = "";
    body.classList.remove("present", "read", "print");
    mode = m;
    body.classList.add(m);
    if (m !== "present" && m !== "read") closeMenu();
    if (m === "read") {
      const head = document.createElement("p");
      head.className = "readhead";
      head.innerHTML = "<b></b> &mdash; reading view. [<a href=\"?mode=present#1\">present</a>] " +
        "[<a href=\"" + esc(body.dataset.index || "index.html") + "\">index</a>] " +
        "[<a href=\"#\" data-action=\"menu\">all decks</a>]";
      head.querySelector("b").textContent = document.title;
      deck.before(head);
      slides.forEach((s) => {
        const slot = document.createElement("div");
        slot.className = "slot";
        s.before(slot);
        slot.appendChild(s);
        const notes = document.createElement("div");
        notes.className = "notes-out";
        s.querySelectorAll("aside.notes").forEach((n) => notes.insertAdjacentHTML("beforeend", n.innerHTML));
        slot.after(notes);
      });
    }
    if (m === "print") {
      document.querySelectorAll("details").forEach((d) => (d.open = true));
    }
    fit();
    if (m === "present") show(cur, step);
    document.dispatchEvent(new CustomEvent("deckmode", { detail: { mode: m } }));
    if (m === "read") scrollToSlide(cur);
  }

  // ---------------------------------------------------------------- series menu (M)
  const menu = document.createElement("div");
  menu.className = "menu";
  menu.setAttribute("role", "dialog");
  menu.setAttribute("aria-label", "All decks");
  menu.addEventListener("click", (e) => {
    if (e.target === menu || e.target.closest("[data-close]")) { e.preventDefault(); closeMenu(); }
  });
  const menuOpen = () => menu.classList.contains("on");
  function closeMenu() { menu.classList.remove("on"); }
  function buildMenu() {
    const S = window.SERIES;
    const index = new URL("../index.html", LIB).href;
    let html = '<div class="menupanel"><p class="menuhead"><b></b> ' +
      '[<a href="' + esc(index) + '">index</a>] [<a href="#" data-close>close</a>]</p>';
    if (!Array.isArray(S)) {
      html += "<p>lib/series.js is not loaded.</p></div>";
      menu.innerHTML = html;
      return;
    }
    const here = seriesIndex();
    html += '<div class="menucols">';
    (S.parts || []).forEach((p) => {
      html += '<div class="menupart"><p class="parttitle">Part ' + esc(p.id) + ". " + esc(p.title) + "</p><ul>";
      S.forEach((d, i) => {
        if (d.part !== p.id) return;
        const num = '<span class="dn">' + String(d.n).padStart(2, "0") + "</span> ";
        if (i === here) html += '<li class="here">' + num + "<b>" + esc(d.title) + "</b> (this deck)</li>";
        else if (d.status === "planned") html += '<li class="planned">' + num + esc(d.title) + " (planned)</li>";
        else html += "<li>" + num + '<a href="' + esc(new URL("../" + d.file, LIB).href) + '">' + esc(d.title) + "</a></li>";
      });
      html += "</ul></div>";
    });
    html += "</div></div>";
    menu.innerHTML = html;
    menu.querySelector(".menuhead b").textContent = S.title || "Series";
  }
  function toggleMenu() {
    if (menuOpen()) { closeMenu(); return; }
    if (mode === "print") return;
    if (window.Code && window.Code.isOpen()) window.Code.close();
    help.classList.remove("on");
    buildMenu();
    menu.classList.add("on");
  }

  // ---------------------------------------------------------------- input
  const codeOpen = () => !!(window.Code && window.Code.isOpen());
  function currentSlide() { return slides[mode === "read" ? visibleIndex() : cur]; }
  function toggleCode() {
    if (mode === "print" || !window.Code) return;
    closeMenu();
    window.Code.toggle(currentSlide());
  }

  function onKey(e) {
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    if (k === "Escape") {
      if (menuOpen() || help.classList.contains("on")) { closeMenu(); help.classList.remove("on"); e.preventDefault(); }
      return;  // lib/code.js closes its own overlay on Escape
    }
    if (k === "c" || k === "C") { toggleCode(); e.preventDefault(); return; }
    if (k === "m" || k === "M") { toggleMenu(); e.preventDefault(); return; }
    if (menuOpen() || codeOpen()) return;  // no slide changes behind an open panel
    if (k === "r" || k === "R") { setMode(mode === "read" ? "present" : "read"); return; }
    if (k === "?") { help.classList.toggle("on"); return; }
    if (mode !== "present") return;
    if (/^[0-9]$/.test(k)) { typed += k; return; }
    if (k === "Enter" && typed) { show(parseInt(typed, 10) - 1, 0); typed = ""; e.preventDefault(); return; }
    typed = "";
    if (["ArrowRight", "ArrowDown", "PageDown", " ", "Enter", "n"].includes(k)) { next(); e.preventDefault(); }
    else if (["ArrowLeft", "ArrowUp", "PageUp", "p"].includes(k)) { prev(); e.preventDefault(); }
    else if (k === "Home") show(0, 0);
    else if (k === "End") show(slides.length - 1, "end");
    else if (k === "f" || k === "F") {
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen && document.documentElement.requestFullscreen();
    }
  }

  let touchX = null;
  function onTouchStart(e) {
    if (e.target.closest("input, select, svg .handle, details, .menu, .codebox")) { touchX = null; return; }
    touchX = e.touches[0].clientX;
  }
  function onTouchEnd(e) {
    if (touchX === null || mode !== "present" || menuOpen() || codeOpen()) return;
    const dx = e.changedTouches[0].clientX - touchX;
    if (Math.abs(dx) > 50) (dx < 0 ? next : prev)();
    touchX = null;
  }

  const help = document.createElement("div");
  help.className = "help";
  help.innerHTML =
    "<b>Keys</b><table>" +
    "<tr><td><kbd>&rarr;</kbd> <kbd>space</kbd></td><td>next</td></tr>" +
    "<tr><td><kbd>&larr;</kbd></td><td>previous</td></tr>" +
    "<tr><td><kbd>Home</kbd> <kbd>End</kbd></td><td>first / last</td></tr>" +
    "<tr><td>12 <kbd>Enter</kbd></td><td>go to slide 12</td></tr>" +
    "<tr><td><kbd>R</kbd></td><td>reading view</td></tr>" +
    "<tr><td><kbd>C</kbd></td><td>code behind this slide</td></tr>" +
    "<tr><td><kbd>M</kbd></td><td>all decks</td></tr>" +
    "<tr><td><kbd>F</kbd></td><td>fullscreen</td></tr>" +
    "<tr><td><kbd>Esc</kbd></td><td>close a panel</td></tr>" +
    "<tr><td><kbd>?</kbd></td><td>this help</td></tr></table>";

  // ---------------------------------------------------------------- pending / settled
  const tasks = [];   // {promise, done}
  let settleRun = 0;
  function pending(p) {
    const task = { done: false };
    task.promise = Promise.resolve(p).then(
      () => { task.done = true; },
      (err) => { task.done = true; console.error("deck.js: a pending task failed:", err); });
    tasks.push(task);
    if (inited) settle();
    return p;
  }
  function settle() {
    const run = ++settleRun;
    if (root.dataset.settled === "1") root.dataset.settled = "0";
    const drain = async () => {
      let n = -1;
      while (n !== tasks.length) {  // promises registered while waiting are waited for too
        n = tasks.length;
        await Promise.all(tasks.map((t) => t.promise));
        await new Promise((r) => setTimeout(r, 0));
      }
      return "done";
    };
    let timer = null;
    const cap = new Promise((r) => { timer = setTimeout(() => r("timeout"), 15000); });
    Promise.race([drain(), cap]).then((why) => {
      clearTimeout(timer);
      if (run !== settleRun) return;  // a newer registration restarted the wait
      if (why === "timeout") {
        const left = tasks.filter((t) => !t.done).length;
        console.warn("deck.js: " + left + " pending task(s) unsettled after 15 s; marking the deck settled");
      }
      root.dataset.settled = "1";
      document.dispatchEvent(new CustomEvent("decksettled"));
      if (mode === "read" && scrolledTo !== null && Math.abs(scrollY - scrolledTo) < 2) scrollToSlide(cur);
    });
  }

  // Sibling libraries as ordered classic scripts (async = false keeps insertion order).
  function load(name, have) {
    if (have()) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = LIB + name;
      s.async = false;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error("could not load " + s.src));
      document.head.appendChild(s);
    });
  }

  // ---------------------------------------------------------------- init
  function init() {
    slides = Array.from(deck.querySelectorAll(":scope > section.slide"));
    renderMath();
    buildFooters();
    body.appendChild(help);
    body.appendChild(menu);
    const h = slideIndex(fragment(location.hash));
    cur = h < 0 ? 0 : h;
    setMode(mode);
    addEventListener("resize", fit);
    addEventListener("keydown", onKey);
    addEventListener("touchstart", onTouchStart, { passive: true });
    addEventListener("touchend", onTouchEnd);
    addEventListener("hashchange", () => {
      const i = slideIndex(fragment(location.hash));
      if (i < 0) return;
      if (mode === "present") {
        if (i !== cur) show(i, 0);
        scrollTo(0, 0);
      } else if (mode === "read") {
        cur = i;
        scrollToSlide(i);
      }
    });
    let before = null;
    addEventListener("beforeprint", () => { before = mode; if (mode !== "print") setMode("print"); });
    addEventListener("afterprint", () => { if (before && before !== "print") setMode(before); before = null; });
    document.querySelectorAll("[data-action=print]").forEach((a) =>
      a.addEventListener("click", (e) => { e.preventDefault(); print(); }));
    document.addEventListener("click", (e) => {
      const a = e.target.closest && e.target.closest("[data-action=menu]");
      if (a) { e.preventDefault(); toggleMenu(); }
    });

    const fonts = document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve();
    const ready = () => { root.dataset.ready = "1"; };
    fonts.then(ready, ready);
    pending(fonts);
    pending(load("series.js", () => Array.isArray(window.SERIES)).then(refreshFooters));
    pending(load("codemap.js", () => !!window.CODEMAP));
    pending(load("code.js", () => !!window.Code));
    // Promises a deck queued before this file ran; later pushes go straight to pending().
    const early = Array.isArray(window.DECK_PENDING) ? window.DECK_PENDING : [];
    early.forEach(pending);
    window.DECK_PENDING = { push: (...ps) => { ps.forEach(pending); return tasks.length; } };
    inited = true;
    settle();
    document.dispatchEvent(new CustomEvent("deckready"));
  }

  window.Deck = {
    next: () => next(),
    prev: () => prev(),
    // show(12) opens slide 12 (1-based); show("some-id") opens the slide with that id.
    show: (i) => {
      const k = slideIndex(typeof i === "number" ? String(i) : i);
      if (k < 0) return false;
      if (mode === "present") show(k, 0); else { cur = k; if (mode === "read") scrollToSlide(k); }
      return true;
    },
    setMode,
    pending,
    mode: () => mode,
    slides: () => slides.slice(),
    current: () => currentSlide(),
    index: () => (mode === "read" ? visibleIndex() : cur),
    menu: () => toggleMenu(),
    code: () => toggleCode(),
    lib: LIB,
    macros: MACROS,
  };
  init();
})();
