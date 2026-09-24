/* deck.js — a small, dependency-free slide engine.
 *
 * A deck is a plain HTML file:
 *
 *   <body data-short="2. The Friction Cone" data-index="index.html"
 *         data-prev="11_coulomb_friction.html" data-next="13_contact_wrench_cone.html">
 *     <main class="deck">
 *       <section class="slide"> ... </section>
 *       ...
 *     </main>
 *
 * Modes (pick with ?mode=present|read|print, or press R to toggle):
 *   present  one slide at a time, scaled to fit the window (default)
 *   read     all slides stacked like a web page, reader notes shown
 *   print    one slide per 1280x720 page, for PDF export
 *
 * Keys: right/space/PgDn next, left/PgUp prev, Home/End, digits+Enter jump,
 * R read mode, F fullscreen, ? help. Elements with class "step" appear one
 * at a time (data-step="2" groups or reorders them). <aside class="notes"> is
 * shown only in read mode.
 */
(function () {
  "use strict";

  const W = 1280, H = 720;
  const params = new URLSearchParams(location.search);
  let mode = params.get("mode") || (params.has("print") ? "print" : "present");

  let slides = [];
  let cur = 0;      // current slide index
  let step = 0;     // number of revealed steps on current slide
  let typed = "";   // digits typed for jump

  const body = document.body;
  const deck = document.querySelector(".deck");

  // Shared KaTeX macros; a deck may add more via window.DECK_MACROS.
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
  }, window.DECK_MACROS || {});

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

  function buildFooters() {
    const short = body.dataset.short || document.title;
    const index = body.dataset.index || "index.html";
    slides.forEach((s, i) => {
      if (s.hasAttribute("data-nofoot")) return;
      const prev = i > 0 ? "#" + i : (body.dataset.prev || null);
      const next = i < slides.length - 1 ? "#" + (i + 2) : (body.dataset.next || null);
      const f = document.createElement("div");
      f.className = "foot";
      f.innerHTML =
        '<span class="deckname"></span>' +
        '<span><span class="nav">' +
        (prev ? '[<a href="' + prev + '" data-go="prev">&lt;&lt; prev</a>] ' : "[&lt;&lt; prev] ") +
        '[<a href="' + index + '">index</a>] ' +
        (next ? '[<a href="' + next + '" data-go="next">next &gt;&gt;</a>]' : "[next &gt;&gt;]") +
        '</span><span class="num">' + (i + 1) + " / " + slides.length + "</span></span>";
      f.querySelector(".deckname").textContent = short;
      s.appendChild(f);
    });
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
    cur = Math.max(0, Math.min(slides.length - 1, i));
    const s = slides[cur];
    const n = groups(s).length;
    step = st === "end" ? n : Math.max(0, Math.min(n, st || 0));
    slides.forEach((x, j) => x.classList.toggle("current", j === cur));
    reveal(s, step);
    history.replaceState(null, "", "#" + (cur + 1));
    s.dispatchEvent(new CustomEvent("slideshown", { bubbles: true }));
  }

  function next() {
    if (step < groups(slides[cur]).length) {
      step++;
      reveal(slides[cur], step);
    } else if (cur < slides.length - 1) {
      show(cur + 1, 0);
    } else if (body.dataset.next) {
      location.href = body.dataset.next;
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

  // ---------------------------------------------------------------- modes
  function setMode(m) {
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
    if (m === "read") {
      const head = document.createElement("p");
      head.className = "readhead";
      head.innerHTML = "<b></b> &mdash; reading view. [<a href=\"?mode=present#1\">present</a>] " +
        "[<a href=\"" + (body.dataset.index || "index.html") + "\">index</a>]";
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
  }

  // ---------------------------------------------------------------- input
  function onKey(e) {
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
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
    if (e.target.closest("input, select, svg .handle, details")) { touchX = null; return; }
    touchX = e.touches[0].clientX;
  }
  function onTouchEnd(e) {
    if (touchX === null || mode !== "present") return;
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
    "<tr><td><kbd>F</kbd></td><td>fullscreen</td></tr>" +
    "<tr><td><kbd>?</kbd></td><td>this help</td></tr></table>";

  // ---------------------------------------------------------------- init
  function init() {
    slides = Array.from(deck.querySelectorAll(":scope > section.slide"));
    renderMath();
    buildFooters();
    body.appendChild(help);
    const h = parseInt((location.hash || "#1").slice(1), 10);
    cur = isNaN(h) ? 0 : Math.max(0, Math.min(slides.length - 1, h - 1));
    setMode(mode);
    addEventListener("resize", fit);
    addEventListener("keydown", onKey);
    addEventListener("touchstart", onTouchStart, { passive: true });
    addEventListener("touchend", onTouchEnd);
    addEventListener("hashchange", () => {
      const n = parseInt(location.hash.slice(1), 10);
      if (!isNaN(n) && n - 1 !== cur && mode === "present") show(n - 1, 0);
    });
    let before = null;
    addEventListener("beforeprint", () => { before = mode; if (mode !== "print") setMode("print"); });
    addEventListener("afterprint", () => { if (before && before !== "print") setMode(before); before = null; });
    document.querySelectorAll("[data-action=print]").forEach((a) =>
      a.addEventListener("click", (e) => { e.preventDefault(); print(); }));
    const ready = () => { document.documentElement.dataset.ready = "1"; };
    (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(ready, ready);
  }

  window.Deck = { next: () => next(), prev: () => prev(), show: (i) => show(i - 1, 0), setMode };
  init();
})();
