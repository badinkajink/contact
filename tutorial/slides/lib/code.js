/* code.js — links from slides to the notebook code that computes their numbers.
 *
 * deck.js loads this file (and lib/codemap.js, which tools/sync.py generates) by itself.
 *
 *   <section class="slide" id="newton-step" data-code="01_optimization_fundamentals.py:run_newton">
 *     ... the footer gets a [code] link
 *   <code data-code="01_optimization_fundamentals.py:run_newton">run_newton</code>
 *     ... an inline element becomes a link to the same overlay
 *   <figure data-code="04_friction_cones.py:push_box"> ... </figure>
 *     ... a block element gets a small [code] link at its end (in the figcaption of a figure)
 *
 * A data-code value is one or more "FILE:name" references separated by spaces or commas; FILE
 * is relative to tutorial/ (e.g. "labs/hybrid_servoing/hfvc.py:solve_ochs").
 *
 * Clicking opens an overlay with the function's source from window.CODEMAP, its file and line
 * range, a GitHub link to those lines, the command that opens the notebook locally, and molab
 * links that run it in the browser. C toggles the overlay for the current slide (deck.js),
 * Esc closes it. In the reading view the sources appear under each slide's notes inside
 * <details>. Print mode shows none of this.
 *
 * API: Code.open(slideElement | "FILE:name ..." | [refs]), Code.close(), Code.toggle(slide),
 *      Code.isOpen(), Code.refs(slide) -> [refs], Code.github(ref) -> URL or null.
 */
(function () {
  "use strict";
  if (window.Code) return;

  const REPO = "badinkajink/contact";
  const GITHUB = "https://github.com/" + REPO + "/blob/main/tutorial/";
  const MOLAB = "https://molab.marimo.io/github/" + REPO + "/blob/main/tutorial/";
  // Elements that turn into a link themselves; any other element gets a [code] link appended.
  const INLINE = new Set(["A", "ABBR", "B", "CITE", "CODE", "DFN", "EM", "I", "KBD", "LABEL", "MARK",
    "Q", "S", "SAMP", "SMALL", "SPAN", "STRONG", "SUB", "SUP", "TIME", "TT", "U", "VAR"]);

  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const refsOf = (el) => (el && el.getAttribute("data-code") || "").split(/[,\s]+/).filter(Boolean);
  const entry = (ref) => (window.CODEMAP && Object.prototype.hasOwnProperty.call(window.CODEMAP, ref) ? window.CODEMAP[ref] : null);
  const nameOf = (ref) => ref.slice(ref.lastIndexOf(":") + 1);
  const fileOf = (ref) => { const e = entry(ref); return e ? e.file : ref.slice(0, Math.max(0, ref.lastIndexOf(":"))); };
  const github = (ref) => { const e = entry(ref); return e ? GITHUB + e.file + "#L" + e.start + "-L" + e.end : null; };
  const allSlides = () => (window.Deck && window.Deck.slides ? window.Deck.slides() : Array.from(document.querySelectorAll("section.slide")));
  const lines = (e) => (e.start === e.end ? "line " + e.start : "lines " + e.start + "&ndash;" + e.end);

  // Every reference a slide links: the section's own first, then inline ones, without repeats.
  function slideRefs(slide) {
    if (!slide) return [];
    const out = [];
    const add = (r) => { if (!out.includes(r)) out.push(r); };
    refsOf(slide).forEach(add);
    slide.querySelectorAll("[data-code]").forEach((el) => refsOf(el).forEach(add));
    return out;
  }

  // ---------------------------------------------------------------- one reference, as HTML
  function sourceHTML(e) {
    const rows = String(e.src).replace(/\n+$/, "").split("\n");
    return rows.map((l, k) => '<span class="ln">' + (e.start + k) + "</span>" + esc(l)).join("\n");
  }
  function itemHTML(ref) {
    const e = entry(ref);
    if (!e) {
      return '<p class="codehead"><b>' + esc(ref) + "</b> is not in lib/codemap.js. Run " +
        "<code>uv run tools/sync.py</code> in <code>tutorial/slides</code>, or fix the reference.</p>";
    }
    const nb = e.marimo !== false;
    let h = '<p class="codehead"><b>' + esc(nameOf(ref)) + "</b> in <code>tutorial/" + esc(e.file) + "</code>, " + lines(e) + "</p>";
    h += '<p class="codelinks">[<a href="' + esc(github(ref)) + '" target="_blank" rel="noopener">GitHub, ' + lines(e) + "</a>]";
    if (nb) {
      h += ' [<a href="' + esc(MOLAB + e.file) + '" target="_blank" rel="noopener">run in molab</a>]' +
        ' [<a href="' + esc(MOLAB + e.file + "/wasm") + '" target="_blank" rel="noopener">run in the browser (WebAssembly)</a>]';
    }
    h += "</p>";
    h += '<p class="codecmd">local: <code>' + (nb ? "uvx marimo edit --sandbox tutorial/" : "uv run tutorial/") + esc(e.file) + "</code></p>";
    h += '<pre class="codesrc">' + sourceHTML(e) + "</pre>";
    return h;
  }

  // ---------------------------------------------------------------- overlay
  let box = null, lastFocus = null;
  function ensureBox() {
    if (box) return box;
    box = document.createElement("div");
    box.className = "codebox";
    box.setAttribute("role", "dialog");
    box.setAttribute("aria-modal", "true");
    box.setAttribute("aria-label", "Source code");
    box.addEventListener("click", (ev) => {
      if (ev.target === box || ev.target.closest("[data-close]")) { ev.preventDefault(); close(); }
    });
    document.body.appendChild(box);
    return box;
  }
  const isOpen = () => !!(box && box.classList.contains("on"));

  function open(arg, slide) {
    const mode = window.Deck && window.Deck.mode ? window.Deck.mode() : "present";
    if (mode === "print") return;
    let refs;
    if (typeof arg === "string") refs = arg.split(/[,\s]+/).filter(Boolean);
    else if (Array.isArray(arg)) refs = arg.slice();
    else { slide = slide || arg; refs = slideRefs(arg); }
    const b = ensureBox();
    const list = allSlides();
    const n = slide ? list.indexOf(slide) + 1 : 0;
    const title = slide ? (slide.querySelector("h1, h2") || {}).textContent || "" : "";
    let h = '<div class="codepanel" tabindex="-1"><p class="codetop"><b>' +
      (n ? "Code behind slide " + n : "Code") + "</b>" + (title ? " &nbsp;<i>" + esc(title.trim()) + "</i>" : "") +
      ' &nbsp;[<a href="#" data-close>close</a>] <span class="muted">(Esc)</span></p>';
    if (!refs.length) {
      h += '<p>No code is linked from this slide. Add <code>data-code="NOTEBOOK.py:function"</code> to its ' +
        "<code>&lt;section&gt;</code> or to an inline element.</p>";
    }
    refs.forEach((r, i) => { h += (i ? "<hr>" : "") + '<div class="codeitem">' + itemHTML(r) + "</div>"; });
    b.innerHTML = h + "</div>";
    lastFocus = document.activeElement;
    b.classList.add("on");
    const p = b.querySelector(".codepanel");
    p.scrollTop = 0;
    p.focus({ preventScroll: true });
  }
  function close() {
    if (!isOpen()) return;
    box.classList.remove("on");
    if (lastFocus && lastFocus.focus && document.contains(lastFocus)) lastFocus.focus({ preventScroll: true });
    lastFocus = null;
  }
  function toggle(slide) {
    if (isOpen()) { close(); return; }
    open(slide || (window.Deck && window.Deck.current ? window.Deck.current() : null));
  }

  // ---------------------------------------------------------------- links on the slides
  function footLink(slide) {
    const nav = slide.querySelector(":scope > .foot .nav");
    if (!nav || nav.querySelector(".codenav") || !refsOf(slide).length) return;  // data-code="" links nothing
    const span = document.createElement("span");
    span.className = "codenav";
    const href = github(refsOf(slide)[0] || "") || "#";
    span.innerHTML = '[<a class="codeopen" href="' + esc(href) + '">code</a>] ';
    nav.insertBefore(span, nav.firstChild);
  }
  function wireElement(el) {
    if (el.matches("section.slide") || el.closest(".codebox") || el.hasAttribute("data-code-wired")) return;
    const refs = refsOf(el);
    if (!refs.length) return;
    el.setAttribute("data-code-wired", "");
    const href = github(refs[0] || "") || "#";
    const missing = refs.some((r) => !entry(r));
    const svg = typeof SVGElement !== "undefined" && el instanceof SVGElement;
    if (INLINE.has(el.tagName) || svg) {
      el.classList.add("codelink");
      if (missing) el.classList.add("codelink-missing");
      if (el.tagName === "A") { if (!el.hasAttribute("href")) el.setAttribute("href", href); }
      else { el.setAttribute("role", "link"); el.setAttribute("tabindex", "0"); }
      if (!el.hasAttribute("title")) el.setAttribute("title", "source: " + refs.join(", "));
    } else {
      const a = document.createElement("a");
      a.className = "codelink-extra";
      a.href = href;
      a.textContent = "[code]";
      const host = (el.tagName === "FIGURE" && el.querySelector(":scope > figcaption")) || el;
      host.appendChild(document.createTextNode(" "));
      host.appendChild(a);
    }
  }

  // Reading view: each slide's sources under its notes, closed by default.
  function addReadBlocks() {
    allSlides().forEach((s) => {
      const refs = slideRefs(s);
      const slot = s.parentElement;
      const notes = slot && slot.classList.contains("slot") ? slot.nextElementSibling : null;
      if (!refs.length || !notes || !notes.classList.contains("notes-out") || notes.querySelector(".code-read")) return;
      refs.forEach((r) => {
        const e = entry(r);
        const d = document.createElement("details");
        d.className = "code-read";
        d.innerHTML = "<summary>code: " + esc(nameOf(r)) + " (" + esc(fileOf(r) || r) +
          (e ? ", " + lines(e) : ", not in lib/codemap.js") + ")</summary><div>" + itemHTML(r) + "</div>";
        notes.appendChild(d);
      });
    });
  }

  function onMode(m) {
    if (m !== "present" && m !== "read") close();
    if (m === "read") addReadBlocks();
  }

  function start() {
    document.querySelectorAll("section.slide[data-code]").forEach(footLink);
    document.querySelectorAll("[data-code]").forEach(wireElement);
    onMode(window.Deck && window.Deck.mode ? window.Deck.mode() : "present");
    document.addEventListener("deckmode", (ev) => onMode(ev.detail && ev.detail.mode));
  }

  document.addEventListener("click", (ev) => {
    const t = ev.target;
    const a = t && t.closest && t.closest(".codeopen, .codelink, .codelink-extra");
    if (!a || a.closest(".codebox")) return;
    const href = a.getAttribute("href");
    if (a.tagName === "A" && href && href !== "#" && (ev.ctrlKey || ev.metaKey || ev.shiftKey || ev.button !== 0)) return;
    ev.preventDefault();
    const slide = a.closest("section.slide");
    if (a.classList.contains("codeopen")) open(slideRefs(slide), slide);
    else open(refsOf(a.closest("[data-code]")), slide);
  });
  // Enter on a focused inline link opens it (and must not also advance the slide).
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && ev.target && ev.target.classList && ev.target.classList.contains("codelink")) {
      ev.preventDefault();
      ev.stopPropagation();
      open(refsOf(ev.target.closest("[data-code]")), ev.target.closest("section.slide"));
    }
  });
  addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && isOpen()) { ev.preventDefault(); close(); }
  });

  window.Code = { open, close, toggle, isOpen, refs: slideRefs, github };

  if (window.Deck && window.Deck.slides) start();
  else if (document.querySelector('script[src*="deck.js"]')) document.addEventListener("deckready", start, { once: true });
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
  else start();
})();
