(() => {
  "use strict";
  const script = document.currentScript;
  const cfg = script ? script.dataset : {};
  const DEFAULTS = {
    check: cfg.check !== "off",
    flash: cfg.flash !== "off",
    fullscreen: cfg.fullscreen !== "off",
    auto: cfg.auto === "on",
    ui: cfg.ui !== "off",
    upload: cfg.upload || null,
    name: null,
    selector: "[data-gaze-aoi]",
    autoSelector: "a, button, input, select, textarea, img, video, h1, h2, h3, nav, header, footer, form, [role=button]",
    sampleMs: 200,
    pointMs: 1600,
  };
  const POINTS = [[0.5, 0.5], [0.08, 0.1], [0.92, 0.1], [0.08, 0.9], [0.92, 0.9], [0.5, 0.1], [0.08, 0.5], [0.92, 0.5], [0.5, 0.9]];
  const S = { opts: null, log: null, recorder: null, chunks: [], stream: null, t0: 0, timer: null, last: "", overlay: null, button: null, on: [] };
  const now = () => Math.round((performance.now() - S.t0) * 10) / 10;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function viewport() {
    return { width: innerWidth, height: innerHeight, fullscreen: !!document.fullscreenElement,
             browser_fullscreen: innerWidth === screen.width && outerHeight === innerHeight,
             screen_x: screenX, screen_y: screenY, outer_width: outerWidth, outer_height: outerHeight };
  }

  function label(el) {
    if (el.dataset.gazeAoi) return el.dataset.gazeAoi;
    if (el.id) return `#${el.id}`;
    const text = (el.getAttribute("aria-label") || el.alt || el.innerText || el.value || el.name || "").trim().replace(/\s+/g, " ").slice(0, 40);
    return text ? `${el.tagName.toLowerCase()}: ${text}` : el.tagName.toLowerCase();
  }

  function elements() {
    const sel = S.opts.auto ? `${S.opts.selector}, ${S.opts.autoSelector}` : S.opts.selector;
    const seen = {};
    const out = [];
    for (const el of document.querySelectorAll(sel)) {
      if (el.closest("[data-gaze-ignore]")) continue;
      let name = label(el);
      seen[name] = (seen[name] || 0) + 1;
      if (seen[name] > 1) name = `${name} (${seen[name]})`;
      const r = el.getBoundingClientRect();
      if (r.width < 1 || r.height < 1 || r.bottom < 0 || r.right < 0 || r.top > innerHeight || r.left > innerWidth) continue;
      const item = { name, x: Math.round(r.left * 10) / 10, y: Math.round(r.top * 10) / 10, w: Math.round(r.width * 10) / 10, h: Math.round(r.height * 10) / 10 };
      if (el.tagName === "IMG" && el.naturalWidth) { item.natural_w = el.naturalWidth; item.natural_h = el.naturalHeight; }
      out.push(item);
    }
    return out;
  }

  function snapshot() {
    if (!S.log) return;
    const snap = { url: location.href, scroll_x: scrollX, scroll_y: scrollY, viewport: viewport(), aoi: elements() };
    const key = JSON.stringify(snap);
    if (key === S.last) return;
    S.last = key;
    S.log.snapshots.push({ t: now(), ...snap });
  }

  function overlay(color) {
    if (!S.overlay) {
      S.overlay = document.createElement("canvas");
      S.overlay.dataset.gazeIgnore = "";
      Object.assign(S.overlay.style, { position: "fixed", inset: "0", width: "100vw", height: "100vh", zIndex: 2147483647, cursor: "none" });
      document.documentElement.appendChild(S.overlay);
    }
    const c = S.overlay;
    c.width = innerWidth * devicePixelRatio;
    c.height = innerHeight * devicePixelRatio;
    const ctx = c.getContext("2d");
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, innerWidth, innerHeight);
    return ctx;
  }

  function closeOverlay() {
    if (S.overlay) S.overlay.remove();
    S.overlay = null;
  }

  async function flash(label) {
    overlay("#000");
    await sleep(500);
    const start = now();
    overlay("#fff");
    await sleep(300);
    overlay("#000");
    S.log.flashes.push({ label, t_start: start, t_end: now() });
    await sleep(500);
    closeOverlay();
  }

  function text(ctx, msg) {
    ctx.fillStyle = "#ddd";
    ctx.font = "20px -apple-system, Helvetica, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(msg, innerWidth / 2, innerHeight / 2 - 40);
  }

  async function check() {
    text(overlay("#000"), "Look at each yellow dot until it disappears");
    await sleep(2200);
    for (const [fx, fy] of POINTS) {
      const x = fx * innerWidth, y = fy * innerHeight;
      const start = now();
      const t = performance.now();
      while (performance.now() - t < S.opts.pointMs) {
        const ctx = overlay("#000");
        const r = 8 + 22 * Math.max(0, 1 - (performance.now() - t) / 350);
        ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.strokeStyle = "#f5c84c"; ctx.lineWidth = 2.5; ctx.stroke();
        ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 2 * Math.PI); ctx.fillStyle = "#fff"; ctx.fill();
        await new Promise(requestAnimationFrame);
      }
      S.log.check.push({ t_start: start, t_end: now(), x: Math.round(x * 10) / 10, y: Math.round(y * 10) / 10 });
    }
    closeOverlay();
  }

  function listen(target, type, fn, opt) {
    target.addEventListener(type, fn, opt);
    S.on.push([target, type, fn, opt]);
  }

  function onClick(e) {
    if (e.target.closest && e.target.closest("[data-gaze-ignore]")) return;
    const el = e.target.closest ? e.target.closest(S.opts.auto ? `${S.opts.selector}, ${S.opts.autoSelector}` : S.opts.selector) : null;
    snapshot();
    S.log.events.push({ type: "click", t: now(), x: e.clientX, y: e.clientY, element: el ? label(el) : null });
  }

  function pickMime() {
    const options = ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
    return options.find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || "";
  }

  async function start(options = {}) {
    if (S.log) return;
    S.opts = { ...DEFAULTS, ...options };
    S.stream = await navigator.mediaDevices.getUserMedia({ audio: false,
      video: { facingMode: "user", width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } } });
    if (S.opts.fullscreen && !document.fullscreenElement) await document.documentElement.requestFullscreen().catch(() => {});
    await sleep(400);
    const settings = S.stream.getVideoTracks()[0].getSettings();
    const mimeType = pickMime();
    S.chunks = [];
    S.recorder = new MediaRecorder(S.stream, mimeType ? { mimeType, videoBitsPerSecond: 6000000 } : undefined);
    S.recorder.ondataavailable = (e) => { if (e.data && e.data.size) S.chunks.push(e.data); };
    const started = new Promise((r) => { S.recorder.onstart = r; });
    S.recorder.start(1000);
    await started;
    S.t0 = performance.now();
    S.last = "";
    S.log = { format: "gaze-session-1", created: new Date().toISOString(), url: location.href, user_agent: navigator.userAgent,
              screen: { width: screen.width, height: screen.height, device_pixel_ratio: devicePixelRatio },
              time_origin: "MediaRecorder start event (ms)",
              video: { mime_type: S.recorder.mimeType, width: settings.width, height: settings.height, frame_rate: settings.frameRate },
              flashes: [], check: [], snapshots: [], events: [] };
    listen(window, "scroll", snapshot, { passive: true, capture: true });
    listen(window, "resize", snapshot);
    listen(document, "fullscreenchange", snapshot);
    listen(window, "click", onClick, true);
    S.timer = setInterval(snapshot, S.opts.sampleMs);
    snapshot();
    updateButton();
    if (S.opts.flash) await flash("start");
    if (S.opts.check) await check();
    mark("start");
  }

  function mark(name) {
    if (!S.log) return;
    snapshot();
    S.log.events.push({ type: "mark", t: now(), label: name });
  }

  async function stop() {
    if (!S.log) return null;
    mark("end");
    if (S.opts.flash) await flash("end");
    clearInterval(S.timer);
    for (const [target, type, fn, opt] of S.on) target.removeEventListener(type, fn, opt);
    S.on = [];
    const stopped = new Promise((r) => { S.recorder.onstop = r; });
    S.log.duration_ms = now();
    S.recorder.stop();
    await stopped;
    S.stream.getTracks().forEach((t) => t.stop());
    if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
    const log = S.log;
    S.log = null;
    updateButton();
    return save(log);
  }

  async function save(log) {
    const stamp = new Date().toISOString().replace(/[-:]/g, "").replace("T", "_").slice(0, 15);
    const name = S.opts.name || `session_${stamp}`;
    const ext = (log.video.mime_type || "").includes("mp4") ? "mp4" : "webm";
    log.video_file = `${name}.${ext}`;
    const files = [[new Blob(S.chunks, { type: log.video.mime_type || "video/webm" }), log.video_file],
                   [new Blob([JSON.stringify(log)], { type: "application/json" }), `${name}.json`]];
    if (S.opts.upload) {
      try {
        for (const [blob, file] of files) {
          const r = await fetch(`${S.opts.upload}?name=${encodeURIComponent(file)}`, { method: "POST", body: blob });
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
        }
        return { uploaded: files.map((f) => f[1]), log };
      } catch (e) {}
    }
    for (const [blob, file] of files) {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = file;
      document.body.appendChild(a);
      a.click();
      a.remove();
    }
    return { downloaded: files.map((f) => f[1]), log };
  }

  function updateButton() {
    if (!S.button) return;
    S.button.textContent = S.log ? "■ Stop gaze recording" : "● Record gaze";
    S.button.style.background = S.log ? "#7a1f1f" : "#1f4fa8";
  }

  function addButton() {
    S.opts = S.opts || { ...DEFAULTS };
    if (!S.opts.ui || S.button) return;
    const b = document.createElement("button");
    b.dataset.gazeIgnore = "";
    Object.assign(b.style, { position: "fixed", right: "14px", bottom: "14px", zIndex: 2147483646, color: "#fff", border: "0",
                             borderRadius: "18px", padding: "8px 14px", font: "13px -apple-system, Helvetica, sans-serif",
                             boxShadow: "0 2px 8px rgba(0,0,0,.25)", cursor: "pointer", opacity: "0.85" });
    b.addEventListener("click", () => (S.log ? stop() : start()));
    S.button = b;
    updateButton();
    document.body.appendChild(b);
  }

  window.GazeRecorder = { start, stop, mark, snapshot, recording: () => !!S.log };
  if (DEFAULTS.ui) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", addButton);
    else addButton();
  }
})();
