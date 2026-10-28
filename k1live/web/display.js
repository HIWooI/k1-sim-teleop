// Audience display. Normal mode: status over SSE (/api/events), latest JPEG pairs, fallback video when the service
// says view = "fallback". Preview mode (?preview=1, inside the operator panel): state comes from the parent page by
// postMessage and the images are polled as single JPEGs, so the preview holds no long-lived connections.
"use strict";
const $ = (id) => document.getElementById(id);
const preview = new URLSearchParams(location.search).has("preview");
const cam = $("cam"), robot = $("robot"), fb = $("fb");
let cfg = null, view = null, fbList = [], fbPos = 0, lastMsgT = 0, streamsOn = false;

if (preview) document.body.classList.add("preview");

async function loadConfig() {
  for (;;) {
    try {
      cfg = await (await fetch("/api/config", { cache: "no-store" })).json();
      break;
    } catch (e) {
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
  cam.classList.toggle("flip", !!cfg.display_cam_flip);
  cam.classList.toggle("contain", cfg.cam_fit === "contain");
  robot.classList.toggle("contain", cfg.robot_fit === "contain");
  fbList = cfg.fallback.filter((v) => v.exists).map((v) => v.i);
}

// ---------------------------------------------------------------- live images
// Fetch only the newest pair; decode both before committing on the same animation frame.
// One in-flight pair prevents a slow tab from accumulating an MJPEG playback backlog.
let polling = false, imageAbort = null, imageUrls = [];
function startStreams() {
  streamsOn = true;
  pollImages();
}
function stopStreams() {
  streamsOn = false;
  if (imageAbort) imageAbort.abort();
  cam.removeAttribute("src");
  robot.removeAttribute("src");
  imageUrls.forEach((url) => URL.revokeObjectURL(url));
  imageUrls = [];
}
function pollImages() {
  if (polling) return;
  polling = true;
  const period = preview ? 200 : 1000 / 30;
  const loop = async () => {
    if (!streamsOn) { polling = false; return; }
    const started = performance.now(), urls = [];
    imageAbort = new AbortController();
    const controller = imageAbort;
    const timeout = setTimeout(() => controller.abort(), 1500);
    let committed = false;
    try {
      const results = await Promise.allSettled([cam, robot].map(async (img) => {
        const response = await fetch(`/${img.id}.jpg?_=${Date.now()}`, {
          cache: "no-store", signal: controller.signal,
        });
        if (!response.ok) throw new Error("image unavailable");
        const url = URL.createObjectURL(await response.blob());
        urls.push(url);
        const next = new Image();
        next.src = url;
        await next.decode();
        return next;
      }));
      if (results.some((r) => r.status === "rejected")) throw new Error("incomplete image pair");
      const images = results.map((r) => r.value);
      if (streamsOn && !controller.signal.aborted) {
        await new Promise((resolve) => requestAnimationFrame(() => {
          if (streamsOn && !controller.signal.aborted) {
            cam.src = images[0].src;
            robot.src = images[1].src;
            imageUrls.forEach((url) => URL.revokeObjectURL(url));
            imageUrls = urls;
            committed = true;
          }
          resolve();
        }));
      }
    } catch (e) { /* Keep the last complete pair during a brief interruption. */ }
    finally {
      clearTimeout(timeout);
      if (!committed) urls.forEach((url) => URL.revokeObjectURL(url));
    }
    setTimeout(loop, Math.max(0, period - (performance.now() - started)));
  };
  loop();
}

// ---------------------------------------------------------------- fallback video
function playFallback() {
  if (!fbList.length) {
    $("rest").hidden = false;
    return;
  }
  fbPos %= fbList.length;
  fb.src = `/fallback/${fbList[fbPos]}.mp4`;
  fb.play().catch(() => {});
}
fb.addEventListener("ended", () => { fbPos += 1; playFallback(); });
fb.addEventListener("playing", () => { $("rest").hidden = true; });
fb.addEventListener("error", () => {
  if (view !== "fallback" || preview) return;
  $("rest").hidden = false;
  fbPos += 1;
  setTimeout(() => { if (view === "fallback") playFallback(); }, 2000);
});

function setView(v) {
  if (v === view) return;
  view = v;
  const fallback = v === "fallback";
  $("fallback-view").hidden = !fallback;
  $("live-view").style.visibility = fallback ? "hidden" : "visible";
  if (fallback) {
    stopStreams();
    if (preview) {
      fb.removeAttribute("src");
      $("rest").hidden = false;
      $("rest").textContent = "잠시 후 다시 시작합니다";
    } else {
      playFallback();
    }
  } else {
    fb.pause();
    fb.removeAttribute("src");
    fb.load();
    startStreams();
  }
}

// ---------------------------------------------------------------- state
function render(s) {
  lastMsgT = Date.now();
  if (!cfg) return;
  setView(s.view);
  if ($("live-word")) $("live-word").textContent = s.cam_ok && s.sim_ok ? "LIVE MOTION" : "STANDBY";
  cam.style.visibility = s.cam_ok === false ? "hidden" : "visible";
}

function connect() {
  if (preview) {
    window.addEventListener("message", (e) => {
      if (e.origin === location.origin && e.data && e.data.k1state) render(e.data.k1state);
    });
    return;
  }
  const es = new EventSource("/api/events");
  es.onmessage = (e) => {
    try { render(JSON.parse(e.data)); } catch (err) { /* ignore a torn message */ }
  };
  es.onopen = () => { if (view === "live") startStreams(); };
}

// service unreachable for a while: keep the screen calm (fallback video if it is still buffered, else a card)
setInterval(() => {
  if (preview || !cfg) return;
  if (Date.now() - lastMsgT > 4000 && view !== "fallback") setView("fallback");
}, 1000);

loadConfig().then(() => {
  setView("fallback");
  connect();
});

// F key or double-click: full screen (move the app window to the projector first)
function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  else document.documentElement.requestFullscreen().catch(() => {});
}
if (!preview) {
  document.addEventListener("keydown", (e) => { if (e.code === "KeyF") toggleFullscreen(); });
  document.addEventListener("dblclick", toggleFullscreen);
}
