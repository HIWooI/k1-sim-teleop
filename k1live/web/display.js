// Audience display. Normal mode: status over SSE (/api/events), MJPEG relays, fallback video when the service
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
  $("title").textContent = cfg.title;
  $("label-cam").textContent = cfg.label_cam;
  $("label-robot").textContent = cfg.label_robot;
  $("label-fb").textContent = cfg.label_fallback;
  cam.classList.toggle("flip", !!cfg.display_cam_flip);
  cam.classList.toggle("contain", cfg.cam_fit === "contain");
  robot.classList.toggle("contain", cfg.robot_fit === "contain");
  fbList = cfg.fallback.filter((v) => v.exists).map((v) => v.i);
}

// ---------------------------------------------------------------- live images
function startStreams() {
  streamsOn = true;
  if (preview) return pollImages();
  const t = Date.now();
  cam.src = `/cam.mjpg?fps=30&_=${t}`;
  robot.src = `/robot.mjpg?fps=30&_=${t}`;
}
function stopStreams() {
  streamsOn = false;
  cam.removeAttribute("src");
  robot.removeAttribute("src");
}
for (const img of [cam, robot]) {
  img.addEventListener("error", () => {
    if (!streamsOn || preview) return;
    setTimeout(() => { if (streamsOn) img.src = `/${img.id}.mjpg?fps=30&_=${Date.now()}`; }, 1000);
  });
}
let polling = false;
function pollImages() {
  if (polling) return;
  polling = true;
  const one = (img) => new Promise((res) => {
    const n = new Image();
    n.onload = () => { img.src = n.src; res(); };
    n.onerror = () => res();
    n.src = `/${img.id}.jpg?_=${Date.now()}`;
  });
  const loop = async () => {
    if (!streamsOn) { polling = false; return; }
    await Promise.all([one(cam), one(robot)]);
    setTimeout(loop, 200);
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
      $("rest").textContent = "대체 영상 재생 중 (관객 화면)";
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
  $("title").textContent = s.title || cfg.title;
  $("subtitle").textContent = s.subtitle || "";
  $("guide").textContent = s.guide || cfg.guide;
  const sim = s.sim || {};
  $("live").classList.toggle("on", s.view === "live" && !!sim.streaming);
  $("walk").classList.toggle("on", s.view === "live" && !!s.walking);
  const msg = s.audience_msg || "";
  if (msg) $("msg").textContent = msg;
  $("msg").classList.toggle("on", !!msg);
  setView(s.view);
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
