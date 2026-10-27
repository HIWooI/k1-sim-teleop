// Operator panel: status over SSE, commands over POST /api/cmd and /api/set, keyboard shortcuts, drive pad.
"use strict";
const $ = (id) => document.getElementById(id);
let cfg = null, st = null, lastMsgT = 0;

// ---------------------------------------------------------------- server calls
function toast(msg, ms = 2500) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.h);
  toast.h = setTimeout(() => { t.hidden = true; }, ms);
}
async function post(path, body, quiet = false) {
  try {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) });
    const j = await r.json();
    if (!j.ok && !quiet) toast(j.msg || "실패");
    return j;
  } catch (e) {
    if (!quiet) toast("서비스에 연결할 수 없습니다");
    return { ok: false };
  }
}
const cmd = (c, args = {}, quiet = false) => post("/api/cmd", { cmd: c, ...args }, quiet);
const setv = (obj) => post("/api/set", obj);

// ---------------------------------------------------------------- config-driven controls
async function loadConfig() {
  for (;;) {
    try { cfg = await (await fetch("/api/config", { cache: "no-store" })).json(); break; }
    catch (e) { await new Promise((r) => setTimeout(r, 1000)); }
  }
  const g = $("gestures");
  g.innerHTML = "";
  for (const x of cfg.gestures) {
    const b = document.createElement("button");
    b.textContent = x.label;
    b.dataset.name = x.name;
    b.title = x.name;
    b.onclick = () => cmd("gesture", { name: x.name });
    g.appendChild(b);
  }
  const sel = $("source");
  sel.innerHTML = "";
  const og1 = document.createElement("optgroup");
  og1.label = "카메라";
  for (const c of cfg.cameras) og1.appendChild(new Option(`카메라 ${c}`, `camera:${c}`));
  sel.appendChild(og1);
  if (cfg.videos.length) {
    const og2 = document.createElement("optgroup");
    og2.label = "리허설 영상";
    for (const v of cfg.videos) og2.appendChild(new Option(v.name, v.path));
    sel.appendChild(og2);
  }
  $("max-speed").max = cfg.speed_limit;
}

// ---------------------------------------------------------------- rendering
const fmt = (v, d = 0) => (v === null || v === undefined || v === "" ? "–" : Number(v).toFixed(d));
function hms(s) {
  s = Math.floor(s || 0);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
  return `${h}:${String(m).padStart(2, "0")}:${String(x).padStart(2, "0")}`;
}
function tile(id, value, cls, small) {
  const t = $(id);
  t.querySelector("span").textContent = value;
  t.className = "tile" + (cls ? " " + cls : "");
  if (small !== undefined) t.querySelector("small").textContent = small;
}
function seg(id, value, disabled = {}) {
  for (const b of $(id).querySelectorAll("button")) {
    b.classList.toggle("on", b.dataset.v === String(value));
    b.disabled = !!disabled[b.dataset.v];
    b.title = disabled[b.dataset.v] ? String(disabled[b.dataset.v]) : "";
  }
}
const busy = (el) => document.activeElement === el || el.dataset.dragging === "1";
const WALK_LABEL = { visitor: "체험객", presenter: "강연자" };
const PROC_LABEL = { sender: "카메라·인식 (sender)", sim: "시뮬레이션 (sim_server)" };
const STATE_LABEL = { running: "실행 중", backoff: "재시작 대기", stopped: "정지", disabled: "꺼짐" };
const EVENT_LABEL = {
  service_start: "서비스 시작", service_stop: "서비스 종료", proc_start: "프로세스 시작", proc_healthy: "프로세스 정상",
  proc_fail: "프로세스 실패", proc_restart: "프로세스 재시작", proc_stop: "프로세스 정지", resync: "설정 재전송",
  cmd: "명령", set: "설정", fall: "낙상", watchdog_reset: "낙상 감시 리셋", auto_idle: "자동 대기",
  auto_calibrate: "자동 보정", auto_start: "자동 시작", calibrated: "보정 완료", calibrate_failed: "보정 실패",
  view: "관객 화면", camera_restart: "카메라 재시작", backend_failed: "백엔드 전환 실패", resend: "재전송",
  memory_restart: "메모리 한도 재시작", gemx_fallback: "GEM-X 불가 → MediaPipe",
};
const PERC_LABEL = { mediapipe: "MediaPipe", gemx: "GEM-X" };

function render(s) {
  st = s;
  lastMsgT = Date.now();
  const sim = s.sim || {}, snd = s.snd || {}, d = s.desired, c = s.counters;
  const pr = s.procs || {};
  const healthy = pr.sender && pr.sender.healthy && pr.sim && pr.sim.healthy;
  const svc = $("svc");
  svc.textContent = healthy ? (s.view === "live" ? "정상" : "정상 · 대체 영상") : "프로세스 확인 필요";
  svc.className = "pill " + (healthy ? (s.view === "live" ? "ok" : "warn") : "bad");
  $("mock").hidden = !s.mock;
  $("uptime").textContent = "가동 " + hms(s.uptime_s);

  // tiles
  const src = (d.source || "").startsWith("camera:") ? `카메라 ${d.source.split(":")[1]}`
    : (d.source || "").split("/").pop();
  const perc = s.perception || { want: "mediapipe", active: "mediapipe", note: "" };
  tile("t-cam", s.cam_ok ? "OK" : (snd.note === "loading" ? "로딩" : "없음"), s.cam_ok ? "ok" : "bad",
    `${src} · ${fmt(snd.fps)} fps · ${PERC_LABEL[perc.active] || perc.active}`);
  tile("t-person", s.present ? "감지" : "없음", s.present ? "ok" : "",
    (snd.calibrated ? "보정됨" : "보정 필요") + (s.absent_s ? ` · ${fmt(s.absent_s)}초 없음` : ""));
  const p95 = s.lat_ms.p95;
  tile("t-lat", p95 == null ? "–" : `${fmt(p95)} ms`, p95 == null ? "" : p95 <= 150 ? "ok" : p95 <= 250 ? "warn" : "bad",
    `p50 ${fmt(s.lat_ms.p50)} · p95 · 30초`);
  const hz = sim.ctrl_hz;
  tile("t-hz", hz == null ? "–" : `${fmt(hz, 1)} Hz`, hz == null ? "" : Math.abs(hz - 50) <= 2 ? "ok" : "warn",
    `실시간 비율 ${fmt(sim.rt_factor, 2)}`);
  tile("t-fall", `${c.falls} / ${fmt(sim.resets)}`, sim.fallen ? "bad" : "",
    sim.fallen ? "넘어짐 — 리셋 대기" : `감시 리셋 ${c.svc_resets}`);
  const hph = { stand: "서기(팔 집중)", walk: "걷기(SONIC)", settle: "멈추는 중" }[((sim.note || "").match(/hybrid:(\w+)/) || [])[1]];
  tile("t-backend", (sim.backend || "–") + (sim.backend === "hybrid" && hph ? ` · ${hph}` : ""), "",
       `사용 가능: ${(s.backends || []).join(", ")}`);
  tile("t-walk", !d.walk ? "꺼짐" : s.walking ? "걷는 중" : (WALK_LABEL[d.loco_profile] || "켜짐"), s.walking ? "ok" : "",
    (d.walk ? `${WALK_LABEL[d.loco_profile] || ""} · ` : "") + `최대 ${fmt(d.max_speed, 2)} m/s`);
  const lc = sim.loco_cmd || [0, 0, 0];
  tile("t-speed", `${fmt(Math.hypot(lc[0] || 0, lc[1] || 0), 2)} m/s`, s.walking ? "ok" : "",
    `vx ${fmt(lc[0], 2)} · vy ${fmt(lc[1], 2)} · wz ${fmt(lc[2], 2)}`);
  tile("t-view", s.view === "live" ? "실시간" : "대체 영상", s.view === "live" ? "ok" : "warn",
    s.view_reason || (s.settings.fallback === "auto" ? "자동" : s.settings.fallback === "on" ? "항상 대체 영상" : "대체 영상 끔"));

  // banners
  const bn = [];
  if (d.estop) bn.push(["bad", "비상정지 중 — 시작(Space)을 누르면 해제됩니다"]);
  if (sim.fallen) bn.push(["bad", "로봇이 넘어졌습니다 — 자동 리셋을 기다리거나 리셋(R)"]);
  if (s.attention) bn.push(["warn", "사람 감지됨 (자동 대기 중) — 보정(C) 후 시작(Space)"]);
  else if (s.auto_idle) bn.push(["ok", "자동 대기: 사람이 없어 인사 동작을 반복합니다"]);
  if (s.calib === "running") bn.push(["ok", "보정 중… 체험자는 그대로 서 있어 주세요"]);
  if (s.calib === "done") bn.push(["ok", "보정 완료"]);
  if (s.calib === "failed") bn.push(["bad", "보정 실패 — 사람이 보이는지 확인하고 다시 C"]);
  for (const [n, p] of Object.entries(pr)) {
    if (p.state === "backoff") bn.push(["warn", `${PROC_LABEL[n]} 재시작 대기 ${fmt(p.retry_in_s, 0)}초 — ${p.last_reason}`]);
    else if (p.state === "running" && !p.healthy) bn.push(["warn", `${PROC_LABEL[n]} 시작 중…`]);
  }
  if (perc.want === "gemx" && perc.active !== "gemx")
    bn.push(["warn", `GEM-X를 쓸 수 없어 MediaPipe로 인식 중 — ${escapeHtml(perc.note || "")} (설정 › 인식에서 GEM-X를 다시 누르면 재시도)`]);
  else if (perc.active === "gemx" && snd.note === "loading") bn.push(["ok", "GEM-X 인식 모델 불러오는 중 (20–40초) — 관객 화면은 대체 영상"]);
  if (s.view === "fallback" && s.settings.fallback === "auto") bn.push(["warn", `관객 화면: 대체 영상 재생 중 — ${s.view_reason || ""}`]);
  $("banners").innerHTML = bn.map(([k, t]) => `<div class="banner ${k}">${t}</div>`).join("");

  // controls
  const tg = $("b-toggle");
  tg.innerHTML = (d.running ? "■ 정지" : "▶ 시작") + "<kbd>Space</kbd>";
  tg.classList.toggle("running", d.running);
  seg("mode", s.auto_idle ? "idle" : d.mode);
  for (const b of $("gestures").querySelectorAll("button")) b.classList.toggle("on", sim.gesture === b.dataset.name);
  $("playlist").checked = !!s.playlist;
  seg("walk", d.walk ? d.loco_profile : "off");
  renderCrop(d);
  seg("mirror", d.mirror ? "1" : "0");
  seg("backend", d.backend, {
    sonic: (s.backends || []).includes("sonic") ? "" : "sonic 체크포인트(또는 sonic 백엔드)가 없어 split만 쓸 수 있습니다",
    sonic_human: (s.backends || []).includes("sonic_human") ? "" : "사람 전신 인코더 체크포인트(models/k1/k1s_human/deploy)가 아직 없습니다",
    hybrid: (s.backends || []).includes("hybrid") ? "" : "SONIC 보행 체크포인트(models/k1/k1s_walk3/deploy/k1s_walk3_005000) 또는 플래너가 없습니다",
  });
  seg("fallback", s.settings.fallback);
  seg("arm_path", s.settings.arm_path);
  seg("perception", perc.want);
  if (s.arm) $("arm_path").title = `${s.arm.arm_ik} IK · 팔 ${s.arm.arm_mode} · ${s.arm.arm_vmax} rad/s — 바꾸면 카메라·시뮬레이션이 몇 초 다시 시작합니다`;
  const ms = $("max-speed");
  if (!busy(ms)) ms.value = d.max_speed;
  $("max-speed-v").textContent = `${Number(ms.value).toFixed(2)} m/s`;
  const ia = $("idle-after");
  if (!busy(ia)) ia.value = s.settings.idle_after_s;
  $("auto-calib").checked = !!s.settings.auto_calibrate;
  $("auto-start").checked = !!s.settings.auto_start;
  const sel = $("source");
  if (!busy(sel) && !sel.dataset.pending) {
    if (![...sel.options].some((o) => o.value === d.source)) sel.appendChild(new Option(src, d.source));
    sel.value = d.source;
  }
  $("view-reason").textContent = s.view === "live" ? "실시간" : `대체 영상 · ${s.view_reason || ""}`;
  $("session").textContent = (s.session || "").split("/").slice(-2).join("/");

  // processes
  $("procs").innerHTML = Object.entries(pr).map(([n, p]) => {
    const cls = p.healthy ? "ok" : p.state === "running" ? "warn" : "bad";
    return `<tr><td><span class="dot ${cls}"></span>${PROC_LABEL[n] || n}</td>
      <td>${STATE_LABEL[p.state] || p.state}${p.uptime_s ? " · " + hms(p.uptime_s) : ""}</td>
      <td>재시작 ${p.restarts}</td><td>${p.rss_mb ? fmt(p.rss_mb) + " MB" : ""}</td>
      <td><button class="small" data-restart="${n}">재시작</button> <button class="small ghost" data-log="${n}">로그</button></td></tr>`;
  }).join("");

  // events
  if (s.events) {
    $("events").innerHTML = s.events.slice().reverse().filter((e) => e.kind !== "resend").map((e) => {
      const t = new Date(e.t * 1000).toTimeString().slice(0, 8);
      const rest = Object.entries(e).filter(([k, v]) => k !== "t" && k !== "kind" && v !== "" && v !== null)
        .map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" ");
      return `<li><time>${t}</time><span><b>${EVENT_LABEL[e.kind] || e.kind}</b> ${escapeHtml(rest).slice(0, 160)}</span></li>`;
    }).join("");
  }
  const fr = $("preview");
  if (fr.contentWindow) fr.contentWindow.postMessage({ k1state: s }, location.origin);
}
function escapeHtml(x) {
  return x.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// ---------------------------------------------------------------- wiring
function connect() {
  const es = new EventSource("/api/events");
  es.onmessage = (e) => { try { render(JSON.parse(e.data)); } catch (err) { console.error(err); } };
}
setInterval(() => {
  if (Date.now() - lastMsgT > 3000) {
    $("svc").textContent = "서비스 연결 끊김";
    $("svc").className = "pill bad";
  }
}, 1000);

document.addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.dataset.restart) cmd("restart", { proc: b.dataset.restart });
  if (b.dataset.log) showLog(b.dataset.log);
  setTimeout(() => b.blur(), 0);  // Space must never re-press a focused button
});
async function showLog(name) {
  const pre = $("logtail");
  if (!pre.hidden && pre.dataset.proc === name) { pre.hidden = true; return; }
  const j = await (await fetch(`/api/log?proc=${name}`)).json();
  pre.dataset.proc = name;
  pre.textContent = j.lines.join("\n") || "(로그 없음)";
  pre.hidden = false;
  pre.scrollTop = pre.scrollHeight;
}

$("b-toggle").onclick = () => cmd("toggle");
$("b-calib").onclick = () => cmd("calibrate");
$("b-reset").onclick = () => cmd("reset");
$("b-estop").onclick = () => cmd("estop");
for (const b of $("mode").querySelectorAll("button")) b.onclick = () => cmd("mode", { mode: b.dataset.v });
$("playlist").onchange = (e) => cmd("playlist", { on: e.target.checked });
for (const b of $("walk").querySelectorAll("button")) {
  b.onclick = () => setv(b.dataset.v === "off" ? { walk: false } : { walk: true, loco_profile: b.dataset.v });
}
for (const b of $("mirror").querySelectorAll("button")) b.onclick = () => setv({ mirror: b.dataset.v === "1" });
for (const b of $("backend").querySelectorAll("button")) b.onclick = () => setv({ backend: b.dataset.v });
for (const b of $("arm_path").querySelectorAll("button")) b.onclick = () => setv({ arm_path: b.dataset.v });
for (const b of $("perception").querySelectorAll("button")) b.onclick = () => setv({ perception: b.dataset.v });
for (const b of $("fallback").querySelectorAll("button")) b.onclick = () => setv({ fallback: b.dataset.v });
const ms = $("max-speed");
ms.addEventListener("pointerdown", () => { ms.dataset.dragging = "1"; });
ms.addEventListener("input", () => { $("max-speed-v").textContent = `${Number(ms.value).toFixed(2)} m/s`; });
ms.addEventListener("change", async () => { await setv({ max_speed: Number(ms.value) }); ms.dataset.dragging = ""; ms.blur(); });
$("idle-after").addEventListener("change", (e) => { setv({ idle_after_s: Number(e.target.value) }); e.target.blur(); });
$("auto-calib").onchange = (e) => setv({ auto_calibrate: e.target.checked });
$("auto-start").onchange = (e) => setv({ auto_start: e.target.checked });
$("source").addEventListener("change", (e) => { e.target.dataset.pending = "1"; });
$("b-source").onclick = async () => {
  const sel = $("source");
  const j = await setv({ source: sel.value });
  sel.dataset.pending = "";
  if (j.ok) toast("입력 소스 변경: " + sel.options[sel.selectedIndex].text);
};
let quitArmed = 0;
$("b-quit").onclick = () => {
  if (Date.now() - quitArmed < 4000) { cmd("shutdown"); $("quit-hint").textContent = "종료 중…"; return; }
  quitArmed = Date.now();
  $("quit-hint").textContent = "한 번 더 누르면 서비스가 종료됩니다 (4초 안에)";
  setTimeout(() => { if (Date.now() - quitArmed >= 4000) $("quit-hint").textContent = ""; }, 4100);
};

// ---------------------------------------------------------------- person search region (sender --crop)
// The preview is /cam.mjpg, mirrored like the robot when mirror is on; the region is stored in raw camera image
// coordinates (0..1), so x flips when mirror is on.
function cropToView(c, mirror) {
  return mirror ? [1 - c[2], c[1], 1 - c[0], c[3]] : c;
}
function place(el, b) {
  el.style.left = `${b[0] * 100}%`; el.style.top = `${b[1] * 100}%`;
  el.style.width = `${(b[2] - b[0]) * 100}%`; el.style.height = `${(b[3] - b[1]) * 100}%`;
}
function renderCrop(d) {
  const r = $("crop-rect");
  r.hidden = !d.crop;
  if (d.crop) place(r, cropToView(d.crop, d.mirror));
  $("crop-v").textContent = d.crop ? `x ${d.crop[0].toFixed(2)}–${d.crop[2].toFixed(2)} · y ${d.crop[1].toFixed(2)}–${d.crop[3].toFixed(2)}`
    : "전체 화면";
}
const cimg = $("crop-img");
function camPreview() {
  cimg.src = `/cam.mjpg?fps=8&t=${Date.now()}`;
}
cimg.onerror = () => setTimeout(camPreview, 2000);
camPreview();
let drag = null;
const at = (e) => {
  const r = cimg.getBoundingClientRect();
  return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
};
const box = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
$("cropbox").addEventListener("pointerdown", (e) => {
  if (!cimg.naturalWidth) { toast("카메라 화면이 아직 없습니다"); return; }
  e.preventDefault();
  $("cropbox").setPointerCapture(e.pointerId);
  drag = { a: at(e) };
  place($("crop-draft"), box(drag.a, drag.a));
  $("crop-draft").hidden = false;
});
$("cropbox").addEventListener("pointermove", (e) => {
  if (drag) place($("crop-draft"), box(drag.a, at(e)));
});
const endDrag = async (e, cancel) => {
  if (!drag) return;
  const b = box(drag.a, at(e));
  drag = null;
  $("crop-draft").hidden = true;
  if (cancel) return;
  if (b[2] - b[0] < 0.1 || b[3] - b[1] < 0.1) { toast("영역이 너무 작습니다 (가로·세로 10 % 이상)"); return; }
  const mirror = st ? st.desired.mirror : true;
  const raw = cropToView(b, mirror).map((v) => Math.round(v * 1000) / 1000);
  const j = await setv({ crop: raw });
  if (j.ok) toast("사람 찾을 영역 지정");
};
$("cropbox").addEventListener("pointerup", (e) => endDrag(e, false));
$("cropbox").addEventListener("pointercancel", (e) => endDrag(e, true));
$("b-crop-clear").onclick = async () => { const j = await setv({ crop: null }); if (j.ok) toast("영역 지움: 전체 화면"); };

// ---------------------------------------------------------------- keyboard + operator drive
const DRIVE_KEYS = { ArrowUp: "fwd", KeyW: "fwd", ArrowDown: "back", KeyS: "back", ArrowLeft: "left", KeyA: "left",
  ArrowRight: "right", KeyD: "right", KeyQ: "turnL", KeyE: "turnR" };
const held = new Set(), padHeld = new Set();
const typing = (el) => el && (el.tagName === "INPUT" && el.type !== "checkbox" && el.type !== "range"
  || el.tagName === "SELECT" || el.tagName === "TEXTAREA");
function flash(id) { const b = $(id); b.classList.add("flash"); setTimeout(() => b.classList.remove("flash"), 200); }
document.addEventListener("keydown", (e) => {
  if (e.code === "Escape") { e.preventDefault(); cmd("estop"); flash("b-estop"); return; }
  if (e.metaKey || e.ctrlKey || e.altKey || typing(e.target)) return;
  if (e.code === "Space") { e.preventDefault(); if (!e.repeat) { cmd("toggle"); flash("b-toggle"); } }
  else if (e.code === "KeyC") { if (!e.repeat) { cmd("calibrate"); flash("b-calib"); } }
  else if (e.code === "KeyR") { if (!e.repeat) { cmd("reset"); flash("b-reset"); } }
  else if (DRIVE_KEYS[e.code]) { e.preventDefault(); held.add(DRIVE_KEYS[e.code]); }
});
document.addEventListener("keyup", (e) => {
  if (e.code === "Space") e.preventDefault();
  if (DRIVE_KEYS[e.code]) held.delete(DRIVE_KEYS[e.code]);
});
window.addEventListener("blur", () => { held.clear(); padHeld.clear(); });
for (const b of $("pad").querySelectorAll("button")) {
  const on = (e) => { e.preventDefault(); b.setPointerCapture(e.pointerId); padHeld.add(b.dataset.d); };
  const off = () => padHeld.delete(b.dataset.d);
  b.addEventListener("pointerdown", on);
  b.addEventListener("pointerup", off);
  b.addEventListener("pointercancel", off);
  b.addEventListener("lostpointercapture", off);
}
let driving = false;
setInterval(() => {
  const h = new Set([...held, ...padHeld]);
  for (const b of $("pad").querySelectorAll("button")) b.classList.toggle("on", h.has(b.dataset.d));
  const v = st ? st.desired.max_speed : 0.3, w = cfg ? cfg.turn_speed : 0.6;
  const vx = ((h.has("fwd") ? 1 : 0) - (h.has("back") ? 1 : 0)) * v;
  const vy = ((h.has("left") ? 1 : 0) - (h.has("right") ? 1 : 0)) * v;
  const wz = ((h.has("turnL") ? 1 : 0) - (h.has("turnR") ? 1 : 0)) * w;
  const moving = vx !== 0 || vy !== 0 || wz !== 0;
  if (moving || driving) cmd("drive", { vx, vy, wz }, true);
  driving = moving;
}, 100);

// the display gets the other loopback name so its streams never compete with the panel's connections
const dispHost = location.hostname === "localhost" ? "127.0.0.1" : "localhost";
document.querySelector("a.link").href = `${location.protocol}//${dispHost}:${location.port}/display`;

loadConfig().then(connect);
