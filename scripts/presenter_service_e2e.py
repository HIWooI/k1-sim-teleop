"""Walking mode / person region end to end: the real service + sender + sim_server (a presenter clip as the fake
camera, MuJoCo only, no camera, no browser). Settings changed through the panel API (/api/set) must reach the sender
at once, survive a sender restart and a service restart (logs/k1live/settings.json; an existing one is set aside and
put back afterwards). Prints PASS/FAIL per step and writes a JSON report.

    K1LIVE_PORT_OFFSET=700 .venv/bin/python scripts/presenter_service_e2e.py logs/presenter/service_e2e.json \
        <event_videos>/presenter/pexels_6952134.mp4
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from k1live import protocol as P  # noqa: E402

UI, CAM = P.UI_HTTP_PORT, P.CAM_HTTP_PORT
out_path, video = sys.argv[1], sys.argv[2]
res = {"ports": {"ui": UI, "cam": CAM}, "video": video, "steps": []}


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as r:
        return json.loads(r.read())


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{UI}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def wait(pred, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            v = pred()
            if v:
                return round(time.time() - t0, 2), v
        except Exception:
            pass
        time.sleep(0.1)
    return None, None


def step(name, ok, **kw):
    res["steps"].append({"step": name, "ok": bool(ok), **kw})
    print(("PASS " if ok else "FAIL ") + name, json.dumps(kw, ensure_ascii=False, default=str), flush=True)


def start_service(tag):
    sess = ROOT / "logs" / "presenter" / f"svc_e2e_{tag}"
    sess.parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("K1LIVE_LOCO_PROFILE", "K1LIVE_CROP")}
    p = subprocess.Popen([sys.executable, "-m", "k1live.service", "--source", video, "--fallback", "off",
                          "--session-dir", str(sess)], cwd=ROOT, stdout=open(f"{sess}.out", "w"),
                         stderr=subprocess.STDOUT, start_new_session=True, env=env)
    return p


def stop_service(p):
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)


settings = ROOT / "logs" / "k1live" / "settings.json"
if settings.exists():
    settings.rename(settings.with_suffix(".json.bak"))   # start from event.yaml defaults
svc = start_service("1")
try:
    t, s = wait(lambda: (lambda st: st if (st["procs"]["sender"]["healthy"] and st["procs"]["sim"]["healthy"]
                                          and st["snd"] and st["sim"]) else None)(get(UI, "/api/state")), 120)
    step("service up (sender + sim healthy)", t is not None, after_s=t)
    t, snd = wait(lambda: (lambda x: x if x.get("loco_profile") == "presenter" else None)(get(CAM, "/state")), 15)
    step("event.yaml default reaches the sender (presenter, whole frame)", t is not None and snd["crop"] is None,
         after_s=t, sender=snd and {k: snd[k] for k in ("loco_profile", "crop", "fps", "present")})
    st = get(UI, "/api/state")
    step("panel state: presenter guide, no bubble", st["guide"].startswith("발표자") and st["audience_msg"] == "",
         guide=st["guide"], msg=st["audience_msg"])
    box = [0.0, 0.0, 0.55, 1.0]
    t0 = time.time()
    r = post("/api/set", {"crop": box})
    t, snd = wait(lambda: (lambda x: x if x.get("crop") == box else None)(get(CAM, "/state")), 10)
    step("set crop -> sender", r.get("ok") and t is not None, api=r, applied_s=t)
    t, snd = wait(lambda: (lambda x: x if x.get("present") else None)(get(CAM, "/state")), 10)
    step("person tracked inside the region", t is not None, after_s=t)
    r = post("/api/set", {"walk": True, "loco_profile": "visitor"})
    t, snd = wait(lambda: (lambda x: x if x.get("loco_profile") == "visitor" else None)(get(CAM, "/state")), 10)
    st = get(UI, "/api/state")
    step("walking mode visitor -> sender", r.get("ok") and t is not None, applied_s=t, guide=st["guide"])
    r = post("/api/set", {"walk": False})
    t, st = wait(lambda: (lambda x: x if x["sim"] and x["sim"]["walk_enabled"] is False else None)(
        get(UI, "/api/state")), 10)
    step("walking off -> sim walk_enabled false", r.get("ok") and t is not None, applied_s=t,
         guide=st and st["guide"])
    r = post("/api/set", {"walk": True, "loco_profile": "presenter"})
    t, st = wait(lambda: (lambda x: x if x["sim"]["walk_enabled"] and x["desired"]["loco_profile"] == "presenter"
                          else None)(get(UI, "/api/state")), 10)
    t2, snd = wait(lambda: (lambda x: x if x.get("loco_profile") == "presenter" else None)(get(CAM, "/state")), 10)
    step("walking mode presenter -> sim + sender", t is not None and t2 is not None, applied_s=[t, t2])
    r = post("/api/set", {"crop": [0.9, 0, 0.1, 1]})
    step("bad region refused by the service", not r.get("ok"), api=r)
    r = post("/api/cmd", {"cmd": "start"})
    t, st = wait(lambda: (lambda x: x if x["sim"]["streaming"] else None)(get(UI, "/api/state")), 10)
    time.sleep(5)
    st = get(UI, "/api/state")
    step("following a presenter: no walking from the person", t is not None,
         walking=st["walking"], loco_cmd=st["sim"]["loco_cmd"], present=st["present"], msg=st["audience_msg"])
    # sender restart: the service re-sends everything
    r = post("/api/cmd", {"cmd": "restart", "proc": "sender"})
    time.sleep(1.0)
    t, snd = wait(lambda: (lambda x: x if (x.get("note") != "loading" and x.get("loco_profile") == "presenter"
                                          and x.get("crop") == box) else None)(get(CAM, "/state")), 90)
    step("sender restart: profile + region re-applied", t is not None, after_s=t,
         restarts=get(UI, "/api/state")["procs"]["sender"]["restarts"])
    res["settings_json"] = json.loads(settings.read_text())
    step("settings.json", res["settings_json"].get("loco_profile") == "presenter"
         and res["settings_json"].get("crop") == box and res["settings_json"].get("walk") is True,
         saved={k: res["settings_json"].get(k) for k in ("walk", "loco_profile", "crop")})
finally:
    stop_service(svc)
# service restart: restored from settings.json
time.sleep(2)
svc = start_service("2")
try:
    t, st = wait(lambda: (lambda x: x if x["snd"] and x["sim"] else None)(get(UI, "/api/state")), 120)
    t2, snd = wait(lambda: (lambda x: x if (x.get("loco_profile") == "presenter" and x.get("crop") == box) else None)(
        get(CAM, "/state")), 30)
    step("service restart: restored and re-applied", t is not None and t2 is not None,
         desired={k: st["desired"][k] for k in ("walk", "loco_profile", "crop")} if st else None, sender_after_s=t2)
finally:
    stop_service(svc)
bak = settings.with_suffix(".json.bak")
if bak.exists():
    bak.replace(settings)
res["passed"] = sum(s["ok"] for s in res["steps"])
res["total"] = len(res["steps"])
Path(out_path).write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(f"== {res['passed']}/{res['total']}")
