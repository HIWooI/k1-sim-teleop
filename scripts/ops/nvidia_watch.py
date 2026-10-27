"""Watchdog for GPU-fenced runs (scripts/ops/cpu_only.env): every INTERVAL s, look at every process of this user
whose environment carries K1_FENCE=<token>; if one maps an NVIDIA / CUDA library or holds an NVIDIA device node open
(/dev/nvidia*, the DRM card / render node whose driver is nvidia), SIGKILL all of them, write the evidence and exit 3.
Standard library only, reads /proc.

    python3 scripts/ops/nvidia_watch.py TOKEN OUT.json [--seconds 900] [--interval 1]
exit 0 = run ended clean (no fenced process left for 10 s, or --seconds reached), 3 = violation (everything killed)
"""
import glob
import json
import os
import signal
import sys
import time

LIBS = ("libnvidia", "libcuda", "libnvcuvid", "libEGL_nvidia", "libGLX_nvidia", "libnvoptix")


def nvidia_nodes():
    nodes = set(glob.glob("/dev/nvidia*"))
    for d in glob.glob("/sys/class/drm/*/device/driver"):
        if os.path.basename(os.path.realpath(d)) == "nvidia":
            name = d.split("/")[4]
            if name.startswith(("card", "renderD")) and "-" not in name:
                nodes.add(f"/dev/dri/{name}")
    return nodes


def fenced(token):
    me, out = os.getuid(), []
    for d in glob.glob("/proc/[0-9]*"):
        pid = int(d[6:])
        if pid == os.getpid():
            continue
        try:
            if os.stat(d).st_uid != me:
                continue
            env = open(f"{d}/environ", "rb").read()
        except OSError:
            continue
        if f"K1_FENCE={token}".encode() in env.split(b"\0"):
            out.append(pid)
    return out


def touches(pid, nodes):
    hits = []
    try:
        maps = open(f"/proc/{pid}/maps").read()
        hits += sorted({l.split()[-1] for l in maps.splitlines() if any(x in l for x in LIBS)})
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                tgt = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if tgt in nodes:
                hits.append(tgt)
    except OSError:
        pass
    return hits


def main():
    token, out = sys.argv[1], sys.argv[2]
    seconds = float(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else 900.0
    interval = float(sys.argv[sys.argv.index("--interval") + 1]) if "--interval" in sys.argv else 1.0
    nodes = nvidia_nodes()
    t0, seen, empty_since, scans = time.time(), set(), None, 0
    while time.time() - t0 < seconds:
        pids = fenced(token)
        seen.update(pids)
        scans += 1
        bad = {}
        for p in pids:
            h = touches(p, nodes)
            if h:
                try:
                    cmd = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")[:200]
                except OSError:
                    cmd = "?"
                bad[p] = {"cmd": cmd, "hits": h}
        if bad:
            for p in pids:
                try:
                    os.kill(p, signal.SIGKILL)
                except OSError:
                    pass
            rep = {"violation": True, "t": time.time(), "after_s": round(time.time() - t0, 1), "offenders": bad,
                   "killed": pids, "nodes": sorted(nodes)}
            json.dump(rep, open(out, "w"), indent=1)
            print(f"[nvidia_watch] VIOLATION: {json.dumps(bad)[:400]} -> killed {len(pids)} processes", flush=True)
            return 3
        if pids:
            empty_since = None
        elif seen:
            empty_since = empty_since or time.time()
            if time.time() - empty_since > 10:
                break
        time.sleep(interval)
    rep = {"violation": False, "scans": scans, "processes_checked": len(seen), "seconds": round(time.time() - t0, 1),
           "nodes": sorted(nodes), "libs": LIBS}
    json.dump(rep, open(out, "w"), indent=1)
    print(f"[nvidia_watch] clean: {len(seen)} processes, {scans} scans", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
