"""Machine telemetry next to a rehearsal / soak run (Linux + NVIDIA, standard library only): every INTERVAL s one CSV row
with GPU memory (whole card and our processes), temperature, power, utilisation, SM clock, the CPU package temperature
(k10temp Tctl / coretemp), the 1-minute load and, if a cgroup is given, its CPU use and throttling counters.

    python3 scripts/ops/telemetry.py OUT.csv [--interval 2] [--gpu-every 5] [--cgroup /sys/fs/cgroup/.../x.scope]
                                              [--match <bundle dir>] [--seconds 660]
    python3 scripts/ops/telemetry.py --summary OUT.csv [--dmon gpu_dmon.log]

GPU numbers come from ONE long-running `nvidia-smi --query-gpu ... -lms` process (and the per-process memory from a
query every 30 s), not two nvidia-smi calls per row, to keep driver queries rare while CUDA work runs. --gpu-every 0 =
no GPU query at all (CPU-only runs on a machine whose GPU must stay untouched) and the default: the GPU
is logged by a separate `nvidia-smi dmon -s pucm -d 5 -o DT` (scripts/ops/soak.sh), summarised with --dmon.
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import threading
import time


class Gpu(threading.Thread):
    """The latest row of a single `nvidia-smi --query-gpu ... -lms` stream."""
    Q = "memory.used,memory.total,temperature.gpu,power.draw,utilization.gpu,clocks.sm"

    def __init__(self, every_s):
        super().__init__(daemon=True)
        self.last, self.every_s = [""] * 6, every_s
        self.proc = None

    def run(self):
        try:
            self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={self.Q}", "--format=csv,noheader,nounits",
                                          f"-lms={int(self.every_s * 1000)}"], stdout=subprocess.PIPE, text=True)
            for line in self.proc.stdout:
                self.last = [x.strip() for x in line.split(",")]
        except OSError:
            pass


def gpu_apps():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return {int(p): float(m) for p, m in (l.split(",") for l in out.strip().splitlines() if l.strip())}
    except Exception:
        return {}


def ours(match):
    pids = set()
    for d in glob.glob("/proc/[0-9]*"):
        try:
            cmd = open(f"{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if match in cmd:
            pids.add(int(d[6:]))
    return pids


def cpu_temp():
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            name = open(f"{h}/name").read().strip()
        except OSError:
            continue
        if name in ("k10temp", "coretemp"):
            try:
                return round(int(open(f"{h}/temp1_input").read()) / 1000, 1)
            except OSError:
                pass
    return ""


def cg_stat(cg):
    if not cg:
        return "", "", ""
    try:
        d = dict(l.split() for l in open(os.path.join(cg, "cpu.stat")))
        return d.get("usage_usec", ""), d.get("nr_throttled", ""), d.get("throttled_usec", "")
    except OSError:
        return "", "", ""


def dmon_summary(path):
    """`nvidia-smi dmon -o DT` log -> median / max per numeric column (pwr, gtemp, sm, mem, fb, ...) + first / last time."""
    names, rows = None, []
    for line in open(path, errors="replace"):
        if line.startswith("#"):
            if names is None:
                names = line.lstrip("#").split()
            continue
        v = line.split()
        if names and len(v) == len(names):
            rows.append(dict(zip(names, v)))
    out = {"samples": len(rows)}
    if rows:
        out["first"] = " ".join(rows[0].get(k, "") for k in ("Date", "Time")).strip()
        out["last"] = " ".join(rows[-1].get(k, "") for k in ("Date", "Time")).strip()
    for k in (names or []):
        try:
            vals = sorted(float(r[k]) for r in rows if r[k] not in ("-", ""))
        except ValueError:
            continue
        if vals and k not in ("gpu", "Date", "Time"):
            out[k] = {"median": vals[len(vals) // 2], "max": vals[-1]}
    return out


def summary(path):
    rows = list(csv.DictReader(open(path)))

    def col(k):
        return [float(r[k]) for r in rows if r.get(k) not in ("", None)]

    out = {"samples": len(rows), "minutes": round((float(rows[-1]["t"]) - float(rows[0]["t"])) / 60, 1) if rows else 0}
    for k in ("gpu_mem_ours_mb", "gpu_mem_used_mb", "gpu_temp_c", "gpu_power_w", "gpu_util_pct", "gpu_sm_mhz",
              "cpu_temp_c", "load1"):
        v = sorted(col(k))
        if v:
            out[k] = {"median": v[len(v) // 2], "max": v[-1]}
    th = col("throttled_usec")
    if len(th) > 1:
        out["throttled_s"] = round((th[-1] - th[0]) / 1e6, 1)
    us, ts = col("cg_usage_usec"), [float(r["t"]) for r in rows if r.get("cg_usage_usec") not in ("", None)]
    if len(us) > 10:   # cores used by the whole scope: mean, and the busiest 1-minute stretch
        out["cores_mean"] = round((us[-1] - us[0]) / 1e6 / (ts[-1] - ts[0]), 2)
        k = max(1, int(round(60 / max(1e-6, (ts[-1] - ts[0]) / (len(ts) - 1)))))
        out["cores_max_1min"] = round(max((us[i + k] - us[i]) / 1e6 / (ts[i + k] - ts[i])
                                          for i in range(len(us) - k)), 2) if len(us) > k else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--gpu-every", type=float, default=0.0, help="seconds between GPU samples; 0 (default) = never touch "
                    "the GPU (GPU runs log it with nvidia-smi dmon instead)")
    ap.add_argument("--dmon", default="", help="with --summary: also summarise this nvidia-smi dmon log")
    ap.add_argument("--cgroup", default="")
    ap.add_argument("--match", default=os.getcwd())
    ap.add_argument("--seconds", type=float, default=3600)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        import json
        out = summary(a.out)
        if a.dmon and os.path.exists(a.dmon):
            out["gpu_dmon"] = dmon_summary(a.dmon)
        print(json.dumps(out, indent=1))
        return 0
    f = open(a.out, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_temp_c", "gpu_power_w", "gpu_util_pct", "gpu_sm_mhz",
                "gpu_mem_ours_mb", "cpu_temp_c", "load1", "cg_usage_usec", "nr_throttled", "throttled_usec"])
    g = Gpu(a.gpu_every) if a.gpu_every > 0 else None
    if g:
        g.start()
    apps, t_apps = {}, 0.0
    t_end = time.time() + a.seconds
    try:
        while time.time() < t_end:
            t = time.time()
            if g and t - t_apps >= 30:
                apps, t_apps = gpu_apps(), t
            mine = ours(a.match) if apps else set()
            w.writerow([round(t, 1), *(g.last if g else [""] * 6),
                        round(sum(m for p, m in apps.items() if p in mine), 0) if g else "", cpu_temp(),
                        os.getloadavg()[0], *cg_stat(a.cgroup)])
            f.flush()
            time.sleep(max(0.0, a.interval - (time.time() - t)))
    finally:
        if g and g.proc:
            g.proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
