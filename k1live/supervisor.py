"""Child-process supervisor for the K1 live service (sender, sim_server).

Each child runs in its own process group with stdout/stderr appended to a log file. The supervisor restarts a child
when it exits or when its status topic goes quiet (the service calls `beat(name)` for every status message):

* start-up grace: no status yet -> wait up to `grace_s` (model loading) before calling it stale
* after the first status: stale after `stale_s` without one (3 s by default)
* restart back-off 1, 2, 4, 8, 15 s; the failure count resets once a child has stayed healthy for `stable_s`

Pure standard library (subprocess + `ps` for memory), so it also runs where psutil is not installed.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

BACKOFF = (1.0, 2.0, 4.0, 8.0, 15.0)


@dataclass
class ProcSpec:
    name: str
    argv: list
    log_path: Path
    env: dict = field(default_factory=dict)
    cwd: Optional[str] = None
    grace_s: float = 30.0      # first status must arrive within this (model loading)
    stale_s: float = 3.0       # then: restart when no status for this long
    stable_s: float = 60.0     # healthy this long -> back-off counter resets
    backoff: tuple = BACKOFF


class Proc:
    def __init__(self, spec: ProcSpec, on_event: Callable[[str, str, dict], None]):
        self.spec = spec
        self.on_event = on_event
        self.popen: Optional[subprocess.Popen] = None
        self.state = "stopped"          # stopped | running | backoff | disabled
        self.started_at = 0.0
        self.last_beat: Optional[float] = None
        self.first_beat_at: Optional[float] = None
        self.restarts = 0               # restarts after a failure (crash / stale), not manual ones
        self.fails_in_row = 0
        self.next_start = 0.0
        self.last_exit: Optional[int] = None
        self.last_reason = ""
        self.rss_mb: Optional[float] = None
        self.argv_override: Optional[list] = None
        self._log = None

    # ------------------------------------------------------------------------------------------------
    @property
    def pid(self):
        return self.popen.pid if self.popen is not None and self.popen.poll() is None else None

    def _open_log(self):
        self.spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._log is None or self._log.closed:
            self._log = open(self.spec.log_path, "ab", buffering=0)
        return self._log

    def start(self, now=None):
        now = time.time() if now is None else now
        argv = self.argv_override or self.spec.argv
        log = self._open_log()
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} start: {' '.join(map(str, argv))}\n".encode())
        env = dict(os.environ)
        env.update({"PYTHONUNBUFFERED": "1"})
        env.update({k: str(v) for k, v in self.spec.env.items()})
        try:
            self.popen = subprocess.Popen([str(a) for a in argv], stdout=log, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, cwd=self.spec.cwd, env=env,
                                          start_new_session=True)
        except OSError as e:  # missing interpreter etc.: treat like a crash, back off
            self.popen = None
            self._failed(now, f"실행 실패: {e}")
            return
        self.state = "running"
        self.started_at = now
        self.last_beat = None
        self.first_beat_at = None
        self.on_event(self.spec.name, "start", {"pid": self.popen.pid})

    def _kill(self, timeout=3.0):
        p = self.popen
        if p is None:
            return None
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
                os.killpg(p.pid, signal.SIGCONT)  # a frozen (stopped) process only sees SIGTERM once resumed
            except (ProcessLookupError, PermissionError):
                pass
            try:
                p.wait(timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                p.wait(2.0)
        else:  # leader gone: make sure no grandchildren of its group linger
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.popen = None
        return p.returncode

    def stop(self, reason="stop", disable=True):
        code = self._kill()
        self.state = "disabled" if disable else "stopped"
        self.on_event(self.spec.name, "stop", {"reason": reason, "code": code})

    def restart(self, reason="manual", now=None):
        """Operator restart: immediate, does not count as a failure."""
        now = time.time() if now is None else now
        self._kill()
        self.on_event(self.spec.name, "restart", {"reason": reason})
        self.fails_in_row = 0
        self.start(now)

    def beat(self, now=None):
        now = time.time() if now is None else now
        self.last_beat = now
        if self.first_beat_at is None:
            self.first_beat_at = now
            self.on_event(self.spec.name, "healthy", {"after_s": round(now - self.started_at, 2)})

    def _failed(self, now, reason):
        self.restarts += 1
        self.fails_in_row += 1
        delay = self.spec.backoff[min(self.fails_in_row - 1, len(self.spec.backoff) - 1)]
        self.next_start = now + delay
        self.state = "backoff"
        self.last_reason = reason
        self.on_event(self.spec.name, "fail", {"reason": reason, "retry_in_s": delay, "restarts": self.restarts})

    def poll(self, now=None):
        now = time.time() if now is None else now
        if self.state == "backoff":
            if now >= self.next_start:
                self.start(now)
            return
        if self.state != "running":
            return
        code = self.popen.poll() if self.popen is not None else -1
        if code is not None:
            self.last_exit = code
            if self.popen is not None:  # the leader is gone: nothing it spawned (e.g. a renderer) may linger
                try:
                    os.killpg(self.popen.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            self.popen = None
            self._failed(now, f"종료됨 (exit {code})")
            return
        if self.last_beat is None:
            if now - self.started_at > self.spec.grace_s:
                self.last_exit = self._kill(timeout=1.0)
                self._failed(now, f"시작 후 {self.spec.grace_s:.0f} s 동안 상태 없음")
            return
        if now - self.last_beat > self.spec.stale_s:
            self.last_exit = self._kill(timeout=1.0)  # hung: do not wait long for a polite exit
            self._failed(now, f"상태 {now - self.last_beat:.1f} s 끊김")
            return
        if self.fails_in_row and now - self.first_beat_at > self.spec.stable_s:
            self.fails_in_row = 0

    def status(self, now=None):
        now = time.time() if now is None else now
        return {
            "name": self.spec.name, "state": self.state, "pid": self.pid,
            "uptime_s": round(now - self.started_at, 1) if self.state == "running" else None,
            "beat_age_s": round(now - self.last_beat, 2) if self.last_beat else None,
            "healthy": self.state == "running" and self.last_beat is not None
            and now - self.last_beat <= self.spec.stale_s,
            "restarts": self.restarts, "last_exit": self.last_exit, "last_reason": self.last_reason,
            "retry_in_s": round(max(0.0, self.next_start - now), 1) if self.state == "backoff" else None,
            "rss_mb": self.rss_mb,
        }


def rss_mb(pids):
    """Resident memory (MB) per pid via `ps` (macOS and Linux)."""
    pids = [p for p in pids if p]
    if not pids:
        return {}
    try:
        out = subprocess.run(["ps", "-o", "pid=,rss=", "-p", ",".join(map(str, pids))],
                             capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    res = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            res[int(parts[0])] = int(parts[1]) / 1024.0
    return res


def group_rss_mb(pgids):
    """Resident memory (MB) summed over each process group (a child and everything it spawned, e.g. a renderer)."""
    want = {p for p in pgids if p}
    if not want:
        return {}
    try:
        out = subprocess.run(["ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    res = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and int(parts[0]) in want:
            res[int(parts[0])] = res.get(int(parts[0]), 0.0) + int(parts[1]) / 1024.0
    return res


class Supervisor:
    def __init__(self, specs, on_event=None, poll_s=0.2, mem_every_s=5.0):
        self.on_event = on_event or (lambda name, ev, d: None)
        self.procs = {s.name: Proc(s, self._event) for s in specs}
        self.poll_s = poll_s
        self.mem_every_s = mem_every_s
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_mem = 0.0

    def _event(self, name, ev, d):
        try:
            self.on_event(name, ev, d)
        except Exception as e:  # never let a logging hook kill the supervisor
            print(f"[supervisor] event hook error: {e}", flush=True)

    def start(self, names=None):
        with self._lock:
            for n, p in self.procs.items():
                if names is None or n in names:
                    p.start()
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="supervisor", daemon=True)
            self._thread.start()

    def _loop(self):
        while not self._stop.wait(self.poll_s):
            now = time.time()
            with self._lock:
                for p in self.procs.values():
                    try:
                        p.poll(now)
                    except Exception as e:
                        print(f"[supervisor] poll {p.spec.name}: {e}", flush=True)
                if now - self._last_mem >= self.mem_every_s:
                    self._last_mem = now
                    mem = group_rss_mb([p.pid for p in self.procs.values()])  # pid == pgid (own session)
                    for p in self.procs.values():
                        p.rss_mb = round(mem[p.pid], 1) if p.pid in mem else None

    def beat(self, name, now=None):
        with self._lock:
            p = self.procs.get(name)
            if p is not None and p.state == "running":
                p.beat(now)

    def restart(self, name, reason="manual", argv=None):
        with self._lock:
            p = self.procs[name]
            if argv is not None:
                p.argv_override = list(argv)
            p.restart(reason)

    def set_argv(self, name, argv):
        """Used by the next (re)start, e.g. a new input source baked into the sender's command line."""
        with self._lock:
            self.procs[name].argv_override = list(argv)

    def status(self):
        now = time.time()
        with self._lock:
            return {n: p.status(now) for n, p in self.procs.items()}

    def stop_watching(self):
        """No more restarts from here on (children may now exit on their own)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)

    def shutdown(self):
        self.stop_watching()
        with self._lock:
            for p in self.procs.values():
                if p.state in ("running", "backoff"):
                    p.stop("shutdown")
                if p._log is not None:
                    p._log.close()
