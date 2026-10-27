"""Offline rehearsal guard: every Python process started with this folder on PYTHONPATH (children inherit it) refuses
network connections and name lookups to anything but this machine, and logs each attempt.

    PYTHONPATH=$PWD/scripts/offline_guard K1_OFFLINE_LOG=$PWD/logs/offline_attempts.log bash K1Demo.command

Used where a real network cut is not possible (no `unshare -n` without root on Ubuntu 24.04: AppArmor restricts
unprivileged user namespaces; pulling the cable would also drop the SSH session). It covers Python sockets and
getaddrinfo (huggingface_hub, urllib, requests, torch.hub, pip) — not native code that opens its own sockets (libzmq
is loopback-only here, Chrome is not under test). The real check stays: unplug / Wi-Fi off at the venue rehearsal.
"""
import errno
import os
import socket
import sys
import time
import traceback

_LOG = os.environ.get("K1_OFFLINE_LOG", "")
_LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback", socket.gethostname()}


def _local(host):
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode(errors="replace")
    host = str(host).strip("[]").lower()
    return (host in _LOCAL_NAMES or host == "" or host.startswith("127.") or host in ("::1", "0.0.0.0", "::")
            or host.startswith("::ffff:127."))


def _note(what, target):
    if not _LOG:
        return
    try:
        stack = "".join(traceback.format_stack(limit=40)[:-2])
        with open(_LOG, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} pid={os.getpid()} {' '.join(sys.argv[:3])[:160]} "
                    f"{what} {target!r}\n{stack}\n")
    except OSError:
        pass


_connect, _connect_ex, _getaddrinfo = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo


def _guard_connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and not _local(address[0]):
        _note("connect", address)
        raise OSError(errno.ENETUNREACH, f"offline guard: no network ({address[0]})")
    return _connect(self, address)


def _guard_connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and not _local(address[0]):
        _note("connect_ex", address)
        return errno.ENETUNREACH
    return _connect_ex(self, address)


def _guard_getaddrinfo(host, *args, **kw):
    if not _local(host):
        _note("getaddrinfo", host)
        raise socket.gaierror(socket.EAI_NONAME, f"offline guard: no DNS ({host})")
    return _getaddrinfo(host, *args, **kw)


def _guard_name(fn):
    def wrapped(host, *args, **kw):
        if not _local(host):
            _note(fn.__name__, host)
            raise socket.gaierror(socket.EAI_NONAME, f"offline guard: no DNS ({host})")
        return fn(host, *args, **kw)
    return wrapped


socket.socket.connect = _guard_connect
socket.socket.connect_ex = _guard_connect_ex
socket.getaddrinfo = _guard_getaddrinfo
socket.gethostbyname = _guard_name(socket.gethostbyname)
socket.gethostbyname_ex = _guard_name(socket.gethostbyname_ex)
