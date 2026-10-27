"""Fetch BONES-SEED (or any HF file) straight onto the GPU server without the HF token ever leaving the Mac.
BONES-SEED is under the BONES-SEED license (https://bones.studio/info/seed-license) and is never committed here.

    SONIC_K1_HOST=<ssh host alias> .venv/bin/python -I sonic_k1_train/box/bones_fetch.py \
        --files metadata/seed_metadata_v004.parquet g1.tar.gz soma_uniform.tar.gz    # -> $SONIC_K1_WORK/data/bones_seed

Per round, the Mac asks the HF resolve endpoint (with the locally cached token, in this process only) for the file's
redirect target - a short-lived pre-signed CDN URL that needs no token - and hands it to the server over ssh **stdin**.
The server (remote helper below, run with the system python3) fetches the missing 1 GB byte ranges with --par parallel
curl processes, each reading the URL from its own stdin (`curl --config -`), so the URL never appears in a command
line, a process list, a file or a log. Partial chunks resume. When a URL expires mid-round the failed chunks are
retried in the next round with a fresh URL. At the end the server concatenates the chunks and checks the LFS sha256 from
the HF API. Printed: sizes, chunk counts, URL lifetime in seconds, throughput - never the token or the URL.

Requires access to the gated repo (the user accepts the BONES-SEED license on huggingface.co themselves).
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REMOTE = r'''
import concurrent.futures as cf, hashlib, json, os, subprocess, sys
dest, name, size, chunk, par, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
parts = os.path.join(dest, ".parts", name.replace("/", "__"))
os.makedirs(parts, exist_ok=True)
n = (size + chunk - 1) // chunk
def span(k):
    a = k * chunk
    return a, min(size, a + chunk) - 1
def have(k):
    p = os.path.join(parts, "%05d" % k)
    return os.path.getsize(p) if os.path.exists(p) else 0
if mode == "status":
    final = os.path.join(dest, name)
    print(json.dumps({"done": os.path.exists(final) and os.path.getsize(final) == size,
                      "missing": [k for k in range(n) if have(k) != span(k)[1] - span(k)[0] + 1]}))
    sys.exit(0)
if mode == "fetch":
    url = sys.stdin.readline().strip()
    ks = json.loads(sys.argv[7])
    cfg = 'url = "%s"\n' % url.replace("\\", "\\\\").replace('"', '\\"')
    def get(k):
        a, b = span(k)
        h = have(k)
        if h == b - a + 1:
            return k, 0
        if h > b - a + 1:
            os.remove(os.path.join(parts, "%05d" % k)); h = 0
        with open(os.path.join(parts, "%05d" % k), "ab") as f:
            r = subprocess.run(["curl", "-sS", "--fail", "--retry", "2", "--connect-timeout", "20",
                                "-r", "%d-%d" % (a + h, b), "--config", "-"], input=cfg.encode(), stdout=f,
                               stderr=subprocess.DEVNULL)
        return k, r.returncode
    with cf.ThreadPoolExecutor(par) as ex:
        res = list(ex.map(get, ks))
    print(json.dumps({"rc": {str(k): rc for k, rc in res}}))
    sys.exit(0)
if mode == "join":
    final = os.path.join(dest, name)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    hsh = hashlib.sha256()
    with open(final + ".tmp", "wb") as out:
        for k in range(n):
            p = os.path.join(parts, "%05d" % k)
            with open(p, "rb") as f:
                while True:
                    b = f.read(1 << 24)
                    if not b:
                        break
                    hsh.update(b); out.write(b)
    ok = hsh.hexdigest() == sys.argv[7]
    if ok:
        os.replace(final + ".tmp", final)
        for k in range(n):
            os.remove(os.path.join(parts, "%05d" % k))
        os.rmdir(parts)
    print(json.dumps({"sha256_ok": ok, "size": os.path.getsize(final if ok else final + ".tmp")}))
'''


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def signed_url(repo: str, filename: str, repo_type: str):
    """Resolve -> pre-signed CDN URL (no token needed to use it) and its lifetime in s (None if unknown)."""
    from huggingface_hub import get_token, hf_hub_url

    url = hf_hub_url(repo, filename, repo_type=repo_type)
    token = get_token()
    opener = urllib.request.build_opener(_NoRedirect)
    for _ in range(5):
        req = urllib.request.Request(url, method="HEAD")
        if token and urllib.parse.urlparse(url).hostname.endswith("huggingface.co"):
            req.add_header("Authorization", f"Bearer {token}")
        try:
            resp = opener.open(req, timeout=30)
            if urllib.parse.urlparse(url).hostname.endswith("huggingface.co"):
                raise RuntimeError(f"{filename}: no redirect from the resolve endpoint (HTTP {resp.status})")
            break
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                url = urllib.parse.urljoin(url, e.headers["Location"])
                if not urllib.parse.urlparse(url).hostname.endswith("huggingface.co"):
                    break
                continue
            raise RuntimeError(f"{filename}: HTTP {e.code} from the resolve endpoint (license accepted?)") from None
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    life = None
    if "Expires" in q:
        life = int(q["Expires"][0]) - int(time.time())
    elif "X-Amz-Expires" in q:
        life = int(q["X-Amz-Expires"][0])
    return url, life


def file_info(repo: str, filename: str, repo_type: str):
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo, repo_type=repo_type, files_metadata=True)
    for s in info.siblings:
        if s.rfilename == filename:
            return s.size, (s.lfs.sha256 if s.lfs else None)
    raise FileNotFoundError(filename)


def remote(host: str, args: list, stdin: str = "") -> str:
    cmd = ["ssh", "-o", "ServerAliveInterval=30", host,
           "nice -n 10 python3 -c " + shlex.quote(REMOTE) + " " + " ".join(shlex.quote(str(a)) for a in args)]
    r = subprocess.run(cmd, input=stdin.encode(), capture_output=True, timeout=None)
    if r.returncode != 0:
        raise RuntimeError(f"remote {args[5]} failed: {r.stderr.decode()[-400:]}")
    return r.stdout.decode().strip().splitlines()[-1]


def fetch(host, dest, repo, repo_type, filename, chunk, par, max_rounds=200):
    import json

    size, sha = file_info(repo, filename, repo_type)
    n = (size + chunk - 1) // chunk
    print(f"[fetch] {filename}: {size / 1e9:.2f} GB, {n} chunk(s), sha256 {sha[:12] if sha else '-'}", flush=True)
    t0 = time.time()
    for rnd in range(max_rounds):
        st = json.loads(remote(host, [dest, filename, size, chunk, par, "status"]))
        if st["done"]:
            print(f"[fetch] {filename}: already complete", flush=True)
            return True
        missing = st["missing"]
        if not missing:
            break
        url, life = signed_url(repo, filename, repo_type)
        # a URL lives `life` s: hand out only as many chunks as fit comfortably (12 MB/s per curl measured)
        take = missing if life is None else missing[:max(par, int(max(life - 120, 60) * 12e6 * par / chunk))]
        t1 = time.time()
        res = json.loads(remote(host, [dest, filename, size, chunk, par, "fetch", json.dumps(take)], stdin=url + "\n"))
        del url
        bad = [k for k, rc in res["rc"].items() if rc != 0]
        got = (len(take) - len(bad)) * chunk
        print(f"[fetch] {filename}: round {rnd + 1}, URL lifetime {life} s, {len(take) - len(bad)}/{len(take)} chunks ok"
              f" ({got / max(time.time() - t1, 1e-3) / 1e6:.0f} MB/s), {len(missing) - len(take) + len(bad)} left",
              flush=True)
        if bad and len(bad) == len(take):
            time.sleep(10)
    out = json.loads(remote(host, [dest, filename, size, chunk, par, "join", sha or ""]))
    print(f"[fetch] {filename}: joined, sha256 ok = {out['sha256_ok']}, {out['size'] / 1e9:.2f} GB, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return bool(out["sha256_ok"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True)
    # on the server; a relative path is under the server's home (SONIC_K1_WORK default sonic_k1 = $HOME/sonic_k1 there)
    ap.add_argument("--dest", default=os.environ.get("SONIC_K1_WORK", "sonic_k1") + "/data/bones_seed")
    ap.add_argument("--host", default=os.environ.get("SONIC_K1_HOST"), required=not os.environ.get("SONIC_K1_HOST"),
                    help="ssh host alias of the GPU server (default $SONIC_K1_HOST)")
    ap.add_argument("--repo", default="bones-studio/seed")
    ap.add_argument("--repo-type", default="dataset")
    ap.add_argument("--chunk-mb", type=int, default=1024)
    ap.add_argument("--par", type=int, default=8)
    a = ap.parse_args()
    ok = all(fetch(a.host, a.dest, a.repo, a.repo_type, f, a.chunk_mb << 20, a.par) for f in a.files)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
