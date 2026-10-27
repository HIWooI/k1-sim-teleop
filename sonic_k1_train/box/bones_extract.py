"""Extract only the selected BONES-SEED takes from the big archives (runs on the GPU server with the system python3,
stdlib only; streams the .tar.gz once, nothing else is unpacked). BONES-SEED is under the BONES-SEED license
(https://bones.studio/info/seed-license) and is never committed to this repository.

    nice -n 10 python3 bones_extract.py --archive $SONIC_K1_WORK/data/bones_seed/soma_uniform.tar.gz \
        --selection selection.json --field bvh --dest $SONIC_K1_WORK/data/bones_seed/subset
    python3 bones_extract.py --archive <...>.tar.gz --list 20      # first member names (layout check)

A member is taken when its path ends with a selected relative path (metadata move_*_path, e.g.
g1/csv/210531/<name>.csv) or with its last two components (<date>/<name>.ext), so an extra top-level folder in the
archive does not matter. Only regular files are written, under --dest/<selected relative path> (no absolute paths,
no '..'): the archive is untrusted input.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time


def safe_rel(p: str) -> str:
    parts = [x for x in p.replace("\\", "/").split("/") if x not in ("", ".")]
    if any(x == ".." for x in parts):
        raise ValueError(f"unsafe path {p!r}")
    return "/".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--selection", default="")
    ap.add_argument("--field", choices=["g1", "bvh"], default="g1")
    ap.add_argument("--dest", default="")
    ap.add_argument("--list", type=int, default=0)
    a = ap.parse_args()
    if a.list:
        with tarfile.open(a.archive, "r|gz") as t:
            for i, m in enumerate(t):
                print(m.name, m.size)
                if i + 1 >= a.list:
                    break
        return 0
    clips = json.load(open(a.selection))["clips"]
    want = {}
    for c in clips:
        rel = safe_rel(c[a.field])
        want[rel] = rel
        want["/".join(rel.split("/")[-2:])] = rel
    todo = set(safe_rel(c[a.field]) for c in clips)
    got, t0, seen = set(), time.time(), 0
    with tarfile.open(a.archive, "r|gz") as t:
        for m in t:
            seen += 1
            if not m.isfile():
                continue
            name = safe_rel(m.name)
            rel = want.get(name) or want.get("/".join(name.split("/")[-2:]))
            if rel is None or rel in got:
                continue
            out = os.path.join(a.dest, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            src = t.extractfile(m)
            with open(out + ".tmp", "wb") as f:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            os.replace(out + ".tmp", out)
            got.add(rel)
            if len(got) % 500 == 0:
                print(f"[extract] {len(got)}/{len(todo)} after {seen} members, {time.time() - t0:.0f} s", flush=True)
            if got >= todo:
                break
    missing = sorted(todo - got)
    print(json.dumps({"extracted": len(got), "wanted": len(todo), "missing": len(missing), "missing_first": missing[:10],
                      "seconds": round(time.time() - t0)}))
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
