#!/usr/bin/env python3
# Portions derived from NVIDIA GR00T-WholeBodyControl (GEAR-SONIC), Copyright (c) NVIDIA CORPORATION & AFFILIATES,
# licensed under the Apache License 2.0 (https://github.com/NVlabs/GR00T-WholeBodyControl). Modified for the K1 port.
"""ZMQ Protocol v3 compatibility check: our MediaPipe sender vs the official SONIC C++ deploy.

The C++ deploy (`deploy.sh --input-type zmq`) consumes `[topic][1280-byte JSON header][binary fields]`
messages. This tool decodes messages with a line-by-line Python port of the official parser:

  P1  ZMQPackedMessageSubscriber::PollOnce / DecodeHeaderJSON / FieldInfo::ComputeByteSize
        gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_packed_message_subscriber.hpp
  P2  ZMQEndpointInterface::DecodeIntoMotionSequence (field lookup, v1/v2/v3 required fields, shapes,
      frame counts, dtype decode, frame_index, timestamp_monotonic, catch_up)
        .../input_interface/zmq_endpoint_interface.hpp
  P3  StreamedMotionMerger::ValidateIncomingData + the cross-message frame_index rules of CalculateSlidingWindow
        .../input_interface/streamed_motion_merger.hpp

and, independently, compares bytes against the official Python packer
(gear_sonic/utils/teleop/zmq/zmq_planner_sender.py:pack_pose_message, which webcam_stream.py uses).
linux/compat/zmq_v3_check.cpp runs the *real* C++ header on the same bytes (replay shim on macOS, live on Linux).

  selftest   synthetic messages: our sonic.stream.v3_message vs official pack_pose_message (byte diff),
             decode with P1-P3, negative cases (the port must reject what C++ rejects). Needs numpy only.
  e2e        start scripts/webcam_sender.py on a video, subscribe like the deploy, validate every message.
  listen     subscribe to a running sender (e.g. the Mac) like the deploy does: --host <sender-ip>.
  check      validate a capture file written by --save (u32-LE length + message, repeated).

Exit code 0 = compatible, 1 = incompatibility found.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
HEADER_SIZE = 1280                                   # ZMQPackedMessageSubscriber::HEADER_SIZE
ELEM = {"f64": 8, "i64": 8, "f32": 4, "i32": 4, "i16": 2, "f16": 2, "i8": 1, "u8": 1, "bool": 1}  # GetElementSize, default 4
NP = {"f32": "<f4", "f64": "<f8", "i32": "<i4", "i64": "<i8", "bool": "u1", "u8": "u1"}
MAX_GAP_FRAMES, HISTORY_FRAMES = 200, 5              # StreamedMotionMerger
STREAMING_DATA_ABSENT_THRESHOLD_S = 0.150            # g1_deploy_onnx_ref.cpp


class Reject(Exception):
    """The C++ code would drop this message (returns false / empty result)."""


# ----------------------------------------------------------------------------- P1
def poll_once(msg: bytes, topic: str = "pose"):
    t = topic.encode()
    if t:
        if len(msg) < len(t):
            raise Reject("packed frame shorter than topic")
        if msg[: len(t)] != t:
            raise Reject("topic mismatch")
        msg = msg[len(t):]
    if len(msg) < HEADER_SIZE:
        raise Reject(f"Packed frame too small: {len(msg)} < {HEADER_SIZE}")
    raw = msg[:HEADER_SIZE]
    nul = raw.find(b"\x00")
    json_len = HEADER_SIZE if nul < 0 else nul        # strnlen(.., HEADER_SIZE)
    try:
        j = json.loads(raw[:json_len].decode("utf-8"))
        hdr = {"version": 0, "endian": "", "count": -1, "fields": []}
        if "v" in j:
            if not isinstance(j["v"], int) or isinstance(j["v"], bool):
                raise TypeError("v")                   # j["v"].get<int>() throws -> parse failure
            hdr["version"] = j["v"]
        if "endian" in j:
            hdr["endian"] = str(j["endian"]) if isinstance(j["endian"], str) else _throw("endian")
        if "count" in j:
            hdr["count"] = j["count"] if isinstance(j["count"], int) else _throw("count")
        for f in j.get("fields", []) if isinstance(j.get("fields"), list) else []:
            shape = f.get("shape", []) if isinstance(f.get("shape"), list) else []
            if any((not isinstance(d, int)) or isinstance(d, bool) or d < 0 for d in shape):
                raise TypeError("shape")               # dim.get<size_t>() throws on negatives / non-ints
            hdr["fields"].append({"name": f.get("name", ""), "dtype": f.get("dtype", ""),
                                  "shape": list(shape), "optional": bool(f.get("optional", False))})
    except (ValueError, TypeError, UnicodeDecodeError) as e:
        raise Reject(f"JSON parse failed ({e})") from None
    data = msg[HEADER_SIZE:]
    bufs, off = [], 0
    for f in hdr["fields"]:
        n = 0 if not f["shape"] else int(np.prod(f["shape"])) * ELEM.get(f["dtype"], 4)   # ComputeByteSize
        if off + n > len(data):
            raise Reject(f"Field {f['name']} exceeds data bounds")
        bufs.append(data[off: off + n])
        off += n
    if hdr["endian"] not in ("", "le"):
        raise Reject("big-endian payload (C++ would byte-swap; our sender never does this)")
    return hdr, bufs, {"header_bytes": json_len, "data_bytes": off, "trailing_bytes": len(data) - off}


def _throw(what):
    raise TypeError(what)


# ----------------------------------------------------------------------------- P2 + P3 (single message)
def decode_motion(hdr, bufs):
    v = hdr["version"]
    idx = {}
    for i, f in enumerate(hdr["fields"]):                  # last occurrence wins, aliases as in C++
        name = {"body_quat_w": "body_quat", "last_smpl_global_frames": "frame_index"}.get(f["name"], f["name"])
        idx[name] = i
    F = lambda k: hdr["fields"][idx[k]]                     # noqa: E731
    warns = []
    if v == 4:
        if "token_state" not in idx:
            raise Reject("Version 4 missing required field 'token_state'")
        return {"version": 4}, warns
    if "body_quat" not in idx:
        raise Reject("Missing required field 'body_quat' (or 'body_quat_w')")
    if "frame_index" not in idx:
        raise Reject("Missing required field 'frame_index' (or 'last_smpl_global_frames')")
    if v in (2, 3):
        for k in ("smpl_joints", "smpl_pose") + (("joint_pos", "joint_vel") if v == 3 else ()):
            if k not in idx:
                raise Reject(f"Version {v} missing required field '{k}'")
    elif v == 1:
        if "joint_pos" not in idx or "joint_vel" not in idx:
            raise Reject("Version 1 missing required fields (joint_pos, joint_vel)")
    else:
        raise Reject(f"Unsupported protocol version: {v}")

    if v in (2, 3):
        sj = F("smpl_joints")["shape"]
        if len(sj) < 2:
            raise Reject("Invalid smpl_joints shape")
        n = sj[0]
        if n <= 0:
            raise Reject(f"Invalid number of frames from smpl_joints: {n}")
        if v == 3:
            jp, jv = F("joint_pos")["shape"], F("joint_vel")["shape"]
            if len(jp) != 2:
                raise Reject("Version 3 has invalid joint_pos shape (expected [N, num_joints])")
            if len(jv) != 2:
                raise Reject("Version 3 has invalid joint_vel shape (expected [N, num_joints])")
            if jp[0] != n:
                raise Reject(f"Version 3 frame count mismatch between smpl_joints ({n}) and joint_pos ({jp[0]})")
            if jv[0] != n:
                raise Reject(f"Version 3 frame count mismatch between smpl_joints ({n}) and joint_vel ({jv[0]})")
    else:
        jp, jv = F("joint_pos")["shape"], F("joint_vel")["shape"]
        if len(jp) != 2 or len(jv) != 2:
            raise Reject("Invalid joint_pos/joint_vel shape")
        n = jp[0]
        if n != jv[0]:
            raise Reject("Frame count mismatch between joint_pos and joint_vel")
    if n <= 0:
        raise Reject(f"Invalid number of frames: {n}")

    def arr(k, per_frame_shape):
        """f32/f64 are decoded; any other dtype leaves the C++ buffers at their defaults (zeros / identity quat)."""
        f, b = F(k), bufs[idx[k]]
        if f["dtype"] not in ("f32", "f64"):
            warns.append(f"{k}: dtype {f['dtype']} is not decoded by C++ (values stay at defaults)")
            return None
        a = np.frombuffer(b, dtype=NP[f["dtype"]]).astype(np.float64)
        need = n * int(np.prod(per_frame_shape))
        if a.size < need:
            raise Reject(f"{k}: buffer has {a.size} values, C++ reads {need} (out-of-bounds memcpy)")
        return a[:need].reshape((n, *per_frame_shape))

    out = {"version": v, "num_frames": n}
    num_joints = 0
    if "joint_pos" in idx and "joint_vel" in idx and len(F("joint_pos")["shape"]) == 2 and len(F("joint_vel")["shape"]) == 2:
        num_joints = F("joint_pos")["shape"][1]
        if num_joints <= 0:
            raise Reject(f"Invalid number of joints: {num_joints}")
        out["joint_pos"], out["joint_vel"] = arr("joint_pos", (num_joints,)), arr("joint_vel", (num_joints,))
    qs = F("body_quat")["shape"]
    nb = qs[1] if len(qs) == 3 else 1
    out["body_quat"] = arr("body_quat", (nb, 4))
    if "smpl_joints" in idx:
        s = F("smpl_joints")["shape"]
        if len(s) in (2, 3):
            out["smpl_joints"] = arr("smpl_joints", (s[1] if len(s) == 3 else 1, 3))
        else:
            warns.append("smpl_joints: invalid shape dims, skipped by C++")
    if "smpl_pose" in idx:
        s = F("smpl_pose")["shape"]
        if len(s) in (2, 3):
            out["smpl_pose"] = arr("smpl_pose", (s[1] if len(s) == 3 else 1, 3))
    fi, fb = F("frame_index"), bufs[idx["frame_index"]]
    if fi["dtype"] in ("i32", "i64"):
        out["frame_indices"] = np.frombuffer(fb, dtype=NP[fi["dtype"]]).astype(np.int64)
    else:
        out["frame_indices"] = np.zeros(0, np.int64)
    if "timestamp_monotonic" in idx:
        tf, tb = F("timestamp_monotonic"), bufs[idx["timestamp_monotonic"]]
        out["timestamp_monotonic"] = struct.unpack("<d", tb[:8])[0] if tf["dtype"] == "f64" and len(tb) >= 8 else 0.0
        if tf["dtype"] != "f64":
            warns.append("timestamp_monotonic: only f64 is read by C++")
    out["catch_up"] = True
    if "catch_up" in idx:
        cf, cb = F("catch_up"), bufs[idx["catch_up"]]
        fmt = {"bool": "<B", "u8": "<B", "i32": "<i", "i64": "<q"}.get(cf["dtype"])
        if fmt and len(cb) >= struct.calcsize(fmt):
            out["catch_up"] = struct.unpack(fmt, cb[: struct.calcsize(fmt)])[0] != 0

    # P3 StreamedMotionMerger::ValidateIncomingData (operates on the decoded vectors)
    if len(out["frame_indices"]) == 0:
        raise Reject("[StreamedMotionMerger] Missing required fields (body_quat or frame_indices)")
    if v == 3:
        if out.get("smpl_joints") is None or out.get("smpl_pose") is None:
            raise Reject("[StreamedMotionMerger] Protocol v3 missing smpl_joints or smpl_pose")
        if out.get("joint_pos") is None or out.get("joint_vel") is None:
            raise Reject("[StreamedMotionMerger] Protocol v3 missing joint_pos or joint_vel")
    return out, warns


def semantic_checks(m):
    """Beyond the parser: what the G1 SONIC encoder (mode 2) expects from a v3 stream."""
    issues = []
    if m["version"] != 3:
        issues.append(f"version {m['version']} (SONIC SMPL teleop uses v3)")
        return issues
    if m["num_frames"] != 1:
        issues.append(f"{m['num_frames']} frames per message (webcam_stream.py / ours send 1)")
    if m.get("joint_pos") is not None and m["joint_pos"].shape[1] != 29:
        issues.append(f"joint_pos has {m['joint_pos'].shape[1]} joints, G1 motion buffer has 29")
    if m.get("smpl_joints") is not None and m["smpl_joints"].shape[1] != 24:
        issues.append(f"smpl_joints has {m['smpl_joints'].shape[1]} joints, encoder expects 24")
    if m.get("smpl_pose") is not None and m["smpl_pose"].shape[1] != 21:
        issues.append(f"smpl_pose has {m['smpl_pose'].shape[1]} poses, protocol says 21")
    q = m.get("body_quat")
    if q is not None:
        nrm = np.linalg.norm(q[:, 0], axis=-1)
        if np.any(np.abs(nrm - 1) > 1e-3):
            issues.append(f"body_quat not unit (|q|={nrm.min():.4f}..{nrm.max():.4f})")
    for k in ("smpl_joints", "body_quat", "joint_pos"):
        if m.get(k) is not None and not np.all(np.isfinite(m[k])):
            issues.append(f"{k} has NaN/inf")
    if m.get("smpl_joints") is not None:
        span = np.ptp(m["smpl_joints"][0], axis=0)
        if span.max() < 0.3 or span.max() > 3.0:
            issues.append(f"smpl_joints extent {span.round(2).tolist()} m looks off for a human skeleton")
    return issues


class StreamTracker:
    """Cross-message rules: constant protocol version, frame_index progression (merger catch-up / gaps)."""

    def __init__(self):
        self.version = None
        self.last_fi = None
        self.n = 0
        self.resets = 0
        self.gaps = 0
        self.t_recv, self.t_send = [], []
        self.errors = []

    def add(self, m, t_recv):
        self.n += 1
        if self.version is None:
            self.version = m["version"]
        elif m["version"] != self.version:
            self.errors.append(f"Protocol version changed from {self.version} to {m['version']} (deploy force-disables ZMQ mode)")
        fi = m["frame_indices"]
        if self.last_fi is not None:
            step = int(fi[0]) - self.last_fi
            if step <= 0:
                self.resets += 1      # incoming_frame_start <= window start -> catch-up reset (playback jumps)
            elif step > 1:
                self.gaps += 1
                if step > MAX_GAP_FRAMES + HISTORY_FRAMES:
                    self.resets += 1
        self.last_fi = int(fi[-1])
        self.t_recv.append(t_recv)
        if "timestamp_monotonic" in m:
            self.t_send.append(m["timestamp_monotonic"])

    def summary(self):
        s = {"messages": self.n, "protocol_version": self.version, "frame_index_resets": self.resets,
             "frame_index_gaps": self.gaps, "errors": self.errors}
        if len(self.t_recv) > 1:
            dt = np.diff(self.t_recv)
            s["rate_hz"] = round(1.0 / max(np.mean(dt), 1e-9), 2)
            s["recv_interval_ms_p50_p95_max"] = [round(1000 * float(np.percentile(dt, p)), 1) for p in (50, 95)] + [round(1000 * float(dt.max()), 1)]
            s["intervals_over_150ms"] = int((dt > STREAMING_DATA_ABSENT_THRESHOLD_S).sum())
        if self.t_send:
            s["has_timestamp_monotonic"] = True
        return s


# ----------------------------------------------------------------------------- official packer
def load_official_packer(sonic_root: Path):
    """Import pack_pose_message by file path (gear_sonic/__init__ would pull in torch etc.)."""
    p = sonic_root / "gear_sonic/utils/teleop/zmq/zmq_planner_sender.py"
    spec = importlib.util.spec_from_file_location("_official_zmq_planner_sender", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.pack_pose_message, p


def official_v3_fields(smpl_joints, body_quat, frame_index, wrists=None):
    """Exactly SonicV3Publisher.publish() in gear_sonic/examples/live_camera_teleop/soma_to_smpl.py."""
    joint_pos = np.zeros((1, 29), dtype=np.float32)
    joint_pos[:, 23:29] = np.zeros((1, 6), np.float32) if wrists is None else wrists
    return {
        "smpl_joints": np.asarray(smpl_joints, np.float32).reshape(1, 24, 3),
        "smpl_pose": np.zeros((1, 21, 3), dtype=np.float32),
        "joint_pos": joint_pos,
        "joint_vel": np.zeros((1, 29), dtype=np.float32),
        "body_quat": np.asarray(body_quat, np.float32).reshape(1, 4),
        "frame_index": np.array([frame_index], dtype=np.int64),
    }


def default_sonic_root():
    for c in (os.environ.get("SONIC_ROOT"), BUNDLE / "vendor/GR00T-WholeBodyControl",
              Path.home() / "sonic/GR00T-WholeBodyControl"):
        if c and (Path(c) / "gear_sonic").is_dir():
            return Path(c)
    return None


def write_capture(path, msgs):
    with open(path, "wb") as f:
        for m in msgs:
            f.write(struct.pack("<I", len(m)))
            f.write(m)


def read_capture(path):
    out, b = [], Path(path).read_bytes()
    off = 0
    while off + 4 <= len(b):
        (n,) = struct.unpack_from("<I", b, off)
        out.append(b[off + 4: off + 4 + n])
        off += 4 + n
    return out


# ----------------------------------------------------------------------------- commands
def cmd_selftest(a):
    sys.path.insert(0, str(BUNDLE))
    from sonic.stream import HEADER_SIZE as OUR_HEADER, unpack_pose_message, v3_message
    root = Path(a.sonic_root) if a.sonic_root else default_sonic_root()
    if root is None:
        sys.exit("SONIC checkout not found: pass --sonic-root or set SONIC_ROOT")
    pack, ppath = load_official_packer(root)
    print(f"official packer : {ppath}")
    print(f"our packer      : {BUNDLE / 'sonic/stream.py'}:v3_message")
    fails = 0
    assert OUR_HEADER == HEADER_SIZE

    rng = np.random.default_rng(0)
    base = np.linspace([-0.2, -0.2, -0.9], [0.2, 0.2, 0.6], 24)   # wire test only; real skeletons come from e2e/listen
    msgs = []
    for k in range(a.n):
        joints = base + 0.01 * rng.standard_normal((24, 3))
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        t = time.monotonic()
        ours = v3_message(joints, q, k, t_mono=t)
        ours_no_ts = v3_message(joints, q, k)
        official = pack(official_v3_fields(joints, q, k), topic="pose", version=3)
        if ours_no_ts != official:
            fails += 1
            i = next(i for i in range(min(len(ours_no_ts), len(official))) if ours_no_ts[i] != official[i])
            print(f"  [FAIL] message {k}: bytes differ from official packer at offset {i}")
        hdr, bufs, info = poll_once(ours)
        m, warns = decode_motion(hdr, bufs)
        for w in warns:
            print(f"  [WARN] {w}")
        if not (np.allclose(m["smpl_joints"][0], joints.astype(np.float32)) and np.allclose(m["body_quat"][0, 0], q.astype(np.float32))
                and m["frame_indices"][0] == k and abs(m["timestamp_monotonic"] - t) < 1e-9):
            fails += 1
            print(f"  [FAIL] message {k}: C++-port decode does not round-trip")
        v, d = unpack_pose_message(ours)
        if v != 3 or not np.array_equal(d["smpl_joints"], m["smpl_joints"].astype(np.float32)):
            fails += 1
            print(f"  [FAIL] message {k}: our own unpacker disagrees with the C++ port")
        msgs.append(ours)
    hdr, _, info = poll_once(msgs[0])
    print(f"wire layout     : topic 'pose' (4 B) + header {HEADER_SIZE} B (JSON {info['header_bytes']} B, NUL padded) + data {info['data_bytes']} B"
          f" = {len(msgs[0])} B, trailing {info['trailing_bytes']} B")
    print("header fields   : " + ", ".join(f"{f['name']}:{f['dtype']}{f['shape']}" for f in hdr["fields"]))
    print(f"byte-identical to official pack_pose_message (minus our extra trailing timestamp_monotonic field): "
          f"{'yes' if fails == 0 else 'NO'} ({a.n} messages)")
    if a.save:
        write_capture(a.save, msgs)
        print(f"wrote {len(msgs)} messages -> {a.save}")

    # negative cases: the port must reject exactly what the C++ code rejects
    good = official_v3_fields(base, [1, 0, 0, 0], 0)
    cases = {
        "v3 without joint_vel": ({k: v for k, v in good.items() if k != "joint_vel"}, 3, "Version 3 missing required field 'joint_vel'"),
        "v3 without smpl_pose": ({k: v for k, v in good.items() if k != "smpl_pose"}, 3, "missing required field 'smpl_pose'"),
        "no frame_index": ({k: v for k, v in good.items() if k != "frame_index"}, 3, "frame_index"),
        "frame count mismatch": ({**good, "joint_pos": np.zeros((2, 29), np.float32)}, 3, "frame count mismatch"),
        "frame_index as f32": ({**good, "frame_index": np.array([0.0], np.float32)}, 3, "frame_indices"),
        "protocol v5": (good, 5, "Unsupported protocol version"),
    }
    for name, (fields, ver, expect) in cases.items():
        try:
            decode_motion(*poll_once(pack(fields, topic="pose", version=ver))[:2])
            print(f"  [FAIL] negative case '{name}' was accepted")
            fails += 1
        except Reject as e:
            hit = expect in str(e)
            fails += 0 if hit else 1
            print(f"  [{'ok' if hit else 'FAIL'}] rejects {name}: {e}")
    for name, msg, expect in [("wrong topic", b"pozz" + msgs[0][4:], "topic mismatch"),
                              ("truncated payload", msgs[0][:-100], "exceeds data bounds"),
                              ("short header", msgs[0][:900], "too small")]:
        try:
            poll_once(msg)
            print(f"  [FAIL] negative case '{name}' was accepted")
            fails += 1
        except Reject as e:
            fails += 0 if expect in str(e) else 1
            print(f"  [{'ok' if expect in str(e) else 'FAIL'}] rejects {name}: {e}")
    print("SELFTEST", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 0 if fails == 0 else 1


def validate_stream(msgs_with_t, topic="pose", verbose=False):
    tr, rejects, sem, warns_all = StreamTracker(), 0, {}, set()
    first = None
    for msg, t in msgs_with_t:
        try:
            hdr, bufs, info = poll_once(msg, topic)
            m, warns = decode_motion(hdr, bufs)
        except Reject as e:
            rejects += 1
            print(f"  [REJECT] {e}")
            continue
        warns_all.update(warns)
        for s in semantic_checks(m):
            sem[s] = sem.get(s, 0) + 1
        tr.add(m, t)
        if first is None:
            first = (hdr, m)
    return tr, rejects, sem, warns_all, first


def report(tr, rejects, sem, warns, first):
    ok = rejects == 0 and not tr.errors and tr.n > 0 and not sem
    if first:
        hdr, m = first
        print("first message   : v%d, fields %s" % (hdr["version"], ", ".join(f"{f['name']}:{f['dtype']}{f['shape']}" for f in hdr["fields"])))
        print("  body_quat     :", np.round(m["body_quat"][0, 0], 4).tolist(), " frame_index:", int(m["frame_indices"][0]))
        print("  smpl_joints[0..3]:", np.round(m["smpl_joints"][0, :4], 3).tolist())
    s = tr.summary()
    print("stream          :", json.dumps(s))
    for w in sorted(warns):
        print(f"  [WARN] {w}")
    for k, v in sem.items():
        print(f"  [SEMANTIC] {k}  ({v} msgs)")
    if s.get("has_timestamp_monotonic"):
        print("  note: timestamp_monotonic present. The deploy uses it only when --zmq-host is the literal string "
              "'localhost' (then it must be the same machine's CLOCK_MONOTONIC). From another machine or an SSH tunnel, "
              "use --zmq-host <ip> / 127.0.0.1 so the deploy falls back to its own receive time.")
    print("RESULT", "COMPATIBLE" if ok else "INCOMPATIBLE")
    return 0 if ok else 1


def subscribe(host, port, topic, seconds, max_msgs, save=None, conflate=False):
    import zmq
    ctx = zmq.Context()
    s = ctx.socket(zmq.SUB)               # same options as ZMQPackedMessageSubscriber::Connect (rcvhwm 3, no conflate)
    s.setsockopt(zmq.RCVTIMEO, 1000)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.RCVHWM, 1 if conflate else 3)
    if conflate:
        s.setsockopt(zmq.CONFLATE, 1)
    s.connect(f"tcp://{host}:{port}")
    s.setsockopt(zmq.SUBSCRIBE, topic.encode())
    print(f"subscribed to '{topic}' at tcp://{host}:{port} for {seconds:.0f} s")
    out, t_end = [], time.monotonic() + seconds
    while time.monotonic() < t_end and len(out) < max_msgs:
        try:
            out.append((s.recv(), time.monotonic()))
        except zmq.Again:
            continue
    s.close(0)
    ctx.term()
    if save:
        write_capture(save, [m for m, _ in out])
        print(f"saved {len(out)} messages -> {save}")
    return out


def cmd_listen(a):
    msgs = subscribe(a.host, a.port, a.topic, a.seconds, a.max_msgs, a.save)
    if not msgs:
        print("no messages received (is the sender running and reachable? firewall on the sender side?)")
        return 1
    return report(*validate_stream(msgs, a.topic))


def cmd_check(a):
    msgs = read_capture(a.capture)
    tr, rej, sem, warns, first = validate_stream([(m, None) for m in msgs], a.topic)
    tr.t_recv = []                       # a capture file has no receive times
    return report(tr, rej, sem, warns, first)


def cmd_e2e(a):
    py = a.python or str(BUNDLE / ".venv/bin/python")
    cmd = [py, "-u", str(BUNDLE / "scripts/webcam_sender.py"), "--video", a.video, "--no-window",
           "--max-seconds", str(a.seconds + 4), "--port", str(a.port)] + (["--loop"] if a.loop else [])
    print("sender:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        msgs = subscribe("localhost", a.port, "pose", a.seconds, 10 ** 9, a.save)
    finally:
        proc.terminate()
        try:
            out = proc.communicate(timeout=10)[0]
        except subprocess.TimeoutExpired:
            proc.kill()
            out = proc.communicate()[0]
    print("sender log:", " | ".join(l for l in out.splitlines() if l.startswith("[sender]"))[:400])
    if not msgs:
        print("no messages received")
        return 1
    return report(*validate_stream(msgs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selftest")
    p.add_argument("--sonic-root", default=None)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--save", default=None, help="write the synthetic messages as a capture file")
    p.set_defaults(fn=cmd_selftest)
    p = sub.add_parser("e2e")
    p.add_argument("--video", required=True)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--python", default=None, help="python with mediapipe (default: <bundle>/.venv/bin/python)")
    p.add_argument("--save", default=None)
    p.set_defaults(fn=cmd_e2e)
    p = sub.add_parser("listen")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--topic", default="pose")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--max-msgs", type=int, default=10 ** 9)
    p.add_argument("--save", default=None)
    p.set_defaults(fn=cmd_listen)
    p = sub.add_parser("check")
    p.add_argument("capture")
    p.add_argument("--topic", default="pose")
    p.set_defaults(fn=cmd_check)
    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
