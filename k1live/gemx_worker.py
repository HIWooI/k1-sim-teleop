"""GEM-X real-time worker (GPU): one camera frame in -> SOMA-77 skeleton out. Runs in the GEM-X virtualenv.

The GEM-X real-time path (perception/demo_webcam.py, RTX 5080: 30 FPS, ~133 ms perceived delay with --no_imgfeat),
imported from a GEM-X perception folder and used as is: VitPose SOMA-77 2D on a box propagated from the previous
keypoints (bootstrap / re-detect on the person region) -> GEM regression over a sliding window (64 frames, starts at
16) -> SOMA forward kinematics of the window's last frame -> joints in the camera frame (m). No SAM-3D-Body tokens
(--no_imgfeat: the stutter source), identity frozen per person (re-measured on reset), global trajectory skipped.

This file must run with only the GEM-X environment (torch, cv2, numpy, gem): nothing from k1live is imported. The
k1live side (k1live/sender_gemx.py, MuJoCo-free, any venv) talks to it over stdin / stdout:

    request   REQ  (magic GXQ1, seq, t_capture, w, h, fmt 0 raw BGR | 1 JPEG, flags 1 reset | 2 re-measure identity,
                    person region x0 y0 x1 y1 (-1 = whole frame), payload bytes) + payload
    reply     REP  (magic GXR1, seq, t_capture, status 0 nobody | 1 2D only (window warming up) | 2 3D,
                    flags 1 tracking restarted, vitpose / predict / fk / total ms) + RES_FLOATS float32:
                    kp2d 77x3 (px, conf) | joints 77x3 (camera frame m) | body_pose 228 | transl 3 | K 3x3
    start-up  b"GXRD" + u32 + JSON {device, load_s, ...} when ready, or b"GXER" + u32 + JSON {error} and exit code
              EXIT_NO_CUDA (5) / EXIT_NO_GEMX (6)

One request in flight at a time (the caller always sends its newest frame). The same pipes work through ssh, so the
GPU can be another machine on the LAN: sender_gemx --gemx-cmd "ssh gpu-host <python> <this file> --serve" --gemx-fmt jpeg.

    <GEM-X>/.venv/bin/python k1live/gemx_worker.py --serve                      # used by k1live/sender_gemx.py
    <GEM-X>/.venv/bin/python k1live/gemx_worker.py --dump a.mp4 b.mp4 --out dir/   # every frame, causal (reel / eval)
    <GEM-X>/.venv/bin/python k1live/gemx_worker.py --dump clip.mp4 --out t.npz --realtime   # camera-rate, drops
    <GEM-X>/.venv/bin/python k1live/gemx_worker.py --check                      # CUDA + import + checkpoint, exit code

Paths: --perception (default $K1LIVE_GEMX_PERCEPTION or <repo>/third_party/gemx_perception: a folder with demo_webcam.py
whose GEM-X/ subfolder is the GEM-X checkout; not part of this repository, docs/setup.md). Read-only: nothing is
written there.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

MAGIC_REQ, MAGIC_REP, MAGIC_READY, MAGIC_ERR = b"GXQ1", b"GXR1", b"GXRD", b"GXER"
REQ = struct.Struct("<4sIdHHBB4fI")          # magic seq t w h fmt flags crop[4] nbytes
REP = struct.Struct("<4sIdBBH4f")             # magic seq t status flags pad ms[4]
N_KP, N_J, N_BP = 77 * 3, 77 * 3, 228
RES_FLOATS = N_KP + N_J + N_BP + 3 + 9
FLAG_RESET, FLAG_IDENTITY = 1, 2
ST_NONE, ST_2D, ST_3D = 0, 1, 2
FMT_RAW, FMT_JPEG = 0, 1
EXIT_NO_CUDA, EXIT_NO_GEMX = 5, 6
DEFAULT_PERCEPTION = os.environ.get("K1LIVE_GEMX_PERCEPTION",
                                    str(Path(__file__).resolve().parents[1] / "third_party" / "gemx_perception"))


def hub_cache_first(torch):
    """GEM-X's VitPose builds its DINOv3 backbone with torch.hub.load("facebookresearch/dinov3", ...) and no branch, so
    every start asks github.com for the default branch even with the repo cached. Offline that falls back to the cache,
    but on a venue network that drops packets urlopen (no timeout) waits for the TCP timeout (~2 min, longer than the
    service's start-up grace). Use the cached checkout first, as torch.hub's own offline branch would (on the
    event PC the only network access of a GEM-X start, found with scripts/offline_guard)."""
    hub = torch.hub
    parse = hub._parse_repo_info

    def cached_first(github):
        if ":" not in github:
            owner, name = github.split("/")
            for ref in ("main", "master"):
                if os.path.isdir(os.path.join(hub.get_dir(), f"{owner}_{name}_{ref}")):
                    return owner, name, ref
        return parse(github)

    hub._parse_repo_info = cached_first


def soma_kernel_cache(log=None):
    """SOMA's skinning (soma/geometry/lbs_warp.py) calls get_kernel() on every forward pass, and get_kernel defines a
    fresh @wp.kernel each time: Warp parses its source again and keeps every Kernel object, so the GEM-X worker's
    memory grew ~3.3 GB/h, linearly, as long as a person was tracked (event PC, 20 min: 4.05 -> 4.85 GB; traced
    on a GPU server with tracemalloc: warp Kernel / ast.parse from lbs_warp.get_kernel). Same arguments -> same kernel,
    so memoise it (keyed by the bone count and the dtypes; Warp vector types by length + scalar type). Patched here
    instead of in the GEM-X checkout, which is not ours. Returns True when the cache is in place."""
    try:
        from soma.geometry import lbs_warp
    except Exception as e:  # another GEM-X / SOMA version: nothing to patch
        (log or print)(f"[gemx] SOMA kernel cache not installed ({type(e).__name__}: {e})")
        return False
    if getattr(lbs_warp.get_kernel, "_k1live_cached", False):
        return True
    orig, cache = lbs_warp.get_kernel, {}

    def tkey(t):
        n = getattr(t, "_length_", None)
        return (n, getattr(t, "_wp_scalar_type_", None)) if n is not None else t

    def get_kernel(max_bones_count, *args, **kw):
        key = (max_bones_count, tuple(tkey(a) for a in args), tuple(sorted((k, tkey(v)) for k, v in kw.items())))
        k = cache.get(key)
        if k is None:
            k = cache[key] = orig(max_bones_count, *args, **kw)
        return k

    get_kernel._k1live_cached = True
    get_kernel.cache = cache
    lbs_warp.get_kernel = get_kernel
    return True


# ------------------------------------------------------------------------------------------------ wire format
def pack_request(seq, t, frame, fmt=FMT_RAW, flags=0, crop=None, jpeg_quality=85):
    """frame: HxWx3 uint8 BGR -> bytes to write."""
    h, w = frame.shape[:2]
    if fmt == FMT_JPEG:
        import cv2

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        payload = buf.tobytes()
    else:
        payload = np.ascontiguousarray(frame, np.uint8).tobytes()
    c = tuple(float(v) for v in crop) if crop is not None else (-1.0, -1.0, -1.0, -1.0)
    return REQ.pack(MAGIC_REQ, seq & 0xFFFFFFFF, float(t), w, h, fmt, flags, *c, len(payload)) + payload


def read_exact(f, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise EOFError(f"pipe closed after {len(buf)} of {n} bytes")
        buf += chunk
    return bytes(buf)


def read_request(f):
    """-> (seq, t, frame BGR, flags, crop or None)."""
    magic, seq, t, w, h, fmt, flags, x0, y0, x1, y1, n = REQ.unpack(read_exact(f, REQ.size))
    if magic != MAGIC_REQ:
        raise ValueError(f"bad request magic {magic!r}")
    payload = read_exact(f, n)
    if fmt == FMT_JPEG:
        import cv2

        frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    else:
        frame = np.frombuffer(payload, np.uint8).reshape(h, w, 3)
    crop = None if x0 < 0 else (x0, y0, x1, y1)
    return seq, t, frame, flags, crop


def pack_reply(seq, t, res):
    ms = res.get("ms", {})
    head = REP.pack(MAGIC_REP, seq & 0xFFFFFFFF, float(t), int(res["status"]), int(res.get("flags", 0)), 0,
                    *(float(ms.get(k, 0.0)) for k in ("vitpose", "predict", "fk", "total")))
    body = np.zeros(RES_FLOATS, np.float32)
    o = 0
    for key, n in (("kp2d", N_KP), ("joints", N_J), ("body_pose", N_BP), ("transl", 3), ("K", 9)):
        v = res.get(key)
        if v is not None:
            body[o:o + n] = np.asarray(v, np.float32).reshape(-1)[:n]
        o += n
    return head + body.tobytes()


def read_reply(f):
    magic, seq, t, status, flags, _, *ms = REP.unpack(read_exact(f, REP.size))
    if magic != MAGIC_REP:
        raise ValueError(f"bad reply magic {magic!r}")
    body = np.frombuffer(read_exact(f, RES_FLOATS * 4), np.float32).astype(np.float64)
    o = 0
    out = {"seq": seq, "t": t, "status": status, "flags": flags,
           "ms": dict(zip(("vitpose", "predict", "fk", "total"), ms))}
    for key, n, shape in (("kp2d", N_KP, (77, 3)), ("joints", N_J, (77, 3)), ("body_pose", N_BP, (N_BP,)),
                          ("transl", 3, (3,)), ("K", 9, (3, 3))):
        out[key] = body[o:o + n].reshape(shape)
        o += n
    return out


def write_msg(f, magic, obj):
    b = json.dumps(obj).encode()
    f.write(magic + struct.pack("<I", len(b)) + b)
    f.flush()


def read_msg(f):
    magic = read_exact(f, 4)
    n = struct.unpack("<I", read_exact(f, 4))[0]
    return magic, json.loads(read_exact(f, n).decode())


# ------------------------------------------------------------------------------------------------ engine
class GemxEngine:
    """demo_webcam.py's per-frame loop body as an object (one person, newest frame each call)."""

    LOST_AFTER = 3      # frames without a box before the window is dropped (demo_webcam.py)

    def __init__(self, perception=DEFAULT_PERCEPTION, window=64, min_window=16, ckpt=None, hfov_deg=0.0,
                 detector="auto", log=None, box_from_3d=True, flip_test=True):
        self.log = log or (lambda *a: print(*a, file=sys.stderr, flush=True))
        t0 = time.time()
        perception = str(Path(perception).expanduser().resolve())
        if perception not in sys.path:
            sys.path.insert(0, perception)
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
        hub_cache_first(torch)
        import demo_webcam as dw   # chdir()s into GEM-X, patches the pipeline (global trajectory off), CUDA tensors

        soma_kernel_cache(self.log)   # before the first forward pass: SOMA re-creates its Warp kernel per call

        from gem.utils.cam_utils import estimate_K
        from gem.utils.geo_transform import get_bbx_xys_from_xyxy
        from gem.utils.vitpose_extractor import VitPoseExtractor

        self.torch, self.dw = torch, dw
        self.estimate_K, self.bbx_from_xyxy = estimate_K, get_bbx_xys_from_xyxy
        self.vitpose = VitPoseExtractor(device="cuda:0", pose_type="soma", tqdm_leave=False)
        # flip test on: +3.5 ms on the RTX 5080 (VitPose 17.7 -> 21.2 ms, batched) and it keeps the upper body on a dark,
        # loose garment where single-pass VitPose collapsed it for 4 s ('toward' test clip: low-confidence shoulders
        # 65 -> 8 %)
        self.vitpose.flip_test = bool(flip_test)
        self.model = dw.build_model(ckpt)
        self.soma = self.model.body_model.soma if hasattr(self.model.body_model, "soma") else self.model.body_model
        self.window, self.min_window, self.hfov_deg = int(window), int(min_window), float(hfov_deg)
        self.box_from_3d = bool(box_from_3d)
        self.yolox = None
        if detector in ("auto", "yolox"):
            try:
                from gem.utils.yolox_detector import YOLOXDetector

                self.yolox = YOLOXDetector(device="cuda:0")
            except Exception as e:   # demo_webcam: "YOLOX is ONNX-only and onnxruntime is broken here" on some boxes
                if detector == "yolox":
                    raise
                self.log(f"[gemx] no YOLOX ({type(e).__name__}: {e}); acquisition by VitPose bootstrap")
        from collections import deque

        self.kp_win, self.bbx_win = deque(maxlen=self.window), deque(maxlen=self.window)
        self.K, self.wh, self.bbx, self.n_miss = None, None, None, 0
        self.n_reset = 0
        self.load_s = time.time() - t0
        self.info = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "load_s": round(self.load_s, 1),
                     "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1),
                     "window": self.window, "min_window": self.min_window, "box_from_3d": self.box_from_3d,
                     "flip_test": bool(flip_test),
                     "detector": "yolox" if self.yolox else
                     "vitpose-bootstrap", "perception": perception, "ckpt": str(ckpt) if ckpt else "default (HF)"}

    # -------------------------------------------------------------------------------------------- helpers
    def _K(self, w, h):
        if self.hfov_deg > 0:
            f = 0.5 * w / np.tan(np.radians(self.hfov_deg) / 2)
            return self.torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=self.torch.float32)
        return self.estimate_K(w, h)

    def _region_box(self, w, h, crop):
        x0, y0, x1, y1 = (0.0, 0.0, 1.0, 1.0) if crop is None else crop
        xyxy = self.torch.tensor([[x0 * (w - 1), y0 * (h - 1), x1 * (w - 1), y1 * (h - 1)]])
        return xyxy

    def _acquire(self, frame, w, h, crop, rounds=3):
        """First box of a person: YOLOX (largest person whose centre is in the region) when available, else VitPose
        re-run on its own output starting from the region (demo_webcam.bootstrap_bbox, generalised to a region)."""
        torch = self.torch
        if self.yolox is not None:
            try:
                from gem.utils.yolox_detector import detect_and_track

                xyxy, _ = detect_and_track(np.ascontiguousarray(frame[..., ::-1])[None], self.yolox)  # RGB in
                xyxy = torch.as_tensor(np.asarray(xyxy, np.float32)).reshape(-1, 4)
                if len(xyxy) and float((xyxy[0, 2:] - xyxy[0, :2]).min()) > 4:
                    c = 0.5 * (xyxy[0, :2] + xyxy[0, 2:])
                    r = self._region_box(w, h, crop)[0]
                    if crop is None or (r[0] <= c[0] <= r[2] and r[1] <= c[1] <= r[3]):
                        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, w - 1)
                        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, h - 1)
                        return self.bbx_from_xyxy(xyxy[:1], base_enlarge=1.2).float()[0]
            except Exception as e:
                self.log(f"[gemx] YOLOX failed ({type(e).__name__}: {e}); VitPose bootstrap")
        bbx = self.bbx_from_xyxy(self._region_box(w, h, crop))[0].float()
        was, self.vitpose.flip_test = self.vitpose.flip_test, True
        try:
            for _ in range(rounds):
                with torch.autocast("cuda", dtype=torch.float16):
                    kp = self.vitpose.extract(frame[None], bbx[None], path_type="np", batch_size=1)[0]
                nb = self.dw.bbox_from_kps(kp, w, h)
                if nb is None:
                    break
                bbx = nb
        finally:
            self.vitpose.flip_test = was
        return bbx

    def warm_up(self, w=1280, h=720):
        """VitPose, GEM predict and FK once each on a synthetic window (kernel selection, allocations), so the first
        person in front of the camera is not seconds late. Leaves the engine reset."""
        torch = self.torch
        t0 = time.perf_counter()
        frame = np.full((h, w, 3), 127, np.uint8)
        self.wh, self.K = (w, h), self._K(w, h)
        with torch.no_grad():
            box = self.bbx_from_xyxy(self._region_box(w, h, (0.35, 0.1, 0.65, 0.95)))[0].float()
            with torch.autocast("cuda", dtype=torch.float16):
                kp = self.vitpose.extract(frame[None], box[None], path_type="np", batch_size=1)[0]
            kp = kp.clone()
            kp[:, 0] = w / 2 + (torch.arange(77) % 9 - 4) * 12.0
            kp[:, 1] = h * 0.1 + torch.arange(77) * (0.8 * h / 77)
            kp[:, 2] = 0.9
            for _ in range(self.min_window):
                self.kp_win.append(kp.cuda())
                self.bbx_win.append(box.cuda())
            for _ in range(2):
                pred = self.model.predict(self.dw.make_data(self.kp_win, self.bbx_win, self.K), static_cam=True,
                                          postproc=False)
                p = {k: v[-1:].cuda() for k, v in pred["body_params_incam"].items()}
                poses = torch.cat([p["global_orient"][:, None], p["body_pose"].reshape(1, 76, 3)], dim=1)
                self.model.body_model.static_forward(poses, p["identity_coeffs"], p["scale_params"], p["transl"],
                                                     return_joints_only=True)
            self._sync()
        self.reset(identity=True)
        self.wh = None
        self.info["warmup_s"] = round(time.perf_counter() - t0, 2)

    BODY = [i for i in range(77) if not (15 <= i <= 38 or 43 <= i <= 66)]   # SOMA joints without fingers

    def _box_3d(self, joints, kp2d, w, h, nbx):
        """Next frame's box from the confident VitPose keypoints AND this frame's GEM-X skeleton projected (scale +
        shift aligned to those keypoints). demo_webcam.py propagates the box from confident keypoints only: when the head and
        shoulders turn uncertain (dark clothes on a bright wall, a person walking toward the camera) the box shrinks to
        the legs and never sees the head again (a 'toward' test clip: shoulders at confidence 0.05 for 4 s). The GEM-X
        skeleton always has a head, so the box keeps the whole body."""
        J = np.asarray(joints, np.float64)
        K = self.K.numpy() if hasattr(self.K, "numpy") else np.asarray(self.K)
        uv = J @ K.T
        uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-3)
        kp = np.asarray(kp2d, np.float64)
        ok = kp[:, 2] > self.dw.CONF_THR
        if ok.sum() >= 6:
            a, b = uv[ok], kp[ok, :2]
            ma, mb = a.mean(0), b.mean(0)
            sc = float(((a - ma) * (b - mb)).sum() / max(((a - ma) ** 2).sum(), 1e-9))
            uv = (uv - ma) * (sc if 0.5 < sc < 2.0 else 1.0) + mb
        pts = np.concatenate([uv[self.BODY], kp[ok, :2]])
        if not np.all(np.isfinite(pts)):
            return nbx
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        xyxy = self.torch.tensor([[np.clip(x0, 0, w - 1), np.clip(y0, 0, h - 1), np.clip(x1, 0, w - 1),
                                   np.clip(y1, 0, h - 1)]], dtype=self.torch.float32)
        if float(xyxy[0, 2] - xyxy[0, 0]) < 4 or float(xyxy[0, 3] - xyxy[0, 1]) < 4:
            return nbx
        return self.bbx_from_xyxy(xyxy, base_enlarge=1.2).float()[0]

    def reset(self, identity=True):
        self.kp_win.clear()
        self.bbx_win.clear()
        self.bbx, self.n_miss = None, 0
        if identity:
            self.soma._identity_frozen = False   # demo_webcam.freeze_identity: the next FK re-measures the body
            self.model._rest_R = None

    # -------------------------------------------------------------------------------------------- one frame
    @staticmethod
    def _sync():
        import torch

        torch.cuda.synchronize()

    def process(self, frame, crop=None, flags=0, t=None):
        torch, dw = self.torch, self.dw
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        out = {"status": ST_NONE, "flags": 0, "ms": {}}
        if self.wh != (w, h):
            self.wh, self.K = (w, h), self._K(w, h)
            self.reset(identity=True)
            out["flags"] |= FLAG_RESET
        if flags & FLAG_RESET:
            self.reset(identity=True)
            out["flags"] |= FLAG_RESET
        elif flags & FLAG_IDENTITY:
            self.soma._identity_frozen = False
        with torch.no_grad():
            if self.bbx is None:
                self.bbx = self._acquire(frame, w, h, crop)
            ts = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                kp2d = self.vitpose.extract(frame[None], self.bbx[None], path_type="np", batch_size=1)[0]
            out["ms"]["vitpose"] = (time.perf_counter() - ts) * 1e3
            nbx = dw.bbox_from_kps(kp2d, w, h)
            self.n_miss = 0 if nbx is not None else self.n_miss + 1
            out["kp2d"] = kp2d.numpy()
            if nbx is None and self.n_miss >= self.LOST_AFTER:   # really gone: re-detect next frame
                self.n_reset += 1
                self.reset(identity=False)
                out["flags"] |= FLAG_RESET
                out["ms"]["total"] = (time.perf_counter() - t0) * 1e3
                return out
            self.kp_win.append(kp2d.cuda())
            self.bbx_win.append(self.bbx.cuda())
            if nbx is not None:
                self.bbx = nbx
            out["status"] = ST_2D
            if len(self.kp_win) >= self.min_window:
                ts = time.perf_counter()
                pred = self.model.predict(dw.make_data(self.kp_win, self.bbx_win, self.K), static_cam=True,
                                          postproc=False)
                self._sync()
                out["ms"]["predict"] = (time.perf_counter() - ts) * 1e3
                ts = time.perf_counter()
                p = {k: v[-1:].cuda() for k, v in pred["body_params_incam"].items()}
                poses = torch.cat([p["global_orient"][:, None], p["body_pose"].reshape(1, 76, 3)], dim=1)
                joints = self.model.body_model.static_forward(poses, p["identity_coeffs"], p["scale_params"],
                                                              p["transl"], return_joints_only=True)["joints"][0]
                out["joints"] = joints.float().cpu().numpy()
                out["body_pose"] = p["body_pose"].reshape(-1).float().cpu().numpy()
                out["transl"] = p["transl"].reshape(-1).float().cpu().numpy()
                out["ms"]["fk"] = (time.perf_counter() - ts) * 1e3
                out["status"] = ST_3D
                if nbx is not None and self.box_from_3d:
                    self.bbx = self._box_3d(out["joints"], out["kp2d"], w, h, nbx)
        out["K"] = self.K.numpy() if hasattr(self.K, "numpy") else np.asarray(self.K)
        out["ms"]["total"] = (time.perf_counter() - t0) * 1e3
        return out


class ReplayEngine:
    """A saved GEM-X track (k1live.gemx.load_track formats) answered by video time instead of GPU inference: the
    k-th result is the track frame at round((t - t_first) * fps) (wrapping). NumPy only - tests and Mac rehearsals
    of the whole GEM-X chain without CUDA (--serve --replay <track>)."""

    def __init__(self, path, log=None):
        p = Path(path)
        z = np.load(p / "soma_joints.npz" if p.is_dir() else p, allow_pickle=True)
        self.J = np.asarray(z["joints_incam"], np.float64)
        self.KP = np.asarray(z["kp2d_vitpose"], np.float64) if "kp2d_vitpose" in z.files else np.zeros(self.J.shape)
        self.BP = np.asarray(z["body_pose"], np.float64) if "body_pose" in z.files else np.zeros((len(self.J), N_BP))
        self.valid = np.asarray(z["valid"], bool) if "valid" in z.files else np.isfinite(self.J[:, 0, 0])
        self.K = np.asarray(z["K"], np.float64)
        self.wh = tuple(int(v) for v in z["image_wh"])
        self.fps = float(z["fps"])
        self.t_first = None
        self.n_reset = 0
        self.info = {"device": "replay (no GPU)", "track": str(path), "frames": len(self.J), "load_s": 0.0}

    def warm_up(self, *a, **k):
        pass

    def reset(self, identity=True):
        self.t_first = None

    def process(self, frame, crop=None, flags=0, t=None):
        t0 = time.perf_counter()
        if flags & FLAG_RESET:
            self.t_first = None
        if self.t_first is None:
            self.t_first = t
        k = int(round((t - self.t_first) * self.fps)) % len(self.J) if t is not None else 0
        h, w = frame.shape[:2]
        s = np.array([w / self.wh[0], h / self.wh[1], 1.0])
        out = {"status": ST_NONE, "flags": 0, "ms": {}, "kp2d": self.KP[k] * s, "K": np.diag(s) @ self.K}
        if self.valid[k]:
            out.update(status=ST_3D, joints=self.J[k], body_pose=self.BP[k], transl=self.J[k, 0])
        out["ms"]["total"] = (time.perf_counter() - t0) * 1e3
        return out


# ------------------------------------------------------------------------------------------------ modes
def make_engine(a, log):
    if getattr(a, "ckpt", ""):
        a.ckpt = str(Path(a.ckpt).expanduser().resolve())   # before demo_webcam's chdir
    if getattr(a, "replay", ""):
        return ReplayEngine(a.replay, log), 0, ""
    try:
        import torch
    except Exception as e:
        return None, EXIT_NO_GEMX, f"torch import failed: {e}"
    if not torch.cuda.is_available():
        return None, EXIT_NO_CUDA, "no CUDA device (torch.cuda.is_available() is False)"
    try:
        eng = GemxEngine(a.perception, a.window, a.min_window, a.ckpt or None, a.hfov, a.detector, log,
                         box_from_3d=not a.box_kp_only, flip_test=not a.no_flip_test)
    except Exception as e:
        return None, EXIT_NO_GEMX, f"GEM-X load failed: {type(e).__name__}: {e}"
    return eng, 0, ""


def serve(a):
    out = sys.stdout.buffer
    sys.stdout = sys.stderr      # stray prints (hydra, GEM-X) must not corrupt the binary stream
    log = lambda *x: print(*x, file=sys.stderr, flush=True)  # noqa: E731
    eng, code, err = make_engine(a, log)
    if eng is None:
        write_msg(out, MAGIC_ERR, {"error": err, "code": code})
        return code
    eng.warm_up()
    write_msg(out, MAGIC_READY, eng.info)
    inp = sys.stdin.buffer
    while True:
        try:
            seq, t, frame, flags, crop = read_request(inp)
        except EOFError:
            return 0
        try:
            res = eng.process(frame, crop, flags, t=t)
        except Exception as e:   # one bad frame must not kill the stream: report nobody, start over
            log(f"[gemx] frame {seq}: {type(e).__name__}: {e}")
            eng.reset(identity=False)
            res = {"status": ST_NONE, "flags": FLAG_RESET, "ms": {}}
        out.write(pack_reply(seq, t, res))
        out.flush()


def dump(a):
    """Every frame of each video through the engine (causal: frame k sees frames <= k only) -> track npz for
    k1live.gemx.load_track (kind "stream"). --realtime: frames arrive at the clip's rate and the engine takes the
    newest one when it is free (what the live sender sees), with each result's completion time. The models load once
    for all videos; the person / window / identity are reset between videos. Existing outputs are skipped."""
    log = lambda *x: print(*x, file=sys.stderr, flush=True)  # noqa: E731
    # absolute paths first: importing demo_webcam chdir()s into the GEM-X checkout
    videos = [str(Path(v).expanduser().resolve()) for v in a.dump]
    out_arg = Path(a.out).expanduser().resolve()
    if len(videos) > 1 or out_arg.suffix != ".npz":
        outs = [out_arg / (Path(v).stem + a.suffix + ".npz") for v in videos]
    else:
        outs = [out_arg]
    todo = [(v, o) for v, o in zip(videos, outs) if not (o.exists() and o.stat().st_size > 0)]
    if not todo:
        log("[gemx] all outputs exist")
        return 0
    eng, code, err = make_engine(a, log)
    if eng is None:
        log(f"[gemx] {err}")
        return code
    crop = tuple(float(v) for v in a.crop.split(",")) if a.crop else None
    rc = 0
    for v, o in todo:
        try:
            meta = dump_one(eng, v, o, crop, a, log)
            log(json.dumps({"video": v, **{k: meta[k] for k in ("processed", "ms_median", "fps_wall", "resets")}}))
        except Exception as e:
            log(f"[gemx] {v}: {type(e).__name__}: {e}")
            rc = 2
    return rc


def dump_one(eng, video, out, crop, a, log):
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if a.max_h and f.shape[0] > a.max_h:
            s = a.max_h / f.shape[0]
            f = cv2.resize(f, (int(round(f.shape[1] * s)), a.max_h), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"cannot read {video}")
    h, w = frames[0].shape[:2]
    rows = []
    eng.warm_up(w, h)
    eng.n_reset = 0
    t_wall0 = time.perf_counter()
    k = 0
    while k < len(frames):
        if a.realtime:   # the camera's newest frame at this moment
            k = max(k, int((time.perf_counter() - t_wall0) * fps))
            if k >= len(frames):
                break
        res = eng.process(frames[k], crop)
        t_done = time.perf_counter() - t_wall0
        rows.append((k, t_done, res))
        k += 1
        if len(rows) % 300 == 0:
            log(f"[gemx] {Path(video).name}: {len(rows)} frames, "
                f"{np.median([r[2]['ms']['total'] for r in rows[-300:]]):.1f} ms/frame")
    N = len(rows)
    J = np.full((N, 77, 3), np.nan, np.float32)
    KP = np.zeros((N, 77, 3), np.float32)
    BP = np.zeros((N, N_BP), np.float32)
    TR = np.zeros((N, 3), np.float32)
    valid = np.zeros(N, bool)
    ms = np.zeros((N, 4), np.float32)
    for i, (_, _, r) in enumerate(rows):
        if r.get("kp2d") is not None:
            KP[i] = r["kp2d"]
        if r["status"] == ST_3D:
            J[i], BP[i], TR[i], valid[i] = r["joints"], r["body_pose"], r["transl"], True
        ms[i] = [r["ms"].get(x, 0.0) for x in ("vitpose", "predict", "fk", "total")]
    K = rows[-1][2].get("K", eng.K)
    K = K.numpy() if hasattr(K, "numpy") else np.asarray(K)
    meta = {**eng.info, "video": str(video), "realtime": bool(a.realtime), "crop": crop, "frames": len(frames),
            "processed": N, "valid_frac": round(float(valid.mean()), 3),
            "ms_median": dict(zip(("vitpose", "predict", "fk", "total"), np.round(np.median(ms, 0), 2).tolist())),
            "ms_p95_total": round(float(np.percentile(ms[:, 3], 95)), 2), "resets": eng.n_reset,
            "fps_wall": round(N / max(rows[-1][1], 1e-6), 2), "image_wh": [w, h]}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(out).with_suffix(".tmp.npz")
    np.savez_compressed(tmp, joints_incam=J, kp2d_vitpose=KP, body_pose=BP, transl=TR, valid=valid,
                        source_frame_index=np.array([r[0] for r in rows]), t_done=np.array([r[1] for r in rows]),
                        ms=ms, K=np.asarray(K, np.float32), image_wh=np.array([w, h]), fps=fps, kind="stream",
                        meta=json.dumps(meta))
    tmp.replace(out)
    return meta


def light_check(a):
    """CUDA + the GEM-X / demo_webcam imports + the checkpoint on disk, without loading any model (a few seconds).
    -> (info dict, exit code)."""
    if getattr(a, "replay", ""):
        return {"device": "replay (no GPU)", "track": a.replay}, 0
    try:
        import torch
    except Exception as e:
        return {"error": f"torch import failed: {e}"}, EXIT_NO_GEMX
    if not torch.cuda.is_available():
        return {"error": "no CUDA device (torch.cuda.is_available() is False)", "torch": torch.__version__}, EXIT_NO_CUDA
    info = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)}
    perception = str(Path(a.perception).expanduser().resolve())
    if not (Path(perception) / "demo_webcam.py").exists():
        return {**info, "error": f"GEM-X perception folder (demo_webcam.py) not found: {perception}"}, EXIT_NO_GEMX
    sys.path.insert(0, perception)
    try:
        import demo_webcam  # noqa: F401  (GEM-X + gem imports, CUDA tensors at import)
    except Exception as e:
        return {**info, "error": f"GEM-X import failed: {type(e).__name__}: {e}"}, EXIT_NO_GEMX
    if a.ckpt:
        ok = Path(a.ckpt).expanduser().exists()
        info["ckpt"] = a.ckpt
    else:   # GEM-X's default checkpoint, from the Hugging Face cache only (no download here)
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            from gem.utils.hf_utils import download_checkpoint

            info["ckpt"] = str(download_checkpoint())
            ok = Path(info["ckpt"]).exists()
        except Exception as e:
            info["ckpt_error"] = f"{type(e).__name__}: {e}"
            ok = False
    if not ok:
        return {**info, "error": "GEM-X checkpoint not on disk (run the worker once online to download it)"}, \
            EXIT_NO_GEMX
    return info, 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--serve", action="store_true", help="stdin/stdout worker for k1live/sender_gemx.py")
    m.add_argument("--dump", metavar="VIDEO", nargs="+", help="process every frame of each VIDEO -> --out (one .npz, "
                   "or a folder: <out>/<stem><suffix>.npz)")
    m.add_argument("--check", action="store_true", help="load everything, print JSON, exit 0 / 5 / 6")
    ap.add_argument("--out", default="gemx_stream.npz")
    ap.add_argument("--suffix", default="", help="--dump into a folder: appended to each output stem (e.g. _rt)")
    ap.add_argument("--realtime", action="store_true", help="--dump at the clip's frame rate, dropping frames")
    ap.add_argument("--perception", default=DEFAULT_PERCEPTION,
                    help="GEM-X perception folder (demo_webcam.py, GEM-X/ inside)")
    ap.add_argument("--ckpt", default="", help="GEM-X checkpoint (default: GEM-X's own / Hugging Face cache)")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--min-window", type=int, default=16)
    ap.add_argument("--hfov", type=float, default=0.0, help="camera horizontal FOV in degrees (0 = GEM-X estimate_K)")
    ap.add_argument("--detector", default="auto", choices=["auto", "yolox", "vitpose"])
    ap.add_argument("--no-flip-test", action="store_true", help="single-pass VitPose (demo_webcam.py live: -3.5 ms, loses "
                    "the upper body more often)")
    ap.add_argument("--box-kp-only", action="store_true", help="demo_webcam.py's box propagation (confident keypoints "
                    "only) instead of keypoints + the GEM-X skeleton")
    ap.add_argument("--crop", default="", help="--dump: person region x0,y0,x1,y1 (0..1)")
    ap.add_argument("--max-h", type=int, default=1080, help="--dump: downscale frames taller than this")
    ap.add_argument("--replay", default="", help="--serve / --check from a saved track instead of the GPU (tests)")
    ap.add_argument("--light", action="store_true", help="--check without loading the models (CUDA, imports, files)")
    a = ap.parse_args(argv)
    if a.serve:
        return serve(a)
    if a.dump:
        return dump(a)
    if a.light:
        info, code = light_check(a)
        print(json.dumps(info))
        return code
    t0 = time.time()
    eng, code, err = make_engine(a, lambda *x: print(*x, file=sys.stderr, flush=True))
    if eng is not None:
        eng.warm_up()
        eng.info["ready_s"] = round(time.time() - t0, 1)
    print(json.dumps(eng.info if eng is not None else {"error": err, "code": code}))
    return code


if __name__ == "__main__":
    sys.exit(main())
