"""Demo reel: person video | K1 in MuJoCo, the live chain run offline on recorded clips (EGL).

Per segment (the robot respawns at the origin and holds the clip's first arm pose for --preroll s before the cut):
the clip plays at its own frame rate as the fake camera -> perception -> k1live.human.HumanMapper (arm IK, waist, bow)
[+ walking intent for walking segments] -> 50 Hz: arm reference through the server's ArmLimiter, walking command
through VelocityLimiter.for_backend(backend) + the server's heading hold -> backend.step(). Mirror off (person and
robot side by side, both facing the viewer: same side = same arm).

Perception (--perception):
  mediapipe  MediaPipe Pose full (scripts/run_video.py cache), k1live.locomotion.LocoEstimator   (the reel so far)
  gemx       GEM-X SOMA skeleton per clip from --gemx-root (k1live.gemx.load_track: an offline GEM-X run
             <root>/<segment name>/soma_joints.npz, or a dump of the real-time worker k1live/gemx_worker.py --dump
             <root>/<segment name>.npz) -> k1live.gemx.soma_det (MediaPipe-33 layout: VitPose 2D for the frontal-plane
             arm angles, GEM-X 3D for depth) -> the same HumanMapper(perception="gemx");
             walking intent k1live.gemx.GemxLocoEstimator (camera-frame pelvis track). Orange SOMA skeleton.
  compare    both chains in lockstep on the same frames: person (MediaPipe green + GEM-X orange) | K1 <- MediaPipe |
             K1 <- GEM-X (3 panels)
The person panel carries a small tag naming the perception exactly: "MediaPipe", "GEM-X (오프라인)" (whole clip at
once, not causal) or "GEM-X (실시간 경로)" (the real-time worker's causal sliding window).

Segment modes (upper body first when standing, a clean gait when the legs move):
  arms    standing upper-body clip: split backend (student arms follow the person, loco legs stand; walking off)
  walk    walking clip: K1 SONIC walk3 5000 (kinematic-planner legs, person's arms laid over) — the natural gait;
          walking intent (profile --walk-profile, homed at the segment start like the operator's calibrate button)
  --backend hybrid  every segment on the hybrid backend (k1live/backends/hybrid.py): split while no walking
          command is held, K1 SONIC walk3 + planner while walking; arms segments keep walking intent off
  <name>  any other k1live backend name (split, sonic, ...)

Every segment is rendered with the same scene (split render model, qpos copied joint by joint) and the live follow
camera (k1live.render.FollowCam), so the robot looks the same whichever policy drives it. Labels are small: the
action name on the person side, "K1 · MuJoCo" on the robot side (no text about any institution).

    python scripts/demo_v2/render_reel.py --out videos/k1_demo_v2_reel.mp4 \
        --seg "wave=<clip>@0-9|손 흔들기|arms" --seg "toward=<clip>@1-12|걸어오기|walk|visitor" ...
    python scripts/demo_v2/render_reel.py --perception gemx --gemx-root logs/demo_v2/gemx ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DT = 0.02
ARM_VMAX = 6.0
HEADING_GAIN, HEADING_WMAX = 2.0, 0.4   # k1live.sim_server --heading-hold 2.0, HEADING_WMAX
ZERO = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
WALK3 = ROOT / "models" / "k1" / "k1s_walk3" / "deploy" / "k1s_walk3_005000"
# walking-intent settings per segment (k1live.locomotion.LocoEstimator profile + overrides)
LOCO = {"visitor": ("visitor", {}),                  # "floor joystick": step away from home -> walk
        "presenter": ("presenter", {}),              # walk along, stops ~1 s after the person stops (event default)
        # walk along for slow stock walk-ins (~0.2 m/s side-on): home follows faster, lower side threshold
        "presenter_slow": ("presenter", dict(home_follow_s=2.0, side_start=0.2, side_stop=0.1))}
MODE_TAG = {"arms": "상체 따라하기", "walk": "걷기 따라하기"}
PERCEPTION_TAG = {"mediapipe": "인식: MediaPipe", "mediapipe_crop": "인식: MediaPipe (사람 영역 지정)",
                  "offline": "인식: GEM-X (오프라인)", "stream": "인식: GEM-X (실시간 경로)"}
MP_GREEN, GEMX_ORANGE = (80, 220, 120), (40, 150, 255)   # BGR skeleton colours


def make(mode, walk_variant, backend=None, arm_mode="student"):
    """-> (backend, its name, walking intent on). backend="hybrid" drives every segment with the hybrid
    backend (it is the split backend while no walking command is held); arms segments still get no walking intent.
    arm_mode: "pd" = the PD arm path for split / hybrid standing (sim_server --arm-mode pd)."""
    from k1live.backends.base import make_backend

    if backend == "hybrid":
        return make_backend("hybrid", variant_dir=walk_variant, arm_mode=arm_mode), "hybrid", mode != "arms"
    if mode == "arms":
        return make_backend("split", arm_mode=arm_mode), "split", False
    if mode == "walk":
        return make_backend("sonic", variant_dir=walk_variant, walk=True), "sonic", True
    be = make_backend(mode)
    return be, mode, bool(be.supports.get("walk"))


class QposMap:
    """Copy a backend's qpos into the render model by joint name (free joint + hinges)."""

    def __init__(self, src_model, dst_model):
        import mujoco

        self.pairs = []
        for j in range(dst_model.njnt):
            name = dst_model.joint(j).name
            jid = mujoco.mj_name2id(src_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if dst_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                fj = [k for k in range(src_model.njnt) if src_model.jnt_type[k] == mujoco.mjtJoint.mjJNT_FREE][0]
                self.pairs.append((src_model.jnt_qposadr[fj], dst_model.jnt_qposadr[j], 7))
            elif jid >= 0:
                self.pairs.append((src_model.jnt_qposadr[jid], dst_model.jnt_qposadr[j], 1))

    def __call__(self, q_src, q_dst):
        for a, b, n in self.pairs:
            q_dst[b:b + n] = q_src[a:a + n]
        return q_dst


class Stage:
    """Render model (split scene) shared by every segment and chain; each chain has its own qpos map + camera."""

    def __init__(self, w, h):
        import mujoco

        from k1live.backends.split import build_render_model
        from k1live.render import FollowCam

        self.model = build_render_model()
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, w)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, h)
        self.data = mujoco.MjData(self.model)
        self.r = mujoco.Renderer(self.model, height=h, width=w)
        self.opt = mujoco.MjvOption()
        self.FollowCam = FollowCam

    def attach(self, be):
        """-> per-chain view state (qpos map + follow camera)."""
        return {"map": QposMap(be.model, self.model), "cam": self.FollowCam()}

    def grab(self, be, view, t):
        import mujoco

        view["map"](be.data.qpos, self.data.qpos)
        mujoco.mj_forward(self.model, self.data)
        self.r.update_scene(self.data, camera=view["cam"].update(self.data.qpos, t), scene_option=self.opt)
        return self.r.render().copy()

    def close(self):
        self.r.close()


# ------------------------------------------------------------------------------------------------ perception
class MPFrames:
    """MediaPipe landmarks of a clip (the live sender's PoseTracker, scripts/run_video.py cache)."""

    name, kind = "mediapipe", "mediapipe"

    def __init__(self, path, crop=None):
        import run_video as g1rv

        self.crop = crop   # the sender's person region (--crop), normalised x0, y0, x1, y1
        self.kind = "mediapipe_crop" if crop else "mediapipe"
        self.W, self.V, self.I, self.fps = g1rv.extract(path, "full", crop, 1, 1280)
        self.n = len(self.W)

    def det(self, k):
        if k < 0 or k >= self.n or np.isnan(self.W[k, 0, 0]):
            return None
        return {"world": self.W[k], "vis": self.V[k], "image": self.I[k]}

    def span_arrays(self):
        return self.I, self.V

    @staticmethod
    def draw(img, det):
        from sonic.pose_mp import draw_overlay

        return draw_overlay(img, det, color=MP_GREEN)


class GemxFrames:
    """GEM-X SOMA track of a clip (k1live.gemx.load_track) as per-frame k1live.human dets."""

    name = "gemx"

    def __init__(self, root, seg_name, path, fps, n, wh):
        from k1live.gemx import load_track

        root = Path(root)
        for cand in (root / seg_name, root / f"{seg_name}.npz", root / seg_name / "gemx_stream.npz",
                     root / Path(path).stem, root / f"{Path(path).stem}.npz"):
            if cand.exists() and (cand.is_file() or (cand / "soma_joints.npz").exists()):
                break
        else:
            raise SystemExit(f"no GEM-X track for segment {seg_name!r} under {root} "
                             f"(<name>/soma_joints.npz or <name>.npz)")
        self.src = str(cand)
        self.track = load_track(cand, n_video=n)
        self.kind = self.track.kind
        self.fps, self.n, self.wh = fps, n, wh
        if abs(self.track.fps - fps) > 0.5:
            print(f"[reel] {seg_name}: GEM-X track {self.track.fps} fps vs clip {fps} fps", flush=True)
        self.dets = [self.track.det(k, scale_to=wh) for k in range(n)]

    def det(self, k):
        return self.dets[k] if 0 <= k < self.n else None

    def span_arrays(self):
        I = np.full((self.n, 33, 3), np.nan)
        V = np.zeros((self.n, 33))
        for k, d in enumerate(self.dets):
            if d is not None:
                I[k], V[k] = d["image"], d["vis"]
        return I, V

    def draw(self, img, det):
        from k1live.gemx import draw_soma

        if det is None:
            return img
        h, w = img.shape[:2]
        uv = det["soma_uv"] * np.array([w / self.wh[0], h / self.wh[1]])
        return draw_soma(img, uv, color=GEMX_ORANGE)


def video_info(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return wh, fps, n


# ------------------------------------------------------------------------------------------------ one chain
class Chain:
    """Perception frames -> HumanMapper (+ walking intent) -> 50 Hz limits -> backend, with the live server's rules."""

    def __init__(self, frames, mode, loco, args, wh, stage):
        from k1live.human import HumanMapper
        from k1live.locomotion import LocoEstimator, VelocityLimiter
        from k1live.sim_server import ArmLimiter

        self.frames, self.mode, self.args, self.wh = frames, mode, args, wh
        self.be, self.bname, self.walks = make(mode, args.walk_variant, args.backend, args.arm_mode)
        self.lim = VelocityLimiter.for_backend(getattr(self.be, "limiter_preset", self.bname))
        self.hm = HumanMapper(mirror=False, loco=False, smpl=False, arm_ik=args.arm_ik, perception=frames.name)
        self.arm_lim = ArmLimiter(self.be.arm_default)  # the server's second-order arm reference
        prof, kw = LOCO[loco]
        if frames.name == "gemx":
            from k1live.gemx import GemxLocoEstimator as Est

            kw = {**kw, "dist_src": getattr(args, "gemx_dist", "pelvis")}
        else:
            Est = LocoEstimator
        self.est = Est(mirror=False, profile=prof, **kw) if self.walks else None
        self.view = stage.attach(self.be)
        self.v_cmd = dict(ZERO)
        self.yaw_ref = None
        self.phases = []
        self.log = {"t": [], "cmd": [], "lim": [], "xy": [], "yaw": [], "tilt": [], "err": []}
        self.fell = None
        self.last = (None, None)

    def preroll(self, k0, k1, vfps):
        """Settle on the first arm pose (and home the walking estimator) without rendering."""
        first = next((k for k in range(k0, k1) if self.frames.det(k) is not None), k0)
        det0 = self.frames.det(first)
        f0 = self.hm.process(det0, first / vfps, *self.wh)
        be = self.be
        arm = be.arm_default.copy()
        tgt0 = be.arm_default if f0["arm_q"] is None else np.asarray(f0["arm_q"])
        for _ in range(int(self.args.preroll / DT)):
            arm = arm + np.clip(tgt0 - arm, -ARM_VMAX * DT, ARM_VMAX * DT)
            be.step(arm, loco_cmd=(0.0, 0.0, 0.0))
        self.arm_lim.reset(arm)
        if self.est is not None and f0.get("obs") is not None:
            self.est.calibrate(f0["obs"])
        self.last = (det0, f0)
        self.x0, self.y0, _ = be.root_pose()
        return first

    def on_frame(self, k, vfps):
        det = self.frames.det(k)
        f = self.hm.process(det, k / vfps, *self.wh)
        self.last = (det, f)
        if self.est is not None:
            obs = f.get("obs")
            if obs is not None and f["present"]:
                c = self.est.update(obs)
                self.v_cmd = c if c is not None else dict(ZERO)
            elif self.args.loco_lost == "live":
                # live sender (k1live.human.HumanMapper loco): the estimator is not called on a frame without a person
                # (its lost_hold_s bridges short drop-outs); the server walks on 0 for that frame (present False)
                self.v_cmd = dict(ZERO)
            else:   # "reset" (reels before v4): update(None) has no timestamp -> the estimator resets its axes at once
                self.est.update(None)
                self.v_cmd = dict(ZERO)

    def tick(self, t):
        be, f = self.be, self.last[1]
        tgt = be.arm_default if (f is None or f["arm_q"] is None) else np.asarray(f["arm_q"])
        arm, arm_dq = self.arm_lim.step(tgt, vmax=self.args.arm_vmax, amax=self.args.arm_amax)
        v = self.lim(self.v_cmd if self.walks else ZERO, DT)
        walk_cmd = np.array(v, float)  # without the heading hold (hybrid walks on this, k1live.sim_server)
        _, _, yaw_now = be.root_pose()
        if v[2] != 0.0 or self.yaw_ref is None:  # k1live.sim_server heading hold: no turn commanded -> hold heading
            self.yaw_ref = yaw_now
        elif be.supports.get("walk"):
            v[2] = float(np.clip(HEADING_GAIN * ((self.yaw_ref - yaw_now + np.pi) % (2 * np.pi) - np.pi),
                                 -HEADING_WMAX, HEADING_WMAX))
        wy = f.get("waist_yaw") if (f is not None and be.supports.get("waist_yaw")) else None
        bw = f.get("bow") if (f is not None and be.supports.get("bow") and not self.walks) else None
        extra = {"walk_cmd": tuple(walk_cmd)} if be.supports.get("intent") else {}
        be.step(arm, waist_yaw=wy, bow=bw, loco_cmd=tuple(v), arm_dq=arm_dq, **extra)
        if getattr(be, "phase", "stand") != "stand":   # k1live.sim_server: hold the heading it ends up standing at
            self.yaw_ref = be.root_pose()[2]
        self.phases.append(getattr(be, "phase", self.mode))
        x, y, yaw = be.root_pose()
        L = self.log
        L["t"].append(t)
        L["cmd"].append([self.v_cmd["vx"], self.v_cmd["vy"], self.v_cmd["wz"]])
        L["lim"].append(list(walk_cmd))  # walking command before the heading hold
        L["xy"].append([x, y])
        L["yaw"].append(yaw)
        L["tilt"].append(be.tilt_deg())
        L["err"].append(float(np.degrees(np.abs(be.arm_q_now() - arm)).mean()))
        if self.fell is None and be.fallen():
            self.fell = round(t, 2)

    def tag_key(self):
        phase = getattr(self.be, "phase", None)
        return "walk" if (phase in ("walk", "settle") or (phase is None and self.mode == "walk")) else "arms"

    def result(self):
        self.be.close()
        L = {k: np.asarray(v_) for k, v_ in self.log.items()}
        walking = np.abs(L["lim"]).max(1) > 0.02
        ph = self.phases
        return {"perception": self.frames.kind, "backend": self.bname, "fell": self.fell is not None,
                "fell_at": self.fell, "max_tilt_deg": round(float(L["tilt"].max()), 1),
                "arm_err_deg": round(float(L["err"].mean()), 2),
                "moved_m": round(float(np.linalg.norm(L["xy"][-1] - [self.x0, self.y0])), 2),
                "path_m": round(float(np.linalg.norm(np.diff(L["xy"], axis=0), axis=1).sum()), 2),
                "turned_deg": round(float(np.degrees(np.ptp(np.unwrap(L["yaw"])))), 1),
                "walk_frac": round(float(walking.mean()), 3),
                "phase_frac": {p: round(ph.count(p) / max(len(ph), 1), 3) for p in sorted(set(ph))},
                "max_cmd": np.round(np.abs(L["lim"]).max(0), 2).tolist()}


# ------------------------------------------------------------------------------------------------ segment
def pill_overlay(ms, text, x, y, sc, size=22, weight="medium", alpha=200, right=None):
    im = ms.pill([(text, ms.font(weight, int(size * sc)), ms.INK)], pad=(int(16 * sc), int(9 * sc)),
                 radius=int(12 * sc), bar=None, bg=(255, 255, 255, alpha))
    if right is not None:
        x = right - im.size[0]
    return ms.Overlay(im, x, y)


def segment(spec, stage, size, fps_out, writer, overlays, args):
    import cv2
    import run_video as g1rv

    import make_showreel as ms

    name, rest = spec.split("=", 1)
    rest, _, caption = rest.partition("|")
    caption, _, mode = caption.partition("|")
    mode, _, loco = mode.partition("|")
    mode = mode or "arms"
    loco = loco or args.walk_profile
    path, _, rng = rest.partition("@")
    a, _, b = rng.partition("-")
    t0, t1 = float(a or 0), float(b) if b else None
    wh, vfps, nv = video_info(path)
    sources = []
    if args.perception in ("mediapipe", "compare"):
        mp = MPFrames(path)
        vfps, nv = mp.fps, mp.n
        sources.append(mp)
    if args.perception in ("gemx", "compare"):
        sources.append(GemxFrames(args.gemx_root, name, path, vfps, nv, wh))
    k0 = int(t0 * vfps)
    k1 = nv if t1 is None else min(nv, int(t1 * vfps))
    W_, H_ = size
    npan = 1 + len(sources)
    pw = W_ // npan
    chains = [Chain(s, mode, loco, args, wh, stage) for s in sources]
    I, V = sources[-1].span_arrays()   # framing from the chain shown (GEM-X when present)
    fit = g1rv.PanelFit(pw, H_, g1rv.person_span(I[k0:k1], V[k0:k1]))
    reader = g1rv.FrameReader(path, height=H_)
    sc = W_ / 1920.0
    lab = ms.Overlay(ms.pill([(caption, ms.font("semibold", int(26 * sc)), ms.INK)], pad=(int(18 * sc), int(10 * sc)),
                             radius=int(12 * sc), bar=None, bg=(255, 255, 255, 215)), int(32 * sc), int(28 * sc)) \
        if caption else None
    y_tag = int(28 * sc)
    tags = [{k: pill_overlay(ms, t, pw * (j + 1) + int(32 * sc), y_tag, sc) for k, t in MODE_TAG.items()}
            for j in range(len(chains))]
    y_bot = H_ - int(72 * sc)
    if len(chains) == 1:
        ptags = [pill_overlay(ms, PERCEPTION_TAG[sources[0].kind], int(32 * sc), y_bot, sc)]
    else:  # compare: person tag lists both skeletons, each robot panel says what drives it
        ptags = [pill_overlay(ms, "초록 MediaPipe · 주황 " + PERCEPTION_TAG[sources[1].kind].split(": ", 1)[1],
                              int(32 * sc), y_bot, sc)]
        for j, s in enumerate(sources):
            ptags.append(pill_overlay(ms, PERCEPTION_TAG[s.kind] + " → K1", pw * (j + 1) + int(32 * sc), y_bot, sc))
    credit = dict(c.split("=", 1) for c in args.credit).get(name)
    if credit:   # source attribution (CC BY clips) on the person panel, above the perception tag
        ptags.append(pill_overlay(ms, credit, int(32 * sc), y_bot - int(46 * sc), sc, size=17, weight="regular",
                                  alpha=185))
    firsts = [c.preroll(k0, k1, vfps) for c in chains]
    every = max(1, int(round(1.0 / (fps_out * DT))))
    k_last = min(firsts)
    n = int((k1 - k0) / vfps / DT)
    fade = int(args.fade * fps_out)
    n_out = (n + every - 1) // every
    frames = 0
    for i in range(n):
        t = i * DT
        k = k0 + int(t * vfps)
        if k != k_last and k < k1:
            k_last = k
            for c in chains:
                c.on_frame(k, vfps)
        for c in chains:
            c.tick(t)
        if i % every == 0:
            src = reader.get(k)
            if args.skeleton:
                for c, s in zip(chains, sources):
                    if c.last[0] is not None:
                        src = s.draw(src, c.last[0])
            panels = [cv2.cvtColor(fit(src), cv2.COLOR_BGR2RGB)]
            for c in chains:
                panels.append(stage.grab(c.be, c.view, t))
            img = np.concatenate(panels, axis=1)
            for ov in overlays:
                ov.blend(img)
            if lab is not None:
                lab.blend(img)
            for j, c in enumerate(chains):   # robot-side tag: what drives the robot right now (hybrid: its phase)
                tags[j][c.tag_key()].blend(img)
            for ov in ptags:
                ov.blend(img)
            g = min(1.0, (frames + 1) / max(fade, 1), (n_out - frames) / max(fade, 1)) if fade else 1.0
            if g < 1.0:
                img = (img.astype(np.float32) * g).astype(np.uint8)
            writer.append_data(img)
            frames += 1
    res = [c.result() for c in chains]
    out = {"segment": name, "clip": path, "mode": mode, "loco": loco if chains[0].walks else None, "t0": t0, "t1": t1,
           "caption": caption, "credit": credit, "seconds": round(n * DT, 1), "frames": frames,
           "gemx_track": next((s.src for s in sources if s.name == "gemx"), None)}
    if len(res) == 1:
        out.update(res[0])
    else:
        out["chains"] = res
        out["fell"] = any(r["fell"] for r in res)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", action="append", required=True, help="name=path@t0-t1|caption|mode")
    ap.add_argument("--out", default=str(ROOT / "videos" / "k1_demo_v2_reel.mp4"))
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=25, help="output rate; 50 Hz control / fps must be an integer")
    ap.add_argument("--walk-variant", default=str(WALK3))
    ap.add_argument("--walk-profile", default="presenter", choices=list(LOCO), help="default for walk segments "
                    "(per segment: name=path@t0-t1|caption|mode|profile)")
    ap.add_argument("--backend", default=None, choices=[None, "hybrid"], help="hybrid: every segment on the "
                    "hybrid backend (split standing, SONIC walk3 + planner walking); default: split / sonic per segment")
    ap.add_argument("--perception", default="mediapipe", choices=["mediapipe", "gemx", "compare"],
                    help="person -> landmarks: MediaPipe (live default), GEM-X tracks from --gemx-root, or both side "
                    "by side (3 panels)")
    ap.add_argument("--gemx-root", default=str(ROOT / "logs" / "demo_v2" / "gemx"),
                    help="GEM-X tracks: <root>/<segment>/soma_joints.npz (offline) or <root>/<segment>.npz (stream)")
    ap.add_argument("--gemx-dist", default="pelvis", choices=["pelvis", "ground"],
                    help="GEM-X walking distance (k1live.gemx.GemxLocoEstimator dist_src; sender --gemx-dist)")
    ap.add_argument("--preroll", type=float, default=1.0)
    ap.add_argument("--arm-ik", default="shape18", choices=["shape18", "dir"],
                    help="person arm IK (sender --arm-ik; dir = the bone-direction IK)")
    ap.add_argument("--arm-mode", default="student", choices=["student", "pd"],
                    help="split / hybrid standing arms (sim_server --arm-mode; pd = the PD arm path)")
    ap.add_argument("--arm-vmax", type=float, default=ARM_VMAX, help="arm reference speed limit (sim_server --arm-vmax)")
    ap.add_argument("--arm-amax", type=float, default=80.0, help="arm reference accel limit (sim_server --arm-amax)")
    ap.add_argument("--fade", type=float, default=0.25, help="fade in/out per segment, seconds")
    ap.add_argument("--loco-lost", default="live", choices=["live", "reset"],
                    help="frame without a person: live = as the live sender (estimator not called, its lost_hold_s "
                    "bridges short drop-outs; walk on 0 for that frame), reset = the reels before v4 (update(None) "
                    "resets the walking axes at once)")
    ap.add_argument("--credit", action="append", default=[], metavar="NAME=TEXT",
                    help="source credit shown on the person panel of segment NAME (CC BY attribution)")
    ap.add_argument("--no-skeleton", dest="skeleton", action="store_false", help="no skeleton on the person")
    a = ap.parse_args()
    import imageio.v2 as imageio
    from PIL import Image

    import make_showreel as ms

    size = tuple(int(x) for x in a.size.split("x"))
    W_, H_ = size
    sc = W_ / 1920.0
    npan = 3 if a.perception == "compare" else 2
    pw = W_ // npan
    ovs = [pill_overlay(ms, "K1 · MuJoCo 시뮬레이션", 0, int(28 * sc), sc, right=W_ - int(32 * sc))]
    for j in range(1, npan):
        ovs.append(ms.Overlay(Image.new("RGBA", (4, H_), (255, 255, 255, 255)), pw * j - 2, 0))
    stage = Stage(pw, H_)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(a.out, fps=a.fps, quality=8, macro_block_size=1)
    t0 = time.perf_counter()
    res = []
    for spec in a.seg:
        r = segment(spec, stage, size, a.fps, writer, ovs, a)
        print(json.dumps(r, default=float, ensure_ascii=False), flush=True)
        res.append(r)
    writer.close()
    stage.close()
    out = {"video": a.out, "backend": a.backend or "per segment (arms: split, walk: sonic)", "walk_variant": a.walk_variant,
           "walk_profile": a.walk_profile, "mirror": False, "perception": a.perception,
           "gemx_root": a.gemx_root if a.perception != "mediapipe" else None,
           "gemx_dist": a.gemx_dist if a.perception != "mediapipe" else None,
           "arm_ik": a.arm_ik, "arm_mode": a.arm_mode, "arm_vmax": a.arm_vmax, "arm_amax": a.arm_amax,
           "loco_lost": a.loco_lost,
           "segments": res, "seconds": round(sum(r["frames"] for r in res) / a.fps, 1),
           "falls": sum(r["fell"] for r in res), "wall_s": round(time.perf_counter() - t0, 1)}
    Path(a.out).with_suffix(".json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("video", "seconds", "falls", "wall_s")}))


if __name__ == "__main__":
    main()
