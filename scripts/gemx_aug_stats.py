"""GEM-X-like SMPL input augmentation for k1s_bones2: measure the live GEM-X stream, fit the model, check the augmenter.

What the K1 SMPL encoder sees live is the sender_gemx SMPL stream (scripts/gemx_feed.py SenderGemx: soma_det ->
JumpGate -> HumanMapper gemx -> mp2smpl) with live timing, through the sonic_human backend (staleness, PresenceGate,
30 ms smoothing). The BONES training SMPL contains the event clips themselves (offline GEM-X -> the same converter,
<set>/smpl/gemx_<clip>.pkl), so the per-capture error of the live stream is measurable (per
bone, per axis and over time), and the dropouts / re-detections are measurable on every dump.

    # GPU server (numpy + joblib; reads GEM-X real-time worker dumps, writes one small bundle)
    python scripts/gemx_aug_stats.py extract --dumps dumps/full dumps/stream \
        --train-smpl pkg/k1_bones_motions/smpl --bones-sample 120 --out stats/gemx_streams.npz
    # Mac
    python scripts/gemx_aug_stats.py fit --bundle gemx_streams.npz --json sonic_k1_train/data/gemx_aug/gemx_aug_target.json \
        --yaml sonic_k1_train/data/gemx_aug/gemx_aug_params.yaml \
        --config sonic_k1_train/overlay/gear_sonic/config/exp/manager/universal_token/all_modes/sonic_k1_small_la4_bones2.yaml
    python scripts/gemx_aug_stats.py check --bundle gemx_streams.npz --target sonic_k1_train/data/gemx_aug/gemx_aug_target.json \
        --params sonic_k1_train/data/gemx_aug/gemx_aug_params.yaml --json sonic_k1_train/data/gemx_aug/gemx_aug_check.json

Categories of the dumps (by name): full_stand (event_*, upper-body clips), full_walk (walking toward / sideways,
walking clips, the rehearsal loop), waist_up (the 4 Pexels gesture clips filmed from the waist up), turn (a full
turn in place: GEM-X loses a person who turns their back). Profiles of the augmenter: full <- paired event clips +
full_* dropouts, waist_up <- paired Pexels clips + waist_up dropouts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts", ROOT / "sonic_k1_train" / "mac", ROOT / "sonic_k1_train" / "data"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import smpl_aug_stats as S  # noqa: E402

AUG_PY = ROOT / "sonic_k1_train/overlay/gear_sonic/envs/manager_env/mdp/smpl_gemx_aug.py"
DT = 0.02
HUMAN_STALE_S = 0.25          # k1live.sim_server
WAIST_UP = {"pexels_10374292", "pexels_10149176", "pexels_10149187", "pexels_10515017"}
TURN = {"pexels_9558217"}
WALK = {"pexels_6764051", "pexels_3888260", "toward", "walk_in", "in_out", "k1live_rehearsal", "k1live_rehearsal_full1"}
PROFILE_OF = {"full_stand": "full", "full_walk": "full", "waist_up": "waist_up"}


def category(name: str) -> str:
    if name in WAIST_UP:
        return "waist_up"
    if name in TURN:
        return "turn"
    if name in WALK or name.startswith("k1live_rehearsal"):
        return "full_walk"
    return "full_stand"


def load_aug():
    spec = importlib.util.spec_from_file_location("smpl_gemx_aug", AUG_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------ extract (box)
def train_clip(path):
    """BONES SMPL PKL -> root-local joints (T,24,3), root quat (T,4) wxyz, fps (as scripts/gemx_smpl_match.py)."""
    import joblib

    from k1live import smpl_stream

    s = joblib.load(path)
    q = smpl_stream.root_quat_from_aa_yup(s["pose_aa"][:, :3])
    J = smpl_stream.world_joints(s["smpl_joints"], q * np.array([1.0, -1.0, -1.0, -1.0]))
    return J.astype(np.float32), q.astype(np.float32), float(s["fps"])


def live_stream(dump):
    """Every-frame GEM-X dump -> the sender_gemx SMPL stream with live timing + the backend's 50 Hz delivery mask."""
    import gemx_feed as GF
    from k1live import smpl_stream

    rows = GF.load_rows(dump).retime()
    fr = GF.SenderGemx(mirror=False, loco=False, smpl=True).frames(rows)
    n = len(fr)
    tc, ta = np.zeros(n), np.zeros(n)
    ok = np.zeros(n, bool)
    J, Q = np.full((n, 24, 3), np.nan, np.float32), np.full((n, 4), np.nan, np.float32)
    for i, f in enumerate(fr):
        tc[i], ta[i] = f["t_capture"], f["_t_arrive"]
        if f.get("smpl") is not None and f.get("present") and f.get("arm_q") is not None:
            j, q, _ = smpl_stream.from_msg(f["smpl"])
            J[i], Q[i], ok[i] = j, q, True
    # sim_server: the newest arrived frame drives the tick while it is fresh and present (human= only then)
    T = int(math.ceil((ta.max() if n else 0) / DT)) + 1
    tick = np.zeros(T, bool)
    k = -1
    for i in range(T):
        t = i * DT
        while k + 1 < n and ta[k + 1] <= t + 1e-9:
            k += 1
        tick[i] = k >= 0 and ok[k] and (t - tc[k]) < HUMAN_STALE_S
    return {"t": tc, "ok": ok, "J": J, "q": Q, "fps": rows.fps, "tick": tick}


def cmd_extract(a):
    out = {}
    meta = {"clips": {}, "bones": []}
    seen = set()
    for d in a.dumps:
        for p in ([Path(d)] if Path(d).is_file() else sorted(Path(d).glob("*.npz"))):
            name = p.stem
            if name in seen or (a.clips and name not in a.clips):
                continue
            seen.add(name)
            try:
                s = live_stream(p)
            except Exception as e:  # noqa: BLE001 - a broken / foreign npz: report and go on
                print(f"skip {p}: {e}", flush=True)
                continue
            for k, v in s.items():
                if k != "fps":
                    out[f"live/{name}/{k}"] = v
            info = {"dump": str(p), "fps": s["fps"], "category": category(name), "frames": int(len(s["t"])),
                    "found_frac": round(float(s["ok"].mean()), 3) if len(s["ok"]) else 0.0}
            tp = Path(a.train_smpl) / f"gemx_{name}.pkl"
            if tp.exists():
                J, q, fps = train_clip(tp)
                out[f"train/{name}/J"], out[f"train/{name}/q"] = J, q
                info["train_fps"] = fps
            meta["clips"][name] = info
            print(name, json.dumps(info), flush=True)
    if a.bones_sample:
        rng = np.random.default_rng(0)
        cand = sorted(p for p in Path(a.train_smpl).glob("*.pkl") if not p.name.startswith("gemx_"))
        pick = [cand[i] for i in sorted(rng.choice(len(cand), min(a.bones_sample, len(cand)), replace=False))]
        for p in pick:
            J, q, fps = train_clip(p)
            n = int(min(len(J), a.bones_max_s * fps))
            out[f"bones/{p.stem}/J"], out[f"bones/{p.stem}/q"] = J[:n], q[:n]
            meta["bones"].append({"key": p.stem, "fps": fps, "frames": n})
    out["meta"] = np.array(json.dumps(meta))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print("wrote", a.out, len(meta["clips"]), "streams,", len(meta["bones"]), "BONES clips")


# ------------------------------------------------------------------------------------------------ bundle access
class Bundle:
    def __init__(self, path):
        self.z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(self.z["meta"]))

    def live(self, name):
        g = lambda k: self.z[f"live/{name}/{k}"]
        return {k: g(k) for k in ("t", "ok", "J", "q", "tick")} | {"fps": self.meta["clips"][name]["fps"]}

    def train(self, name):
        k = f"train/{name}/J"
        return (self.z[k], self.z[f"train/{name}/q"]) if k in self.z.files else None

    def bones(self):
        for b in self.meta["bones"]:
            yield b["key"], self.z[f"bones/{b['key']}/J"], self.z[f"bones/{b['key']}/q"], b["fps"]


LAG_S = -0.14   # live (causal GEM-X) vs training (offline GEM-X) of the same capture: the best lag of the error-vs-lag
                # curve pooled over the 8 paired clips (curves are flat: +-2 % over +-0.5 s; per-clip minima of a flat
                # curve sit on the search limits or on repeated motions, so one physical lag is used for all clips)


def paired(live, train, fps_t=50.0, lag_s=LAG_S):
    """Delivered live frames vs the training frame at capture time + lag_s -> (live idx, train idx, shift_ms)."""
    ok, t = live["ok"], live["t"]
    k = np.round(t * fps_t).astype(int) + int(round(lag_s * fps_t))
    m = ok & (k >= 0) & (k < len(train[0]))
    if m.sum() < 10:
        return None
    return np.flatnonzero(m), k[m], round(1000.0 * lag_s)


def contiguous_runs(idx, max_step=1):
    """Split sorted frame indices into runs of consecutive frames."""
    if len(idx) == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) > max_step) + 1
    return np.split(idx, cut)


def hf_jitter(J, q, fps, win_s=0.2, min_s=1.0):
    """High-frequency jitter: RMS angle (deg) of each bone direction (in the root frame) around its centred moving
    average over win_s, per group, on runs of >= min_s; root tilt jitter likewise."""
    J = np.asarray(J, float)
    w = max(3, int(round(win_s * fps)) | 1)
    d = S.bone_dirs(J - J[:, :1])
    if len(d) < max(w, int(min_s * fps)):
        return None
    k = np.ones(w) / w
    sm = np.stack([np.stack([np.convolve(d[:, b, c], k, mode="valid") for c in range(3)], -1)
                   for b in range(d.shape[1])], 1)
    h = w // 2
    dv = d[h:len(d) - h]
    sm = sm / np.maximum(np.linalg.norm(sm, axis=-1, keepdims=True), 1e-9)
    ang = np.degrees(np.arccos(np.clip(np.sum(dv * sm, -1), -1, 1)))          # (T', 23)
    out = {g: float(np.sqrt(np.mean(ang[:, [j - 1 for j in js]] ** 2))) for g, js in S.BONE_GROUPS.items()}
    tl = S.tilt_deg(q)
    tls = np.convolve(tl, k, mode="valid")
    out["tilt"] = float(np.sqrt(np.mean((tl[h:len(tl) - h] - tls) ** 2)))
    out["_n"] = int(len(ang))
    return out


def pool(items):
    """Frame-weighted mean of dicts of floats (keys without '_')."""
    items = [x for x in items if x]
    if not items:
        return {}
    w = np.array([x["_n"] for x in items], float)
    return {k: round(float(np.average([x[k] for x in items], weights=w)), 2) for k in items[0] if not k.startswith("_")}


def jumps(live, idx_runs, fps):
    """Re-detection jumps on the capture timeline: last delivered frame before a gap vs the first after it."""
    J, q = live["J"], live["q"]
    out = []
    for a, b in zip(idx_runs[:-1], idx_runs[1:]):
        i, j = a[-1], b[0]
        gap_s = (j - i - 1) / fps
        Ji, Jj = J[i] - J[i][:1], J[j] - J[j][:1]
        dq = S.qmul(S.qconj(q[i][None]), q[j][None])[0]
        out.append({"gap_s": gap_s, "joint_cm": float(100 * np.linalg.norm(Jj - Ji, axis=-1).mean()),
                    "wrist_cm": float(100 * np.linalg.norm(Jj[[20, 21]] - Ji[[20, 21]], axis=-1).mean()),
                    "root_deg": float(np.degrees(np.linalg.norm(S.q_to_rotvec(dq))))})
    return out


def jump_summary(js):
    if not js:
        return {"n": 0}
    a = lambda k, p: round(float(np.percentile([x[k] for x in js], p)), 2)
    return {"n": len(js), "joint_cm_p50_p90": [a("joint_cm", 50), a("joint_cm", 90)],
            "wrist_cm_p50_p90": [a("wrist_cm", 50), a("wrist_cm", 90)], "root_deg_p50_p90": [a("root_deg", 50), a("root_deg", 90)]}


# ------------------------------------------------------------------------------------------------ fit (Mac)
DUPLICATES = {"open_arms": "event_open_arms", "raise_arms": "event_raise_arms"}   # earlier dumps of the same videos


def measure_live(B: Bundle):
    names = sorted(n for n in B.meta["clips"] if DUPLICATES.get(n) not in B.meta["clips"])
    prof_clips = {"full": [], "waist_up": []}
    prof_post = {"full": [], "waist_up": []}
    per_clip, gaps_cat, tracked_cat, jumps_cat, hf_cat = {}, {}, {}, {}, {}
    for n in names:
        cat = B.meta["clips"][n]["category"]
        L = B.live(n)
        fps = float(L["fps"])
        g = S.gaps(L["tick"], DT)
        gaps_cat.setdefault(cat, []).append(g)
        tracked_cat[cat] = tracked_cat.get(cat, 0.0) + float(L["tick"].sum()) * DT
        runs = contiguous_runs(np.flatnonzero(L["ok"]))
        jumps_cat.setdefault(cat, []).extend(jumps(L, runs, fps))
        hf_cat.setdefault(cat, []).extend(hf_jitter(L["J"][r], L["q"][r], fps) for r in runs)
        rec = {"category": cat, "fps": fps, "tick_tracked_s": round(float(L["tick"].sum()) * DT, 1),
               "gaps": len(g), "gap_s_max": round(max(g), 2) if g else 0.0}
        tr = B.train(n)
        if tr is not None and cat in PROFILE_OF and len(runs):
            pm = paired(L, tr)
            if pm is not None:
                li, ti, shift = pm
                res = S.residual(L["J"][li], L["q"][li], tr[0][ti], tr[1][ti])
                prof_clips[PROFILE_OF[cat]].append({"res": res, "dt": 1.0 / fps, "name": n})
                rec.update(shift_ms=shift, joint_cm=round(float(res["joint_cm"].mean()), 2),
                           wrist_cm=round(float(res["joint_cm"][:, [20, 21]].mean()), 2))
                # error right after a re-detection (first 0.5 s of each run after the first) vs the rest
                pos = {int(v): k for k, v in enumerate(li)}
                post = np.zeros(len(li), bool)
                for r in runs[1:]:
                    for v in r[: int(0.5 * fps)]:
                        if int(v) in pos:
                            post[pos[int(v)]] = True
                bm = res["bone_rv"].mean(0)
                e = np.degrees(np.sqrt(np.sum((res["bone_rv"] - bm) ** 2, -1)).mean(-1))
                prof_post[PROFILE_OF[cat]].append((e[post], e[~post]))
        per_clip[n] = rec
    return prof_clips, prof_post, per_clip, gaps_cat, tracked_cat, jumps_cat, hf_cat


def cmd_fit(a):
    B = Bundle(a.bundle)
    prof_clips, prof_post, per_clip, gaps_cat, tracked_cat, jumps_cat, hf_cat = measure_live(B)
    target = {"per_clip": per_clip, "profiles": {}, "gaps": {}, "jumps": {}, "hf_jitter_deg": {}}
    for p, clips in prof_clips.items():
        if not clips:
            continue
        sm = S.summarize(clips, DT)
        sm["clip_names"] = [c["name"] for c in clips]
        post = prof_post[p]
        if post:
            e_post = np.concatenate([x[0] for x in post])
            e_rest = np.concatenate([x[1] for x in post])
            sm["post_redetect_bone_err_deg"] = [round(float(e_post.mean()), 2) if len(e_post) else None,
                                                round(float(e_rest.mean()), 2), int(len(e_post))]
        target["profiles"][p] = sm
    for cat, gl in gaps_cat.items():
        target["gaps"][cat] = S.gap_summary(gl, tracked_cat[cat])
        target["jumps"][cat] = jump_summary(jumps_cat[cat])
        target["hf_jitter_deg"][f"live_{cat}"] = pool(hf_cat[cat])
    hb = {"walk": [], "other": []}
    for key, J, q, fps in B.bones():
        hb["walk" if "walk" in key.lower() else "other"].append(hf_jitter(J, q, fps))
    for k, v in hb.items():
        target["hf_jitter_deg"][f"bones_{k}"] = pool(v)
    params = params_from_target(target)
    params, calib = calibrate(load_aug(), params, B, target, iters=a.iters, reps=a.reps)
    target["calibration"] = calib
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(target, indent=1))
    if a.yaml:
        import yaml

        params = json.loads(json.dumps(params, default=float))
        Path(a.yaml).write_text(yaml.safe_dump({"aug": params}, sort_keys=False))
        if a.config:
            write_config(Path(a.config), params)
    print(json.dumps({"profiles": {p: {k: v for k, v in d.items() if not k.endswith("_acf")}
                                   for p, d in target["profiles"].items()}, "gaps": target["gaps"]}, indent=1))


ARM_GROUPS = ("collar", "upper_arm", "forearm", "hand")
ARM_SCALE = 0.35     # arm bias / slow error kept at 0.35 x GEM-X: live arms are scored against the dir-IK goal built
                     # from the same GEM-X estimate, so an unobservable per-episode arm offset would only teach the
                     # encoder to shrink arm motions; legs, pelvis, dropouts and timing stay at the measured level


def write_config(path: Path, params: dict):
    """Replace the `smpl_gemx_aug:` block (up to the `defaults:` list) of the training experiment yaml."""
    import yaml

    text = path.read_text()
    i0, i1 = text.index("smpl_gemx_aug:\n"), text.index("\ndefaults:")
    block = yaml.safe_dump({"smpl_gemx_aug": params}, sort_keys=False, default_flow_style=None, width=120)
    path.write_text(text[:i0] + block + text[i1:])


def params_from_target(T: dict, bias_scale: float = 1.0, noise_scale: float = 1.0, arm_scale: float = ARM_SCALE) -> dict:
    """The augmenter's `aug` block from the measured target (gaps: full <- full_stand + full_walk)."""
    aug = load_aug()
    out = {"enable": True, "prob": 0.8, "camera_hz": [24.0, 30.0], "smooth_tau": 0.03, "hold_s": 0.5, "quick_s": 0.25,
           "wrist_newest": True, "lowpass_tau": dict(aug.DEFAULT["lowpass_tau"]),
           "root_lowpass_tau": aug.DEFAULT["root_lowpass_tau"], "noise_smooth_tau": aug.DEFAULT["noise_smooth_tau"],
           "profiles": {}}
    gap_src = {"full": ["full_stand", "full_walk"], "waist_up": ["waist_up"]}
    weights = {"full": 0.75, "waist_up": 0.25}
    for p, sm in T["profiles"].items():
        # a fast part below 5 % of the variance is dropped: the live high-frequency jitter (0.1-0.7 deg) leaves no room
        ff_b, ff_r = (f if f >= 0.05 else 0.0 for f in (sm["bone_fast_frac"], sm["root_fast_frac"]))
        slow, fast = {}, {}
        gs = {g: (arm_scale if g in ARM_GROUPS else 1.0) for g in sm["bone_noise_deg"]}
        for g, v in sm["bone_noise_deg"].items():
            s, f = S.split_slow_fast(v * noise_scale * gs[g], ff_b)
            slow[g], fast[g] = round(s, 2), round(f, 2)
        rs, rf = zip(*(S.split_slow_fast(v * noise_scale, ff_r) for v in sm["root_noise_deg"]))
        gl = [T["gaps"][c] for c in gap_src[p] if c in T["gaps"]]
        tracked = sum(g["tracked_s"] for g in gl)
        rate = sum(g.get("per_min_le_hold", 0.0) * g["tracked_s"] for g in gl) / max(tracked, 1e-6)
        mus = [(g["logn_mu"], g["logn_sd"], g["n"]) for g in gl if "logn_mu" in g]
        if mus:
            w = np.array([m[2] for m in mus], float)
            mu, sd = float(np.average([m[0] for m in mus], weights=w)), float(np.average([m[1] for m in mus], weights=w))
        else:
            mu, sd = aug.DEFAULT["profiles"][p]["gap_logn_mu"], aug.DEFAULT["profiles"][p]["gap_logn_sd"]
        out["profiles"][p] = {
            "weight": weights.get(p, 0.5),
            "bone_bias_deg": {g: round(v * bias_scale * gs[g], 2) for g, v in sm["bone_bias_deg"].items()},
            "bone_slow_deg": slow, "bone_fast_deg": fast,
            "tau_slow": sm["bone_tau_slow"], "tau_fast": sm["bone_tau_fast"],
            "root_bias_deg": [round(v * bias_scale, 2) for v in sm["root_bias_deg"]],
            "root_slow_deg": [round(v, 2) for v in rs], "root_fast_deg": [round(v, 2) for v in rf],
            "root_tau_slow": sm["root_tau_slow"], "root_tau_fast": sm["root_tau_fast"],
            "gap_per_min": round(rate, 2), "gap_logn_mu": round(mu, 3), "gap_logn_sd": round(sd, 3), "gap_min_s": 0.02,
            "redetect_resample": REDETECT.get(p, 0.3),
        }
    return out


REDETECT = {"full": 0.1, "waist_up": 0.4}   # P(GEM-X slow error re-drawn at a re-detection): live re-detection jumps
                                            # are small (full_walk p50 0.8 cm) - one-frame losses keep GEM-X's error


def target_scaled(sm: dict, arm_scale=ARM_SCALE) -> dict:
    """What the augmenter should reproduce: the live numbers, arm groups x arm_scale (see ARM_SCALE)."""
    f = lambda g: arm_scale if g in ARM_GROUPS else 1.0
    return {"bone_bias_deg": {g: v * f(g) for g, v in sm["bone_bias_deg"].items()},
            "bone_noise_deg": {g: v * f(g) for g, v in sm["bone_noise_deg"].items()},
            "root_bias_deg": list(sm["root_bias_deg"]), "root_noise_deg": list(sm["root_noise_deg"]),
            "bone_tau_slow": sm["bone_tau_slow"], "root_tau_slow": sm["root_tau_slow"]}


def paired_aug_summary(aug_mod, params, B: Bundle, prof: str, names, reps=16, seed=0):
    """Like-for-like: the augmenter on the clean training SMPL of the paired clips (noise pass: a camera frame every
    tick, no dropouts), sampled at the live capture indices, through the same estimator as the live stream."""
    cfg = single_profile(params, prof, prob=1.0, camera_hz=[50.0, 50.0])
    cfg["profiles"][prof]["gap_per_min"] = 0.0
    clips = []
    for i, n in enumerate(names):
        L, tr = B.live(n), B.train(n)
        li, ti, _ = paired(L, tr)
        rec = run_aug(aug_mod, cfg, [(tr[0], tr[1])], reps=reps, seed=seed + i)
        k = ti - 3                        # tick k's newest window frame is clean index k + F - 1
        for r in range(reps):
            m = (k >= 0) & (k < rec["cam_J"].shape[1])
            m[m] = np.isfinite(rec["cam_J"][r, k[m], 0, 0])     # tick 0 (restart) has no camera frame
            res = S.residual(rec["cam_J"][r, k[m]], rec["cam_q"][r, k[m]], tr[0][ti[m]], tr[1][ti[m]])
            clips.append({"res": res, "dt": 1.0 / float(L["fps"])})
    return S.summarize(clips, DT)


def _with(pp, bias=None, slow=None, tau=None, rtau=None):
    """A copy of profile params with bias / slow-noise scales (None = keep) and slow time constants."""
    q = json.loads(json.dumps(pp))
    if bias is not None:
        q["bone_bias_deg"] = {g: v * bias for g, v in q["bone_bias_deg"].items()}
        q["root_bias_deg"] = [v * bias for v in q["root_bias_deg"]]
    if slow is not None:
        for k in ("bone_slow_deg", "bone_fast_deg"):
            q[k] = {g: v * slow for g, v in q[k].items()}
        for k in ("root_slow_deg", "root_fast_deg"):
            q[k] = [v * slow for v in q[k]]
    if tau is not None:
        q["tau_slow"] = tau
    if rtau is not None:
        q["root_tau_slow"] = rtau
    return q


def calibrate(aug_mod, params, B: Bundle, T: dict, iters=3, reps=16):
    """Fit the generator so that the estimator on the augmented paired clips reproduces the (arm-scaled) live target.

    The estimator's bias / noise mix the generator's bias b and slow error s (a slow error's clip mean reads as bias,
    a bias on a moving bone reads as noise) and the motion itself (low-pass lag): measured^2 ~ m + alpha b^2 + c s^2
    (bias) and m' + gamma b^2 + d s^2 (noise), per bone group and root axis. Three runs (nothing / bias only / slow only)
    give m, alpha..d; a 2x2 solve per group gives b, s. The slow time constants are matched first (noise-only runs)."""
    params = json.loads(json.dumps(params, default=float))
    hist = {}
    run = lambda pp, p, names, seed: paired_aug_summary(aug_mod, dict(params, profiles={p: pp}), B, p, names, reps, seed)
    G_ = list(S.BONE_GROUPS)
    for p, sm in T["profiles"].items():
        tg = target_scaled(sm)
        names = sm["clip_names"]
        pp = params["profiles"][p]
        h = {"target_arm_scaled": tg}
        # 1) slow time constants (bias off)
        tau, rtau = pp["tau_slow"], pp["root_tau_slow"]
        for it in range(iters):
            a = run(_with(pp, bias=0.0, tau=tau, rtau=rtau), p, names, 10 + it)
            tau = float(np.clip(tau * np.clip(tg["bone_tau_slow"] / max(a["bone_tau_slow"], 1e-3), 0.5, 2.0), 0.1, 8.0))
            rtau = float(np.clip(rtau * np.clip(tg["root_tau_slow"] / max(a["root_tau_slow"], 1e-3), 0.5, 2.0), 0.1, 8.0))
        pp["tau_slow"], pp["root_tau_slow"] = round(tau, 3), round(rtau, 3)
        # 2) responses
        A0 = run(_with(pp, bias=0.0, slow=0.0), p, names, 20)
        AB = run(_with(pp, slow=0.0), p, names, 21)
        AS = run(_with(pp, bias=0.0), p, names, 22)

        def solve(tb, tn, m_b, m_n, Bb, Bn, Sb, Sn, b0, s0):
            al, ga = (Bb ** 2 - m_b ** 2) / max(b0 ** 2, 1e-9), (Bn ** 2 - m_n ** 2) / max(b0 ** 2, 1e-9)
            c, d = (Sb ** 2 - m_b ** 2) / max(s0 ** 2, 1e-9), (Sn ** 2 - m_n ** 2) / max(s0 ** 2, 1e-9)
            yb, yn = max(tb ** 2 - m_b ** 2, 0.0), max(tn ** 2 - m_n ** 2, 0.0)
            M = np.array([[max(al, 1e-6), max(c, 0.0)], [max(ga, 0.0), max(d, 1e-6)]])
            x = np.linalg.lstsq(M, np.array([yb, yn]), rcond=None)[0]
            if x[0] < 0:                    # no room for a bias: noise only (fit the noise row)
                x = np.array([0.0, yn / M[1, 1]])
            if x[1] < 0:
                x = np.array([yb / M[0, 0], 0.0])
            return math.sqrt(max(x[0], 0.0)), math.sqrt(max(x[1], 0.0))

        for g in G_:
            b0 = pp["bone_bias_deg"][g]
            s0 = math.hypot(pp["bone_slow_deg"][g], pp["bone_fast_deg"][g])
            b, sl = solve(tg["bone_bias_deg"][g], tg["bone_noise_deg"][g], A0["bone_bias_deg"][g], A0["bone_noise_deg"][g],
                          AB["bone_bias_deg"][g], AB["bone_noise_deg"][g], AS["bone_bias_deg"][g], AS["bone_noise_deg"][g],
                          max(b0, 1e-3), max(s0, 1e-3))
            f = sl / max(s0, 1e-9)
            pp["bone_bias_deg"][g] = round(b, 3)
            pp["bone_slow_deg"][g] = round(pp["bone_slow_deg"][g] * f, 3)
            pp["bone_fast_deg"][g] = round(pp["bone_fast_deg"][g] * f, 3)
        for k in range(3):
            b0 = pp["root_bias_deg"][k]
            s0 = math.hypot(pp["root_slow_deg"][k], pp["root_fast_deg"][k])
            b, sl = solve(tg["root_bias_deg"][k], tg["root_noise_deg"][k], A0["root_bias_deg"][k], A0["root_noise_deg"][k],
                          AB["root_bias_deg"][k], AB["root_noise_deg"][k], AS["root_bias_deg"][k], AS["root_noise_deg"][k],
                          max(b0, 1e-3), max(s0, 1e-3))
            f = sl / max(s0, 1e-9)
            pp["root_bias_deg"][k] = round(b, 3)
            pp["root_slow_deg"][k] = round(pp["root_slow_deg"][k] * f, 3)
            pp["root_fast_deg"][k] = round(pp["root_fast_deg"][k] * f, 3)
        fin = run(pp, p, names, 30)
        keep = ("bone_bias_deg", "bone_noise_deg", "root_bias_deg", "root_noise_deg", "bone_tau_slow", "root_tau_slow", "joint_cm")
        h.update(motion_only={k: A0[k] for k in keep}, final={k: fin[k] for k in keep})
        print(p, "final", json.dumps({k: fin[k] for k in keep}), flush=True)
        hist[p] = h
    return params, hist


# ------------------------------------------------------------------------------------------------ check (Mac)
def run_aug(aug_mod, cfg, clips, reps=4, seed=0, F=4):
    """Play clips [(J (T,24,3), q (T,4))] through SmplGemxAug, `reps` envs per clip. -> per env: clean newest frame,
    the camera frame (pre-gate, per tick, NaN when no new frame), the encoder's newest frame, the tick delivery mask."""
    import torch

    envs = [c for c in clips for _ in range(reps)]
    N = len(envs)
    T = max(len(c[0]) for c in envs)
    g = torch.Generator().manual_seed(seed)
    A = aug_mod.SmplGemxAug(N, F, cfg, generator=g)
    Jall = np.zeros((N, T + F, 24, 3), np.float32)
    qall = np.zeros((N, T + F, 4), np.float32)
    lens = np.array([len(c[0]) for c in envs])
    for i, (J, q) in enumerate(envs):
        n = len(J)
        Jall[i, :n], qall[i, :n] = J, q
        Jall[i, n:], qall[i, n:] = J[-1], q[-1]
    rec = {k: np.full((N, T) + s, np.nan, np.float32) for k, s in
           (("cam_J", (24, 3)), ("cam_q", (4,)), ("out_J", (24, 3)), ("out_q", (4,)))}
    rec["tick"] = np.zeros((N, T), bool)
    for k in range(T):
        idx = np.minimum(k + np.arange(F)[None], lens[:, None] - 1)
        J = torch.from_numpy(Jall[np.arange(N)[:, None], idx])
        q = torch.from_numpy(qall[np.arange(N)[:, None], idx])
        cont = torch.full((N,), k > 0)
        Jo, qo, _ = A.step(J, q, None, cont)
        nc = A.new_cam.numpy() & A.active.numpy()
        rec["cam_J"][nc, k] = A.cam_J.numpy()[nc]
        rec["cam_q"][nc, k] = A.cam_q.numpy()[nc]
        rec["out_J"][:, k], rec["out_q"][:, k] = Jo[:, -1].numpy(), qo[:, -1].numpy()
        rec["tick"][:, k] = ~A.in_gap.numpy()
    clean_idx = np.minimum(np.arange(T)[None] + F - 1, lens[:, None] - 1)
    rec["clean_J"] = Jall[np.arange(N)[:, None], clean_idx]
    rec["clean_q"] = qall[np.arange(N)[:, None], clean_idx]
    rec["lens"], rec["active"] = lens, A.active.numpy()
    return rec


def single_profile(params, p, **over):
    cfg = json.loads(json.dumps(params))
    cfg["profiles"] = {p: dict(cfg["profiles"][p], weight=1.0)}
    cfg.update(over)
    return cfg


def measure_aug(rec, hz_mode="tick"):
    """Same estimator as the live fit: per env, the noise pass (a camera frame every tick, no gaps) -> residual."""
    clips = []
    for i in range(len(rec["lens"])):
        if not rec["active"][i]:
            continue
        n = rec["lens"][i]
        m = np.isfinite(rec["cam_J"][i, :n, 0, 0])
        if m.sum() < 50:
            continue
        res = S.residual(rec["cam_J"][i, :n][m], rec["cam_q"][i, :n][m], rec["clean_J"][i, :n][m], rec["clean_q"][i, :n][m])
        clips.append({"res": res, "dt": DT})
    return S.summarize(clips, DT) if clips else {}


def cmd_check(a):
    import yaml

    aug = load_aug()
    B = Bundle(a.bundle)
    target = json.loads(Path(a.target).read_text())
    params = yaml.safe_load(Path(a.params).read_text())["aug"]
    bones = [(J, q) for _, J, q, _ in B.bones()]
    out = {"profiles": {}, "gaps": {}, "jumps": {}, "hf_jitter_deg": {}}
    for p in params["profiles"]:
        # noise pass: a new (corrupted) camera frame every tick, no dropouts -> per-capture residual vs the clean frame
        cfg = single_profile(params, p, prob=1.0, camera_hz=[50.0, 50.0])
        cfg["profiles"][p]["gap_per_min"] = 0.0
        rec = run_aug(aug, cfg, bones, reps=a.reps, seed=1)
        sm = measure_aug(rec)
        # high-frequency jitter on uniformly sampled frames, subsampled to 25 fps as the live captures
        hf = []
        for i in range(len(rec["lens"])):
            n = rec["lens"][i]
            m = np.isfinite(rec["cam_J"][i, :n, 0, 0])
            idx = np.flatnonzero(m)[::2]
            hf.append(hf_jitter(rec["cam_J"][i, idx], rec["cam_q"][i, idx], 25.0))
        out["hf_jitter_deg"][p] = pool(hf)
        out["profiles"][p] = {k: sm.get(k) for k in ("bone_bias_deg", "bone_noise_deg", "root_bias_deg", "root_noise_deg",
                                                     "bone_tau_slow", "bone_fast_frac", "root_tau_slow", "root_fast_frac",
                                                     "joint_cm")}
        # full pass: the live timing model -> dropouts, re-detection jumps (camera frames)
        rec = run_aug(aug, single_profile(params, p, prob=1.0), bones, reps=a.reps, seed=2)
        gl, tracked, js = [], 0.0, []
        for i in range(len(rec["lens"])):
            n = rec["lens"][i]
            tick = rec["tick"][i, :n]
            gl.append(S.gaps(tick, DT))
            tracked += float(tick.sum()) * DT
            m = np.isfinite(rec["cam_J"][i, :n, 0, 0])
            cam = {"J": rec["cam_J"][i, :n], "q": rec["cam_q"][i, :n]}
            idx = np.flatnonzero(m)
            # camera-frame runs: consecutive camera frames not separated by a gap tick
            gap_ticks = np.flatnonzero(~tick)
            runs, cur = [], [idx[0]] if len(idx) else []
            for u, v in zip(idx[:-1], idx[1:]):
                if np.any((gap_ticks > u) & (gap_ticks < v)):
                    runs.append(np.array(cur))
                    cur = [v]
                else:
                    cur.append(v)
            if cur:
                runs.append(np.array(cur))
            js.extend(jumps(cam, runs, 50.0))
        out["gaps"][p] = S.gap_summary(gl, tracked)
        out["jumps"][p] = jump_summary(js)
    out["target"] = {"profiles": {p: {k: v for k, v in d.items() if k in out["profiles"].get(p, {})}
                                  for p, d in target["profiles"].items()},
                     "gaps": target["gaps"], "jumps": target["jumps"], "hf_jitter_deg": target["hf_jitter_deg"]}
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--dumps", nargs="+", required=True, help="dump folders or files (first one wins per clip name)")
    e.add_argument("--clips", nargs="*", default=[])
    e.add_argument("--train-smpl", required=True)
    e.add_argument("--bones-sample", type=int, default=120)
    e.add_argument("--bones-max-s", type=float, default=20.0)
    e.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--bundle", required=True)
    f.add_argument("--json", required=True)
    f.add_argument("--yaml", default="")
    f.add_argument("--config", default="", help="training yaml whose smpl_gemx_aug block is rewritten")
    f.add_argument("--iters", type=int, default=3)
    f.add_argument("--reps", type=int, default=16)
    c = sub.add_parser("check")
    c.add_argument("--bundle", required=True)
    c.add_argument("--target", required=True)
    c.add_argument("--params", required=True)
    c.add_argument("--reps", type=int, default=2)
    c.add_argument("--json", required=True)
    a = ap.parse_args()
    {"extract": cmd_extract, "fit": cmd_fit, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
