"""Pick the BONES-SEED takes for the K1 human-encoder data (stdlib only: reads metadata/seed_metadata_v004.csv).

    python sonic_k1_train/data/bones_to_k1.py select --meta <bones>/metadata/seed_metadata_v004.csv --out <set>

The demo needs what a visitor does standing in front of a camera with the whole body in view: standing, gestures,
communication / pointing, small steps, walking forward / backward / sideways, turning, starting / stopping.
Rules (each take is judged on its metadata; a take and its mirror are kept or dropped together):
  1. body position "standing" (or "standing, leaning"), no props (content_props = 0)
  2. no DROP word (whole-token prefix: floor work, jumps, jogging / running, kicks, squats / crouches, falls,
     furniture, vehicles, acrobatics, dances, injured / drunk styles; superset of gear_sonic
     filter_and_copy_bones_data.py) in the file name / content name / movement type / short descriptions,
     no injured / old style
  3. 1.5 s <= duration <= 40 s
  4. group by content_type_of_movement: walk = walking / turning / transition, gesture = gesture / action /
     pointing (+ category Gestures / Communication / Looking and Pointing), idle = standing idle / standing / *idle*;
     anything else is dropped. (v004 facts: content_vertical_move is 1 for ~90 % of gestures, so it is not a jump
     flag; category "Baseline" holds jumps / jogs / dances, not idles - neither is used.)
  5. budget: at most --max-hours (mirrors included), split walk 45 % / gesture 40 % / idle 15 %,
     random (seeded) inside each group -> spreads over actors and styles; half of the gesture budget goes
     round-robin over event-style keywords (PRIORITY: wave, greet, clap, bow, point, cheer, salute, ...)
Output: <set>/selection.json (clips with relative g1 / bvh paths, group, seconds) and <set>/selection.md (counts per
rule, per group, top values of the classification columns) - the record of what was chosen and why.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

FPS_SRC = 120.0
# whole-token prefixes ("jump" drops jumping / jumps, "sit" drops sitting but not "position"); "a_b" = phrase
DROP = ["floor", "ground", "lying", "lie", "prone", "supine", "kneel", "sit", "seat", "chair", "bench", "bed", "couch",
        "sofa", "table", "desk", "crawl", "climb", "ladder", "stair", "step_on", "stepdown", "step_up", "step_down",
        "box", "jump", "hop", "leap", "skip", "run", "jog", "sprint", "dash", "kick", "squat", "lunge", "crouch",
        "fall", "trip", "stumble", "roll", "flip", "cartwheel", "handstand", "acrobat", "somersault", "push_up",
        "pushup", "plank", "burpee", "obstacle", "bike", "scooter", "ride", "crutch", "cane", "skate", "swim", "fight",
        "punch", "sword", "gun", "shoot", "rifle", "kozak", "on_box", "on_1m", "off_1m", "edge", "slope", "ramp",
        "mohak", "all_fours", "dance", "dancing", "zombie", "drunk", "injur", "inj", "limp", "baby", "diaper",
        "reaching_down", "stoop", "calf", "ankle", "foot", "shoe", "toe", "one_foot", "one_leg", "puke", "vomit"]
# event-style gestures: half of the gesture budget is filled round-robin over these (rare ones like bow / wave first)
PRIORITY = ["wave", "waving", "greet", "hello", "bye", "clap", "applau", "bow", "point", "welcome", "cheer", "triumph",
            "heart", "thumb", "salute", "celebrat", "victory", "beckon", "come", "shrug", "nod", "present", "invit",
            "thank"]
PRIORITY_SHARE = 0.5
DROP_STYLE = ["injur", "old", "drunk", "zombie"]
POSITIONS = ("standing", "standing, leaning")
WALK_TYPES = ("walking", "turning", "walking, turning", "transition")
GESTURE_TYPES = ("gesture", "action", "pointing", "standing, pointing", "standing, gesture", "interacting")
IDLE_TYPES = ("standing idle", "standing")
GESTURE_CAT = ("gestures", "communication", "looking and pointing")
SHARES = {"walk": 0.45, "gesture": 0.40, "idle": 0.15}


def _tokens(r, *cols):
    txt = " ".join(str(r.get(c, "") or "") for c in cols).lower()
    toks = [t for t in re.split(r"[^a-z0-9]+", txt) if t]
    return toks, "_" + "_".join(toks) + "_"


def _hit(keys, toks, joined):
    for k in keys:
        if "_" in k:
            if f"_{k}_" in joined or f"_{k}" in joined:
                return k
        elif any(t.startswith(k) for t in toks):
            return k
    return None


def _zero(v):
    return str(v or "").strip().lower() in ("", "0", "0.0", "none", "nan", "false", "no")


def classify(r):
    """-> (group | None, reason). r = one metadata row (dict of strings). Note: content_vertical_move is 1 for most
    gestures in v004 (not a jump flag), category "Baseline" holds jumps / jogs / dances -> both unused."""
    pos = str(r.get("content_body_position", "") or "").strip().lower()
    if pos not in POSITIONS:
        return None, f"position:{pos}"
    if not _zero(r.get("content_props")):
        return None, "props"
    move = str(r.get("content_type_of_movement", "") or "").strip().lower()
    toks, joined = _tokens(r, "filename", "content_name", "content_type_of_movement", "content_short_description",
                           "content_short_description_2")
    k = _hit(DROP, toks, joined)
    if k:
        return None, f"keyword:{k}"
    stoks, sjoined = _tokens(r, "content_uniform_style", "content_all_rigplay_styles")
    k = _hit(DROP_STYLE, stoks, sjoined)
    if k:
        return None, f"style:{k}"
    try:
        sec = float(r.get("move_duration_frames") or 0) / FPS_SRC
    except ValueError:
        return None, "duration:?"
    if not 1.5 <= sec <= 40.0:
        return None, "duration"
    cat = str(r.get("category", "") or "").strip().lower()
    if move in WALK_TYPES:
        return "walk", "ok"
    if move in GESTURE_TYPES or (cat in GESTURE_CAT and move in ("", "walking")):
        return "gesture", "ok"
    if move in IDLE_TYPES or "idle" in toks:
        return "idle", "ok"
    return None, f"type:{move or '?'}"


def base_name(r):
    f = str(r.get("filename", ""))
    return f[:-2] if f.endswith("_M") else f


def main(meta, out, max_hours=10.0, seed=0):
    meta, out = Path(meta), Path(out)
    if meta.suffix == ".parquet":  # the CSV next to it has the same rows (no pyarrow / pandas needed)
        meta = meta.with_suffix(".csv")
    csv.field_size_limit(sys.maxsize)
    with open(meta, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    reasons, cols = Counter(), defaultdict(Counter)
    takes = defaultdict(list)
    for r in rows:
        for c in ("package", "category", "content_type_of_movement", "content_body_position", "content_uniform_style"):
            cols[c][str(r.get(c, ""))] += 1
        g, why = classify(r)
        reasons[why.split(":")[0] if g is None else f"kept:{g}"] += 1
        if g is not None:
            takes[(g, base_name(r))].append(r)
    by_group = defaultdict(list)
    for (g, b), rs in takes.items():
        if len(rs) == 2 or len(rs) == 1:   # take (+ its mirror)
            by_group[g].append(rs)
    rng = random.Random(seed)
    clips, budget = [], max_hours * 3600.0
    summary = {}
    for g, share in SHARES.items():
        pool = by_group[g]
        rng.shuffle(pool)
        order, prio_n = pool, 0
        if g == "gesture":  # round-robin over the event-style keywords first, then everything else at random
            buckets = defaultdict(list)
            for rs in pool:
                toks, joined = _tokens(rs[0], "filename", "content_name", "content_short_description")
                buckets[_hit(PRIORITY, toks, joined)].append(rs)
            rest = buckets.pop(None, [])
            prio, prio_left = [], budget * share * PRIORITY_SHARE
            while prio_left > 0 and any(buckets.values()):
                for k in sorted(buckets):
                    if buckets[k] and prio_left > 0:
                        rs = buckets[k].pop()
                        prio.append(rs)
                        prio_left -= sum(float(r["move_duration_frames"]) / FPS_SRC for r in rs)
            prio_n = len(prio)
            order = prio + rest + [rs for b in buckets.values() for rs in b]
        left, n = budget * share, 0
        for rs in order:
            sec = sum(float(r["move_duration_frames"]) / FPS_SRC for r in rs)
            if sec > left:
                continue
            left -= sec
            n += 1
            for r in rs:
                date = str(r.get("take_date", "") or "").split(".")[0]
                clips.append({"key": r["filename"], "group": g, "seconds": round(float(r["move_duration_frames"]) / FPS_SRC, 2),
                              "g1": r.get("move_g1_path") or r.get("move_g1_mujoco_path") or f"g1/csv/{date}/{r['filename']}.csv",
                              "bvh": r.get("move_soma_uniform_path") or f"soma_uniform/bvh/{date}/{r['filename']}.bvh",
                              "category": r.get("category", ""), "movement": r.get("content_type_of_movement", ""),
                              "actor": r.get("actor_uid", ""), "desc": r.get("content_short_description", "")})
        summary[g] = {"takes_available": len(pool), "takes_picked": n, "event_style_takes": prio_n,
                      "hours_picked": round((budget * share - left) / 3600, 2)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps({"meta": str(meta), "max_hours": max_hours, "seed": seed,
                                                    "summary": summary, "clips": clips}, indent=0))
    md = [f"# BONES-SEED selection ({meta.name}, {len(rows)} rows)", "", "| rule outcome | rows |", "|---|---:|"]
    md += [f"| {k} | {v} |" for k, v in reasons.most_common()]
    md += ["", "| group | takes available | takes picked | hours (with mirrors) |", "|---|---:|---:|---:|"]
    md += [f"| {g} | {s['takes_available']} | {s['takes_picked']} | {s['hours_picked']} |" for g, s in summary.items()]
    for c, cnt in cols.items():
        md += ["", f"## {c} (top 25 of {len(cnt)})", ""] + [f"- {k or '(empty)'}: {v}" for k, v in cnt.most_common(25)]
    mv = Counter(c["movement"] for c in clips)
    md += ["", "## picked: movement types (top 40)", ""] + [f"- {k}: {v}" for k, v in mv.most_common(40)]
    (out / "selection.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:40]))
    return clips


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 10.0)
