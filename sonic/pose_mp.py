"""MediaPipe PoseLandmarker wrapper (tasks API) with simple single-person tracking."""
from __future__ import annotations

import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from .params import MODELS

BONES = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (23, 24), (11, 23), (12, 24),
         (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32), (15, 19), (16, 20), (0, 11), (0, 12)]


class PoseTracker:
    def __init__(self, model="full", num_poses=1, pick_x=0.5, min_conf=0.5, max_jump=0.12, crop=None):
        path = MODELS / f"pose_landmarker_{model}.task"
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.lm = vision.PoseLandmarker.create_from_options(opts)
        self.pick_x = pick_x
        self.max_jump = max_jump
        self.crop = crop  # (x0, y0, x1, y1) normalised, or None
        self.prev_center = None
        self.t0 = time.monotonic()
        self.last_ts = -1

    def process(self, frame_bgr, t_sec=None):
        """Returns dict(world (33,3), vis (33,), image (33,3) normalised) or None."""
        H, W = frame_bgr.shape[:2]
        if self.crop is not None:
            x0, y0, x1, y1 = self.crop
            frame_bgr = frame_bgr[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
        rgb = cv2.cvtColor(np.ascontiguousarray(frame_bgr), cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if t_sec is None:
            t_sec = time.monotonic() - self.t0
        ts = int(t_sec * 1000)
        if ts <= self.last_ts:
            ts = self.last_ts + 1
        self.last_ts = ts
        res = self.lm.detect_for_video(img, ts)
        if not res.pose_world_landmarks:
            return None
        cands = []
        for k in range(len(res.pose_world_landmarks)):
            im = np.array([[p.x, p.y, p.z] for p in res.pose_landmarks[k]])
            c = 0.5 * (im[23, :2] + im[24, :2])
            cands.append((k, im, c))
        if self.prev_center is None:
            k, im, c = min(cands, key=lambda e: abs(e[2][0] - self.pick_x))
        else:
            k, im, c = min(cands, key=lambda e: np.linalg.norm(e[2] - self.prev_center))
            if np.linalg.norm(c - self.prev_center) > self.max_jump:
                return None  # probably another person: drop the frame, keep tracking the old one
        self.prev_center = c
        if self.crop is not None:  # back to full-frame normalised image coords (for overlays)
            x0, y0, x1, y1 = self.crop
            im = im.copy()
            im[:, 0] = x0 + im[:, 0] * (x1 - x0)
            im[:, 1] = y0 + im[:, 1] * (y1 - y0)
        wl = res.pose_world_landmarks[k]
        world = np.array([[p.x, p.y, p.z] for p in wl])
        vis = np.array([min(p.visibility or 0.0, p.presence or 1.0) for p in res.pose_landmarks[k]])
        return {"world": world, "vis": vis, "image": im}

    def close(self):
        self.lm.close()


def draw_overlay(frame_bgr, det, text="", color=(0, 255, 0)):
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    if det is not None:
        im = det["image"]
        vis = det["vis"]
        for a, b in BONES:
            if vis[a] > 0.3 and vis[b] > 0.3:
                pa = (int(im[a, 0] * w), int(im[a, 1] * h))
                pb = (int(im[b, 0] * w), int(im[b, 1] * h))
                cv2.line(out, pa, pb, color, max(2, w // 300), cv2.LINE_AA)
        for i in range(33):
            if vis[i] > 0.3:
                cv2.circle(out, (int(im[i, 0] * w), int(im[i, 1] * h)), max(2, w // 250), (0, 0, 255), -1)
    if text:
        for i, line in enumerate(text.split("\n")):
            cv2.putText(out, line, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1400), (255, 255, 255), 2, cv2.LINE_AA)
    return out
