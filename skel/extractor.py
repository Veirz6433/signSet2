"""Landmark extraction with MediaPipe Tasks (pose 33, hands 2x21, face 20 of 478).

Hands are assigned to the PERSON's left/right by matching each detected hand's wrist to the
pose model's wrists (MediaPipe's handedness label is only a fallback, because it assumes a
mirrored selfie image). When a hand is missed, the hand model is re-run on a crop around the
pose wrist, extended along the forearm.
"""
from __future__ import annotations

import itertools

import cv2
import numpy as np

from . import landmarks as LM

NAN = np.nan


def _empty():
    return {
        "pose_px": np.full((33, 3), NAN, np.float32), "pose_world": np.full((33, 3), NAN, np.float32),
        "pose_conf": np.zeros(33, np.float32), "pose_code": LM.MISSING,
        "left_hand_px": np.full((21, 3), NAN, np.float32), "left_hand_world": np.full((21, 3), NAN, np.float32),
        "left_hand_conf": 0.0, "left_hand_code": LM.MISSING,
        "right_hand_px": np.full((21, 3), NAN, np.float32), "right_hand_world": np.full((21, 3), NAN, np.float32),
        "right_hand_conf": 0.0, "right_hand_code": LM.MISSING,
        "face_px": np.full((20, 3), NAN, np.float32), "face_conf": 0.0, "face_code": LM.MISSING,
    }


class Extractor:
    def __init__(self, cfg, mode: str = "video"):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp = mp
        d = cfg["detection"]
        models_dir = cfg.path("models_dir")
        self.video = mode == "video"
        rm = vision.RunningMode.VIDEO if self.video else vision.RunningMode.IMAGE

        def base(name):
            return BaseOptions(model_asset_path=str(models_dir / d[name]))

        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base("pose_model"), running_mode=rm, num_poses=1,
            min_pose_detection_confidence=d["min_pose_detection_confidence"],
            min_pose_presence_confidence=d["min_pose_presence_confidence"],
            min_tracking_confidence=d["min_tracking_confidence"], output_segmentation_masks=False))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base("hand_model"), running_mode=rm, num_hands=2,
            min_hand_detection_confidence=d["min_hand_detection_confidence"],
            min_hand_presence_confidence=d["min_hand_presence_confidence"],
            min_tracking_confidence=d["min_tracking_confidence"]))
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=base("face_model"), running_mode=rm, num_faces=1,
            min_face_detection_confidence=d["min_face_detection_confidence"],
            min_face_presence_confidence=d["min_face_presence_confidence"],
            min_tracking_confidence=d["min_tracking_confidence"],
            output_face_blendshapes=False, output_facial_transformation_matrixes=False))
        self.crop_hands = None
        if d["hand_crop_redetect"]:
            self.crop_hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=base("hand_model"), running_mode=vision.RunningMode.IMAGE, num_hands=2,
                min_hand_detection_confidence=d["min_hand_detection_confidence"],
                min_hand_presence_confidence=d["min_hand_presence_confidence"]))
        self.max_side = int(cfg["video"]["detect_max_side"])
        self.mirrored = bool(cfg["video"]["video_is_mirrored"])
        self.crop_scale = float(d["hand_crop_scale"])
        self.crop_min = int(d["hand_crop_min_px"])
        self.max_dist = float(d["hand_assign_max_dist"])
        self.crop_mode = str(d.get("hand_crop_mode", "auto"))
        self.crop_up = int(d.get("hand_crop_upscale_px", 256))
        self.small_px = float(d.get("hand_small_px", 120))
        self.last_ts = -1

    def close(self):
        for m in (self.pose, self.hands, self.face, self.crop_hands):
            if m is not None:
                m.close()

    # ------------------------------------------------------------------ helpers
    def _img(self, arr):
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(arr))

    def _run(self, model, img, ts):
        return model.detect_for_video(img, ts) if self.video else model.detect(img)

    @staticmethod
    def _hand_cands(res, W, H, x0=0.0, y0=0.0, cw=None, ch=None):
        cw = W if cw is None else cw
        ch = H if ch is None else ch
        out = []
        for i, lms in enumerate(res.hand_landmarks or []):
            px = np.array([[x0 + l.x * cw, y0 + l.y * ch, l.z * cw] for l in lms], np.float32)
            wl = res.hand_world_landmarks[i] if res.hand_world_landmarks else None
            world = np.array([[l.x, l.y, l.z] for l in wl], np.float32) if wl else np.full((21, 3), NAN, np.float32)
            cat = res.handedness[i][0] if res.handedness and res.handedness[i] else None
            out.append({"px": px, "world": world, "score": float(cat.score) if cat else 0.0,
                        "label": cat.category_name if cat else ""})
        return out

    def _label_side(self, label):
        # MediaPipe assumes a mirrored (selfie) image: on a normal video "Left" is the person's right.
        if not label:
            return None
        side = "left" if label.lower() == "left" else "right"
        if not self.mirrored:
            side = "right" if side == "left" else "left"
        return side

    @staticmethod
    def _forearm_len(o, side):
        s = LM.SIDE_POSE[side]
        fl = float(np.linalg.norm(o["pose_px"][s["wrist"], :2] - o["pose_px"][s["elbow"], :2]))
        if not np.isfinite(fl) or fl < 4:
            sw = float(np.linalg.norm(o["pose_px"][LM.L_SH, :2] - o["pose_px"][LM.R_SH, :2]))
            fl = 0.6 * sw if np.isfinite(sw) and sw > 4 else 40.0
        return fl

    def _assign(self, o, cands):
        sides = ("left", "right")
        if o["pose_code"] == LM.MISSING:
            cands = sorted(cands, key=lambda c: -c["score"])
            taken = {}
            for c in cands:
                s = self._label_side(c["label"])
                if s in taken or s is None:
                    s = next((x for x in sides if x not in taken), None)
                if s is not None:
                    taken[s] = c
            return taken
        cost = {}
        for ci, c in enumerate(cands):
            for s in sides:
                pw = o["pose_px"][LM.SIDE_POSE[s]["wrist"], :2]
                cost[ci, s] = float(np.linalg.norm(c["px"][0, :2] - pw)) / self._forearm_len(o, s)
        best, best_key = {}, (-1, 0.0)
        options = [None, "left", "right"]
        for combo in itertools.product(options, repeat=len(cands)):
            used = [s for s in combo if s]
            if len(used) != len(set(used)):
                continue
            if any(s and cost[ci, s] > self.max_dist for ci, s in enumerate(combo)):
                continue
            key = (len(used), -sum(cost[ci, s] for ci, s in enumerate(combo) if s))
            if key > best_key:
                best_key, best = key, {s: cands[ci] for ci, s in enumerate(combo) if s}
        # a far-away hand is kept only if that side's pose wrist itself is unreliable
        for c in cands:
            if any(c is v for v in best.values()):
                continue
            s = self._label_side(c["label"])
            if s and s not in best and o["pose_conf"][LM.SIDE_POSE[s]["wrist"]] < 0.5:
                best[s] = c
        return best

    def _crop_detect(self, o, rgb_full, side, around=None):
        """Run the hand model on an UPSCALED square crop.
        around=None: crop centred ahead of the pose wrist (side = scale x forearm length);
        around=hand px (21,3): crop centred on that hand's box (side = 2 x box size).
        Returns the candidate whose wrist is nearest the pose wrist (if within max_dist)."""
        s = LM.SIDE_POSE[side]
        w = o["pose_px"][s["wrist"], :2].astype(np.float64)
        e = o["pose_px"][s["elbow"], :2].astype(np.float64)
        pose_ok = np.all(np.isfinite(w)) and np.all(np.isfinite(e))
        H, W = rgb_full.shape[:2]
        if around is not None:
            xy = around[:, :2].astype(np.float64)
            lo, hi = np.nanmin(xy, 0), np.nanmax(xy, 0)
            c = 0.5 * (lo + hi)
            side_px = int(round(max(self.crop_min, 2.0 * float(np.max(hi - lo)))))
        else:
            if not pose_ok:
                return None
            v = w - e
            if float(np.linalg.norm(v)) < 4:
                return None
            c = w + 0.35 * v
            side_px = int(round(max(self.crop_min, self.crop_scale * float(np.linalg.norm(v)))))
        x0, y0 = int(round(c[0] - side_px / 2)), int(round(c[1] - side_px / 2))
        if x0 >= W or y0 >= H or x0 + side_px <= 0 or y0 + side_px <= 0:
            return None
        crop = np.zeros((side_px, side_px, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + side_px), min(H, y0 + side_px)
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb_full[sy0:sy1, sx0:sx1]
        if side_px < self.crop_up:          # small hands get more pixels (normalised coords are unchanged)
            crop = cv2.resize(crop, (self.crop_up, self.crop_up), interpolation=cv2.INTER_CUBIC)
        res = self.crop_hands.detect(self._img(crop))
        cands = self._hand_cands(res, W, H, x0, y0, side_px, side_px)
        if not cands:
            return None
        ref = w if pose_ok else c
        fl = self._forearm_len(o, side) if pose_ok else side_px / 2.0
        d = [float(np.linalg.norm(cd["px"][0, :2] - ref)) / fl for cd in cands]
        k = int(np.argmin(d))
        return cands[k] if d[k] <= self.max_dist else None

    @staticmethod
    def _box(px):
        xy = px[:, :2]
        return float(np.max(np.nanmax(xy, 0) - np.nanmin(xy, 0)))

    # ------------------------------------------------------------------ main
    def process(self, rgb_display: np.ndarray, det_rgb: np.ndarray, t_s: float = 0.0) -> dict:
        """rgb_display: geometry reference (display frame). det_rgb: same size, possibly enhanced."""
        H, W = rgb_display.shape[:2]
        k = min(1.0, self.max_side / max(H, W))
        det_small = det_rgb if k >= 1.0 else cv2.resize(det_rgb, (int(round(W * k)), int(round(H * k))),
                                                        interpolation=cv2.INTER_AREA)
        img = self._img(det_small)
        ts = int(round(t_s * 1000.0))
        if ts <= self.last_ts:
            ts = self.last_ts + 1
        self.last_ts = ts
        o = _empty()

        pr = self._run(self.pose, img, ts)
        if pr.pose_landmarks:
            lms = pr.pose_landmarks[0]
            o["pose_px"] = np.array([[l.x * W, l.y * H, l.z * W] for l in lms], np.float32)
            o["pose_conf"] = np.array([l.visibility if l.visibility is not None else 0.0 for l in lms], np.float32)
            if pr.pose_world_landmarks:
                o["pose_world"] = np.array([[l.x, l.y, l.z] for l in pr.pose_world_landmarks[0]], np.float32)
            o["pose_code"] = LM.DETECTED

        hr = self._run(self.hands, img, ts)
        assigned = self._assign(o, self._hand_cands(hr, W, H))
        for side, c in assigned.items():
            o[f"{side}_hand_px"], o[f"{side}_hand_world"] = c["px"], c["world"]
            o[f"{side}_hand_conf"], o[f"{side}_hand_code"] = c["score"], LM.DETECTED
        if self.crop_hands is not None:
            for side in ("left", "right"):
                full = o[f"{side}_hand_code"] != LM.MISSING
                if full and self.crop_mode == "missing_only":
                    continue
                if full and self.crop_mode == "auto" and self._box(o[f"{side}_hand_px"]) >= self.small_px:
                    continue                      # hand is big enough: keep the full-frame result
                # two-stage refinement: crop around the found hand, or ahead of the pose wrist
                c = self._crop_detect(o, det_rgb, side, around=o[f"{side}_hand_px"] if full else None)
                if c is None and full:
                    continue                      # crop failed: keep the full-frame result
                if c is not None:
                    o[f"{side}_hand_px"], o[f"{side}_hand_world"] = c["px"], c["world"]
                    o[f"{side}_hand_conf"], o[f"{side}_hand_code"] = c["score"], LM.CROP_DETECTED

        fr = self._run(self.face, img, ts)
        if fr.face_landmarks:
            lms = fr.face_landmarks[0]
            o["face_px"] = np.array([[lms[i].x * W, lms[i].y * H, lms[i].z * W] for i in LM.FACE_MP_IDX], np.float32)
            o["face_conf"], o["face_code"] = 1.0, LM.DETECTED
        return o
