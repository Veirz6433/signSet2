"""Absolute accuracy against human-labelled ground truth: COCO-WholeBody val (body, hands, face).

For each person annotation: crop the person box (expanded), run each model on the crop, map
predictions back to image pixels, and score against the labels.
Models compared (config coco.models):
  mediapipe          - the extractor exactly as configured (incl. the upscaled-crop hand pass)
  mediapipe_v1       - same, but hand crop pass only for missed hands (v1.0 behaviour)
  rtmw               - RTMW whole-body (rtmlib/ONNX, 2D only), if installed
Normalisers: hand = max(w, h) of the labelled hand box; face = inter-ocular distance
(outer eye corners, iBUG 36-45); body = torso length ||mid-shoulder - mid-hip||."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np

from . import landmarks as LM
from .metrics import normalised_errors, summarise_errors
from .quality import QualityMeter

COCO_BODY = [(LM.L_SH, 5), (LM.R_SH, 6), (LM.L_EL, 7), (LM.R_EL, 8), (LM.L_WR, 9), (LM.R_WR, 10)]


class _MP:
    def __init__(self, cfg, crop_mode=None):
        from .extractor import Extractor
        c = cfg
        if crop_mode:
            c = copy.deepcopy(dict(cfg))
            c["detection"] = dict(c["detection"], hand_crop_mode=crop_mode)
            from .config import Config
            c = Config(c)
        self.ex = Extractor(c, "image")
        self.qm = QualityMeter(cfg)

    def __call__(self, crop):
        self.qm.prev = None
        b, _, _ = self.qm.measure(crop)
        det, _ = self.qm.enhance(crop, b)
        o = self.ex.process(crop, det, 0.0)
        return {"left_hand": (o["left_hand_px"][:, :2], o["left_hand_code"] != LM.MISSING),
                "right_hand": (o["right_hand_px"][:, :2], o["right_hand_code"] != LM.MISSING),
                "face": (o["face_px"][:, :2], o["face_code"] != LM.MISSING),
                "pose": (o["pose_px"][:, :2], o["pose_code"] != LM.MISSING)}

    def close(self):
        self.ex.close()


class _RTMW:
    def __init__(self, cfg):
        from rtmlib import Wholebody
        sm = cfg["evaluation"]["second_model"]
        self.min_score = float(sm["min_score"])
        self.model = Wholebody(to_openpose=False, mode=sm["mode"], backend="onnxruntime", device="cpu")

    def __call__(self, crop):
        from .second_model import _to_parts
        kps, scs = self.model(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        if kps is None or len(kps) == 0:
            return None
        c = np.array([crop.shape[1] / 2, crop.shape[0] / 2])
        d = [np.linalg.norm(kp[:17][sc[:17] >= self.min_score].mean(0) - c) if (sc[:17] >= self.min_score).any()
             else np.inf for kp, sc in zip(kps, scs)]
        j = int(np.argmin(d))
        parts = _to_parts(kps[j], scs[j], self.min_score)
        out = {}
        for p in ("left_hand", "right_hand", "face", "pose"):
            arr = parts[p]
            out[p] = (arr, bool(np.isfinite(arr).all(-1).any()))
        return out

    def close(self):
        pass


def _score(pred, a, off, acc, hand_ok, face_ok):
    """Append per-instance normalised errors for one model's prediction on one annotation."""
    def get(part):
        if pred is None:
            n = {"left_hand": 21, "right_hand": 21, "face": 20, "pose": 33}[part]
            return np.full((n, 2), np.nan), False
        xy, ok = pred[part]
        return np.asarray(xy, np.float64) + off, bool(ok)

    for side in ("left", "right"):
        if not hand_ok(a, side):
            continue
        gt = np.array(a[f"{side}hand_kpts"], np.float64).reshape(21, 3)
        box = a[f"{side}hand_box"]
        s = float(max(box[2], box[3]))
        p, ok = get(f"{side}_hand")
        acc["hand"].append(normalised_errors(p[None], gt[None, :, :2], np.array([s]), (gt[:, 2] > 0)[None],
                                             np.array([ok]))[0])
        acc["hand_s"].append(s)
    if face_ok(a):
        g68 = np.array(a["face_kpts"], np.float64).reshape(68, 3)
        s = float(np.linalg.norm(g68[36, :2] - g68[45, :2]))
        if s > 1 and g68[36, 2] > 0 and g68[45, 2] > 0:
            gt = np.array([g68[ib, :2].mean(axis=0) for ib in LM.FACE_IBUG])
            gv = np.array([np.all(g68[ib, 2] > 0) for ib in LM.FACE_IBUG])
            p, ok = get("face")
            acc["face"].append(normalised_errors(p[None], gt[None], np.array([s]), gv[None], np.array([ok]))[0])
    g17 = np.array(a["keypoints"], np.float64).reshape(17, 3)
    if np.all(g17[[5, 6, 11, 12], 2] > 0):
        torso = float(np.linalg.norm(g17[[5, 6], :2].mean(0) - g17[[11, 12], :2].mean(0)))
        if torso > 1:
            gt = np.array([g17[ci, :2] for _, ci in COCO_BODY])
            gv = np.array([g17[ci, 2] > 0 for _, ci in COCO_BODY])
            p, ok = get("pose")
            pb = np.array([p[mi] for mi, _ in COCO_BODY])
            acc["body"].append(normalised_errors(pb[None], gt[None], np.array([torso]), gv[None], np.array([ok]))[0])


def _summarise(acc, c, kw):
    res = {}
    if acc["hand"]:
        E, S = np.array(acc["hand"]), np.array(acc["hand_s"])
        res["hands"] = summarise_errors(E, **kw)
        bins = list(c["hand_size_bins"]) + [np.inf]
        res["hands_by_box_px"] = {}
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (S >= lo) & (S < hi)
            if m.sum() > 1:
                res["hands_by_box_px"][f"[{lo},{hi})"] = summarise_errors(E[m], **kw)
        det = np.where(np.isfinite(E), E, np.nan)
        res["hand_per_joint_nme"] = {LM.HAND_NAMES[j]: float(np.nanmean(det[:, j])) for j in range(21)}
    if acc["face"]:
        res["face"] = summarise_errors(np.array(acc["face"]), **kw)
    if acc["body"]:
        res["body"] = summarise_errors(np.array(acc["body"]), **kw)
    return res


def coco_eval(cfg, out_dir: Path, log=print):
    ann_path, img_dir = cfg.path("coco_annotations"), cfg.path("coco_images_dir")
    if not ann_path.exists() or not img_dir.exists():
        raise FileNotFoundError(f"COCO-WholeBody not found ({ann_path}, {img_dir}). Run: python main.py download-coco")
    c, ev = cfg["coco"], cfg["evaluation"]
    log(f"loading {ann_path.name} ...")
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    imgs = {im["id"]: im for im in data["images"]}

    def hand_ok(a, side):
        b = a.get(f"{side}hand_box") or [0, 0, 0, 0]
        return bool(a.get(f"{side}hand_valid")) and max(b[2], b[3]) >= c["min_hand_box_px"]

    def face_ok(a):
        b = a.get("face_box") or [0, 0, 0, 0]
        return bool(a.get("face_valid")) and max(b[2], b[3]) >= c["min_face_box_px"]

    anns = [a for a in data["annotations"] if not a.get("iscrowd", 0)
            and (hand_ok(a, "left") or hand_ok(a, "right") or face_ok(a))]
    anns.sort(key=lambda a: a["id"])
    rng = np.random.default_rng(int(ev["seed"]))
    sel = [anns[i] for i in sorted(rng.permutation(len(anns))[: int(c["max_samples"])])]
    log(f"{len(anns)} eligible person annotations, evaluating {len(sel)}")

    models = {}
    for name in c.get("models", ["mediapipe"]):
        if name == "mediapipe":
            models[name] = _MP(cfg)
        elif name == "mediapipe_v1":
            models[name] = _MP(cfg, crop_mode="missing_only")
        elif name == "rtmw":
            from .second_model import available
            if available():
                models[name] = _RTMW(cfg)
            else:
                log("  rtmw skipped (pip install -r requirements-optional.txt)")
    acc = {m: {"hand": [], "hand_s": [], "face": [], "body": []} for m in models}
    for n, a in enumerate(sel, 1):
        im = imgs[a["image_id"]]
        bgr = cv2.imread(str(img_dir / im["file_name"]))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x, y, w, h = a["bbox"]
        e = float(c["person_crop_expand"])
        cx, cy = x + w / 2, y + h / 2
        x0, y0 = int(max(0, cx - w * e / 2)), int(max(0, cy - h * e / 2))
        x1, y1 = int(min(rgb.shape[1], cx + w * e / 2)), int(min(rgb.shape[0], cy + h * e / 2))
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        off = np.array([x0, y0], np.float64)
        for name, model in models.items():
            _score(model(crop), a, off, acc[name], hand_ok, face_ok)
        if n % 100 == 0:
            log(f"  {n}/{len(sel)}")
    for m in models.values():
        m.close()

    kw = dict(alphas=ev["pck_alphas"], auc_max=ev["auc_alpha_max"], boot=int(ev["bootstrap_resamples"]),
              seed=int(ev["seed"]))
    res = {"dataset": "COCO-WholeBody v1.0 val2017", "n_person_annotations": len(sel),
           "normalisers": {"hands": "max(w,h) of labelled hand box", "face": "inter-ocular (iBUG 36-45)",
                           "body": "torso = ||mid-shoulder - mid-hip||"},
           "models": {name: _summarise(acc[name], c, kw) for name in models}}
    if "mediapipe" in res["models"]:          # top-level keys = the configured extractor
        res.update(res["models"]["mediapipe"])
    out_dir.mkdir(parents=True, exist_ok=True)
    from .evaluate import _jsonable
    (out_dir / "coco_results.json").write_text(json.dumps(_jsonable(res), indent=2), encoding="utf-8")
    return res
