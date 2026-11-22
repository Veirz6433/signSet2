"""Optional independent second model (RTMW whole-body via rtmlib/ONNX) for the agreement check.

D_j = ||p_MP - p_RTMW|| / s  per joint; agreement = fraction of joints with D < 0.1.
rtmlib downloads its ONNX models on first use (needs internet once)."""
from __future__ import annotations

import cv2
import numpy as np

from . import landmarks as LM
from .metrics import face_scale, hand_scale, normalised_errors, summarise_errors
from .video_io import iter_frames


def available():
    try:
        import rtmlib  # noqa: F401
        return True
    except Exception:
        return False


def _to_parts(kp, sc, min_score):
    """RTMW 133 keypoints -> dict of our parts (2D) with validity."""
    def take(idx):
        p = kp[idx].astype(np.float64)
        v = sc[idx] >= min_score
        p[~v] = np.nan
        return p
    body = np.full((33, 2), np.nan)
    for mp_i, wb_i in LM.WB_BODY.items():
        body[mp_i] = kp[wb_i] if sc[wb_i] >= min_score else np.nan
    face = np.array([np.mean([kp[LM.WB_FACE0 + j] for j in ib], axis=0) if min(sc[LM.WB_FACE0 + j] for j in ib) >= min_score
                     else [np.nan, np.nan] for ib in LM.FACE_IBUG])
    return {"pose": body, "left_hand": take(np.arange(LM.WB_LEFT_HAND0, LM.WB_LEFT_HAND0 + 21)),
            "right_hand": take(np.arange(LM.WB_RIGHT_HAND0, LM.WB_RIGHT_HAND0 + 21)), "face": face}


def agreement(meta, tree, cfg):
    from rtmlib import Wholebody
    sm = cfg["evaluation"]["second_model"]
    model = Wholebody(to_openpose=False, mode=sm["mode"], backend="onnxruntime", device="cpu")
    fi = {int(f): k for k, f in enumerate(tree["time"]["frame_index"])}
    raw = tree["raw"]
    T = len(fi)
    other = {p: np.full((T, n, 2), np.nan) for p, n in LM.PART_SIZES.items()}
    for i, t, rgb in iter_frames(meta):
        if i not in fi:
            continue
        k = fi[i]
        kps, scs = model(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if kps is None or len(kps) == 0:
            continue
        # choose the person whose shoulders are closest to ours
        ours = raw["pose_px"][k, [LM.L_SH, LM.R_SH], :2]
        if np.all(np.isfinite(ours)):
            d = [np.nanmean(np.linalg.norm(kp[[5, 6]] - ours, axis=-1)) for kp in kps]
            j = int(np.nanargmin(d))
        else:
            j = int(np.argmax([s.mean() for s in scs]))
        parts = _to_parts(kps[j], scs[j], float(sm["min_score"]))
        for p in LM.PARTS:
            other[p][k] = parts[p]
    out = {}
    scales = {"pose": tree["norm"]["scale_px"], "left_hand": hand_scale(raw["left_hand_px"]),
              "right_hand": hand_scale(raw["right_hand_px"]), "face": face_scale(raw["face_px"])}
    for p in LM.PARTS:
        J = LM.BODY_EVAL if p == "pose" else slice(None)
        ref = other[p][:, J]
        mine = raw[f"{p}_px"][:, J, :2].astype(np.float64)
        e = normalised_errors(mine, ref, scales[p], np.all(np.isfinite(ref), axis=-1), raw[f"{p}_mask"] != 0)
        out["body" if p == "pose" else p] = summarise_errors(e)
    return out
