"""Hand outlier rejection (runs on raw detections, before gap filling and smoothing).

Four tests per hand and frame; any failure rejects the frame (raw data are kept untouched,
the reason is stored as bit flags in /raw/{side}_hand_reject):

  bit 1  DUPLICATE  both hands on the same spot:
                    mean_j ||L_j - R_j|| / max(box_L, box_R) < dup_tol  -> reject the hand whose
                    wrist is farther from its own pose wrist (or the lower-confidence one)
  bit 2  FAR        ||hand wrist - pose wrist|| / forearm length > wrist_tol
  bit 4  SHAPE      bone-length profile differs from the clip's own median profile:
                    l_b = L_b / sum_b L_b (3D, scale-free), r_b = |l_b / l*_b - 1|,
                    mean of the 5 largest r_b > shape_tol   (l* = the clip's median profile)
  bit 8  SPIKE      Hampel test in time: window W_i = frames within +-hampel_window_s,
                    m_j = median_{k in W_i} p_kj (per coordinate),
                    dev_i = median_j ||p_ij - m_j|| / median_{k in W_i} box_k  > spike_tol
                    (a median of a monotone sequence is its centre value, so fast but smooth
                    motion is never rejected; isolated jumps are)
"""
from __future__ import annotations

import numpy as np

from . import landmarks as LM
from .postprocess import median_dt

DUPLICATE, FAR, SHAPE, SPIKE = 1, 2, 4, 8


def _box(px):
    """max(width, height) of each frame's 2D hand box. px: (T,21,>=2)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        xy = px[..., :2]
        return np.max(np.nanmax(xy, axis=-2) - np.nanmin(xy, axis=-2), axis=-1)


def _forearm(pose_px, side):
    s = LM.SIDE_POSE[side]
    fl = np.linalg.norm(pose_px[:, s["wrist"], :2] - pose_px[:, s["elbow"], :2], axis=-1)
    sw = np.linalg.norm(pose_px[:, LM.L_SH, :2] - pose_px[:, LM.R_SH, :2], axis=-1)
    return np.where(np.isfinite(fl) & (fl > 4), fl, 0.6 * sw)


def reject_hand_outliers(t, raw, cfg):
    """Returns {side: flags (T,) uint8}. raw: dict of raw arrays (pipeline layout)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _reject(np.asarray(t, np.float64), raw, cfg)


def _reject(t, raw, cfg):
    oc = cfg["cleanup"].get("outliers", {})
    if not oc.get("enabled", True):
        return {s: np.zeros(len(t), np.uint8) for s in ("left", "right")}
    dup_tol, wrist_tol = float(oc.get("dup_tol", 0.3)), float(oc.get("wrist_tol", 0.8))
    shape_tol, spike_tol = float(oc.get("shape_tol", 0.35)), float(oc.get("spike_tol", 0.35))
    win_s = float(oc.get("hampel_window_s", 0.08))
    T = len(t)
    pose = raw["pose_px"].astype(np.float64)
    pose_ok = raw["pose_mask"] != LM.MISSING
    px = {s: raw[f"{s}_hand_px"].astype(np.float64) for s in ("left", "right")}
    wd = {s: raw[f"{s}_hand_world"].astype(np.float64) for s in ("left", "right")}
    ok = {s: raw[f"{s}_hand_mask"] != LM.MISSING for s in ("left", "right")}
    flags = {s: np.zeros(T, np.uint8) for s in ("left", "right")}
    box = {s: _box(px[s]) for s in ("left", "right")}

    # distance of each hand's wrist to its own pose wrist, in forearm lengths
    dist = {}
    for s in ("left", "right"):
        pw = pose[:, LM.SIDE_POSE[s]["wrist"], :2]
        dist[s] = np.linalg.norm(px[s][:, 0, :2] - pw, axis=-1) / _forearm(pose, s)
        far = ok[s] & pose_ok & np.isfinite(dist[s]) & (dist[s] > wrist_tol)
        flags[s][far] |= FAR

    both = ok["left"] & ok["right"]
    sep = np.linalg.norm(px["left"][..., :2] - px["right"][..., :2], axis=-1).mean(-1) / \
        np.maximum(np.fmax(box["left"], box["right"]), 1e-6)
    dup = both & (sep < dup_tol)
    for i in np.where(dup)[0]:
        dl, dr = dist["left"][i], dist["right"][i]
        if np.isfinite(dl) and np.isfinite(dr) and pose_ok[i]:
            loser = "left" if dl > dr else "right"
        else:
            loser = "left" if raw["left_hand_conf"][i] < raw["right_hand_conf"][i] else "right"
        flags[loser][i] |= DUPLICATE

    for s in ("left", "right"):
        keep = ok[s] & (flags[s] == 0)
        # shape test (scale-free 3D bone-length profile vs the clip's own median profile)
        L = np.stack([np.linalg.norm(wd[s][:, b] - wd[s][:, a], axis=-1) for a, b in LM.HAND_BONES], axis=1)
        prof = L / np.nansum(L, axis=1, keepdims=True)
        if keep.sum() >= 5:
            ref = np.nanmedian(prof[keep], axis=0)
            rel = np.nan_to_num(np.abs(prof / ref - 1), nan=-1.0)
            top = np.sort(rel, axis=1)[:, -5:]                       # 5 largest relative deviations
            dev = np.nanmean(np.where(top >= 0, top, np.nan), axis=1)
            bad = ok[s] & np.isfinite(dev) & (dev > shape_tol)
            flags[s][bad] |= SHAPE
        # Hampel spike test on surviving frames
        keep = ok[s] & (flags[s] == 0)
        idx = np.where(keep)[0]
        if idx.size >= 3:
            lo = np.searchsorted(t[idx], t[idx] - win_s - 1e-9, "left")
            hi = np.searchsorted(t[idx], t[idx] + win_s + 1e-9, "right")
            for n, i in enumerate(idx):
                w = idx[lo[n]:hi[n]]
                if w.size < 3 or w[0] == i or w[-1] == i:      # need neighbours on BOTH sides
                    continue
                m = np.median(px[s][w, :, :2], axis=0)
                sc = np.median(box[s][w])
                d = np.median(np.linalg.norm(px[s][i, :, :2] - m, axis=-1)) / max(sc, 1e-6)
                if d > spike_tol:
                    flags[s][i] |= SPIKE
    return flags


def accepted(raw, side):
    """Mask of frames kept after rejection."""
    rej = raw.get(f"{side}_hand_reject")
    m = raw[f"{side}_hand_mask"] != LM.MISSING
    return m if rej is None else m & (rej == 0)


def accepted_px(raw, part):
    """(px, valid) for a part with rejected hand frames set to NaN."""
    px = raw[f"{part}_px"].astype(np.float64).copy()
    if part in ("left_hand", "right_hand"):
        v = accepted(raw, part.split("_")[0])
        px[~v] = np.nan
        return px, v
    return px, raw[f"{part}_mask"] != LM.MISSING


def window_frames(t, seconds):
    return max(1, int(round(seconds / median_dt(t))))
