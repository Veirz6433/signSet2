"""Rebuild the skeleton from the HDF5 file ONLY and draw it on the original video.

Pixel coordinates are reconstructed from the stored body-normalised coordinates:
    xy_px = n * s_t + o_t ,  z_px = n_z * s_t
and frames are matched to stored timestamps. Reported: reconstruction error vs the stored
pixel coordinates and timestamp matching error.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import landmarks as LM
from .postprocess import denormalise, median_dt
from .storage import read_h5
from .video_io import file_sha256, iter_frames, probe

COLORS = {"pose": (255, 255, 255), "left_hand": (0, 220, 255), "right_hand": (255, 140, 0), "face": (255, 230, 0)}
EDGES = {"pose": LM.POSE_DRAW, "left_hand": LM.HAND_DRAW, "right_hand": LM.HAND_DRAW, "face": LM.FACE_DRAW}
POSE_POINTS = sorted({i for e in LM.POSE_DRAW for i in e})


def reconstruct_px(tree):
    nr = tree["norm"]
    return {p: denormalise(nr[p].astype(np.float64), nr["origin_px"], nr["scale_px"]) for p in LM.PARTS}


def render(h5_path, video_path, out_path, cfg):
    tree = read_h5(h5_path)
    vmeta = tree["video"]["__attrs__"]
    if file_sha256(video_path) != vmeta["sha256"]:
        print("  WARNING: video file differs from the one this HDF5 was made from (sha256 mismatch)")
    meta = probe(video_path)
    # geometry comes from the FILE (rotation, display size), not from re-probing
    for k in ("rotation_cw", "display_width", "display_height", "sar", "decoder_already_rotated"):
        meta[k] = vmeta[k]
    px = reconstruct_px(tree)
    src = cfg["overlay"]["source"]
    stored = {p: tree["smoothed" if src == "smoothed" else "raw"][f"{p}_px"].astype(np.float64) for p in LM.PARTS}
    if src == "raw":       # raw coordinates are drawn directly (norm is built from smoothed)
        px = stored
    masks = {p: tree["smoothed" if src == "smoothed" else "raw"][f"{p}_mask"] for p in LM.PARTS}

    errs = []
    for p in LM.PARTS:
        d = np.linalg.norm(px[p][..., :2] - stored[p][..., :2], axis=-1)
        errs.append(d[np.isfinite(d)])
    errs = np.concatenate(errs) if errs else np.array([0.0])

    ts = tree["time"]["timestamps_s"]
    half = 0.5 * median_dt(ts)
    W, H = int(vmeta["display_width"]), int(vmeta["display_height"])
    fps = float(vmeta.get("fps_avg") or 30.0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*cfg["overlay"]["codec"]), fps, (W, H))
    th = max(1, int(round(min(W, H) / 360)))
    dts, matched, drawn, n_video = [], 0, 0, 0
    for i, t, rgb in iter_frames(meta):
        n_video += 1
        k = int(np.searchsorted(ts, t))
        cand = [j for j in (k - 1, k) if 0 <= j < len(ts)]
        j = min(cand, key=lambda c: abs(ts[c] - t)) if cand else None
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if j is not None and abs(ts[j] - t) <= half:
            matched += 1
            dts.append(abs(ts[j] - t))
            for p in LM.PARTS:
                if masks[p][j] == LM.MISSING:
                    continue
                P = px[p][j]
                col = COLORS[p][::-1]
                for a, b in EDGES[p]:
                    if np.all(np.isfinite(P[[a, b], :2])):
                        cv2.line(bgr, tuple(np.round(P[a, :2]).astype(int)), tuple(np.round(P[b, :2]).astype(int)),
                                 col, th, cv2.LINE_AA)
                idx = POSE_POINTS if p == "pose" else range(P.shape[0])
                filled = masks[p][j] == LM.FILLED
                for q in idx:
                    if np.all(np.isfinite(P[q, :2])):
                        cv2.circle(bgr, tuple(np.round(P[q, :2]).astype(int)), th + 1, col,
                                   1 if filled else -1, cv2.LINE_AA)
            drawn += 1
        cv2.putText(bgr, f"f{i} t={t:.3f}s", (8, 8 + 12 * th), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * th,
                    (0, 255, 0), max(1, th // 2), cv2.LINE_AA)
        writer.write(bgr)
    writer.release()
    report = {"h5": str(h5_path), "video": str(video_path), "overlay": str(out_path), "source": src,
              "frames_in_video": n_video, "frames_matched": matched,
              "frames_in_file": int(len(ts)),
              "timestamp_abs_err_ms_max": float(np.max(dts) * 1000) if dts else float("nan"),
              "timestamp_abs_err_ms_mean": float(np.mean(dts) * 1000) if dts else float("nan"),
              "reconstruction_err_px_max": float(errs.max()) if errs.size else 0.0,
              "reconstruction_err_px_mean": float(errs.mean()) if errs.size else 0.0,
              "rotation_cw": int(vmeta["rotation_cw"]), "rotation_verified": bool(vmeta["rotation_verified"])}
    out_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
