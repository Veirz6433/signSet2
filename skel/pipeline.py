"""Video -> landmarks -> cleanup -> features -> HDF5 tree."""
from __future__ import annotations

import datetime as _dt
import importlib.metadata as _md
import math
import warnings

import numpy as np

from . import landmarks as LM
from .features import compute_features
from .metrics import label_free
from .models import model_hashes
from .outliers import accepted, reject_hand_outliers
from .postprocess import body_frame, fill_gaps, gaussian_smooth, median_dt, normalise
from .quality import QualityMeter
from .video_io import iter_frames, probe

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")

PACKAGES = ("mediapipe", "numpy", "scipy", "opencv-contrib-python", "opencv-python", "av", "h5py", "PyYAML")


def package_versions():
    out = {}
    for p in PACKAGES:
        try:
            out[p] = _md.version(p)
        except Exception:
            pass
    return out


def extract_frames(meta, cfg, extractor, frame_fn=None, frame_select=None, progress=None):
    """Run detection on every (selected) frame. frame_fn(i, rgb) -> rgb of the same size
    (used by the evaluation to apply known transforms / degradations)."""
    qm = QualityMeter(cfg)
    rows = []
    for i, t, rgb in iter_frames(meta):
        if frame_select is not None and not frame_select(i):
            continue
        if frame_fn is not None:
            rgb = frame_fn(i, rgb)
        b, bl, mo = qm.measure(rgb)
        det, enh = qm.enhance(rgb, b)
        o = extractor.process(rgb, det, t)
        o.update(t=t, i=i, brightness=b, blur=bl, motion=mo, enhanced=enh)
        rows.append(o)
        if progress:
            progress()
    if not rows:
        raise RuntimeError("no frames decoded")
    return rows


def clean_track(t, X, valid, conf, gmax, context, sigma):
    """Gap fill (<= gmax frames) then confidence-weighted zero-phase smoothing.
    conf: (T,) or (T,N). Filled frames get weight 0.5. Returns (smoothed, filled_mask)."""
    Xf, filled = fill_gaps(t, X, valid, gmax, context)
    c = np.clip(np.asarray(conf, np.float64), 0.05, 1.0)
    w = np.where(valid[:, None], c, 0.0) if c.ndim == 2 else np.where(valid, c, 0.0)
    w[filled] = 0.5
    return gaussian_smooth(t, Xf, w, sigma), filled


def build_tree(rows, meta, cfg):
    T = len(rows)
    t = np.array([r["t"] for r in rows], np.float64)
    raw = {}
    for part in LM.PARTS:
        raw[f"{part}_px"] = np.stack([r[f"{'face' if part == 'face' else part}_px"] for r in rows]).astype(np.float32)
        raw[f"{part}_mask"] = np.array([r[f"{part}_code"] for r in rows], np.uint8)
        if part != "face":
            raw[f"{part}_world"] = np.stack([r[f"{part}_world"] for r in rows]).astype(np.float32)
        raw[f"{part}_conf"] = (np.stack([r["pose_conf"] for r in rows]) if part == "pose"
                               else np.array([r[f"{part}_conf"] for r in rows])).astype(np.float32)

    # ---- hand outlier rejection (raw arrays stay untouched; reasons stored as bit flags)
    rej = reject_hand_outliers(t, raw, cfg)
    for side in ("left", "right"):
        raw[f"{side}_hand_reject"] = rej[side]

    # ---- cleanup: gap fill + zero-phase smoothing (on accepted detections only)
    cl = cfg["cleanup"]
    dt = median_dt(t)
    gmax = int(math.floor(cl["max_gap_s"] / dt + 1e-9))
    sm = {}
    for part in LM.PARTS:
        if part in ("left_hand", "right_hand"):
            valid = accepted(raw, part.split("_")[0])
        else:
            valid = raw[f"{part}_mask"] != LM.MISSING
        mask = np.where(valid, raw[f"{part}_mask"], LM.MISSING).astype(np.uint8)
        arrays = [f"{part}_px"] + ([f"{part}_world"] if part != "face" else [])
        filled_any = np.zeros(T, bool)
        for key in arrays:
            X = raw[key].astype(np.float64).copy()
            X[~valid] = np.nan
            sm[key], filled = clean_track(t, X, valid, raw[f"{part}_conf"], gmax, int(cl["interp_context_frames"]),
                                          float(cl["smoothing_sigma_s"]))
            sm[key] = sm[key].astype(np.float32)
            filled_any |= filled
        mask[filled_any] = LM.FILLED
        sm[f"{part}_mask"] = mask

    # ---- body-relative normalisation
    pose_ok = sm["pose_mask"] != LM.MISSING
    origin, scale, interp = body_frame(t, sm["pose_px"].astype(np.float64), sm["pose_world"].astype(np.float64),
                                       pose_ok, float(cl["scale_smoothing_sigma_s"]))
    nrm = {part: normalise(sm[f"{part}_px"].astype(np.float64), origin, scale) for part in LM.PARTS}

    feats = compute_features(sm, nrm, t)
    bones = {}
    trees = {"body": LM.BODY_TREE, "left_hand": LM.HAND_BONES, "right_hand": LM.HAND_BONES}
    for name, (root, L, th, ph) in feats["bones"].items():
        bones[name] = {"root": root, "length": L, "theta": th, "phi": ph,
                       "__attrs__": {"tree": [[a if a == "root" else int(a), int(b)] for a, b in trees[name]],
                                     "units": "metres (MediaPipe world), radians",
                                     "source": "smoothed world landmarks"}}

    tree = {
        "video": {"__attrs__": meta},
        "time": {"timestamps_s": t, "frame_index": np.array([r["i"] for r in rows], np.int32)},
        "quality": {"brightness": np.array([r["brightness"] for r in rows], np.float32),
                    "blur": np.array([r["blur"] for r in rows], np.float32),
                    "motion_frac": np.array([r["motion"] for r in rows], np.float32),
                    "enhanced": np.array([r["enhanced"] for r in rows], np.uint8)},
        "raw": raw,
        "smoothed": sm,
        "norm": {"origin_px": origin, "scale_px": scale, "origin_interpolated": interp.astype(np.uint8),
                 **{p: nrm[p].astype(np.float32) for p in LM.PARTS},
                 "__attrs__": {"definition": "(xy_px - origin_px) / scale_px ; z_px / scale_px",
                               "source": "smoothed px"}},
        "features": {"angles": feats["angles"].astype(np.float32),
                     "distances": feats["distances"].astype(np.float32),
                     **{k: v.astype(np.float32) for k, v in feats["extra"].items()},
                     "__attrs__": {"angle_names": feats["angle_names"], "angle_units": "degrees",
                                   "distance_names": feats["distance_names"]}},
        "bones": bones,
        "names": {"__attrs__": {"pose": LM.POSE_NAMES, "hand": LM.HAND_NAMES, "face": LM.FACE_NAMES,
                                "face_mediapipe_index": LM.FACE_MP_IDX.tolist(),
                                "mask_codes": {"0": "missing", "1": "detected", "2": "detected_on_upscaled_crop",
                                               "3": "interpolated"},
                                "hand_reject_bits": {"1": "duplicate_of_other_hand", "2": "far_from_pose_wrist",
                                                     "4": "shape_differs_from_clip_profile",
                                                     "8": "temporal_spike_hampel"}}},
    }
    meta["n_frames"] = T
    meta["duration_s"] = float(t[-1] - t[0] + dt) if T > 1 else 0.0
    meta["fps_measured"] = float(1.0 / dt) if dt > 0 else 0.0
    tree["qc"] = {"__attrs__": label_free(tree)}
    return tree


def file_attrs(cfg, meta):
    return {"pipeline_version": cfg["pipeline_version"], "config_hash": cfg.extraction_hash(),
            "config": cfg.extraction_dict(), "packages": package_versions(),
            "models": model_hashes(cfg.path("models_dir")), "source_sha256": meta["sha256"],
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}


def process_video(path, cfg, extractor=None, frame_fn=None, frame_select=None, meta=None, progress=None):
    """Full pipeline for one video. Returns (tree, meta)."""
    from .extractor import Extractor
    meta = dict(meta) if meta is not None else probe(path)
    own = extractor is None
    ex = extractor or Extractor(cfg, "video")
    try:
        rows = extract_frames(meta, cfg, ex, frame_fn, frame_select, progress)
    finally:
        if own:
            ex.close()
    return build_tree(rows, meta, cfg), meta
