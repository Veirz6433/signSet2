"""Video probing and decoding with explicit rotation, pixel-aspect and per-frame timestamps.

Rotation handling (phone videos store sideways frames + a rotation tag):
  1. read the tag ourselves with PyAV (display matrix side data, frame.rotation, or the
     legacy "rotate" metadata), converted to a clockwise angle;
  2. rotate frames ourselves (never rely on a decoder's auto-rotation);
  3. verify: OpenCV is opened with auto-orientation ON and its first frame is compared with
     our rotated first frame for k in {0, 90, 180, 270} (normalised cross-correlation);
  4. store rotation, its source and the verification result in the HDF5 file.
"""
from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

try:
    import av  # PyAV
except Exception:  # pragma: no cover - optional at import time, required for best results
    av = None


def file_sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _rot_cw(img, deg):
    k = (int(deg) // 90) % 4
    return img if k == 0 else np.ascontiguousarray(np.rot90(img, k=-k))


def _ncc(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _displaymatrix_cw(frame):
    """Clockwise rotation from a frame's DISPLAYMATRIX side data, or None."""
    try:
        for sd in frame.side_data:
            name = str(getattr(sd, "type", "")).upper()
            if "DISPLAYMATRIX" in name:
                m = np.frombuffer(bytes(sd), dtype="<i4")[:9].astype(np.float64) / 65536.0
                s0, s1 = np.hypot(m[0], m[3]), np.hypot(m[1], m[4])
                if s0 == 0 or s1 == 0:
                    return None
                ccw = -np.degrees(np.arctan2(m[1] / s1, m[0] / s0))   # av_display_rotation_get
                return int(round(-ccw / 90.0)) * 90 % 360
    except Exception:
        return None
    return None


def _opencv_first_frame(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, None
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass
    meta = cap.get(cv2.CAP_PROP_ORIENTATION_META) if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else 0
    ok, bgr = cap.read()
    cap.release()
    return (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None), int(round(meta or 0)) % 360


def _match_rotation(raw_rgb, ref_rgb):
    """Which clockwise rotation of raw_rgb best matches ref_rgb. Returns (deg, score, margin)."""
    ref = cv2.resize(cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY), (64, 64), interpolation=cv2.INTER_AREA)
    ref_aspect = ref_rgb.shape[1] / ref_rgb.shape[0]
    scores = {}
    for deg in (0, 90, 180, 270):
        cand = _rot_cw(raw_rgb, deg)
        aspect = cand.shape[1] / cand.shape[0]
        if abs(aspect / ref_aspect - 1) > 0.03:
            scores[deg] = -1.0
            continue
        g = cv2.resize(cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY), (64, 64), interpolation=cv2.INTER_AREA)
        scores[deg] = _ncc(g, ref)
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    best, s1 = ordered[0]
    margin = s1 - ordered[1][1]
    return best, s1, margin


def probe(path) -> dict:
    """Read all metadata needed to decode, rotate and later overlay the video."""
    path = Path(path)
    meta = {"path": str(path), "file_name": path.name, "sha256": file_sha256(path),
            "decoder": "pyav" if av is not None else "opencv"}
    first_raw = None
    meta_rot, rot_source = 0, "none"
    if av is not None:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            cc = stream.codec_context
            meta.update(codec=cc.name, coded_width=int(cc.width), coded_height=int(cc.height))
            rate = stream.average_rate or stream.guessed_rate or Fraction(30, 1)
            meta["fps_avg"] = float(rate)
            meta["n_frames_header"] = int(stream.frames or 0)
            if stream.duration is not None and stream.time_base is not None:
                meta["duration_s_header"] = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                meta["duration_s_header"] = float(container.duration / 1_000_000)
            sar = stream.sample_aspect_ratio or getattr(cc, "sample_aspect_ratio", None)
            meta["sar"] = float(sar) if sar else 1.0
            legacy = stream.metadata.get("rotate") if stream.metadata else None
            for frame in container.decode(stream):
                first_raw = frame.to_ndarray(format="rgb24")
                r = _displaymatrix_cw(frame)
                if r is not None:
                    meta_rot, rot_source = r, "displaymatrix"
                elif getattr(frame, "rotation", None) not in (None, 0):
                    meta_rot, rot_source = int(round(-float(frame.rotation) / 90.0)) * 90 % 360, "frame.rotation"
                elif legacy is not None:
                    meta_rot, rot_source = int(round(float(legacy) / 90.0)) * 90 % 360, "legacy_rotate_tag"
                break
    cv_first, cv_meta = _opencv_first_frame(path)
    if av is None:
        if cv_first is None:
            raise IOError(f"cannot open video: {path}")
        cap = cv2.VideoCapture(str(path))
        meta.update(codec="unknown", coded_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    coded_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    fps_avg=float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
                    n_frames_header=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), sar=1.0)
        cap.release()
        meta_rot, rot_source = cv_meta, "opencv_orientation_meta"
    if first_raw is None and av is not None:
        raise IOError(f"no decodable video frames: {path}")

    # ---- decide and verify rotation
    already_rotated = False
    verified, note = False, ""
    if av is not None:
        h0, w0 = first_raw.shape[:2]
        if (h0, w0) == (meta["coded_width"], meta["coded_height"]) and meta_rot in (90, 270) \
                and meta["coded_width"] != meta["coded_height"]:
            already_rotated = True        # decoder applied rotation itself; do not rotate twice
        rotation = meta_rot
        if cv_first is not None:
            # raw (unrotated) first frame; if the decoder already rotated, undo it for the match
            raw0 = _rot_cw(first_raw, (360 - rotation) % 360) if already_rotated else first_raw
            best, score, margin = _match_rotation(raw0, cv_first)   # image-implied CW rotation
            meta["rotation_image_match"] = {"best_cw": int(best), "ncc": round(score, 4),
                                            "margin": round(margin, 4)}
            if cv_meta == meta_rot:
                verified = (best == rotation) and score > 0.8
                note = "opencv_agrees" if verified else f"image_match_disagrees(best={best},ncc={score:.3f})"
            elif meta_rot != 0 and cv_meta == 0:
                note = "opencv_reported_no_rotation_tag; kept PyAV value (cannot cross-check)"
            elif meta_rot == 0 and cv_meta != 0:
                if best == cv_meta and score > 0.8 and margin > 0.05:
                    rotation, rot_source, verified = cv_meta, "opencv_orientation_meta", True
                    note = "pyav_missed_tag; opencv tag confirmed by image match"
                else:
                    note = f"tags_disagree(pyav=0,opencv={cv_meta}); kept 0"
            else:
                note = f"tags_disagree(pyav={meta_rot},opencv={cv_meta}); image_best={best}"
                if best in (meta_rot, cv_meta) and score > 0.8 and margin > 0.05:
                    rotation, verified = best, True
    else:
        rotation, already_rotated, note = cv_meta, True, "opencv_decoder_autorotates"
    meta.update(rotation_cw=int(rotation), rotation_source=rot_source, rotation_verified=bool(verified),
                rotation_note=note, decoder_already_rotated=bool(already_rotated),
                opencv_orientation_meta=int(cv_meta or 0))

    # ---- display geometry (square pixels, rotated)
    w_sq = int(round(meta["coded_width"] * meta["sar"]))
    h_sq = int(meta["coded_height"])
    if rotation in (90, 270):
        w_sq, h_sq = h_sq, w_sq
    meta["display_width"], meta["display_height"] = int(w_sq), int(h_sq)
    if av is None and cv_first is not None:      # OpenCV already rotated: trust its frame size
        meta["display_height"], meta["display_width"] = int(cv_first.shape[0]), int(cv_first.shape[1])
    return meta


def iter_frames(meta: dict):
    """Yield (frame_index, timestamp_s, rgb uint8 [display_height, display_width, 3])."""
    W, H = meta["display_width"], meta["display_height"]
    rot, sar = meta["rotation_cw"], meta.get("sar", 1.0)
    fps = meta.get("fps_avg") or 30.0
    last_t = -np.inf
    if meta["decoder"] == "pyav" and av is not None:
        with av.open(meta["path"]) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for i, frame in enumerate(container.decode(stream)):
                img = frame.to_ndarray(format="rgb24")
                if abs(sar - 1.0) > 1e-3 and not meta["decoder_already_rotated"]:
                    img = cv2.resize(img, (int(round(img.shape[1] * sar)), img.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
                if not meta["decoder_already_rotated"]:
                    img = _rot_cw(img, rot)
                if img.shape[:2] != (H, W):
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                t = frame.time if frame.time is not None else i / fps
                if t <= last_t:
                    t = last_t + 1e-6
                last_t = t
                yield i, float(t), np.ascontiguousarray(img)
    else:
        cap = cv2.VideoCapture(meta["path"])
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        except Exception:
            pass
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not np.isfinite(t) or t <= 0 and i > 0:
                t = i / fps
            if t <= last_t:
                t = last_t + 1e-6
            last_t = t
            yield i, float(t), img
            i += 1
        cap.release()
