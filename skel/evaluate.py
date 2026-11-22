"""Pilot evaluation on a few clips (no hand labels needed - Option B).

Per clip:
  clean run (saved to HDF5 + overlay MP4), determinism repeat, round trips,
  known-transform equivariance, degradation curves, gap-fill test, smoothing sweep,
  label-free QC, optional second-model agreement.
Across clips: calibration of label-free QC metrics against degradation error.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from . import landmarks as LM
from .config import load_config
from .features import decode_bones
from .metrics import (face_scale, hand_scale, jitter, kneedle, lag_seconds, normalised_errors,
                      peak_loss, spearman, summarise_errors, youden_threshold)
from .postprocess import OneEuro, fill_gaps, gaussian_smooth, median_dt, normalise
from .outliers import accepted_px
from .transforms import Variant, variant_list

CAL_METRICS = ["any_hand_detection_rate", "left_hand_detection_rate", "right_hand_detection_rate",
               "left_hand_accepted_rate", "right_hand_accepted_rate", "hand_rejected_rate",
               "hand_jitter", "body_jitter", "hand_bone_cv", "arm_bone_cv", "symmetry_error",
               "angle_violation_rate", "mean_hand_conf", "mean_body_conf", "mean_brightness",
               "median_blur", "hand_crop_redetect_rate"]


# --------------------------------------------------------------------------- worker
def _run_variant(args):
    clip, meta, cfg_path, kind, param = args
    from .pipeline import process_video
    cfg = load_config(cfg_path)
    v = Variant(kind, param, meta["display_width"], meta["display_height"], cfg)
    t0 = time.time()
    tree, _ = process_video(clip, cfg, frame_fn=v if v.needs_frame_fn() else None,
                            frame_select=v.frame_select(), meta=meta)
    tree.pop("video", None)
    return kind, param, tree, time.time() - t0


def _map(jobs, workers):
    if workers <= 1:
        for j in jobs:
            yield _run_variant(j)
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        yield from pool.imap_unordered(_run_variant, jobs)


# --------------------------------------------------------------------------- comparison
def _part_arrays(tree, part, stage="accepted"):
    """Detections of a part: 'accepted' = after hand outlier rejection, 'raw' = detector output."""
    raw = tree["raw"]
    if stage == "accepted":
        return accepted_px(raw, part)
    return raw[f"{part}_px"].astype(np.float64), raw[f"{part}_mask"] != LM.MISSING


def compare(ref, var, variant: Variant, cfg, stage="accepted"):
    """Per-part normalised error of `var` (mapped back to the original frame) vs `ref`."""
    ev = cfg["evaluation"]
    r_idx = {int(f): k for k, f in enumerate(ref["time"]["frame_index"])}
    v_frames = var["time"]["frame_index"]
    rows = [(r_idx[int(f)], k, int(f)) for k, f in enumerate(v_frames) if int(f) in r_idx]
    rr = np.array([a for a, _, _ in rows])
    vv = np.array([b for _, b, _ in rows])
    ff = np.array([c for _, _, c in rows])
    W, H = variant.W, variant.H
    out = {}
    for part in LM.PARTS:
        src = part
        if variant.mirror and part in ("left_hand", "right_hand"):
            src = "right_hand" if part == "left_hand" else "left_hand"
        rpx, rvalid = _part_arrays(ref, part, stage)
        vpx, vvalid = _part_arrays(var, src, stage)
        rpx, rvalid, vpx, vvalid = rpx[rr], rvalid[rr], vpx[vv], vvalid[vv]
        if variant.mirror and part == "pose":
            vpx = vpx[:, LM.POSE_MIRROR]
        if variant.mirror and part == "face":
            vpx = vpx[:, LM.FACE_MIRROR]
        mapped = vpx[..., :2].copy()
        inside = np.ones(rpx.shape[:2], bool)
        if variant.geometric:
            for n, f in enumerate(ff):
                mapped[n] = variant.inverse_points(int(f), vpx[n, :, :2])
                fwd = variant.forward_points(int(f), rpx[n, :, :2])
                inside[n] = (fwd[:, 0] >= 0) & (fwd[:, 0] < W) & (fwd[:, 1] >= 0) & (fwd[:, 1] < H)
        if part == "pose":
            J = LM.BODY_EVAL
            scale = ref["norm"]["scale_px"][rr]
        elif part == "face":
            J = slice(None)
            scale = face_scale(rpx)
        else:
            J = slice(None)
            scale = hand_scale(rpx)
        rv = rvalid[:, None] & inside
        e = normalised_errors(mapped[:, J], rpx[:, J, :2], scale, rv[:, J], vvalid)
        out["body" if part == "pose" else part] = summarise_errors(e, ev["pck_alphas"], ev["auc_alpha_max"])
        if part in ("left_hand", "right_hand"):
            out.setdefault("_hand_errors", []).append(e)
    he = np.concatenate(out.pop("_hand_errors"), axis=0)
    out["hands"] = summarise_errors(he, ev["pck_alphas"], ev["auc_alpha_max"])
    return out


def determinism(a, b):
    diffs, identical = {}, True
    for grp in ("raw", "smoothed", "norm", "features"):
        for k, x in a[grp].items():
            if k == "__attrs__" or k not in b[grp]:
                continue
            x, y = np.asarray(x), np.asarray(b[grp][k])
            if x.shape != y.shape:
                diffs[f"{grp}/{k}"] = "shape_mismatch"
                identical = False
                continue
            if x.dtype.kind == "f":
                same_nan = np.array_equal(np.isnan(x), np.isnan(y))
                d = np.nanmax(np.abs(x.astype(np.float64) - y)) if np.isfinite(x).any() else 0.0
                diffs[f"{grp}/{k}"] = float(d) if same_nan else "nan_pattern_differs"
                identical &= same_nan and d == 0
            else:
                eq = np.array_equal(x, y)
                diffs[f"{grp}/{k}"] = 0.0 if eq else "differs"
                identical &= eq
    numeric = [v for v in diffs.values() if isinstance(v, float)]
    return {"bitwise_identical": bool(identical), "max_abs_diff": float(max(numeric) if numeric else 0.0),
            "non_numeric_differences": [k for k, v in diffs.items() if not isinstance(v, float)]}


# --------------------------------------------------------------------------- round trips
def roundtrips(file_tree):
    """Computed from the tree READ BACK FROM THE HDF5 FILE."""
    from .postprocess import denormalise
    from .retarget import retarget
    nr, sm = file_tree["norm"], file_tree["smoothed"]
    px_err = []
    for p in LM.PARTS:
        rec = denormalise(nr[p].astype(np.float64), nr["origin_px"], nr["scale_px"])
        d = np.linalg.norm(rec[..., :2] - sm[f"{p}_px"][..., :2].astype(np.float64), axis=-1)
        px_err.append(d[np.isfinite(d)])
    px_err = np.concatenate(px_err)
    out = {"norm_to_px_err_px_max": float(px_err.max()) if px_err.size else 0.0,
           "norm_to_px_err_px_mean": float(px_err.mean()) if px_err.size else 0.0}
    b = file_tree["bones"]
    fk = {}
    for name, tree_def, world_key in (("body", LM.BODY_TREE, "pose_world"),
                                      ("left_hand", LM.HAND_BONES, "left_hand_world"),
                                      ("right_hand", LM.HAND_BONES, "right_hand_world")):
        g = b[name]
        n = 33 if name == "body" else 21
        rec = decode_bones(tree_def, g["root"].astype(np.float64), g["length"].astype(np.float64),
                           g["theta"].astype(np.float64), g["phi"].astype(np.float64), n)
        true = sm[world_key].astype(np.float64)
        nodes = sorted({x for e in tree_def for x in e if x != "root"})
        d = np.linalg.norm(rec[:, nodes] - true[:, nodes], axis=-1)
        d = d[np.isfinite(d)]
        fk[name] = {"max_mm": float(d.max() * 1000) if d.size else float("nan"),
                    "mean_mm": float(d.mean() * 1000) if d.size else float("nan")}
    out["angles_lengths_to_coords"] = fk
    # avatar retarget: lengths fixed -> CV must be 0, directions unchanged
    rt = retarget(b)
    cvs, ang = [], []
    for name, tree_def in (("body", LM.BODY_TREE), ("left_hand", LM.HAND_BONES), ("right_hand", LM.HAND_BONES)):
        P = rt[name]
        g = b[name]
        for k, (a, c) in enumerate(tree_def):
            pa = np.zeros_like(P[:, c]) if a == "root" else P[:, a]
            v = P[:, c] - pa
            L = np.linalg.norm(v, axis=-1)
            ok = np.isfinite(L)
            if ok.sum() > 2:
                cvs.append(L[ok].std() / L[ok].mean())
                th = np.arccos(np.clip(v[ok, 2] / L[ok], -1, 1))
                ang.append(np.nanmax(np.abs(th - g["theta"][ok, k])))
    out["retarget_bone_length_cv_max"] = float(max(cvs)) if cvs else float("nan")
    out["retarget_direction_err_deg_max"] = float(np.degrees(max(ang))) if ang else float("nan")
    return out


# --------------------------------------------------------------------------- gap-fill test
def gap_test(tree, cfg):
    """G4 in OUTPUT space: truth = smooth(fill(accepted track)); test = the same pipeline after
    hiding g accepted frames. e = ||out - truth|| / box(truth) at the hidden frames."""
    from .pipeline import clean_track
    gt = cfg["evaluation"]["gap_test"]
    ctx = int(cfg["cleanup"]["interp_context_frames"])
    sigma = float(cfg["cleanup"]["smoothing_sigma_s"])
    rng = np.random.default_rng(int(cfg["evaluation"]["seed"]))
    t = tree["time"]["timestamps_s"]
    dt = median_dt(t)
    res = []
    for g in range(1, int(gt["max_gap_frames"]) + 1):
        errs = []
        for side in ("left_hand", "right_hand"):
            px, valid = _part_arrays(tree, side)
            conf = tree["raw"][f"{side}_conf"]
            truth, _ = clean_track(t, px, valid, conf, g, ctx, sigma)
            T = len(t)
            starts = [a for a in range(ctx, T - g - ctx) if valid[a - ctx:a + g + ctx].all()]
            if not starts:
                continue
            pick = rng.choice(starts, min(len(starts), int(gt["trials_per_gap"]) // 2), replace=False)
            for a in pick:
                v2 = valid.copy()
                v2[a:a + g] = False
                X = px.copy()
                X[a:a + g] = np.nan
                out, _ = clean_track(t, X, v2, conf, g, ctx, sigma)
                tr = truth[a:a + g]
                s = hand_scale(tr)
                e = np.linalg.norm(out[a:a + g, :, :2] - tr[..., :2], axis=-1) / s[:, None]
                errs.append(e.ravel())
        if not errs:
            res.append({"gap_frames": g, "gap_s": g * dt, "n": 0})
            continue
        e = np.concatenate(errs)
        e = e[np.isfinite(e)]
        res.append({"gap_frames": g, "gap_s": g * dt, "n": int(e.size), "nme": float(np.mean(e)),
                    "pck": float((e < gt["pck_alpha"]).mean()), "p95_err": float(np.percentile(e, 95))})
    g_star = 0
    for r in res:
        if r.get("n") and r["pck"] >= gt["required_pck"]:
            g_star = r["gap_frames"]
        elif r.get("n"):
            break
    return {"dt_s": dt, "curve": res, "max_fillable_frames": g_star, "max_fillable_s": g_star * dt,
            "criterion": f"PCK@{gt['pck_alpha']} >= {gt['required_pck']}", "space": "output (filled + smoothed)"}


# --------------------------------------------------------------------------- smoothing sweep
def smoothing_sweep(tree, cfg):
    sw = cfg["evaluation"]["smoothing_sweep"]
    t = tree["time"]["timestamps_s"]
    o, s = tree["norm"]["origin_px"], tree["norm"]["scale_px"]
    dt = median_dt(t)
    gmax = int(np.floor(cfg["cleanup"]["max_gap_s"] / dt + 1e-9))
    Xs, Ws, hs = [], [], []
    for side in ("left_hand", "right_hand"):
        px, valid = _part_arrays(tree, side)
        Xf, filled = fill_gaps(t, px, valid, gmax, int(cfg["cleanup"]["interp_context_frames"]))
        Xs.append(normalise(Xf, o, s)[..., :2])
        Ws.append(np.repeat((valid | filled)[:, None], 21, axis=1).astype(float))
        hs.append(np.repeat((hand_scale(px) / s)[:, None], 21, axis=1))
    X, Wt, hsz = np.concatenate(Xs, 1), np.concatenate(Ws, 1), np.concatenate(hs, 1)
    J0 = jitter(t, X)

    def row(label, kind, param, Y):
        J = jitter(t, Y)
        d = np.linalg.norm(Y - X, axis=-1) / hsz
        return {"label": label, "kind": kind, "param": param, "jitter": J,
                "jitter_reduction": float(1 - J / J0) if J0 > 0 else float("nan"),
                "lag_ms": lag_seconds(t, X, Y, sw["max_lag_s"]) * 1000.0,
                "peak_loss": peak_loss(t, X, Y),
                "fidelity_nme": float(np.nanmean(d)) if np.isfinite(d).any() else float("nan")}

    rows = []
    for sg in sw["offline_sigma_s"]:
        rows.append(row(f"gauss sigma={sg}s", "offline", sg, gaussian_smooth(t, X, Wt, float(sg))))
    for mc in sw["live_min_cutoff_hz"]:
        Y = np.concatenate([OneEuro(mc, sw["live_beta"], sw["live_d_cutoff_hz"])
                            .run(t, X[:, h * 21:(h + 1) * 21].reshape(len(t), -1)).reshape(len(t), 21, 2)
                            for h in (0, 1)], axis=1)
        rows.append(row(f"one-euro fc_min={mc}Hz", "live", mc, Y))
    off = [r for r in rows if r["kind"] == "offline"]
    k = kneedle([r["peak_loss"] for r in off], [r["jitter_reduction"] for r in off])
    live = [r for r in rows if r["kind"] == "live"]
    kl = kneedle([r["peak_loss"] for r in live], [r["jitter_reduction"] for r in live])
    return {"raw_jitter": J0, "rows": rows, "recommended_offline_sigma_s": off[k]["param"],
            "recommended_live_min_cutoff_hz": live[kl]["param"]}


# --------------------------------------------------------------------------- calibration
def calibrate(points, cfg):
    """points: list of (qc_metrics dict, hand PCK@0.1 vs clean, clip baseline b_c).
    y = PCK / b_c (robustness relative to how stable the clean clip itself is, from G2);
    a variant is bad if y < bad_if_below_fraction_of_baseline."""
    c = cfg["evaluation"]["calibration"]
    frac = float(c.get("bad_if_below_fraction_of_baseline", 0.85))
    y = np.array([p[1] / p[2] if len(p) > 2 and p[2] and np.isfinite(p[2]) else p[1] for p in points], float)
    bad = y < frac
    out = {"n_variants": int(len(points)), "n_bad": int(bad.sum()), "bad_if_below_fraction_of_baseline": frac,
           "baselines": sorted({round(float(p[2]), 4) for p in points if len(p) > 2 and p[2] is not None}),
           "metrics": {}}
    for m in CAL_METRICS:
        x = np.array([p[0].get(m, np.nan) for p in points], float)
        rho = spearman(x, y)
        higher_is_worse = bool(rho < 0) if np.isfinite(rho) else True
        thr, J = youden_threshold(x, bad, higher_is_worse)
        out["metrics"][m] = {"spearman_rho_vs_hand_pck": rho, "higher_is_worse": higher_is_worse,
                             "threshold": thr, "youden_J": J,
                             "used_for_qc": bool(np.isfinite(rho) and abs(rho) >= c["min_abs_spearman"]
                                                 and np.isfinite(thr))}
    return out


def qc_status(qc: dict, calibration: dict | None, cfg) -> dict:
    hard = float(cfg["qc"]["hard_min_any_hand_detection"])
    flags = []
    if calibration:
        for m, d in calibration["metrics"].items():
            if not d["used_for_qc"] or m not in qc or not np.isfinite(qc[m]):
                continue
            if (qc[m] >= d["threshold"]) if d["higher_is_worse"] else (qc[m] <= d["threshold"]):
                flags.append(m)
    status = "pass" if not flags else ("warn" if len(flags) == 1 else "fail")
    if qc.get("any_hand_detection_rate", 0) < hard:
        status = "fail"
        flags.append("any_hand_detection_rate<hard_min")
    return {"status": status, "flags": flags, "calibrated": bool(calibration)}


# --------------------------------------------------------------------------- driver
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def evaluate_clips(clips, cfg, out_dir: Path, log=print):
    from .overlay import render
    from .pipeline import file_attrs, process_video
    from .storage import read_h5, write_h5
    from .video_io import probe
    from . import second_model

    out_dir.mkdir(parents=True, exist_ok=True)
    workers = int(cfg["evaluation"]["workers"])
    variants = variant_list(cfg)
    cal_points, summaries = [], []
    use_second = cfg["evaluation"]["second_model"]["enabled"]
    if use_second == "auto":
        use_second = second_model.available()
    elif use_second and not second_model.available():
        raise RuntimeError("second_model.enabled=true but rtmlib is not installed (pip install -r requirements-optional.txt)")

    for clip in clips:
        clip = Path(clip)
        cdir = out_dir / clip.stem
        cdir.mkdir(parents=True, exist_ok=True)
        log(f"\n=== {clip.name}")
        meta = probe(clip)
        log(f"  {meta['display_width']}x{meta['display_height']} rot={meta['rotation_cw']} "
            f"verified={meta['rotation_verified']} fps~{meta['fps_avg']:.2f}")
        t0 = time.time()
        ref, meta = process_video(clip, cfg, meta=meta)
        log(f"  clean extraction: {len(ref['time']['timestamps_s'])} frames in {time.time() - t0:.1f}s")
        h5 = cdir / f"{clip.stem}.h5"
        write_h5(h5, ref, file_attrs(cfg, meta))
        file_tree = read_h5(h5)
        ov = render(h5, clip, cdir / f"{clip.stem}_overlay.mp4", cfg)
        log(f"  overlay written: {ov['overlay']}")
        res = {"clip": clip.name, "sha256": meta["sha256"], "video": {k: meta.get(k) for k in (
            "display_width", "display_height", "rotation_cw", "rotation_source", "rotation_verified",
            "rotation_note", "fps_avg", "fps_measured", "n_frames", "duration_s", "codec", "decoder")},
            "h5_size_bytes": h5.stat().st_size, "label_free": ref["qc"]["__attrs__"], "overlay": ov,
            "roundtrip": roundtrips(file_tree), "gap_test": gap_test(ref, cfg),
            "smoothing_sweep": smoothing_sweep(ref, cfg), "transforms": {}, "degradations": {}}
        jobs = [(str(clip), meta, cfg["_path"], k, p) for k, p in variants]
        clip_points = []
        for n, (kind, param, tree, secs) in enumerate(_map(jobs, workers), 1):
            v = Variant(kind, param, meta["display_width"], meta["display_height"], cfg)
            if kind == "repeat":
                res["determinism"] = determinism(ref, tree)
            elif v.geometric:
                res["transforms"][v.name] = compare(ref, tree, v, cfg)
                res["transforms"][v.name]["hands_raw"] = compare(ref, tree, v, cfg, stage="raw")["hands"]
            else:
                cmp_ = compare(ref, tree, v, cfg)
                res["degradations"].setdefault(kind, []).append(
                    {"level": param, "vs_clean": cmp_, "label_free": tree["qc"]["__attrs__"]})
                clip_points.append((tree["qc"]["__attrs__"], cmp_["hands"]["pck@0.1"]))
            log(f"  [{n}/{len(jobs)}] {v.name} done ({secs:.1f}s)")
        for kind in res["degradations"]:
            res["degradations"][kind].sort(key=lambda r: r["level"])
        rigid = [c["hands"]["pck@0.1"] for n_, c in res["transforms"].items()
                 if n_.split("=")[0] in ("shift", "rotate", "scale") and np.isfinite(c["hands"].get("pck@0.1", np.nan))]
        res["hand_stability_baseline"] = float(np.median(rigid)) if rigid else float("nan")
        cal_points += [(q, p_, res["hand_stability_baseline"]) for q, p_ in clip_points]
        if use_second:
            log("  second model (RTMW) agreement ...")
            res["second_model_agreement"] = second_model.agreement(meta, ref, cfg)
        (cdir / "results.json").write_text(json.dumps(_jsonable(res), indent=2), encoding="utf-8")
        summaries.append(res)

    cal = calibrate(cal_points, cfg)
    (out_dir / "calibration.json").write_text(json.dumps(_jsonable(cal), indent=2), encoding="utf-8")
    for res in summaries:
        res["qc_status"] = qc_status(res["label_free"], cal, cfg)
    summary = {"clips": [s["clip"] for s in summaries], "config_hash": cfg.extraction_hash(),
               "pipeline_version": cfg["pipeline_version"], "calibration": cal,
               "qc": {s["clip"]: s["qc_status"] for s in summaries}}
    (out_dir / "pilot_summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    for res in summaries:
        (out_dir / Path(res["clip"]).stem / "results.json").write_text(json.dumps(_jsonable(res), indent=2), encoding="utf-8")
    return summaries, cal
