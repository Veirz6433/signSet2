"""Evaluation metrics. Every formula here is written out in report.md."""
from __future__ import annotations

import numpy as np
from scipy import stats

from . import landmarks as LM
from .features import angle_limit_key
from .postprocess import median_dt, normalise


# --------------------------------------------------------------------------- accuracy vs a reference
def normalised_errors(pred, ref, scale, ref_valid, pred_valid):
    """e = ||p_hat - p||_2 / s for every (sample, joint) where the reference exists.
    Missing predictions get e = +inf (they count as failures in PCK, excluded from NME).
    pred, ref: (S, J, 2); scale: (S,); *_valid: (S,) or (S, J) bool."""
    S, J = ref.shape[:2]
    rv = np.broadcast_to(ref_valid if np.ndim(ref_valid) == 2 else np.asarray(ref_valid)[:, None], (S, J))
    pv = np.broadcast_to(pred_valid if np.ndim(pred_valid) == 2 else np.asarray(pred_valid)[:, None], (S, J))
    rv = rv & np.all(np.isfinite(ref[..., :2]), axis=-1) & np.isfinite(scale)[:, None] & (scale[:, None] > 0)
    pv = pv & np.all(np.isfinite(pred[..., :2]), axis=-1)
    e = np.linalg.norm(pred[..., :2] - ref[..., :2], axis=-1) / np.where(scale > 0, scale, np.nan)[:, None]
    e = np.where(pv, e, np.inf)
    return np.where(rv, e, np.nan)          # NaN = not scored


def summarise_errors(e, alphas=(0.05, 0.1, 0.2), auc_max=0.2, boot=0, seed=0, by_sample=True):
    """NME over detected joints, PCK@alpha with misses as failures, AUC of PCK on [0, auc_max],
    detection rate, and optional bootstrap 95% CIs (resampling samples = rows)."""
    scored = np.isfinite(e) | np.isinf(e)
    det = np.isfinite(e)
    out = {"n_samples": int(scored.any(axis=1).sum()), "n_joints": int(scored.sum()),
           "detection_rate": float(det.sum() / max(scored.sum(), 1))}
    out["nme"] = float(np.nanmean(np.where(det, e, np.nan))) if det.any() else float("nan")
    for a in alphas:
        out[f"pck@{a}"] = float((e[scored] < a).mean()) if scored.any() else float("nan")
    grid = np.linspace(0, auc_max, 21)
    if scored.any():
        pcks = np.array([(e[scored] < a).mean() for a in grid])
        out[f"auc@{auc_max}"] = float(np.trapezoid(pcks, grid) / auc_max) if hasattr(np, "trapezoid") \
            else float(np.trapz(pcks, grid) / auc_max)
    if boot and out["n_samples"] > 1:
        rows = np.where(scored.any(axis=1))[0]
        rng = np.random.default_rng(seed)
        nmes, pck1 = [], []
        for _ in range(boot):
            r = e[rng.choice(rows, len(rows), replace=True)]
            sc, dt_ = np.isfinite(r) | np.isinf(r), np.isfinite(r)
            nmes.append(np.nanmean(np.where(dt_, r, np.nan)) if dt_.any() else np.nan)
            pck1.append((r[sc] < 0.1).mean())
        out["nme_ci95"] = [float(np.nanpercentile(nmes, 2.5)), float(np.nanpercentile(nmes, 97.5))]
        out["pck@0.1_ci95"] = [float(np.percentile(pck1, 2.5)), float(np.percentile(pck1, 97.5))]
    return out


def hand_scale(px):
    """Hand size s_h = max(width, height) of the 2D bounding box of the 21 points. px: (T,21,>=2)."""
    import warnings
    xy = px[..., :2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(np.nanmax(xy, axis=1) - np.nanmin(xy, axis=1), axis=-1)


def face_scale(px):
    a, b = LM.INTEROCULAR
    return np.linalg.norm(px[:, a, :2] - px[:, b, :2], axis=-1)


# --------------------------------------------------------------------------- temporal
def derivatives(t, X):
    """Non-uniform finite differences: v_i = (x_{i+1}-x_i)/(t_{i+1}-t_i),
    a_i = (v_i - v_{i-1}) / ((t_{i+1} - t_{i-1})/2).  X: (T, ..., D)."""
    t = np.asarray(t, np.float64)
    dt = np.diff(t).reshape((-1,) + (1,) * (X.ndim - 1))
    v = np.diff(X, axis=0) / dt
    dt2 = (0.5 * (t[2:] - t[:-2])).reshape((-1,) + (1,) * (X.ndim - 1))
    a = np.diff(v, axis=0) / dt2
    return v, a


def jitter(t, X):
    """Mean acceleration magnitude J = mean_{i,j} ||a_ij|| over frames where x_{i-1}, x_i, x_{i+1}
    all exist. X in body units (shoulder widths) -> J in shoulder widths / s^2."""
    if len(t) < 3:
        return float("nan")
    _, a = derivatives(t, np.asarray(X, np.float64))
    m = np.linalg.norm(a, axis=-1)
    return float(np.nanmean(m)) if np.isfinite(m).any() else float("nan")


def speed_series(t, X):
    v, _ = derivatives(t, np.asarray(X, np.float64)) if len(t) >= 3 else (None, None)
    if v is None:
        return np.array([]), np.array([])
    sp = np.linalg.norm(v, axis=-1)
    sp = np.nanmean(sp.reshape(sp.shape[0], -1), axis=1) if sp.ndim > 1 else sp
    tm = 0.5 * (np.asarray(t)[1:] + np.asarray(t)[:-1])
    return tm, sp


def lag_seconds(t, ref, test, max_lag_s=0.3):
    """Delay of `test` relative to `ref`: tau* = argmax_tau sum_c corr(x_c(t), y_c(t + tau)) over all
    joint coordinates c (mean-removed, resampled to a uniform grid at the median frame interval).
    Positive = test lags behind ref."""
    t = np.asarray(t, np.float64)
    R = np.asarray(ref, np.float64).reshape(len(t), -1)
    S = np.asarray(test, np.float64).reshape(len(t), -1)
    dt = median_dt(t)
    grid = np.arange(t[0], t[-1], dt)
    A, B = [], []
    for c in range(R.shape[1]):
        ok = np.isfinite(R[:, c]) & np.isfinite(S[:, c])
        if ok.sum() < 10:
            continue
        a = np.interp(grid, t[ok], R[ok, c])
        b = np.interp(grid, t[ok], S[ok, c])
        A.append(a - a.mean())
        B.append(b - b.mean())
    if not A:
        return float("nan")
    a, b = np.array(A), np.array(B)
    K = int(round(max_lag_s / dt))

    def corr(k):
        x, y = (a[:, :a.shape[1] - k], b[:, k:]) if k >= 0 else (a[:, -k:], b[:, :b.shape[1] + k])
        if x.shape[1] < 5:
            return -np.inf
        d = np.sqrt((x * x).sum() * (y * y).sum())
        return (x * y).sum() / d if d > 0 else -np.inf

    cs = {k: corr(k) for k in range(-K - 1, K + 2)}
    best_k = max(range(-K, K + 1), key=lambda k: cs[k])
    c0, cm, cp = cs[best_k], cs[best_k - 1], cs[best_k + 1]
    den = cm - 2 * c0 + cp
    delta = 0.5 * (cm - cp) / den if np.isfinite(den) and den < 0 else 0.0   # parabolic sub-frame peak
    return float((best_k + np.clip(delta, -0.5, 0.5)) * dt)


def peak_loss(t, ref, test, q=95):
    """1 - P95(|v_test|) / P95(|v_ref|): how much fast motion smoothing removes."""
    _, s_ref = speed_series(t, ref)
    _, s_tst = speed_series(t, test)
    ok = np.isfinite(s_ref) & np.isfinite(s_tst)
    if ok.sum() < 5:
        return float("nan")
    return float(1 - np.percentile(s_tst[ok], q) / max(np.percentile(s_ref[ok], q), 1e-12))


# --------------------------------------------------------------------------- physical plausibility
def bone_cv(world, bones):
    """Coefficient of variation of each 3D bone length over time, CV_b = sigma_b / mu_b."""
    out = []
    for a, b in bones:
        L = np.linalg.norm(world[:, b] - world[:, a], axis=-1)
        L = L[np.isfinite(L)]
        if L.size >= 3 and L.mean() > 0:
            out.append(L.std() / L.mean())
    return float(np.mean(out)) if out else float("nan")


def symmetry_error(pw, lhw, rhw):
    """|1 - median(L_left) / median(L_right)| averaged over upper arm, forearm, palm."""
    pairs = [((LM.L_SH, LM.L_EL), (LM.R_SH, LM.R_EL), pw, pw), ((LM.L_EL, LM.L_WR), (LM.R_EL, LM.R_WR), pw, pw),
             ((0, 9), (0, 9), lhw, rhw)]
    errs = []
    for (a1, b1), (a2, b2), A, B in pairs:
        l1 = np.linalg.norm(A[:, b1] - A[:, a1], axis=-1)
        l2 = np.linalg.norm(B[:, b2] - B[:, a2], axis=-1)
        m1, m2 = np.nanmedian(l1) if np.isfinite(l1).any() else np.nan, \
            np.nanmedian(l2) if np.isfinite(l2).any() else np.nan
        if np.isfinite(m1) and np.isfinite(m2) and m2 > 0:
            errs.append(abs(1 - m1 / m2))
    return float(np.mean(errs)) if errs else float("nan")


def violation_rate(angle_names, angles):
    """Fraction of finite (frame, joint-angle) values outside the anatomical range."""
    bad = tot = 0
    for j, n in enumerate(angle_names):
        key = angle_limit_key(n)
        if key is None:
            continue
        lo, hi = LM.ANGLE_LIMITS[key]
        v = angles[:, j]
        v = v[np.isfinite(v)]
        tot += v.size
        bad += int(((v < lo) | (v > hi)).sum())
    return float(bad / tot) if tot else float("nan")


# --------------------------------------------------------------------------- statistics helpers
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def youden_threshold(score, bad, higher_is_worse=True):
    """Threshold maximising J = TPR - FPR for flagging bad samples."""
    s, b = np.asarray(score, float), np.asarray(bad, bool)
    ok = np.isfinite(s)
    s, b = s[ok], b[ok]
    if b.sum() == 0 or (~b).sum() == 0:
        return float("nan"), float("nan")
    if not higher_is_worse:
        s = -s
    best_j, best_t = -1.0, float("nan")
    for thr in np.unique(s):
        flag = s >= thr
        j = flag[b].mean() - flag[~b].mean()
        if j > best_j:
            best_j, best_t = j, thr
    return float(best_t if higher_is_worse else -best_t), float(best_j)


def kneedle(x, y):
    """Knee of an increasing curve: normalise x, y to [0,1]; knee = argmax (y_n - x_n)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return int(np.nanargmax(y)) if np.isfinite(y).any() else 0
    xn = (x - np.nanmin(x[ok])) / max(np.nanmax(x[ok]) - np.nanmin(x[ok]), 1e-12)
    yn = (y - np.nanmin(y[ok])) / max(np.nanmax(y[ok]) - np.nanmin(y[ok]), 1e-12)
    d = np.where(ok, yn - xn, -np.inf)
    return int(np.argmax(d))


# --------------------------------------------------------------------------- per-video label-free QC
def label_free(tree) -> dict:
    t = tree["time"]["timestamps_s"]
    raw, nr = tree["raw"], tree["norm"]
    o, s = nr["origin_px"], nr["scale_px"]
    out = {}
    for part in ("pose", "left_hand", "right_hand", "face"):
        m = raw[f"{part}_mask"]
        out[f"{part}_detection_rate"] = float((m != LM.MISSING).mean()) if m.size else 0.0
    out["any_hand_detection_rate"] = float(((raw["left_hand_mask"] != 0) | (raw["right_hand_mask"] != 0)).mean())
    out["hand_crop_redetect_rate"] = float(np.mean([(raw[f"{h}_mask"] == LM.CROP_DETECTED).mean()
                                                    for h in ("left_hand", "right_hand")]))
    out["hand_filled_rate"] = float(np.mean([(tree["smoothed"][f"{h}_mask"] == LM.FILLED).mean()
                                             for h in ("left_hand", "right_hand")]))
    from .outliers import accepted_px
    acc = {h: accepted_px(raw, h)[0] for h in ("left_hand", "right_hand")}
    rej = [raw[f"{h}_reject"] for h in ("left_hand", "right_hand") if f"{h}_reject" in raw]
    if rej:
        det = [(raw[f"{h}_mask"] != 0) for h in ("left_hand", "right_hand")]
        out["hand_rejected_rate"] = float(sum(((r != 0) & d).sum() for r, d in zip(rej, det)) /
                                          max(sum(d.sum() for d in det), 1))
        out["left_hand_accepted_rate"] = float(((raw["left_hand_mask"] != 0) & (rej[0] == 0)).mean())
        out["right_hand_accepted_rate"] = float(((raw["right_hand_mask"] != 0) & (rej[1] == 0)).mean())
    for tag, src in (("", acc), ("_raw", {h: raw[f"{h}_px"].astype(np.float64) for h in acc})):
        jh = [jitter(t, normalise(src[h], o, s)[..., :2]) for h in ("left_hand", "right_hand")]
        out[f"hand_jitter{tag}"] = float(np.nanmean(jh)) if np.isfinite(jh).any() else float("nan")
    out["body_jitter"] = jitter(t, normalise(raw["pose_px"].astype(np.float64), o, s)[:, LM.BODY_EVAL, :2])
    out["face_jitter"] = jitter(t, normalise(raw["face_px"].astype(np.float64), o, s)[..., :2])
    from .outliers import accepted as _acc
    for tag, use_acc in (("", True), ("_raw", False)):
        cv = []
        for h in ("left", "right"):
            wld = raw[f"{h}_hand_world"].astype(np.float64).copy()
            if use_acc:
                wld[~_acc(raw, h)] = np.nan
            cv.append(bone_cv(wld, LM.HAND_BONES))
        out[f"hand_bone_cv{tag}"] = float(np.nanmean(cv)) if np.isfinite(cv).any() else float("nan")
    out["arm_bone_cv"] = bone_cv(raw["pose_world"].astype(np.float64),
                                 [(LM.L_SH, LM.L_EL), (LM.L_EL, LM.L_WR), (LM.R_SH, LM.R_EL), (LM.R_EL, LM.R_WR)])
    out["symmetry_error"] = symmetry_error(raw["pose_world"].astype(np.float64),
                                           raw["left_hand_world"].astype(np.float64),
                                           raw["right_hand_world"].astype(np.float64))
    f = tree["features"]
    out["angle_violation_rate"] = violation_rate(f["__attrs__"]["angle_names"], f["angles"])
    hc = np.concatenate([raw[f"{h}_conf"][raw[f"{h}_mask"] != 0] for h in ("left_hand", "right_hand")])
    out["mean_hand_conf"] = float(hc.mean()) if hc.size else float("nan")
    pc = raw["pose_conf"][:, LM.BODY_EVAL][raw["pose_mask"] != 0]
    out["mean_body_conf"] = float(pc.mean()) if pc.size else float("nan")
    q = tree["quality"]
    out["mean_brightness"] = float(np.mean(q["brightness"]))
    out["median_blur"] = float(np.median(q["blur"]))
    out["mean_motion"] = float(np.mean(q["motion_frac"]))
    out["enhanced_rate"] = float(np.mean(q["enhanced"]))
    return out
