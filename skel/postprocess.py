"""Gap filling, smoothing and body-relative normalisation. All functions are timestamp-aware
(variable frame rate safe) and never invent data across long gaps."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


def median_dt(t):
    d = np.diff(np.asarray(t, np.float64))
    return float(np.median(d)) if d.size else 1 / 30


def runs(mask):
    """Contiguous runs of True in a 1-D bool array -> list of (start, end_inclusive)."""
    m = np.asarray(mask, bool).astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def fill_gaps(t, X, valid, max_gap_frames, context=3):
    """Interpolate gaps of <= max_gap_frames missing frames that have valid frames on both sides.

    X: (T, ...) with NaN where invalid; valid: (T,) bool.
    PCHIP (shape-preserving cubic) through up to `context` valid frames on each side.
    Returns (X_filled, filled_mask)."""
    t = np.asarray(t, np.float64)
    X = np.array(X, dtype=np.float64, copy=True)
    valid = np.asarray(valid, bool)
    filled = np.zeros(len(t), bool)
    for a, b in runs(~valid):
        g = b - a + 1
        if a == 0 or b == len(t) - 1 or g > max_gap_frames:
            continue
        left = [i for i in range(a - 1, max(-1, a - 1 - context), -1) if valid[i]][::-1]
        right = [i for i in range(b + 1, min(len(t), b + 1 + context)) if valid[i]]
        # keep only the contiguous valid frames adjacent to the gap
        left = [i for i in left if all(valid[i:a])]
        right = [i for i in right if all(valid[b + 1:i + 1])]
        idx = left + right
        if len(idx) < 2:
            continue
        flat = X[idx].reshape(len(idx), -1)
        tq = t[a:b + 1]
        if len(idx) == 2:
            w = (tq - t[idx[0]]) / (t[idx[1]] - t[idx[0]])
            vals = flat[0][None] * (1 - w[:, None]) + flat[1][None] * w[:, None]
        else:
            vals = PchipInterpolator(t[idx], flat, axis=0)(tq)
        X[a:b + 1] = vals.reshape((g,) + X.shape[1:])
        filled[a:b + 1] = True
    return X, filled


def gaussian_smooth(t, X, weights, sigma_s):
    """Zero-phase Gaussian kernel regression in time (no lag by construction).

    out_i = sum_j w_ij X_j / sum_j w_ij,  w_ij = exp(-(t_i-t_j)^2 / (2 sigma^2)) * c_j,
    over |t_i - t_j| <= 3 sigma, where c_j is the confidence weight (0 = invalid).
    weights: (T,) or (T, N) broadcastable to X[..., 0]. Invalid frames stay NaN."""
    X = np.asarray(X, np.float64)
    t = np.asarray(t, np.float64)
    w = np.asarray(weights, np.float64)
    if w.ndim == 1 and X.ndim > 2:
        w = w.reshape((-1,) + (1,) * (X.ndim - 2))
    W = np.broadcast_to(w, X.shape[:-1]).copy()
    W[~np.all(np.isfinite(X), axis=-1)] = 0.0
    out = np.full_like(X, np.nan)
    if sigma_s <= 0:
        out[W > 0] = X[W > 0]
        return out
    Xz = np.where(np.isfinite(X), X, 0.0)
    T = len(t)
    lo = np.searchsorted(t, t - 3 * sigma_s, "left")
    hi = np.searchsorted(t, t + 3 * sigma_s, "right")
    for i in range(T):
        if not np.any(W[i] > 0):
            continue
        sl = slice(lo[i], hi[i])
        k = np.exp(-0.5 * ((t[sl] - t[i]) / sigma_s) ** 2)
        k = k.reshape((-1,) + (1,) * (W.ndim - 1)) * W[sl]
        den = k.sum(axis=0)
        num = (k[..., None] * Xz[sl]).sum(axis=0)
        ok = (den > 0) & (W[i] > 0)
        res = np.full(X.shape[1:], np.nan)
        res[ok] = num[ok] / den[ok][..., None]
        out[i] = res
    return out


class OneEuro:
    """One Euro filter (Casiez et al., 2012), vectorised, causal (for live use).
    alpha(fc, dt) = 1 / (1 + tau/dt), tau = 1/(2 pi fc), fc = min_cutoff + beta * |dx_hat|."""

    def __init__(self, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
        self.mc, self.beta, self.dc = float(min_cutoff), float(beta), float(d_cutoff)

    @staticmethod
    def _alpha(fc, dt):
        tau = 1.0 / (2 * np.pi * fc)
        return 1.0 / (1.0 + tau / dt)

    def run(self, t, X):
        X = np.asarray(X, np.float64)
        out = np.full_like(X, np.nan)
        x_prev = dx_prev = None
        t_prev = None
        for i in range(len(t)):
            x = X[i]
            if not np.all(np.isfinite(x)):
                x_prev = dx_prev = t_prev = None      # reset across gaps
                continue
            if x_prev is None:
                out[i], x_prev, dx_prev, t_prev = x, x, np.zeros_like(x), t[i]
                continue
            dt = max(t[i] - t_prev, 1e-6)
            dx = (x - x_prev) / dt
            a_d = self._alpha(self.dc, dt)
            dx_hat = a_d * dx + (1 - a_d) * dx_prev
            fc = self.mc + self.beta * np.abs(dx_hat)
            a = self._alpha(fc, dt)
            x_hat = a * x + (1 - a) * x_prev
            out[i], x_prev, dx_prev, t_prev = x_hat, x_hat, dx_hat, t[i]
        return out


def _interp_hold(t, y, valid):
    """Linear interpolation over invalid frames, holding edge values."""
    out = np.array(y, np.float64, copy=True)
    if valid.sum() == 0:
        return out
    if out.ndim == 1:
        out[~valid] = np.interp(t[~valid], t[valid], out[valid])
    else:
        for d in range(out.shape[1]):
            out[~valid, d] = np.interp(t[~valid], t[valid], out[valid, d])
    return out


def body_frame(t, pose_px, pose_world, pose_valid, scale_sigma_s):
    """Per-frame origin (mid-shoulder, px) and scale (yaw-corrected shoulder width, px).

    s_t = ||sh_L - sh_R||_2D / clip(c_t, 0.5, 1),  c_t = ||(W_L - W_R)_xy|| / ||W_L - W_R||
    (c_t = cos of the shoulder line's angle out of the image plane, from 3D world landmarks),
    then Gaussian-smoothed in time. Frames without a pose are linearly interpolated (flagged)."""
    from . import landmarks as LM
    sl, sr = pose_px[:, LM.L_SH, :2].astype(np.float64), pose_px[:, LM.R_SH, :2].astype(np.float64)
    origin = 0.5 * (sl + sr)
    w2d = np.linalg.norm(sl - sr, axis=1)
    dw = pose_world[:, LM.L_SH].astype(np.float64) - pose_world[:, LM.R_SH].astype(np.float64)
    c = np.linalg.norm(dw[:, :2], axis=1) / np.maximum(np.linalg.norm(dw, axis=1), 1e-9)
    c = np.where(np.isfinite(c), c, 1.0)
    scale = w2d / np.clip(c, 0.5, 1.0)
    valid = pose_valid & np.isfinite(scale) & (scale > 1) & np.all(np.isfinite(origin), axis=1)
    scale_s = gaussian_smooth(t, scale[:, None], valid.astype(float), scale_sigma_s)[:, 0]
    origin = _interp_hold(t, np.where(valid[:, None], origin, np.nan), valid)
    scale = _interp_hold(t, np.where(valid, scale_s, np.nan), valid)
    return origin.astype(np.float64), scale.astype(np.float64), ~valid


def normalise(px, origin, scale):
    """(x, y, z) px -> body units: ((x,y) - o_t) / s_t, z / s_t."""
    out = np.empty(px.shape, np.float64)
    shp = (-1,) + (1,) * (px.ndim - 2)
    out[..., :2] = (px[..., :2] - origin.reshape(shp + (2,))) / scale.reshape(shp + (1,))
    out[..., 2] = px[..., 2] / scale.reshape(shp)
    return out


def denormalise(nrm, origin, scale):
    out = np.empty(nrm.shape, np.float64)
    shp = (-1,) + (1,) * (nrm.ndim - 2)
    out[..., :2] = nrm[..., :2] * scale.reshape(shp + (1,)) + origin.reshape(shp + (2,))
    out[..., 2] = nrm[..., 2] * scale.reshape(shp)
    return out
