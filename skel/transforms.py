"""Known geometric transforms (exact ground truth by construction) and controlled degradations.

Every variant is described by (kind, param) so it can be rebuilt in a worker process.
Geometric variants define a per-frame affine A_i; predictions on the transformed video are
mapped back with A_i^{-1} and compared with predictions on the original video.
"""
from __future__ import annotations

import cv2
import numpy as np

GEOMETRIC = {"shift", "rotate", "scale", "shake", "mirror"}


def variant_list(cfg):
    ev = cfg["evaluation"]
    tr, dg = ev["transforms"], ev["degradations"]
    out = [("repeat", 0)]
    out += [("shift", v) for v in tr["shift_frac"]]
    out += [("rotate", v) for v in tr["rotate_deg"]]
    out += [("scale", v) for v in tr["scale"]]
    out += [("shake", 1)]
    if tr["mirror"]:
        out += [("mirror", 1)]
    for kind, key in (("brightness", "brightness_factor"), ("gaussian_blur", "gaussian_blur_sigma_720p"),
                      ("motion_blur", "motion_blur_len_720p"), ("noise", "noise_sigma"),
                      ("downscale", "downscale_factor"), ("jpeg", "jpeg_quality"),
                      ("fps_drop", "fps_drop_keep_every")):
        out += [(kind, v) for v in dg[key]]
    return out


class Variant:
    def __init__(self, kind, param, W, H, cfg):
        self.kind, self.param, self.W, self.H = kind, param, int(W), int(H)
        self.seed = int(cfg["evaluation"]["seed"])
        self.mirror = kind == "mirror"
        self.geometric = kind in GEOMETRIC
        sk = cfg["evaluation"]["transforms"]["shake"]
        self.sk = (float(sk["translation_frac"]), float(sk["rotation_deg"]), float(sk["smoothness"]))
        self._shake = []
        if kind == "motion_blur":
            L = max(3, int(round(param * H / 720)))
            k = np.zeros((L, L), np.float32)
            c = (L - 1) / 2
            ang = np.radians(30)
            for s in np.linspace(-c, c, 4 * L):
                x, y = int(round(c + s * np.cos(ang))), int(round(c + s * np.sin(ang)))
                k[y, x] = 1
            self.kernel = k / k.sum()

    @property
    def name(self):
        return f"{self.kind}={self.param}"

    # ---------------------------------------------------------------- affine per frame
    def affine(self, i):
        W, H, c = self.W, self.H, (self.W / 2.0, self.H / 2.0)
        if self.kind == "shift":
            return np.array([[1, 0, self.param * W], [0, 1, self.param * H]], np.float64)
        if self.kind == "rotate":
            return cv2.getRotationMatrix2D(c, float(self.param), 1.0)
        if self.kind == "scale":
            return cv2.getRotationMatrix2D(c, 0.0, float(self.param))
        if self.kind == "mirror":
            return np.array([[-1, 0, W - 1], [0, 1, 0]], np.float64)
        if self.kind == "shake":
            # stationary AR(1): x_i = rho x_{i-1} + sqrt(1-rho^2) eps_i, eps ~ N(0,1), seeded
            tf, rd, rho = self.sk
            while len(self._shake) <= i:
                rng = np.random.default_rng(self.seed + 7919 * len(self._shake))
                eps = rng.standard_normal(3)
                prev = self._shake[-1] if self._shake else rng.standard_normal(3)
                self._shake.append(rho * prev + np.sqrt(1 - rho * rho) * eps)
            x = self._shake[i]
            A = cv2.getRotationMatrix2D(c, float(rd * x[2]), 1.0)
            A[0, 2] += tf * W * x[0]
            A[1, 2] += tf * H * x[1]
            return A
        return np.array([[1, 0, 0], [0, 1, 0]], np.float64)

    def inverse_points(self, i, xy):
        """Map points from the transformed frame back to the original frame. xy: (..., 2)."""
        if not self.geometric:
            return xy
        Ai = cv2.invertAffineTransform(self.affine(i))
        return xy @ Ai[:, :2].T + Ai[:, 2]

    def forward_points(self, i, xy):
        if not self.geometric:
            return xy
        A = self.affine(i)
        return xy @ A[:, :2].T + A[:, 2]

    # ---------------------------------------------------------------- frame functions
    def frame_select(self):
        if self.kind == "fps_drop":
            k = int(self.param)
            return lambda i: i % k == 0
        return None

    def __call__(self, i, rgb):
        k, p = self.kind, self.param
        if self.geometric:
            return cv2.warpAffine(rgb, self.affine(i), (self.W, self.H), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        if k == "brightness":
            return np.clip(rgb.astype(np.float32) * float(p), 0, 255).astype(np.uint8)
        if k == "gaussian_blur":
            return cv2.GaussianBlur(rgb, (0, 0), float(p) * self.H / 720.0)
        if k == "motion_blur":
            return cv2.filter2D(rgb, -1, self.kernel, borderType=cv2.BORDER_REFLECT)
        if k == "noise":
            rng = np.random.default_rng(self.seed + 104729 * i)
            return np.clip(rgb.astype(np.float32) + rng.normal(0, float(p), rgb.shape), 0, 255).astype(np.uint8)
        if k == "downscale":
            w, h = max(8, int(round(self.W * p))), max(8, int(round(self.H * p)))
            small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            return cv2.resize(small, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if k == "jpeg":
            ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(p)])
            return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return rgb      # repeat, fps_drop

    def needs_frame_fn(self):
        return self.kind not in ("repeat", "fps_drop")
