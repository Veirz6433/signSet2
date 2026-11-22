"""Per-frame quality measures and detector-only enhancement (geometry is never changed)."""
from __future__ import annotations

import cv2
import numpy as np


class QualityMeter:
    def __init__(self, cfg):
        q = cfg["quality"]
        self.h = int(q["analysis_height"])
        self.dark = float(q["dark_threshold"])
        self.target = float(q["target_brightness"])
        self.clahe = cv2.createCLAHE(clipLimit=float(q["clahe_clip"]), tileGridSize=(8, 8))
        self.prev = None

    def _gray(self, rgb):
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        scale = self.h / g.shape[0]
        return cv2.resize(g, (max(8, int(round(g.shape[1] * scale))), self.h), interpolation=cv2.INTER_AREA)

    def measure(self, rgb):
        """brightness = mean luma; blur = var(Laplacian) at analysis height;
        motion = global translation between consecutive frames (phase correlation) / height."""
        g = self._gray(rgb)
        brightness = float(g.mean())
        blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
        gf = g.astype(np.float32)
        if self.prev is not None and self.prev.shape == gf.shape:
            (dx, dy), _ = cv2.phaseCorrelate(self.prev, gf)
            motion = float(np.hypot(dx, dy) / self.h)
        else:
            motion = 0.0
        self.prev = gf
        return brightness, blur, motion

    def enhance(self, rgb, brightness):
        """Return (image for the detector, enhanced flag). Only intensities change."""
        if brightness >= self.dark:
            return rgb, False
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L = self.clahe.apply(lab[..., 0])
        m = max(float(L.mean()), 1.0)
        gamma = float(np.clip(np.log(self.target / 255.0) / np.log(m / 255.0), 0.35, 1.0))
        lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
        lab[..., 0] = cv2.LUT(L, lut)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), True
