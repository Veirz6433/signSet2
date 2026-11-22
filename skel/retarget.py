"""Avatar-ready retargeting: keep the measured bone DIRECTIONS, replace bone LENGTHS with fixed
human proportions, so limbs can never stretch ("no Slenderman arms").

    c = p + L_std * u(theta, phi)

Body ratios are fractions of standing height (approximate anthropometric segment ratios,
e.g. upper arm ~0.186 H, forearm ~0.146 H, hand ~0.108 H). Finger segments are fractions of
hand length. Edit these if the avatar has different proportions."""
from __future__ import annotations

import numpy as np

from . import landmarks as LM
from .features import decode_bones

BODY_RATIOS = {          # fraction of height
    ("root", LM.P["nose"]): 0.13, ("root", LM.L_SH): 0.115, ("root", LM.R_SH): 0.115,
    (LM.L_SH, LM.L_EL): 0.186, (LM.L_EL, LM.L_WR): 0.146,
    (LM.R_SH, LM.R_EL): 0.186, (LM.R_EL, LM.R_WR): 0.146,
}
HAND_LENGTH_RATIO = 0.108
HAND_RATIOS = {          # fraction of hand length
    (0, 1): 0.12, (1, 2): 0.25, (2, 3): 0.20, (3, 4): 0.16,
    (0, 5): 0.37, (5, 6): 0.27, (6, 7): 0.14, (7, 8): 0.10,
    (0, 9): 0.37, (9, 10): 0.28, (10, 11): 0.17, (11, 12): 0.11,
    (0, 13): 0.34, (13, 14): 0.26, (14, 15): 0.16, (15, 16): 0.11,
    (0, 17): 0.30, (17, 18): 0.21, (18, 19): 0.12, (19, 20): 0.09,
}
TWIST_LIMIT_DEG = 90.0


def standard_lengths(height_m=1.70):
    body = np.array([BODY_RATIOS[b] * height_m for b in LM.BODY_TREE])
    hand_len = HAND_LENGTH_RATIO * height_m
    hand = np.array([HAND_RATIOS[b] * hand_len for b in LM.HAND_BONES])
    return body, hand


def retarget(bones: dict, height_m=1.70):
    """bones: tree['bones'] from the HDF5 file. Returns 3D positions with fixed lengths.
    Hands are attached at the avatar's wrists (directions only are taken from the video)."""
    Lb, Lh = standard_lengths(height_m)
    b = bones["body"]
    T = b["theta"].shape[0]
    root = np.zeros((T, 3))
    body = decode_bones(LM.BODY_TREE, root, b["length"], b["theta"], b["phi"], 33, lengths_override=Lb)
    out = {"body": body}
    for side, wr in (("left", LM.L_WR), ("right", LM.R_WR)):
        h = bones[f"{side}_hand"]
        out[f"{side}_hand"] = decode_bones(LM.HAND_BONES, body[:, wr], h["length"], h["theta"], h["phi"], 21,
                                           lengths_override=Lh)
    return out


def clamp_twist(twist_deg):
    return np.clip(twist_deg, -TWIST_LIMIT_DEG, TWIST_LIMIT_DEG)
