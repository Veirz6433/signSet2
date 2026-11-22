"""Angles, distances, palm orientation, forearm twist and the bone (angle + length) encoding.

3D quantities use MediaPipe world landmarks (metres, camera-aligned axes: x right, y down,
z away from camera). Body and each hand have their own world origin, so only DIRECTIONS are
combined across them (never positions). Hand-to-body distances use the 2D body-normalised
coordinates instead.
"""
from __future__ import annotations

import numpy as np

from . import landmarks as LM

DEG = 180.0 / np.pi


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 1e-12, v / n, np.nan)


def dot(a, b):
    return np.sum(a * b, axis=-1)


def angle_between(a, b):
    return np.degrees(np.arccos(np.clip(dot(unit(a), unit(b)), -1.0, 1.0)))


def side_sign(side):
    return 1.0 if side == "right" else -1.0


# --------------------------------------------------------------------------- palm frame
def hand_frame(hw, side):
    """Palm frame from hand world landmarks (T,21,3).
    x = unit(p9 - p0) (distal), n = (p5 - p0) x (p17 - p0), z = s_h * n made orthogonal to x
    (palmar normal; s_h = +1 right, -1 left), y = z x x. Returns x, y, z (T,3)."""
    p0, p5, p9, p17 = hw[:, 0], hw[:, 5], hw[:, 9], hw[:, 17]
    x = unit(p9 - p0)
    n = np.cross(p5 - p0, p17 - p0) * side_sign(side)
    z = unit(n - dot(n, x)[:, None] * x)
    y = np.cross(z, x)
    return x, y, z


def quat_from_frame(x, y, z):
    """Rotation matrix R = [x y z] (columns) -> unit quaternion (w, x, y, z), sign-continuous in time."""
    R = np.stack([x, y, z], axis=-1)
    T = R.shape[0]
    q = np.full((T, 4), np.nan)
    for i in range(T):
        m = R[i]
        if not np.all(np.isfinite(m)):
            continue
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = unit(q)
    prev = None
    for i in range(T):
        if np.all(np.isfinite(q[i])):
            if prev is not None and dot(q[i], prev) < 0:
                q[i] = -q[i]
            prev = q[i]
    return q


# --------------------------------------------------------------------------- arm + wrist
def arm_angles(pw, hw, side):
    """Elbow bend, shoulder angle, wrist flexion / deviation and forearm twist (deg).

    f = unit(W - E) forearm axis, u = E - S upper arm, l = S - S_other (outward shoulder line).
    z_f = palmar normal projected perpendicular to f, y_f = z_f x f.
    flex  = atan2(x_h . z_f, x_h . f)                (+ = toward palm)
    dev   = -s_h atan2(x_h . y_f, x_h . f)           (+ = toward thumb, both hands)
    medial reference r: blend of r_e = -s_h unit(u x f) (elbow hinge axis) and r_s = -l
    projected perpendicular to f, weight w = clip(|unit(u) x f| / sin 30deg, 0, 1).
    twist = -s_h atan2((r x z_f) . f, r . z_f)        (0 = palm facing the body midline,
                                                         + = pronation, both hands)"""
    s = LM.SIDE_POSE[side]
    sh = side_sign(side)
    S, E, Wr, So = pw[:, s["shoulder"]], pw[:, s["elbow"]], pw[:, s["wrist"]], pw[:, s["other_shoulder"]]
    u, f = E - S, unit(Wr - E)
    out = {"elbow_bend": angle_between(u, Wr - E), "shoulder_angle": angle_between(S - So, u)}
    xh, yh, zh = hand_frame(hw, side)
    zf = unit(zh - dot(zh, f)[:, None] * f)
    yf = np.cross(zf, f)
    out["wrist_flex"] = np.degrees(np.arctan2(dot(xh, zf), dot(xh, f)))
    out["wrist_dev"] = -sh * np.degrees(np.arctan2(dot(xh, yf), dot(xh, f)))
    cross_uf = np.cross(unit(u), f)
    w = np.clip(np.linalg.norm(cross_uf, axis=-1) / np.sin(np.radians(30)), 0, 1)[:, None]
    r_e = -sh * unit(cross_uf)
    l_out = S - So
    r_s = unit(-(l_out - dot(l_out, f)[:, None] * f))
    r_e = np.where(np.isfinite(r_e), r_e, 0.0)
    r = w * r_e + (1 - w) * np.where(np.isfinite(r_s), r_s, 0.0)
    r = unit(r - dot(r, f)[:, None] * f)
    out["forearm_twist"] = -sh * np.degrees(np.arctan2(dot(np.cross(r, zf), f), dot(r, zf)))
    perp_s = np.linalg.norm(np.cross(unit(l_out), f), axis=-1)
    out["twist_conf"] = np.abs(zh[:, 2]) * np.maximum(w[:, 0], perp_s)
    return out, (xh, yh, zh)


# --------------------------------------------------------------------------- fingers
def finger_angles(hw, side):
    """Signed flexion (deg) about the knuckle axis a = s_h unit(p5 - p17):
    theta = atan2((b1 x b2) . a, b1 . b2), b1 = p_j - p_prev, b2 = p_next - p_j.
    Thumb joints: unsigned bend arccos(b1 . b2). Spread: angle between neighbouring proximal segments."""
    a = unit(hw[:, 5] - hw[:, 17]) * side_sign(side)
    names, vals = [], []
    for name, i0, j, i1 in LM.FINGER_JOINTS:
        b1, b2 = unit(hw[:, j] - hw[:, i0]), unit(hw[:, i1] - hw[:, j])
        if name.startswith("thumb"):
            v = np.degrees(np.arccos(np.clip(dot(b1, b2), -1, 1)))
        else:
            v = np.degrees(np.arctan2(dot(np.cross(b1, b2), a), dot(b1, b2)))
        names.append(name)
        vals.append(v)
    for name, (a0, a1), (b0, b1_) in LM.FINGER_SPREAD:
        names.append(name)
        vals.append(angle_between(hw[:, a1] - hw[:, a0], hw[:, b1_] - hw[:, b0]))
    return names, np.stack(vals, axis=1)


def angle_limit_key(name):
    base = name.split("_", 1)[1] if name.startswith(("left_", "right_")) else name
    if base.startswith("thumb"):
        return base
    for k in ("mcp", "pip", "dip"):
        if base.endswith(k):
            return k
    if base.startswith("spread"):
        return "spread"
    if base == "elbow_bend":
        return "elbow_bend"
    return None


# --------------------------------------------------------------------------- bones (angle + length)
def encode_bones(tree, P, root):
    """Each bone (parent -> child): length L = ||c - p||, direction u = (c - p)/L as spherical
    angles theta = arccos(u_z), phi = atan2(u_y, u_x).  P: (T,N,3); root: (T,3)."""
    T = P.shape[0]
    L = np.full((T, len(tree)), np.nan)
    th, ph = L.copy(), L.copy()
    for k, (a, b) in enumerate(tree):
        pa = root if a == "root" else P[:, a]
        v = P[:, b] - pa
        L[:, k] = np.linalg.norm(v, axis=-1)
        u = unit(v)
        th[:, k] = np.arccos(np.clip(u[:, 2], -1, 1))
        ph[:, k] = np.arctan2(u[:, 1], u[:, 0])
    return L, th, ph


def decode_bones(tree, root, L, th, ph, n_points, lengths_override=None):
    """Forward kinematics: c = p + L * (sin th cos ph, sin th sin ph, cos th).
    lengths_override: (B,) fixed lengths (used for the avatar retarget)."""
    T = root.shape[0]
    P = np.full((T, n_points, 3), np.nan)
    for k, (a, b) in enumerate(tree):
        pa = root if a == "root" else P[:, a]
        if a == 0 and np.all(np.isnan(P[:, 0])):
            P[:, 0] = root
            pa = root
        Lk = L[:, k] if lengths_override is None else lengths_override[k]
        u = np.stack([np.sin(th[:, k]) * np.cos(ph[:, k]), np.sin(th[:, k]) * np.sin(ph[:, k]),
                      np.cos(th[:, k])], axis=-1)
        P[:, b] = pa + (Lk[..., None] if np.ndim(Lk) else Lk) * u
    return P


# --------------------------------------------------------------------------- all features
def compute_features(sm, nrm, t=None):
    """sm: smoothed arrays dict (pose_world, left_hand_world, right_hand_world).
    nrm: normalised 2D arrays dict (pose, left_hand, right_hand, face) in body units."""
    pw = sm["pose_world"].astype(np.float64)
    T = pw.shape[0]
    angle_names, angle_cols, extra = [], [], {}
    for side in ("left", "right"):
        hw = sm[f"{side}_hand_world"].astype(np.float64)
        arm, (xh, yh, zh) = arm_angles(pw, hw, side)
        for k in ("elbow_bend", "shoulder_angle", "wrist_flex", "wrist_dev", "forearm_twist"):
            angle_names.append(f"{side}_{k}")
            angle_cols.append(arm[k])
        fn, fv = finger_angles(hw, side)
        angle_names += [f"{side}_{n}" for n in fn]
        angle_cols += [fv[:, i] for i in range(fv.shape[1])]
        extra[f"{side}_hand_quat"] = quat_from_frame(xh, yh, zh)
        extra[f"{side}_twist_conf"] = arm["twist_conf"]
        extra[f"{side}_palm_normal"] = zh
    angles = np.stack(angle_cols, axis=1)

    dist_names, dist_cols = [], []
    for side in ("left", "right"):
        hw = sm[f"{side}_hand_world"].astype(np.float64)
        palm = np.linalg.norm(hw[:, 9] - hw[:, 0], axis=-1)
        for a, b in LM.HAND_DIST_PAIRS:
            d = np.linalg.norm(hw[:, LM.H[a]] - hw[:, LM.H[b]], axis=-1)
            dist_names += [f"{side}_{a}__{b}_m", f"{side}_{a}__{b}_per_palm"]
            dist_cols += [d, d / palm]
    face, pose = nrm["face"][..., :2], nrm["pose"][..., :2]
    targets = {
        "face:nose_tip": face[:, LM.F["nose_tip"]], "face:chin": face[:, LM.F["chin"]],
        "face:mouth_center": face[:, [LM.F["mouth_left"], LM.F["mouth_right"], LM.F["upper_lip_top"],
                                      LM.F["lower_lip_bottom"]]].mean(axis=1),
        "face:right_eye_center": face[:, [LM.F[f"right_eye_{k}"] for k in ("outer", "inner", "upper", "lower")]].mean(axis=1),
        "face:left_eye_center": face[:, [LM.F[f"left_eye_{k}"] for k in ("outer", "inner", "upper", "lower")]].mean(axis=1),
        "pose:left_shoulder": pose[:, LM.L_SH], "pose:right_shoulder": pose[:, LM.R_SH],
        "pose:mid_shoulder": np.zeros((T, 2)),
    }
    for side in ("left", "right"):
        hand = nrm[f"{side}_hand"][..., :2]
        for a, tgt in LM.HAND_BODY_TARGETS:
            dist_names.append(f"{side}_{a}__{tgt.split(':')[1]}_per_shoulder")
            dist_cols.append(np.linalg.norm(hand[:, LM.H[a]] - targets[tgt], axis=-1))
    lh, rh = nrm["left_hand"][..., :2], nrm["right_hand"][..., :2]
    dist_names += ["wrist_left__wrist_right_per_shoulder", "index_tip_left__index_tip_right_per_shoulder"]
    dist_cols += [np.linalg.norm(lh[:, 0] - rh[:, 0], axis=-1), np.linalg.norm(lh[:, 8] - rh[:, 8], axis=-1)]
    distances = np.stack(dist_cols, axis=1)

    bones = {}
    root = 0.5 * (pw[:, LM.L_SH] + pw[:, LM.R_SH])
    bones["body"] = (root,) + encode_bones(LM.BODY_TREE, pw, root)
    for side in ("left", "right"):
        hw = sm[f"{side}_hand_world"].astype(np.float64)
        bones[f"{side}_hand"] = (hw[:, 0],) + encode_bones(LM.HAND_BONES, hw, hw[:, 0])
    return {"angle_names": angle_names, "angles": angles, "distance_names": dist_names,
            "distances": distances, "extra": extra, "bones": bones}
