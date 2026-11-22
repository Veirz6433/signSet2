"""Landmark definitions: names, index maps, bones, mirror permutations.

Convention: "left"/"right" always mean the PERSON's left/right (anatomical),
never the image side.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- pose (MediaPipe, 33)
POSE_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]
P = {n: i for i, n in enumerate(POSE_NAMES)}
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = P["left_shoulder"], P["right_shoulder"], P["left_elbow"], \
    P["right_elbow"], P["left_wrist"], P["right_wrist"]
SIDE_POSE = {
    "left": {"shoulder": L_SH, "elbow": L_EL, "wrist": L_WR, "other_shoulder": R_SH},
    "right": {"shoulder": R_SH, "elbow": R_EL, "wrist": R_WR, "other_shoulder": L_SH},
}
BODY_EVAL = [L_SH, R_SH, L_EL, R_EL, L_WR, R_WR]          # joints scored for "body"


def _mirror_perm(names):
    idx = {n: i for i, n in enumerate(names)}
    out = []
    for n in names:
        if n.startswith("left_"):
            out.append(idx["right_" + n[5:]])
        elif n.startswith("right_"):
            out.append(idx["left_" + n[6:]])
        else:
            out.append(idx[n])
    return np.array(out)


POSE_MIRROR = _mirror_perm(POSE_NAMES)

POSE_DRAW = [(L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR),
             (L_SH, P["left_hip"]), (R_SH, P["right_hip"]), (P["left_hip"], P["right_hip"])]

# --------------------------------------------------------------------------- hands (21)
HAND_NAMES = ["wrist",
              "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
              "index_mcp", "index_pip", "index_dip", "index_tip",
              "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
              "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
              "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]
H = {n: i for i, n in enumerate(HAND_NAMES)}
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
              (0, 17), (17, 18), (18, 19), (19, 20)]
HAND_DRAW = HAND_BONES + [(5, 9), (9, 13), (13, 17)]

# finger joints: (name, a, joint, b); signed flexion about the knuckle axis
FINGER_JOINTS = [
    ("thumb_cmc", 0, 1, 2), ("thumb_mcp", 1, 2, 3), ("thumb_ip", 2, 3, 4),
    ("index_mcp", 0, 5, 6), ("index_pip", 5, 6, 7), ("index_dip", 6, 7, 8),
    ("middle_mcp", 0, 9, 10), ("middle_pip", 9, 10, 11), ("middle_dip", 10, 11, 12),
    ("ring_mcp", 0, 13, 14), ("ring_pip", 13, 14, 15), ("ring_dip", 14, 15, 16),
    ("pinky_mcp", 0, 17, 18), ("pinky_pip", 17, 18, 19), ("pinky_dip", 18, 19, 20),
]
# spread: angle between proximal segments of neighbouring digits
FINGER_SPREAD = [("spread_thumb_index", (2, 3), (5, 6)), ("spread_index_middle", (5, 6), (9, 10)),
                 ("spread_middle_ring", (9, 10), (13, 14)), ("spread_ring_pinky", (13, 14), (17, 18))]

# approximate active range of motion (deg) used only for the "impossible angle" check.
# thumb joints use unsigned bend (sign is ill-defined for the thumb), fingers signed flexion.
ANGLE_LIMITS = {
    "thumb_cmc": (0, 80), "thumb_mcp": (0, 90), "thumb_ip": (0, 100),
    "mcp": (-45, 100), "pip": (-10, 120), "dip": (-30, 95),
    "spread": (0, 70), "elbow_bend": (0, 165),
}

# --------------------------------------------------------------------------- face (20 of 478)
# (name, MediaPipe face-mesh index, iBUG-68 indices (mean if several), mirror name)
FACE_POINTS = [
    ("right_eye_outer", 33, [36], "left_eye_outer"),
    ("right_eye_inner", 133, [39], "left_eye_inner"),
    ("right_eye_upper", 159, [37, 38], "left_eye_upper"),
    ("right_eye_lower", 145, [40, 41], "left_eye_lower"),
    ("left_eye_inner", 362, [42], "right_eye_inner"),
    ("left_eye_outer", 263, [45], "right_eye_outer"),
    ("left_eye_upper", 386, [43, 44], "right_eye_upper"),
    ("left_eye_lower", 374, [46, 47], "right_eye_lower"),
    ("right_brow_outer", 70, [17], "left_brow_outer"),
    ("right_brow_mid", 105, [19], "left_brow_mid"),
    ("right_brow_inner", 107, [21], "left_brow_inner"),
    ("left_brow_inner", 336, [22], "right_brow_inner"),
    ("left_brow_mid", 334, [24], "right_brow_mid"),
    ("left_brow_outer", 300, [26], "right_brow_outer"),
    ("mouth_right", 61, [48], "mouth_left"),
    ("mouth_left", 291, [54], "mouth_right"),
    ("upper_lip_top", 0, [51], "upper_lip_top"),
    ("lower_lip_bottom", 17, [57], "lower_lip_bottom"),
    ("nose_tip", 1, [30], "nose_tip"),
    ("chin", 152, [8], "chin"),
]
FACE_NAMES = [f[0] for f in FACE_POINTS]
FACE_MP_IDX = np.array([f[1] for f in FACE_POINTS])
FACE_IBUG = [f[2] for f in FACE_POINTS]
F = {n: i for i, n in enumerate(FACE_NAMES)}
FACE_MIRROR = np.array([F[f[3]] for f in FACE_POINTS])
FACE_DRAW = [(F["right_eye_outer"], F["right_eye_upper"]), (F["right_eye_upper"], F["right_eye_inner"]),
             (F["right_eye_inner"], F["right_eye_lower"]), (F["right_eye_lower"], F["right_eye_outer"]),
             (F["left_eye_outer"], F["left_eye_upper"]), (F["left_eye_upper"], F["left_eye_inner"]),
             (F["left_eye_inner"], F["left_eye_lower"]), (F["left_eye_lower"], F["left_eye_outer"]),
             (F["right_brow_outer"], F["right_brow_mid"]), (F["right_brow_mid"], F["right_brow_inner"]),
             (F["left_brow_outer"], F["left_brow_mid"]), (F["left_brow_mid"], F["left_brow_inner"]),
             (F["mouth_right"], F["upper_lip_top"]), (F["upper_lip_top"], F["mouth_left"]),
             (F["mouth_left"], F["lower_lip_bottom"]), (F["lower_lip_bottom"], F["mouth_right"])]
INTEROCULAR = (F["right_eye_outer"], F["left_eye_outer"])

# --------------------------------------------------------------------------- body bone tree (pose world)
# "root" is the virtual mid-shoulder point.
BODY_TREE = [("root", P["nose"]), ("root", L_SH), ("root", R_SH),
             (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR)]

# --------------------------------------------------------------------------- distances
HAND_DIST_PAIRS = [("thumb_tip", "index_tip"), ("thumb_tip", "middle_tip"), ("thumb_tip", "ring_tip"),
                   ("thumb_tip", "pinky_tip"), ("index_tip", "middle_tip"), ("wrist", "index_tip"),
                   ("wrist", "middle_tip"), ("wrist", "ring_tip"), ("wrist", "pinky_tip")]
# hand-to-body targets (2D, normalised by shoulder width). Face targets are face points or
# derived centres; body targets are pose indices.
HAND_BODY_TARGETS = [("index_tip", "face:nose_tip"), ("index_tip", "face:chin"),
                     ("index_tip", "face:mouth_center"), ("index_tip", "face:right_eye_center"),
                     ("index_tip", "face:left_eye_center"), ("index_tip", "pose:left_shoulder"),
                     ("index_tip", "pose:right_shoulder"), ("wrist", "face:chin"),
                     ("wrist", "pose:mid_shoulder")]

PARTS = ("pose", "left_hand", "right_hand", "face")
PART_SIZES = {"pose": 33, "left_hand": 21, "right_hand": 21, "face": 20}

# mask codes
MISSING, DETECTED, CROP_DETECTED, FILLED = 0, 1, 2, 3

# --------------------------------------------------------------------------- RTMW / COCO-WholeBody (133)
WB_BODY = {L_SH: 5, R_SH: 6, L_EL: 7, R_EL: 8, L_WR: 9, R_WR: 10}   # MediaPipe pose -> COCO body
WB_LEFT_HAND0, WB_RIGHT_HAND0, WB_FACE0 = 91, 112, 23
