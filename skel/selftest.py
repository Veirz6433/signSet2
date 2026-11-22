"""Self-test: checks the maths and the whole post-detection pipeline on a synthetic person with
KNOWN angles, so a fresh install can be verified without any video or labels.

    python main.py selftest
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np

from . import landmarks as LM

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def rodrigues(v, axis, ang):
    axis = axis / np.linalg.norm(axis)
    return v * np.cos(ang) + np.cross(axis, v) * np.sin(ang) + axis * np.dot(axis, v) * (1 - np.cos(ang))


def hand_template(side):
    rad = -1.0 if side == "right" else 1.0
    pts = np.zeros((21, 3))
    fingers = {1: (0.03, 0.030), 5: (0.09, 0.025), 9: (0.095, 0.0), 13: (0.09, -0.02), 17: (0.08, -0.038)}
    seg = {1: [0.02, 0.02, 0.02], 5: [0.04, 0.025, 0.02], 9: [0.045, 0.028, 0.022], 13: [0.042, 0.026, 0.02],
           17: [0.032, 0.02, 0.018]}
    for base, (x, y) in fingers.items():
        pts[base] = [x, rad * y, 0.0 if base != 1 else 0.01]
        d = np.array([1.0, rad * (0.6 if base == 1 else 0.05 * np.sign(y) if y else 0.0), 0.0])
        d /= np.linalg.norm(d)
        for k, L in enumerate(seg[base], 1):
            pts[base + k] = pts[base + k - 1] + L * d
    return pts


def arm_pose(side, t):
    sh = 1.0 if side == "right" else -1.0
    S = np.array([-0.18 * sh, 0.0, 0.0])
    E = S + np.array([-0.03 * sh, 0.28, 0.0])
    f = np.array([0.15 * sh, -0.35, -0.92])
    f /= np.linalg.norm(f)
    Wr = E + 0.25 * f
    return S, E, Wr


def synth_hand_world(side, S, E, Wr, So, twist_deg, flex_deg):
    sh = 1.0 if side == "right" else -1.0
    f = (Wr - E) / np.linalg.norm(Wr - E)
    u = E - S
    r = -sh * np.cross(u / np.linalg.norm(u), f)
    r /= np.linalg.norm(r)
    zf = rodrigues(r, f, np.radians(-sh * twist_deg))
    xh = f + np.tan(np.radians(flex_deg)) * zf
    xh /= np.linalg.norm(xh)
    zh = zf - np.dot(zf, xh) * xh
    zh /= np.linalg.norm(zh)
    yh = np.cross(zh, xh)
    R = np.stack([xh, yh, zh], axis=1)
    return Wr + hand_template(side) @ R.T


def synth_rows(T=90, fps=30.0, W=1280, H=720, seed=0, noise_px=0.8, affine=None):
    rng = np.random.default_rng(seed)
    k = 900.0
    rows, truth = [], {"left_twist": [], "right_twist": [], "left_flex": [], "right_flex": []}
    face_off = rng.uniform(-0.05, 0.05, (20, 2))
    for i in range(T):
        t = i / fps
        pw = np.zeros((33, 3))
        pw[:] = [0.0, -0.25, -0.08]
        cam = np.array([0.01 * np.sin(2 * np.pi * 0.7 * t), 0.008 * np.cos(2 * np.pi * 0.9 * t)])  # shake
        o = {}
        for side in ("left", "right"):
            S, E, Wr = arm_pose(side, t)
            s = LM.SIDE_POSE[side]
            pw[s["shoulder"]], pw[s["elbow"]], pw[s["wrist"]] = S, E, Wr
        pw[LM.P["left_hip"]], pw[LM.P["right_hip"]] = [0.12, 0.5, 0], [-0.12, 0.5, 0]
        for side in ("left", "right"):
            s = LM.SIDE_POSE[side]
            tw = 40 * np.sin(2 * np.pi * 0.5 * t + (0 if side == "left" else 1.0))
            fl = 25 * np.sin(2 * np.pi * 0.4 * t)
            truth[f"{side}_twist"].append(tw)
            truth[f"{side}_flex"].append(fl)
            hw = synth_hand_world(side, pw[s["shoulder"]], pw[s["elbow"]], pw[s["wrist"]], pw[s["other_shoulder"]], tw, fl)
            o[f"{side}_hw"] = hw

        def proj(P):
            xy = np.stack([W / 2 + k * (P[:, 0] + cam[0]), H * 0.35 + k * (P[:, 1] + cam[1])], axis=1)
            if affine is not None:
                A = affine(i)
                xy = xy @ A[:, :2].T + A[:, 2]
            return np.concatenate([xy + rng.normal(0, noise_px, xy.shape), k * P[:, 2:3]], axis=1)

        row = {"t": t, "i": i, "brightness": 120.0, "blur": 300.0, "motion": 0.0, "enhanced": False,
               "pose_px": proj(pw).astype(np.float32), "pose_world": pw.astype(np.float32),
               "pose_conf": np.full(33, 0.99, np.float32), "pose_code": LM.DETECTED}
        for side in ("left", "right"):
            hw = o[f"{side}_hw"]
            row[f"{side}_hand_px"] = proj(hw).astype(np.float32)
            row[f"{side}_hand_world"] = (hw - hw.mean(0) + rng.normal(0, 0.0005, hw.shape)).astype(np.float32)
            row[f"{side}_hand_conf"], row[f"{side}_hand_code"] = 0.95, LM.DETECTED
        fpts = np.concatenate([np.array([[0.0, -0.25]]) + face_off, np.full((20, 1), -0.08)], axis=1)
        row["face_px"], row["face_conf"], row["face_code"] = proj(fpts).astype(np.float32), 1.0, LM.DETECTED
        if 40 <= i <= 42:      # short gap (fillable)
            row["right_hand_px"][:] = np.nan
            row["right_hand_world"][:] = np.nan
            row["right_hand_code"], row["right_hand_conf"] = LM.MISSING, 0.0
        if 60 <= i <= 70:      # long gap (must stay missing)
            row["left_hand_px"][:] = np.nan
            row["left_hand_world"][:] = np.nan
            row["left_hand_code"], row["left_hand_conf"] = LM.MISSING, 0.0
        rows.append(row)
    meta = {"path": "synthetic", "sha256": "0" * 64, "display_width": W, "display_height": H, "rotation_cw": 0,
            "rotation_verified": True, "rotation_source": "none", "rotation_note": "", "fps_avg": fps, "sar": 1.0,
            "decoder": "synthetic", "decoder_already_rotated": False, "codec": "none"}
    return rows, meta, {k: np.array(v) for k, v in truth.items()}


# --------------------------------------------------------------------------- individual tests
def t_angles():
    from .features import arm_angles
    for side in ("left", "right"):
        S, E, Wr = arm_pose(side, 0)
        So, _, _ = arm_pose("left" if side == "right" else "right", 0)
        errs = []
        for tw in (-80, -30, 0, 30, 80):
            for fl in (-40, 0, 40):
                pw = np.full((1, 33, 3), np.nan)
                s = LM.SIDE_POSE[side]
                pw[0, s["shoulder"]], pw[0, s["elbow"]], pw[0, s["wrist"]], pw[0, s["other_shoulder"]] = S, E, Wr, So
                hw = synth_hand_world(side, S, E, Wr, So, tw, fl)[None]
                a, _ = arm_angles(pw, hw, side)
                errs.append(max(abs(a["forearm_twist"][0] - tw), abs(a["wrist_flex"][0] - fl)))
        check(f"forearm twist + wrist flexion recovered exactly ({side} hand)", max(errs) < 1e-6,
              f"max error {max(errs):.2e} deg over 15 poses")
    # mirror consistency: mirror x and swap sides -> identical angles
    S, E, Wr = arm_pose("right", 0)
    So, _, _ = arm_pose("left", 0)
    hw = synth_hand_world("right", S, E, Wr, So, 35, 20)
    M = np.array([-1.0, 1, 1])
    pw = np.full((1, 33, 3), np.nan)
    pw[0, LM.R_SH], pw[0, LM.R_EL], pw[0, LM.R_WR], pw[0, LM.L_SH] = S, E, Wr, So
    pwm = np.full((1, 33, 3), np.nan)
    pwm[0, LM.L_SH], pwm[0, LM.L_EL], pwm[0, LM.L_WR], pwm[0, LM.R_SH] = S * M, E * M, Wr * M, So * M
    a, _ = arm_angles(pw, hw[None], "right")
    b, _ = arm_angles(pwm, (hw * M)[None], "left")
    d = max(abs(a[k][0] - b[k][0]) for k in ("forearm_twist", "wrist_flex", "wrist_dev", "elbow_bend"))
    check("angles identical after mirroring (left/right symmetric definitions)", d < 1e-6, f"max diff {d:.2e} deg")


def t_fk():
    from .features import decode_bones, encode_bones
    rng = np.random.default_rng(1)
    P = rng.normal(0, 0.1, (50, 21, 3))
    L, th, ph = encode_bones(LM.HAND_BONES, P, P[:, 0])
    R = decode_bones(LM.HAND_BONES, P[:, 0], L, th, ph, 21)
    err = np.abs(R - P).max()
    check("bones (length, theta, phi) -> coordinates round trip", err < 1e-12, f"max error {err:.1e} m")


def t_norm():
    from .postprocess import denormalise, normalise
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1000, (30, 21, 3))
    o, s = rng.uniform(300, 600, (30, 2)), rng.uniform(80, 200, 30)
    err = np.abs(denormalise(normalise(p, o, s), o, s) - p).max()
    check("normalised <-> pixel round trip", err < 1e-9, f"max error {err:.1e} px")
    k, c = 0.5, np.array([640.0, 360.0])          # camera twice as far: everything shrinks about c
    p2 = p.copy()
    p2[..., :2] = c + k * (p[..., :2] - c)
    p2[..., 2] *= k
    o2, s2 = c + k * (o - c), k * s
    err = np.abs(normalise(p2, o2, s2) - normalise(p, o, s)).max()
    check("normalised coordinates invariant to camera distance", err < 1e-9, f"max diff {err:.1e}")


def t_filters():
    from .metrics import jitter, lag_seconds
    from .postprocess import OneEuro, fill_gaps, gaussian_smooth
    t = np.arange(150) / 30.0
    X = np.stack([np.sin(2 * np.pi * 0.8 * t), np.cos(2 * np.pi * 0.5 * t)], axis=1)[:, None, :]
    valid = np.ones(150, bool)
    valid[50:53] = False
    Xn = X.copy()
    Xn[~valid] = np.nan
    Xf, filled = fill_gaps(t, Xn, valid, 4)
    err = np.abs(Xf[50:53] - X[50:53]).max()
    check("PCHIP gap fill on smooth motion", filled[50:53].all() and err < 0.02, f"max error {err:.1e}")
    valid2 = valid.copy()
    valid2[80:90] = False
    _, filled2 = fill_gaps(t, np.where(valid2[:, None, None], X, np.nan), valid2, 4)
    check("long gaps are NOT filled", not filled2[80:90].any())
    rng = np.random.default_rng(3)
    Y = X + rng.normal(0, 0.02, X.shape)
    S = gaussian_smooth(t, Y, np.ones((150, 1)), 0.04)
    E = OneEuro(1.0, 0.0).run(t, Y.reshape(150, -1)).reshape(Y.shape)
    lag_g, lag_e = lag_seconds(t, X, S), lag_seconds(t, X, E)
    check("zero-phase smoothing: lag = 0 and jitter reduced",
          abs(lag_g) < 0.005 and jitter(t, S) < 0.5 * jitter(t, Y), f"lag {lag_g * 1000:.0f} ms")
    check("causal One Euro filter shows positive lag", lag_e > 0, f"lag {lag_e * 1000:.0f} ms")


def t_metrics():
    from .metrics import kneedle, normalised_errors, spearman, summarise_errors, youden_threshold
    ref = np.zeros((4, 2, 2))
    pred = ref.copy()
    pred[0, 0] = [3, 4]                                  # e = 5/50 = 0.1
    pred[1, 1] = [1, 0]                                  # e = 0.02
    e = normalised_errors(pred, ref, np.full(4, 50.0), np.ones(4, bool), np.array([1, 1, 1, 0], bool))
    s = summarise_errors(e)
    ok = abs(s["nme"] - (0.1 + 0.02) / 6) < 1e-12 and abs(s["pck@0.1"] - 5 / 8) < 1e-12 and abs(s["detection_rate"] - 0.75) < 1e-12
    check("NME / PCK / detection rate on a hand-computed example", ok,
          f"nme={s['nme']:.4f} pck@0.1={s['pck@0.1']:.3f} det={s['detection_rate']:.2f}")
    thr, J = youden_threshold([1, 2, 3, 10, 11, 12], [0, 0, 0, 1, 1, 1], True)
    check("Youden threshold separates perfectly", J == 1.0 and 3 < thr <= 10, f"q*={thr}, J={J}")
    check("Spearman of a monotone relation = 1", abs(spearman([1, 2, 3, 4, 5], [2, 4, 8, 16, 32]) - 1) < 1e-12)
    check("Kneedle picks the elbow", kneedle([0, 0.1, 0.2, 0.6, 1.0], [0, 0.7, 0.9, 0.95, 1.0]) == 2)


def t_transforms(cfg):
    from .transforms import Variant
    pts = np.random.default_rng(4).uniform(0, 700, (50, 2))
    worst = 0.0
    for kind, p in (("shift", 0.05), ("rotate", 10), ("scale", 1.15), ("shake", 1), ("mirror", 1)):
        v = Variant(kind, p, 1280, 720, cfg)
        for i in (0, 5, 17):
            worst = max(worst, np.abs(v.inverse_points(i, v.forward_points(i, pts)) - pts).max())
    check("known transforms: A^-1(A(x)) = x", worst < 1e-9, f"max error {worst:.1e} px")
    check("mirror permutations are involutions",
          np.array_equal(LM.POSE_MIRROR[LM.POSE_MIRROR], np.arange(33)) and
          np.array_equal(LM.FACE_MIRROR[LM.FACE_MIRROR], np.arange(20)))


def t_pipeline(cfg, tmp: Path):
    from .evaluate import (_jsonable, calibrate, compare, determinism, gap_test, qc_status, roundtrips,
                           smoothing_sweep)
    from .pipeline import build_tree
    from .transforms import Variant
    rows, meta, truth = synth_rows()
    tree = build_tree(rows, dict(meta), cfg)
    names = tree["features"]["__attrs__"]["angle_names"]
    A = tree["features"]["angles"]
    for side in ("left", "right"):
        tw = A[:, names.index(f"{side}_forearm_twist")]
        fl = A[:, names.index(f"{side}_wrist_flex")]
        ok = np.isfinite(tw)
        e1 = np.abs(tw[ok] - truth[f"{side}_twist"][ok]).max()
        e2 = np.abs(fl[ok] - truth[f"{side}_flex"][ok]).max()
        check(f"pipeline recovers known twist / flexion through noise + smoothing ({side})", e1 < 3 and e2 < 3,
              f"max err twist {e1:.2f} deg, flex {e2:.2f} deg")
    sm = tree["smoothed"]
    check("3-frame gap filled, 11-frame gap left missing",
          (sm["right_hand_mask"][40:43] == LM.FILLED).all() and (sm["left_hand_mask"][60:71] == LM.MISSING).all())
    rt = roundtrips(tree)
    fk = max(v["max_mm"] for v in rt["angles_lengths_to_coords"].values())
    check("file round trips: n->px and (L,theta,phi)->P",
          rt["norm_to_px_err_px_max"] < 0.01 and fk < 0.01,
          f"{rt['norm_to_px_err_px_max']:.1e} px, {fk:.1e} mm")
    check("avatar retarget: fixed lengths (CV = 0), directions kept",
          rt["retarget_bone_length_cv_max"] < 1e-6 and rt["retarget_direction_err_deg_max"] < 1e-3,
          f"CV {rt['retarget_bone_length_cv_max']:.1e}, dir {rt['retarget_direction_err_deg_max']:.1e} deg")
    tree2 = build_tree(synth_rows()[0], dict(meta), cfg)
    d = determinism(tree, tree2)
    check("determinism: identical input -> bitwise identical output", d["bitwise_identical"])
    v = Variant("rotate", 10, meta["display_width"], meta["display_height"], cfg)
    rows_t, _, _ = synth_rows(affine=v.affine)
    var = build_tree(rows_t, dict(meta), cfg)
    c = compare(tree, var, v, cfg)
    check("equivariance error ~ noise level for an exactly transformed input",
          c["hands"]["nme"] < 0.03 and c["hands"]["detection_rate"] > 0.99, f"hands E/s = {c['hands']['nme']:.4f}")
    g = gap_test(tree, cfg)
    sw = smoothing_sweep(tree, cfg)
    check("gap test and smoothing sweep run", g["curve"] and sw["rows"],
          f"g*={g['max_fillable_frames']} frames, sigma*={sw['recommended_offline_sigma_s']} s")
    lf = tree["qc"]["__attrs__"]
    check("label-free QC: bone-length CV small on a rigid synthetic hand", lf["hand_bone_cv"] < 0.05,
          f"CV {lf['hand_bone_cv']:.4f}")
    # report rendering on synthetic results
    vm = tree["video"]["__attrs__"]
    pdir = tmp / "pilot"
    (pdir / "synthetic").mkdir(parents=True)
    res = {"clip": "synthetic.mp4", "sha256": meta["sha256"],
           "video": {**{k: meta.get(k) for k in ("display_width", "display_height", "rotation_cw", "rotation_source",
                                                  "rotation_verified", "rotation_note", "fps_avg", "codec", "decoder")},
                     "n_frames": vm["n_frames"], "duration_s": vm["duration_s"], "fps_measured": vm["fps_measured"]},
           "h5_size_bytes": 0, "label_free": lf, "roundtrip": rt, "gap_test": g, "smoothing_sweep": sw,
           "overlay": {"frames_matched": 90, "frames_in_video": 90, "timestamp_abs_err_ms_max": 0.0,
                       "timestamp_abs_err_ms_mean": 0.0, "reconstruction_err_px_max": 0.0,
                       "reconstruction_err_px_mean": 0.0},
           "determinism": d, "transforms": {v.name: c}, "degradations": {
               "noise": [{"level": 5, "vs_clean": c, "label_free": lf}, {"level": 10, "vs_clean": c, "label_free": lf}]}}
    cal = calibrate([(lf, 0.9), (dict(lf, hand_jitter=lf["hand_jitter"] * 5), 0.5),
                     (dict(lf, hand_jitter=lf["hand_jitter"] * 2), 0.85), (dict(lf, hand_jitter=lf["hand_jitter"] * 9), 0.3)], cfg)
    res["qc_status"] = qc_status(lf, cal, cfg)
    (pdir / "synthetic" / "results.json").write_text(json.dumps(_jsonable(res)), encoding="utf-8")
    (pdir / "pilot_summary.json").write_text(json.dumps(_jsonable(
        {"clips": ["synthetic.mp4"], "config_hash": cfg.extraction_hash(), "pipeline_version": cfg["pipeline_version"],
         "calibration": cal})), encoding="utf-8")
    from .report import build_report
    rp = build_report(pdir)
    txt = rp.read_text(encoding="utf-8")
    check("report.md renders (maths + tables + plots)", "## 3. Ground truth" in txt and (pdir / "plots").exists(),
          f"{len(txt.splitlines())} lines")


def t_outliers(cfg):
    from .outliers import DUPLICATE, SHAPE, SPIKE, reject_hand_outliers
    from .pipeline import build_tree
    rows, meta, _ = synth_rows(T=90)
    spikes = [20, 55]
    for i in spikes:                                   # isolated jumps of 2 hand sizes
        rows[i]["right_hand_px"][:, 0] += 300
    rows[30]["left_hand_px"] = rows[30]["right_hand_px"].copy()     # duplicate: both hands on one spot
    rows[30]["left_hand_world"] = rows[30]["right_hand_world"].copy()
    rows[45]["right_hand_world"][8] += np.array([0.08, 0, 0], np.float32)  # broken finger shape
    rows[45]["right_hand_world"][12] += np.array([0.08, 0, 0], np.float32)
    rows[45]["right_hand_world"][7] += np.array([0.06, 0.03, 0], np.float32)
    tree = build_tree(rows, dict(meta), cfg)
    rl, rr = tree["raw"]["left_hand_reject"], tree["raw"]["right_hand_reject"]
    hit_spike = all(rr[i] & SPIKE or rr[i] for i in spikes)
    hit_dup = bool(rl[30] & DUPLICATE) or bool(rr[30] & DUPLICATE)
    hit_shape = bool(rr[45] & SHAPE)
    planted = set(spikes) | {30, 45}
    false = [i for i in range(90) if i not in planted and ((rl[i] and tree["raw"]["left_hand_mask"][i]) or
                                                          (rr[i] and tree["raw"]["right_hand_mask"][i]))]
    check("outlier rejection catches planted spikes, duplicate hand and broken hand shape",
          hit_spike and hit_dup and hit_shape, f"spike={hit_spike} duplicate={hit_dup} shape={hit_shape}")
    check("outlier rejection keeps clean fast motion (false rejections)", len(false) <= 1,
          f"{len(false)} of {2 * 90 - 4} clean hand frames rejected")
    sm = tree["smoothed"]["right_hand_px"]
    clean_rows, _, _ = synth_rows(T=90)
    err = np.nanmax(np.abs(sm[spikes[0], :, 0] - clean_rows[spikes[0]]["right_hand_px"][:, 0]))
    check("rejected spike replaced by interpolation in the stored output", err < 15, f"max |dx| {err:.1f} px")


def t_video(cfg, tmp: Path):
    from .video_io import iter_frames, probe
    W, H, n = 320, 192, 24
    path = tmp / "synthetic.mp4"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (W, H))
    for i in range(n):
        img = np.zeros((H, W, 3), np.uint8)
        cv2.rectangle(img, (20, 20), (120, 80), (0, 0, 255), -1)       # asymmetric content
        cv2.circle(img, (250, 150), 20 + i, (0, 255, 0), -1)
        wr.write(img)
    wr.release()
    meta = probe(path)
    frames = list(iter_frames(meta))
    ts = np.array([f[1] for f in frames])
    check("video decode: all frames, display size, increasing timestamps",
          len(frames) == n and frames[0][2].shape == (H, W, 3) and np.all(np.diff(ts) > 0),
          f"{len(frames)} frames, decoder={meta['decoder']}")
    ff = shutil.which("ffmpeg")
    if not ff:
        print("  [SKIP] rotation-tag test (ffmpeg not on PATH)")
        return
    rot = tmp / "rotated.mp4"
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-display_rotation", "90", "-i", str(path), "-c", "copy",
                        str(rot)], capture_output=True)
    if r.returncode != 0:
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(path), "-c", "copy",
                            "-metadata:s:v:0", "rotate=270", str(rot)], capture_output=True)
    if r.returncode != 0:
        print("  [SKIP] rotation-tag test (this ffmpeg cannot write rotation tags)")
        return
    m2 = probe(rot)
    f2 = next(iter_frames(m2))[2]
    ok = m2["rotation_cw"] in (90, 270) and f2.shape[:2] == (W, H)
    check("phone-style rotation tag read and applied (frame becomes portrait)", ok,
          f"rotation_cw={m2['rotation_cw']} source={m2['rotation_source']} verified={m2['rotation_verified']} "
          f"note={m2['rotation_note']}")


def t_h5(cfg, tmp: Path):
    try:
        import h5py  # noqa: F401
    except Exception:
        check("HDF5 write/read round trip", False, "h5py not installed")
        return
    from .pipeline import build_tree
    from .storage import read_h5, write_h5
    rows, meta, _ = synth_rows(T=30)
    tree = build_tree(rows, dict(meta), cfg)
    p = tmp / "x.h5"
    write_h5(p, tree, {"config_hash": cfg.extraction_hash()})
    back = read_h5(p)
    a, b = tree["raw"]["left_hand_px"], back["raw"]["left_hand_px"]
    ok = np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b)) == 0
    names_ok = back["features"]["__attrs__"]["angle_names"] == tree["features"]["__attrs__"]["angle_names"]
    check("HDF5 write/read round trip (lossless float32, names, attrs)", ok and names_ok,
          f"{p.stat().st_size / 1024:.0f} KiB for 30 frames")


def t_models(cfg):
    from .models import check_models
    probs = check_models(cfg.path("models_dir"))
    if probs:
        print("  [WARN] " + "; ".join(probs))
        return
    try:
        from .extractor import Extractor
        ex = Extractor(cfg, "video")
        img = np.zeros((480, 640, 3), np.uint8)
        o = ex.process(img, img, 0.0)
        ex.close()
        check("MediaPipe models load and run (blank frame -> nothing detected)",
              o["pose_code"] == LM.MISSING and o["left_hand_code"] == LM.MISSING)
    except Exception as exc:
        check("MediaPipe models load and run", False, f"{type(exc).__name__}: {exc}")


def run(cfg) -> int:
    RESULTS.clear()
    tmp = Path(tempfile.mkdtemp(prefix="skel_selftest_"))
    print("Self-test")
    for fn, args in ((t_angles, ()), (t_fk, ()), (t_norm, ()), (t_filters, ()), (t_metrics, ()),
                     (t_transforms, (cfg,)), (t_outliers, (cfg,)), (t_pipeline, (cfg, tmp)), (t_video, (cfg, tmp)), (t_h5, (cfg, tmp)),
                     (t_models, (cfg,))):
        try:
            fn(*args)
        except Exception as exc:
            check(fn.__name__, False, f"crashed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    shutil.rmtree(tmp, ignore_errors=True)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    return 1 if n_fail else 0
