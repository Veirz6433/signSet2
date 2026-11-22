"""Builds report.md: every quantity is defined mathematically, then the measured values follow."""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from . import plots


def f(x, d=3, pct=False):
    if x is None or (isinstance(x, float) and x != x):
        return "–"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(f(v, d, pct) for v in x) + "]"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int,)) and not pct:
        return str(x)
    return f"{100 * x:.{max(d - 2, 1)}f}%" if pct else f"{x:.{d}g}" if abs(x) < 1e-3 and x != 0 else f"{x:.{d}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


METHOD = r"""
## 1. Notation

| symbol | meaning |
|---|---|
| $i = 1..T$ | frame index; $t_i$ = presentation timestamp (s) read from the container (variable frame rate safe) |
| $\Delta t$ | median of $t_{i+1}-t_i$ |
| $j$ | landmark index (body 33, each hand 21, face 20) |
| $\mathbf p_{ij}\in\mathbb R^2$ | predicted pixel position in the *display* frame (rotation and pixel aspect applied) |
| $\mathbf g_{ij}$ | ground-truth position (definition depends on the ground-truth type, §3) |
| $\mathbf P_{ij}\in\mathbb R^3$ | MediaPipe world coordinates (m), camera-aligned axes ($x$ right, $y$ down, $z$ away) |
| $m_{ij}\in\{0,1,2,3\}$ | mask: missing / detected / detected on wrist crop / interpolated |
| $\mathbb 1[\cdot]$ | indicator function |

## 2. Geometry of the stored data

**2.1 Rotation.** Clockwise rotation $\theta\in\{0,90,180,270\}$ is read from the display-matrix side data
(or `rotate` tag) and applied explicitly: $I_{\text{display}} = \mathrm{rot}_\theta(I_{\text{coded}})$.
Verification against an independent decoder (OpenCV with auto-orientation, image $J$):

$$\hat\theta=\arg\max_{\phi\in\{0,90,180,270\}}\ \mathrm{NCC}\big(\mathrm{rot}_\phi(I_{\text{coded}}),J\big),\qquad
\mathrm{NCC}(a,b)=\frac{\sum_k (a_k-\bar a)(b_k-\bar b)}{\sqrt{\sum_k (a_k-\bar a)^2\sum_k (b_k-\bar b)^2}}$$

on $64\times64$ grey images. *Verified* $\iff \hat\theta=\theta$ and $\mathrm{NCC}>0.8$.

**2.1b Hand detection (two-stage).** Pass 1: hand model on the full frame (video mode, tracking). Pass 2, for every
hand that is missed or whose box $b=\max(w,h)<b_{\text{small}}$: the hand model on a square crop of side
$\max(b_{\min},\,2b)$ around the found hand (or $\max(b_{\min},\,2.2\lVert\mathbf p_W-\mathbf p_E\rVert)$ centred at
$\mathbf p_W+0.35(\mathbf p_W-\mathbf p_E)$ if missed), bicubically upscaled to $256\times256$ when smaller. Crop results
are mapped back exactly ($\mathbf p=\mathbf x_0+\mathbf u\cdot\text{side}$, $\mathbf u\in[0,1]^2$) and get $m_{ij}=2$.
A hand is assigned to the side whose pose wrist is nearest (in forearm lengths, $\le 0.8$).

**2.1c Hand outlier rejection** (before filling and smoothing; raw data are kept, the reason is stored as bit flags).
A detected hand frame $i$ is rejected if any test fails:

| test | statistic | reject if |
|---|---|---|
| duplicate | $\overline{\lVert\mathbf p^{L}_{ij}-\mathbf p^{R}_{ij}\rVert}_j/\max(b^L_i,b^R_i)$ | $<0.3$ (the hand farther from its pose wrist is removed) |
| far | $\lVert\mathbf p_{i,\text{wrist}}-\mathbf p^{\text{pose}}_{i,\text{wrist}}\rVert/\lVert\mathbf p_W-\mathbf p_E\rVert$ | $>0.8$ |
| shape | mean of the 5 largest $r_{ib}=\lvert \ell_{ib}/\ell^*_b-1\rvert$, $\ell_{ib}=L_{ib}/\sum_b L_{ib}$ (3D), $\ell^*_b=\mathrm{med}_i\,\ell_{ib}$ | $>0.35$ |
| spike (Hampel) | $\mathrm{med}_j\lVert\mathbf p_{ij}-\mathrm{med}_{k\in W_i}\mathbf p_{kj}\rVert/\mathrm{med}_{k\in W_i}b_k$, $W_i=\{k:\lvert t_k-t_i\rvert\le0.08\text{ s}\}$ | $>0.35$ |

The temporal median of a monotone sequence equals its centre value, so fast but smooth motion is never rejected by
the spike test; isolated jumps are. Gap filling and smoothing (§2.2–2.3) use accepted frames only.

**2.2 Gap filling.** A run of $g$ consecutive missing frames with valid neighbours on both sides is filled iff
$g\le g_{\max}=\lfloor \tau_{\text{gap}}/\Delta t\rfloor$, by shape-preserving piecewise-cubic Hermite (PCHIP)
interpolation through up to 3 valid frames on each side (linear if only 2 exist). Filled points get $m_{ij}=3$.

**2.3 Smoothing (zero phase).** Gaussian kernel regression in time with confidence weights $c_{j}\in[0,1]$:

$$\tilde{\mathbf p}_{ij}=\frac{\sum_{k:|t_k-t_i|\le3\sigma} w_{ik}\,c_{kj}\,\mathbf p_{kj}}{\sum_{k} w_{ik}\,c_{kj}},\qquad
w_{ik}=\exp\!\Big(-\frac{(t_i-t_k)^2}{2\sigma^2}\Big)$$

The kernel is symmetric, so the phase lag is $0$ by construction. Raw values are stored next to smoothed ones.

**2.4 Body-relative normalisation (camera-distance invariance).**

$$\mathbf o_i=\tfrac12(\mathbf p_{i,\text{LS}}+\mathbf p_{i,\text{RS}}),\qquad
s_i=\frac{\lVert\mathbf p_{i,\text{LS}}-\mathbf p_{i,\text{RS}}\rVert}{\mathrm{clip}(c_i,0.5,1)},\qquad
c_i=\frac{\lVert(\mathbf P_{\text{LS}}-\mathbf P_{\text{RS}})_{xy}\rVert}{\lVert\mathbf P_{\text{LS}}-\mathbf P_{\text{RS}}\rVert}$$

($c_i$ = cosine of the shoulder line's angle out of the image plane, corrects for turning), $s_i$ Gaussian-smoothed
with $\sigma_s=0.25$ s. Stored coordinates $\mathbf n_{ij}=(\mathbf p_{ij}-\mathbf o_i)/s_i$ (unit = one shoulder width).
Scaling the camera distance by $k$ multiplies $\mathbf p-\mathbf o$ and $s$ by $\approx 1/k$, so $\mathbf n$ is unchanged.

**2.5 Angles** (all 3D, from world coordinates; scale-free by definition).

* bend at joint $j$ between bones $\mathbf b_1=\mathbf P_j-\mathbf P_a$, $\mathbf b_2=\mathbf P_b-\mathbf P_j$:
  $\beta=\arccos(\hat{\mathbf b}_1\cdot\hat{\mathbf b}_2)$ (elbow, thumb joints, finger spread).
* signed finger flexion about the knuckle axis $\mathbf a=s_h\,\widehat{\mathbf P_5-\mathbf P_{17}}$
  ($s_h=+1$ right hand, $-1$ left): $\phi=\operatorname{atan2}\big((\hat{\mathbf b}_1\times\hat{\mathbf b}_2)\cdot\mathbf a,\ \hat{\mathbf b}_1\cdot\hat{\mathbf b}_2\big)$.
* palm frame: $\mathbf x_h=\widehat{\mathbf P_9-\mathbf P_0}$, $\mathbf n=s_h(\mathbf P_5-\mathbf P_0)\times(\mathbf P_{17}-\mathbf P_0)$,
  $\mathbf z_h=\widehat{\mathbf n-(\mathbf n\cdot\mathbf x_h)\mathbf x_h}$ (palmar normal), $\mathbf y_h=\mathbf z_h\times\mathbf x_h$;
  stored as the unit quaternion of $R=[\mathbf x_h\ \mathbf y_h\ \mathbf z_h]$ (sign made continuous in time).
* forearm axis $\mathbf f=\widehat{\mathbf P_W-\mathbf P_E}$, upper arm $\mathbf u=\mathbf P_E-\mathbf P_S$,
  $\mathbf z_f=\widehat{\mathbf z_h-(\mathbf z_h\cdot\mathbf f)\mathbf f}$, $\mathbf y_f=\mathbf z_f\times\mathbf f$ (swing–twist split):

$$\text{flex}=\operatorname{atan2}(\mathbf x_h\cdot\mathbf z_f,\ \mathbf x_h\cdot\mathbf f),\qquad
\text{dev}=-s_h\operatorname{atan2}(\mathbf x_h\cdot\mathbf y_f,\ \mathbf x_h\cdot\mathbf f)$$

$$\mathbf r=\widehat{w\,\mathbf r_e+(1-w)\,\mathbf r_s},\quad \mathbf r_e=-s_h\widehat{\mathbf u\times\mathbf f},\quad
\mathbf r_s=-\big(\mathbf l-(\mathbf l\cdot\mathbf f)\mathbf f\big),\quad w=\mathrm{clip}\Big(\frac{\lVert\hat{\mathbf u}\times\mathbf f\rVert}{\sin 30^\circ},0,1\Big)$$

$$\text{twist}=-s_h\operatorname{atan2}\big((\mathbf r\times\mathbf z_f)\cdot\mathbf f,\ \mathbf r\cdot\mathbf z_f\big)$$

  with $\mathbf l$ the outward shoulder line. twist $=0$: palm faces the body midline; $>0$: pronation (both hands).
  Reliability weight stored: $|\mathbf z_{h,z}|\cdot\max(w,\lVert\hat{\mathbf l}\times\mathbf f\rVert)$.

**2.6 Distances.** Hand-internal: $d=\lVert\mathbf P_a-\mathbf P_b\rVert$ (m) and $d/\lVert\mathbf P_9-\mathbf P_0\rVert$ (palm lengths).
Hand-to-body: $d=\lVert\mathbf n_a-\mathbf n_b\rVert$ (shoulder widths).

**2.7 Bone encoding and reconstruction.** For each bone (parent $a$ → child $b$) of the body and hand trees:

$$L=\lVert\mathbf P_b-\mathbf P_a\rVert,\quad \vartheta=\arccos(\hat u_z),\quad \varphi=\operatorname{atan2}(\hat u_y,\hat u_x),\qquad
\mathbf P_b=\mathbf P_a+L\,(\sin\vartheta\cos\varphi,\ \sin\vartheta\sin\varphi,\ \cos\vartheta)$$

The avatar retarget uses the same $(\vartheta,\varphi)$ with fixed human lengths $L^{\text{std}}$ (fractions of height), so
$\mathrm{CV}(L)=0$ by construction.

## 3. Ground truth: four constructions

**G1 – human labels (absolute accuracy).** COCO-WholeBody v1.0 val2017: manually annotated body (17), hands (2×21, same
order as MediaPipe) and face (68 iBUG points; our 20 points map to iBUG indices, averaged where two are given).
$\mathbf g_{ij}$ = label. The extractor runs unchanged (image mode) on each person crop.

**G2 – known transforms (exact, no labels).** Each frame is warped by a known affine map $A_i(\mathbf x)=M_i\mathbf x+\mathbf c_i$
(shift, rotation, scale, seeded AR(1) camera shake, mirror). Since the true positions satisfy $\mathbf g^{T}_{ij}=A_i(\mathbf g_{ij})$,
the equivariance error

$$E_{ij}=\big\lVert A_i^{-1}\big(\mathbf p^{T}_{ij}\big)-\mathbf p_{ij}\big\rVert$$

obeys (triangle inequality) $E_{ij}\le \lVert M_i^{-1}\rVert\, e^{T}_{ij}+e_{ij}$ with $e=\lVert\mathbf p-\mathbf g\rVert$.
For rigid maps $\lVert M_i^{-1}\rVert=1$, hence $\tfrac12(e^{T}_{ij}+e_{ij})\ge \tfrac12E_{ij}$: **$E/2$ is a lower bound on the mean
true error of the two runs**. (A bias that is itself equivariant is not detected; G1 covers that.)
Mirror additionally permutes left/right labels.

**G3 – controlled degradation (robustness).** Reference = clean-video output $\mathbf p^{0}$. Degraded run $\mathbf p^{d}$
(brightness ×, Gaussian blur, motion blur, sensor noise, down-scaling, JPEG quality, frame-rate drop; 5 levels each).
$\Delta^{d}_{ij}=\lVert\mathbf p^{d}_{ij}-\mathbf p^{0}_{ij}\rVert$ gives, by the triangle inequality,
$\Delta^d-e^0\le e^d\le\Delta^d+e^0$: the degraded true error is within $\pm e^0$ (estimated by G1) of the measured change.

**G4 – hidden detections (gap filling, in output space).** Truth $\mathbf q=\mathcal S(\mathcal F(\mathbf p))$ where
$\mathcal F$ = gap filling (§2.2), $\mathcal S$ = smoothing (§2.3), $\mathbf p$ = accepted detections. Accepted frames
$a..a+g-1$ (with $\ge3$ accepted frames on each side) are deleted and the same pipeline is re-run:
$e_{ij}=\lVert\mathcal S(\mathcal F(\mathbf p^{\setminus g}))_{ij}-\mathbf q_{ij}\rVert/b(\mathbf q_i)$ at the deleted frames.
Largest acceptable gap $g^*=\max\{g:\ \mathrm{PCK}_{0.1}(g)\ge 0.95\}$ (first failure stops the search);
$\tau_{\text{gap}}=g^*\Delta t$.

**Identity checks (exact).** (a) $\mathbf n\to\mathbf p$: $\lVert \mathbf n_{ij}s_i+\mathbf o_i-\tilde{\mathbf p}_{ij}\rVert$;
(b) $(L,\vartheta,\varphi)\to\mathbf P$ (§2.7) vs stored $\mathbf P$; (c) overlay timestamp match $|t^{\text{video}}_k-t_{i(k)}|$,
$i(k)=\arg\min_i|t^{\text{video}}_k-t_i|$; (d) determinism: $\max|x^{(1)}-x^{(2)}|$ over every stored array of two runs.
All are computed **from the HDF5 file read back from disk**.

## 4. Scores

Normaliser $s$: hands $s=\max(w,h)$ of the hand's 2D box; face $s=\lVert\mathbf g_{\text{RE,out}}-\mathbf g_{\text{LE,out}}\rVert$;
body $s$ = torso length (G1) or $s_i$ of §2.4 (G2–G3). Normalised error $e_{ij}=\lVert\mathbf p_{ij}-\mathbf g_{ij}\rVert/s_i$;
a missed detection sets $e_{ij}=\infty$.

$$\mathrm{NME}=\frac{1}{|D|}\sum_{(i,j)\in D}e_{ij}\ \ (D=\text{detected}),\qquad
\mathrm{PCK}_\alpha=\frac1{|G|}\sum_{(i,j)\in G}\mathbb 1[e_{ij}<\alpha]\ \ (G=\text{all labelled}),\qquad
\mathrm{AUC}_{0.2}=\frac1{0.2}\int_0^{0.2}\mathrm{PCK}_\alpha\,d\alpha$$

Detection rate $=|D|/|G|$. 95 % CIs: percentile bootstrap over samples (1000 resamples, fixed seed).

Temporal (non-uniform differences, body units):
$\mathbf v_i=\frac{\mathbf n_{i+1}-\mathbf n_i}{t_{i+1}-t_i}$, $\mathbf a_i=\frac{\mathbf v_i-\mathbf v_{i-1}}{(t_{i+1}-t_{i-1})/2}$,

$$J=\overline{\lVert\mathbf a_{ij}\rVert},\qquad
\text{lag}=\arg\max_{\tau\in[-0.3,0.3]}\ \frac{\sum_c\sum_t x_c(t)\,y_c(t+\tau)}{\sqrt{\sum_c\sum_t x_c^2\ \sum_c\sum_t y_c^2}},\qquad
\text{peak loss}=1-\frac{P_{95}(|\mathbf v^{\text{sm}}|)}{P_{95}(|\mathbf v^{\text{raw}}|)}$$

The integer-frame arg max is refined to sub-frame precision by a parabola through the three correlations around it.
($x_c$, $y_c$: mean-removed raw and smoothed coordinate $c$ on a uniform grid.) Fidelity $=\overline{\lVert\tilde{\mathbf n}-\mathbf n\rVert/s_h}$. Smoothing knee (Kneedle): with $x$ = peak loss and $y$ = jitter
reduction $1-J/J_0$ min–max normalised to $[0,1]$, $k^*=\arg\max_k(y_k-x_k)$. Live filter: One Euro,
$\alpha=(1+\tau/\Delta t)^{-1}$, $\tau=1/(2\pi f_c)$, $f_c=f_{\min}+\beta|\dot{\hat x}|$.

Physical plausibility (label-free): bone-length variation $\mathrm{CV}_b=\sigma(L_b)/\mu(L_b)$ (3D, should be $\approx0$);
symmetry $|1-\mathrm{med}(L_{\text{left}})/\mathrm{med}(L_{\text{right}})|$ (upper arm, forearm, palm);
violation rate $=\frac{1}{N}\sum\mathbb 1[\text{angle}\notin[\ell,u]]$ with $[\ell,u]$ the approximate active range of each joint.

Calibration (label-free metric $q$ vs robustness). Clip stability baseline $b_c=\mathrm{med}$ of hand
$\mathrm{PCK}_{0.1}$ over the rigid G2 transforms (shift, rotate, scale) of clip $c$. For every degraded variant $v$ of
clip $c$: $y_v=\mathrm{PCK}^{\text{hands}}_{0.1}(v\text{ vs clean})/b_c$; *bad* $\iff y_v<0.85$.
$\rho_S(q,y)$ = Spearman correlation over all variants of all clips. Threshold $q^*=\arg\max_q\ \mathrm{TPR}(q)-\mathrm{FPR}(q)$
(Youden's $J$). Metrics with $|\rho_S|\ge0.5$ are used for QC:
$\text{flags}=\#\{q:\ q\text{ on the bad side of }q^*\}$; pass $=0$, warn $=1$, fail $\ge2$ or any-hand detection $<0.3$.
"""


def _safe(name):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def _g1_rows(name, r):
    return [name, r["n_samples"], f(r["detection_rate"], 3), f(r["nme"], 4), f(r.get("nme_ci95"), 4),
            f(r.get("pck@0.05"), 3), f(r.get("pck@0.1"), 3), f(r.get("pck@0.1_ci95"), 3), f(r.get("pck@0.2"), 3),
            f(r.get("auc@0.2"), 3)]


def _rigid(r, key="hands", sub="pck@0.1"):
    v = [c.get(key, {}).get(sub) for n, c in r.get("transforms", {}).items() if n.split("=")[0] in ("shift", "rotate", "scale")]
    v = [x for x in v if isinstance(x, (int, float))]
    return sorted(v)[len(v) // 2] if v else None


def _rigid_raw(r):
    v = [c.get("hands_raw", {}).get("pck@0.1") for n, c in r.get("transforms", {}).items()
         if n.split("=")[0] in ("shift", "rotate", "scale")]
    v = [x for x in v if isinstance(x, (int, float))]
    return sorted(v)[len(v) // 2] if v else None


def key_metrics(r):
    """Headline numbers of one clip (used for the summary and for before/after comparison)."""
    lf = r["label_free"]
    ag = r.get("second_model_agreement", {})
    hands_ag = [ag.get(k, {}).get("pck@0.1") for k in ("left_hand", "right_hand")]
    hands_ag = [x for x in hands_ag if isinstance(x, (int, float))]
    return {
        "left hand detected (frames)": lf.get("left_hand_detection_rate"),
        "right hand detected (frames)": lf.get("right_hand_detection_rate"),
        "left hand kept after rejection": lf.get("left_hand_accepted_rate", lf.get("left_hand_detection_rate")),
        "right hand kept after rejection": lf.get("right_hand_accepted_rate", lf.get("right_hand_detection_rate")),
        "hand jitter J (raw)": lf.get("hand_jitter_raw", lf.get("hand_jitter")),
        "hand jitter J (kept)": lf.get("hand_jitter"),
        "hand bone-length CV (raw)": lf.get("hand_bone_cv_raw", lf.get("hand_bone_cv")),
        "hand bone-length CV (kept)": lf.get("hand_bone_cv"),
        "G2 hands PCK@0.1, median rigid (raw)": _rigid_raw(r) if _rigid_raw(r) is not None else _rigid(r),
        "G2 hands PCK@0.1, median rigid (kept)": _rigid(r),
        "G2 body PCK@0.1, median rigid": _rigid(r, "body"),
        "G2 face PCK@0.1, median rigid": _rigid(r, "face"),
        "RTMW agreement hands PCK@0.1": sum(hands_ag) / len(hands_ag) if hands_ag else None,
        "G4 safe gap g* (frames)": r["gap_test"]["max_fillable_frames"],
        "determinism bitwise identical": r.get("determinism", {}).get("bitwise_identical"),
        "rebuild error n→p max (px)": r["roundtrip"]["norm_to_px_err_px_max"],
    }


def build_report(pilot_dir: Path, coco_dir: Path | None = None, out_path: Path | None = None,
                 baseline_dir: Path | None = None) -> Path:
    pilot_dir = Path(pilot_dir)
    summary = json.loads((pilot_dir / "pilot_summary.json").read_text(encoding="utf-8"))
    clips = [json.loads((pilot_dir / Path(c).stem / "results.json").read_text(encoding="utf-8")) for c in summary["clips"]]
    coco = None
    cpath = Path(coco_dir or pilot_dir) / "coco_results.json"
    if cpath.exists():
        coco = json.loads(cpath.read_text(encoding="utf-8"))
    base = {}
    if baseline_dir and not (Path(baseline_dir) / "pilot_summary.json").exists():
        print(f"WARNING: baseline skipped - no pilot_summary.json in {baseline_dir}")
        baseline_dir = None
    if baseline_dir:
        bdir = Path(baseline_dir)
        bs = json.loads((bdir / "pilot_summary.json").read_text(encoding="utf-8"))
        for c in bs["clips"]:
            fp = bdir / Path(c).stem / "results.json"
            if fp.exists():
                base[c] = json.loads(fp.read_text(encoding="utf-8"))
        base_version = bs.get("pipeline_version", "?")
    plot_dir = pilot_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    out_path = Path(out_path or pilot_dir / "report.md")

    L = ["# Skeleton extraction — evaluation report", "",
         table(["item", "value"], [
             ["pipeline version", summary["pipeline_version"]], ["config hash", f"`{summary['config_hash']}`"],
             ["generated (UTC)", _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")],
             ["clips", ", ".join(summary["clips"])],
             ["absolute-accuracy set (G1)", coco["dataset"] if coco else "not run"]])]

    # ---------------------------------------------------------------- 0. summary
    L.append("\n## 0. Summary of measured values\n")
    km = [key_metrics(r) for r in clips]
    if km:
        L.append(table(["quantity"] + [r["clip"] for r in clips],
                       [[k] + [f(m[k], 4) for m in km] for k in km[0]]))
    if coco and coco.get("models"):
        rows = []
        for name, m in coco["models"].items():
            for part in ("hands", "face", "body"):
                if part in m:
                    rows.append([name, part, f(m[part]["pck@0.1"], 3), f(m[part].get("pck@0.1_ci95"), 3),
                                 f(m[part]["nme"], 4), f(m[part]["detection_rate"], 3)])
        L.append("\nAbsolute accuracy against human labels (G1):\n")
        L.append(table(["model", "part", "PCK@0.1", "95% CI", "NME", "det. rate"], rows))
    if base:
        L.append(f"\n### 0.1 Before / after (baseline v{base_version} → v{summary['pipeline_version']})\n")
        for r in clips:
            b = base.get(r["clip"])
            if not b:
                continue
            kb, kn = key_metrics(b), key_metrics(r)
            rows = []
            for k in kn:
                x0, x1 = kb.get(k), kn.get(k)
                d = (x1 - x0) if isinstance(x0, (int, float)) and isinstance(x1, (int, float)) \
                    and not isinstance(x0, bool) else None
                rows.append([k, f(x0, 4), f(x1, 4), f(d, 4) if d is not None else "–"])
            L.append(f"\n`{r['clip']}`\n")
            L.append(table(["quantity", "before", "after", "after − before"], rows))

    L += [METHOD, "## 5. Results"]

    # ---------------------------------------------------------------- G1
    L.append("\n### 5.1 Absolute accuracy (G1, human labels)\n")
    if coco:
        hdr = ["model / part", "n", "det. rate", "NME", "NME 95% CI", "PCK@0.05", "PCK@0.1", "PCK@0.1 95% CI",
               "PCK@0.2", "AUC@0.2"]
        rows = []
        for name, m in (coco.get("models") or {"mediapipe": coco}).items():
            for label, key in (("hands", "hands"), ("face (20 pts)", "face"), ("body (6 pts)", "body")):
                if m.get(key):
                    rows.append(_g1_rows(f"{name}: {label}", m[key]))
            for b_, rr in m.get("hands_by_box_px", {}).items():
                rows.append(_g1_rows(f"{name}: hands, box {b_} px", rr))
        L.append(table(hdr, rows))
        L.append("\n(RTMW outputs 2D points only; MediaPipe outputs 2D + 3D.)\n")
        pj = coco.get("hand_per_joint_nme")
        if pj:
            L.append("\nPer-joint hand NME (mediapipe):\n")
            L.append(table(["joint", "NME"], [[k, f(v, 4)] for k, v in pj.items()]))
    else:
        L.append("G1 not run (`python main.py coco-eval`).")

    # ---------------------------------------------------------------- per clip
    for n, r in enumerate(clips, 1):
        stem = _safe(Path(r["clip"]).stem)
        v, lf, rt = r["video"], r["label_free"], r["roundtrip"]
        L.append(f"\n### 5.{n + 1} Clip `{r['clip']}`\n")
        L.append(table(["quantity", "value"], [
            ["display size (px)", f"{v['display_width']} × {v['display_height']}"],
            ["rotation θ (°), source, verified", f"{v['rotation_cw']}, {v['rotation_source']}, {f(v['rotation_verified'])}"],
            ["frames T, duration (s), measured fps", f"{v['n_frames']}, {f(v['duration_s'], 3)}, {f(v['fps_measured'], 4)}"],
            ["HDF5 size (bytes)", r["h5_size_bytes"]],
            ["detection rate: body / left hand / right hand / face",
             " / ".join(f(lf[k], 3) for k in ("pose_detection_rate", "left_hand_detection_rate",
                                                "right_hand_detection_rate", "face_detection_rate"))],
            ["kept after outlier rejection: left / right hand",
             f"{f(lf.get('left_hand_accepted_rate'), 3)} / {f(lf.get('right_hand_accepted_rate'), 3)}"],
            ["fraction of detected hand frames rejected", f(lf.get("hand_rejected_rate"), 3)],
            ["hand frames from the upscaled-crop pass", f(lf["hand_crop_redetect_rate"], 3)],
            ["hand frames interpolated", f(lf["hand_filled_rate"], 3)],
            ["jitter J (shoulder widths/s²): hands raw / hands kept / body / face",
             " / ".join(f(lf.get(k), 4) for k in ("hand_jitter_raw", "hand_jitter", "body_jitter", "face_jitter"))],
            ["bone-length CV: hands raw / hands kept / arms",
             f"{f(lf.get('hand_bone_cv_raw'), 4)} / {f(lf['hand_bone_cv'], 4)} / {f(lf['arm_bone_cv'], 4)}"],
            ["left/right symmetry error", f(lf["symmetry_error"], 4)],
            ["angle violation rate", f(lf["angle_violation_rate"], 4)],
            ["stability baseline b_c (median rigid G2 hand PCK@0.1)", f(r.get("hand_stability_baseline"), 3)],
            ["QC status (flags)", f"{r['qc_status']['status']} ({', '.join(r['qc_status']['flags']) or 'none'})"],
        ]))
        d = r.get("determinism", {})
        ov = r["overlay"]
        L.append("\nIdentity checks (from the file read back from disk):\n")
        L.append(table(["check", "value"], [
            ["determinism: bitwise identical / max abs difference", f"{f(d.get('bitwise_identical'))} / {f(d.get('max_abs_diff'), 3)}"],
            ["n → p reconstruction error (px): max / mean",
             f"{f(rt['norm_to_px_err_px_max'], 3)} / {f(rt['norm_to_px_err_px_mean'], 3)}"],
            *[[f"(L, ϑ, φ) → P error (mm), {k}: max / mean", f"{f(x['max_mm'], 3)} / {f(x['mean_mm'], 3)}"]
              for k, x in rt["angles_lengths_to_coords"].items()],
            ["avatar retarget: max CV(L) / max direction error (°)",
             f"{f(rt['retarget_bone_length_cv_max'], 3)} / {f(rt['retarget_direction_err_deg_max'], 3)}"],
            ["overlay: frames matched / in video", f"{ov['frames_matched']} / {ov['frames_in_video']}"],
            ["overlay: timestamp error (ms) max / mean",
             f"{f(ov['timestamp_abs_err_ms_max'], 3)} / {f(ov['timestamp_abs_err_ms_mean'], 3)}"],
            ["overlay: reconstruction error (px) max / mean",
             f"{f(ov['reconstruction_err_px_max'], 3)} / {f(ov['reconstruction_err_px_mean'], 3)}"],
        ]))
        if r["transforms"]:
            L.append("\nG2 — known transforms, equivariance error $E/s$ (after outlier rejection; hands also before):\n")
            rows = []
            for name, c in r["transforms"].items():
                for part in ("hands", "body", "face"):
                    x = c.get(part, {})
                    raw_p = c.get("hands_raw", {}).get("pck@0.1") if part == "hands" else None
                    rows.append([name, part, f(x.get("nme"), 4), f(x.get("pck@0.05"), 3), f(x.get("pck@0.1"), 3),
                                 f(raw_p, 3) if part == "hands" else "", f(x.get("detection_rate"), 3),
                                 f(x.get("nme") / 2 if x.get("nme") else None, 4)])
            L.append(table(["transform", "part", "mean E/s", "PCK@0.05", "PCK@0.1", "PCK@0.1 before rejection",
                            "det. agreement", "lower bound on mean true error (E/2s)"], rows))
        if r["degradations"]:
            L.append("\nG3 — degradations, change vs clean $\\Delta/s$:\n")
            rows = []
            for kind, levels in r["degradations"].items():
                for lv in levels:
                    c = lv["vs_clean"]
                    rows.append([kind, lv["level"], *[f(c.get(p, {}).get("pck@0.1"), 3) for p in ("hands", "body", "face")],
                                 *[f(c.get(p, {}).get("nme"), 4) for p in ("hands", "body", "face")],
                                 f(lv["label_free"].get("any_hand_detection_rate"), 3)])
            L.append(table(["degradation", "level", "PCK@0.1 hands", "PCK@0.1 body", "PCK@0.1 face", "Δ/s hands",
                            "Δ/s body", "Δ/s face", "any-hand det."], rows))
            p = plots.degradation_plot(r, plot_dir / f"{stem}_degradation.png")
            if p:
                L.append(f"\n![degradation curves](plots/{p.name})\n")
        g = r["gap_test"]
        L.append(f"\nG4 — gap filling ({g.get('space', 'raw detections')}):\n")
        L.append(table(["gap g (frames)", "gap (s)", "n points", "NME", "PCK@0.1", "P95 error"],
                       [[x["gap_frames"], f(x["gap_s"], 3), x.get("n", 0), f(x.get("nme"), 4), f(x.get("pck"), 3),
                         f(x.get("p95_err"), 4)] for x in g["curve"]]))
        L.append(f"\n$g^*={g['max_fillable_frames']}$ frames, $\\tau_{{gap}}={f(g['max_fillable_s'], 3)}$ s "
                 f"(criterion {g['criterion']}).\n")
        p = plots.gap_plot(g, 0.95, plot_dir / f"{stem}_gap.png")
        if p:
            L.append(f"![gap test](plots/{p.name})\n")
        s = r["smoothing_sweep"]
        L.append(f"\nSmoothing trade-off (hands after rejection, raw $J_0$ = {f(s['raw_jitter'], 4)}):\n")
        L.append(table(["filter", "J", "1 − J/J₀", "lag (ms)", "peak loss", "fidelity"],
                       [[x["label"], f(x["jitter"], 4), f(x["jitter_reduction"], 3), f(x["lag_ms"], 3),
                         f(x["peak_loss"], 3), f(x["fidelity_nme"], 4)] for x in s["rows"]]))
        L.append(f"\nKnee: offline $\\sigma^*={s['recommended_offline_sigma_s']}$ s; "
                 f"live $f_{{\\min}}^*={s['recommended_live_min_cutoff_hz']}$ Hz.\n")
        p = plots.smoothing_plot(s, plot_dir / f"{stem}_smoothing.png")
        L.append(f"![smoothing trade-off](plots/{p.name})\n")
        if r.get("second_model_agreement"):
            a = r["second_model_agreement"]
            L.append("\nIndependent-model agreement (RTMW), $D/s$:\n")
            L.append(table(["part", "mean D/s", "PCK@0.1 agreement", "det. rate"],
                           [[k, f(x.get("nme"), 4), f(x.get("pck@0.1"), 3), f(x.get("detection_rate"), 3)]
                            for k, x in a.items()]))

    # ---------------------------------------------------------------- calibration
    cal = summary["calibration"]
    frac = cal.get("bad_if_below_fraction_of_baseline")
    L.append(f"\n### 5.{len(clips) + 2} Calibration of label-free QC metrics\n")
    L.append(f"{cal['n_variants']} degraded variants, {cal['n_bad']} bad "
             f"($y_v<{frac}$, baselines $b_c$ = {f(cal.get('baselines'), 3)}).\n")
    L.append(table(["metric q", "ρ_S(q, y)", "bad side", "threshold q*", "Youden J", "used for QC"],
                   [[k, f(v["spearman_rho_vs_hand_pck"], 3), "≥ q*" if v["higher_is_worse"] else "≤ q*",
                     f(v["threshold"], 4), f(v["youden_J"], 3), f(v["used_for_qc"])]
                    for k, v in cal["metrics"].items()]))

    # ---------------------------------------------------------------- derived settings
    if clips:
        gs = [c["gap_test"]["max_fillable_s"] for c in clips]
        ss = [c["smoothing_sweep"]["recommended_offline_sigma_s"] for c in clips]
        L.append(f"\n### 5.{len(clips) + 3} Settings derived from the measurements\n")
        L.append(table(["setting", "per clip", "chosen value", "rule"], [
            ["cleanup.max_gap_s", f(gs, 3), f(min(gs), 3), "min over clips of $g^*\\Delta t$ (conservative)"],
            ["cleanup.smoothing_sigma_s", f(ss, 3), f(sorted(ss)[len(ss) // 2], 3), "median over clips of $\\sigma^*$"],
        ]))
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out_path
