# Skeleton extraction, reconstruction and evaluation

Turns any video into a clean, scale-independent skeleton stored in HDF5, rebuilds that skeleton
from the file alone and draws it on the original video, and measures how accurate it is with
maths rather than by eye.

| Checklist item | What does it | Command |
|---|---|---|
| Any video input (size, shape, fps, rotation, dark, blurry, shaky) | rotation-aware decoding, real per-frame timestamps, brightness fix for the detector only, a second hand pass on an upscaled crop for small or missed hands, hand outlier rejection, gap filling, zero-lag smoothing, body-relative coordinates | `extract` |
| Accurate extraction: face, shoulders, elbows, palm, all finger joints, angles, distances | MediaPipe pose (33) + both hands (2×21) + 20 face points; 48 joint angles (incl. forearm twist and wrist bend) and 56 distances, all independent of camera distance | `extract` |
| Best file format | HDF5, one file per video, float32 + compression, about 1–2 MB per 10 s clip at 30 fps | — |
| Rebuild + overlay | skeleton rebuilt from the HDF5 file only and drawn on the video (MP4), with reconstruction and timing error | `overlay` |
| Cross-evaluation | 4 kinds of ground truth (human labels, known transforms, controlled degradation, hidden detections) and every formula, written to `report.md` | `pilot` |
| Bulk and consistent | folder in → files out, skips finished videos, never stops on a bad file, identical results on every run | `extract <folder>` |

---

## 1. Setup (once per machine)

Requirements: **64-bit Python 3.12**, about 2 GB free disk space (plus 1 GB for the COCO evaluation images).
A GPU is not needed: MediaPipe's Python package runs on the CPU.

**Windows (PowerShell)**
```powershell
git clone <repo-url> skeleton-pipeline
cd skeleton-pipeline
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-lock.txt     # if this file exists in the repo, otherwise:
pip install -r requirements.txt
python main.py download-models
python main.py selftest
```

**Linux / macOS**
```bash
git clone <repo-url> skeleton-pipeline && cd skeleton-pipeline
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt  ||  pip install -r requirements.txt
python main.py download-models
python main.py selftest
```

`selftest` needs no video. It checks the maths on a synthetic person with known angles, the video
decoder, HDF5 writing and the models. **Every line must say PASS.** (The rotation-tag check is
skipped if `ffmpeg` is not on PATH; that is fine.)

**First machine only** (the reference machine), after everything passes:
```bash
pip freeze > requirements-lock.txt
git add requirements-lock.txt models/models.lock.json
git commit -m "Lock package versions and model hashes"
```
From then on, every machine installs exactly the same versions, and `download-models` refuses
models whose SHA-256 differs from the committed lock. Same versions + same models + same
config = same numbers.

Optional second model (RTMW) for the agreement check:
`pip install -r requirements-optional.txt` (downloads its weights on first use).

### COCO-WholeBody (for the absolute-accuracy part of the evaluation)
```bash
python main.py download-coco
```
This downloads the COCO val2017 images (about 1 GB) into `~/skeleton_data/coco/val2017`. The label file
can't be downloaded automatically: get `coco_wholebody_val_v1.0.json` from
<https://github.com/jin-s13/COCO-WholeBody> (Google Drive link on that page) and save it as
`~/skeleton_data/coco/coco_wholebody_val_v1.0.json`. If it's missing, the pilot still runs and
the report marks that part "not run".

---

## 2. Pilot: test on 2–3 messy clips first

Put the clips in a folder, then run:
```bash
python main.py pilot path/to/clips
```
Results go to `~/skeleton_outputs/pilot/`:

```
pilot/
  report.md                 <- the evaluation report (all maths + numbers)
  plots/                    <- figures used by report.md
  pilot_summary.json        <- QC status per clip, calibration
  calibration.json          <- thresholds used by bulk QC
  coco_results.json         <- absolute accuracy (if COCO is installed)
  <clip>/
    <clip>.h5               <- extracted skeleton
    <clip>_overlay.mp4      <- skeleton rebuilt from the .h5, drawn on the video
    <clip>_overlay.json     <- overlay reconstruction + timing error
    results.json            <- every number for this clip
```

Runtime: each clip is processed 45 times (clean, repeat, 8 known transforms, 35
degradations). For a 10 s clip that's roughly 10–20 min with `evaluation.workers: 2`. COCO adds
about 5–10 min. Raise `workers` if you have the CPU cores and RAM (about 1.5 GB per worker).

To compare against an earlier run, write the new run to a new folder and pass the old one as baseline.
`report.md` then starts with a before/after table:
```bash
python main.py pilot path/to/clips --out ~/skeleton_outputs/pilot_v1_1 --baseline ~/skeleton_outputs/pilot
```

After the pilot, copy the two derived values from the last table of `report.md` into `config.yaml`:
`cleanup.max_gap_s` and `cleanup.smoothing_sigma_s`. Run `python main.py report` to rebuild the report
from saved results without re-running anything.

---

## 3. Bulk extraction

```bash
python main.py extract path/to/videos              # whole folder, recursive
python main.py extract path/to/video.mp4           # single file
python main.py extract path/to/videos --overlay    # also write an overlay MP4 per video
python main.py extract path/to/videos --workers 4 --out D:/skeletons
```
* Output mirrors the input folder structure: `videos/a/b.mp4` → `<out>/a/b.h5`.
* `manifest.csv` has one row per video: status, QC result, detection rates, rotation, fps,
  sha256, config hash and any error. `errors.log` has full tracebacks.
* Re-running the same command skips videos already done with the same file content (sha256)
  **and** the same settings (config hash). Change a setting and they're redone.
* An interrupted run can be restarted. Files are written to a temp name and renamed, so no
  half-written `.h5` files exist.
* QC (`pass` / `warn` / `fail`) uses the thresholds in `pilot/calibration.json` (from the pilot),
  plus a hard rule: fail if hands are found in fewer than 30 % of frames.

Overlay for any file later:
```bash
python main.py overlay out/clip.h5 clip.mp4 --out clip_overlay.mp4
python main.py inspect out/clip.h5          # list every array in a file
```

**Keep outputs out of OneDrive, Dropbox or Google Drive folders.** Sync clients lock and partially
upload files mid-write. The default output folder is `~/skeleton_outputs` (your home folder).
The program warns if the output path contains "OneDrive".

---

## 4. HDF5 layout (one file per video)

Missing values are `NaN`. "px" = pixels in the display frame (after rotation). "body units" =
shoulder widths. "world" = MediaPipe metric 3D estimate in metres (x right, y down, z away from
the camera; the body and each hand have their own origin).

| Path | Shape | Meaning |
|---|---|---|
| file attributes | — | `pipeline_version`, `config_hash`, `config` (all settings), `packages` (versions), `models` (SHA-256), `source_sha256`, `created_utc` |
| `/video` (attributes) | — | file name, sha256, codec, coded and display size, `rotation_cw`, `rotation_source`, `rotation_verified`, `sar`, fps (header and measured), frame count, duration |
| `/time/timestamps_s` | T | presentation time of each frame (float64, variable-frame-rate safe) |
| `/time/frame_index` | T | decoder frame index |
| `/quality/brightness, blur, motion_frac, enhanced` | T | mean luma, Laplacian variance, camera motion per frame (fraction of height), 1 if the detector got a brightened copy |
| `/raw/pose_px` | T×33×3 | detector output: x, y px and relative depth z (px scale) |
| `/raw/pose_world` | T×33×3 | metres |
| `/raw/pose_conf` | T×33 | visibility 0–1 |
| `/raw/{left,right}_hand_px` | T×21×3 | person's left or right hand |
| `/raw/{left,right}_hand_world` | T×21×3 | metres |
| `/raw/{left,right}_hand_conf` | T | hand presence score |
| `/raw/face_px` | T×20×3 | 20 named face points (eyes, brows, lips, nose tip, chin) |
| `/raw/*_mask` | T | 0 missing, 1 detected, 2 detected on the upscaled-crop pass |
| `/raw/{left,right}_hand_reject` | T | outlier flags (0 = kept): 1 duplicate of the other hand, 2 far from the pose wrist, 4 hand shape wrong, 8 sudden jump; flagged frames are left out of `/smoothed` (or interpolated if the gap is short) |
| `/smoothed/...` | same | outlier frames removed, gaps up to `max_gap_s` filled (mask 3), then zero-lag smoothed |
| `/norm/origin_px`, `/norm/scale_px` | T×2, T | mid-shoulder point and turn-corrected shoulder width |
| `/norm/{pose,left_hand,right_hand,face}` | T×N×3 | (xy_px − origin) / scale; independent of camera distance |
| `/features/angles` | T×48 | degrees, names in attribute `angle_names` (elbow bend, shoulder angle, wrist flexion, wrist deviation, forearm twist, 15 finger joints and 4 spreads per hand) |
| `/features/distances` | T×56 | names in `distance_names`: hand-internal (metres and per palm length) and hand-to-face/body (shoulder widths) |
| `/features/{left,right}_hand_quat` | T×4 | palm orientation quaternion (w, x, y, z) |
| `/features/{left,right}_twist_conf` | T | reliability of the twist angle (low when the palm is edge-on) |
| `/features/{left,right}_palm_normal` | T×3 | unit palm normal (world axes) |
| `/bones/{body,left_hand,right_hand}/root, length, theta, phi` | T×3, T×B | each bone as length and direction angles; rebuilds the 3D skeleton exactly; used for the avatar |
| `/qc` (attributes) | — | label-free quality metrics, plus `qc_status` and `qc_flags` in bulk runs |
| `/names` (attributes) | — | landmark names, MediaPipe face indices, mask codes |

Reading in Python:
```python
from skel.storage import read_h5
d = read_h5("clip.h5")
angles = d["features"]["angles"]; names = d["features"]["__attrs__"]["angle_names"]
right_hand = d["norm"]["right_hand"]          # T x 21 x 3, body units
```

### Avatar (later)
`skel/retarget.py` rebuilds a 3D skeleton with **fixed human bone lengths** (fractions of height)
from the stored bone directions, so limbs can never stretch. Twist is clamped to ±90°. Edit the
ratios at the top of the file to match the avatar's proportions.

---

## 5. How the evaluation works (summary; full maths in `report.md`)

| Ground truth | What it proves | Needs labels? |
|---|---|---|
| **G1** human labels (COCO-WholeBody) | absolute accuracy of hands, face, body: NME, PCK@0.05/0.1/0.2, AUC, 95 % CIs, for MediaPipe, MediaPipe without the crop pass, and RTMW | already labelled |
| **G2** known transforms (shift, rotate, scale, camera shake, mirror) | the true position after a known transform is known exactly, so the error gives a lower bound on true error | no |
| **G3** controlled degradation (dark, blur, motion blur, noise, low resolution, JPEG, low fps; 5 levels each) | how much each problem moves the landmarks; where it breaks | no |
| **G4** hidden detections | how much hiding g frames changes the final (filled + smoothed) output, and from it the longest safe gap | no |
| identity checks | file → pixels, angles + lengths → 3D, overlay timing, run-to-run determinism | no |

Plus label-free checks on every video (bone-length constancy, left/right symmetry, impossible
joint angles, jitter), calibrated against G3 so bulk QC thresholds are measured, not guessed.
The smoothing trade-off (jitter vs lag vs lost fast motion) is a measured curve with its knee.

---

## 6. Configuration

Everything lives in `config.yaml`. The most useful settings:

| Setting | Default | Meaning |
|---|---|---|
| `paths.output_dir` | `~/skeleton_outputs` | where everything is written |
| `video.detect_max_side` | 1280 | frames larger than this are downscaled for detection only (stored coordinates stay full resolution) |
| `quality.dark_threshold` | 80 | below this mean brightness the detector gets an enhanced copy |
| `detection.hand_crop_mode` | auto | second hand pass on an upscaled crop: `auto` = missed hands and hands smaller than `hand_small_px` (120 px), `always`, or `missing_only` |
| `cleanup.outliers.*` | see file | thresholds of the four hand outlier tests |
| `cleanup.max_gap_s` | 0.15 | longest gap filled (replace with the pilot's measured value) |
| `cleanup.smoothing_sigma_s` | 0.03 | smoothing strength (replace with the pilot's knee value) |
| `evaluation.workers` / `bulk.workers` | 2 | parallel processes |

Changing anything under `video`, `quality`, `detection` or `cleanup` changes the config hash
stored in each file, so old and new outputs are never mixed.

---

## 7. Reproducibility

* package versions: `requirements-lock.txt`
* model files: `models/models.lock.json` (SHA-256 checked on every download)
* settings: `config_hash` in every HDF5 file and in `manifest.csv`
* fresh models per video (no tracking state carried between videos, so processing order and
  parallelism don't change results)
* fixed random seeds for every synthetic transform, noise and bootstrap
* the pilot reruns one clip and reports `bitwise identical: yes/no`

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `missing model ...` | `python main.py download-models` |
| `... does not match the committed lock` | someone has different model files; don't delete the lock unless you're switching models on purpose |
| `pip` can't find `mediapipe==0.10.32` | check that Python is 3.12 **64-bit**; if it still fails, `pip install mediapipe` (the code only uses the Tasks API) and re-run `selftest` |
| `ModuleNotFoundError: av` | `pip install av`. Without it OpenCV decodes the video, but rotation can't be cross-checked |
| Windows "path too long" | enable long paths (`git config --system core.longpaths true` and the Windows LongPathsEnabled setting), or use a shorter output folder |
| Files locked or corrupted | the output folder is inside OneDrive or Dropbox; move it (`paths.output_dir`) |
| Overlay looks rotated | check `rotation_cw`, `rotation_verified` and `rotation_note` with `python main.py inspect file.h5` |
| Slow | raise `workers`, lower `video.detect_max_side` (e.g. 960) |

---

## 9. Code map

```
main.py               command line
config.yaml           all settings
skel/video_io.py      probing, rotation (read + verify), per-frame timestamps
skel/quality.py       brightness / blur / motion, detector-only enhancement
skel/extractor.py     MediaPipe pose + hands + face, wrist-matched hand sides, upscaled-crop hand pass
skel/outliers.py      hand outlier rejection (duplicate, far, shape, Hampel spike)
skel/postprocess.py   gap fill (PCHIP), zero-phase smoothing, One Euro, normalisation
skel/features.py      angles, palm frame, forearm twist, distances, bone encoding / decoding
skel/pipeline.py      video -> HDF5 tree
skel/storage.py       HDF5 read / write (atomic)
skel/overlay.py       rebuild from file + draw
skel/transforms.py    known transforms and degradations
skel/metrics.py       NME, PCK, AUC, jitter, lag, bone CV, Spearman, Youden, Kneedle ...
skel/evaluate.py      pilot evaluation, calibration, QC
skel/coco_eval.py     COCO-WholeBody accuracy (MediaPipe, MediaPipe without crop pass, RTMW)
skel/second_model.py  optional RTMW agreement
skel/report.py        report.md
skel/retarget.py      fixed-proportion avatar skeleton
skel/selftest.py      install check
skel/bulk.py          folder processing + manifest
```

## 10. Known limits (also stated in the report)

* Depth (z) from a single camera is estimated, so 3D angles are less exact than 2D positions.
  Distances in metres are estimates; the scale-free versions (per palm length, per shoulder
  width) are the ones to rely on.
* G2 can't detect an error that is itself consistent under the transform; G1 covers that.
* G3 measures change against the clean clip; its true error is within ± the clean error (G1).
* COCO-WholeBody contains still photos, not signing video. Hands are reported by box size so
  the sign-language-sized bin can be read on its own.
* Forearm twist is unreliable when the palm is edge-on to the camera (see `*_twist_conf`).

## Changelog

**1.1.0**
* Hands: second pass on an upscaled crop for hands that are missed or small (`hand_crop_mode: auto`).
* Hands: outlier rejection before filling and smoothing (duplicate, far from wrist, wrong shape, sudden jump);
  reasons stored per frame in `/raw/{side}_hand_reject`.
* Evaluation: gap test measured on the final output; calibration "bad" cutoff relative to each clip's own
  stability; G2 reports hands before and after rejection; COCO compares MediaPipe, MediaPipe without the crop
  pass, and RTMW.
* Report: summary table at the top, `--baseline` before/after table, plot file names without spaces.

**1.0.0** first release.
