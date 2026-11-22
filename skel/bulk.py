"""Bulk extraction: folder in -> HDF5 files out (same relative folder structure) + manifest.csv.

- one bad video never stops the batch (error is logged in the manifest);
- re-running skips videos whose HDF5 already has the same source sha256 AND config hash;
- each video gets fresh models (no tracking state leaks between videos -> order-independent);
- writes are atomic (temp file + rename), so an interrupted run never leaves half files.
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

from .config import load_config

FIELDS = ["relative_path", "status", "qc_status", "qc_flags", "n_frames", "duration_s", "width", "height",
          "rotation_cw", "rotation_verified", "fps_measured", "any_hand_detection_rate", "pose_detection_rate",
          "face_detection_rate", "seconds", "sha256", "config_hash", "h5_path", "error"]


def list_videos(root: Path, exts):
    exts = {e.lower() for e in exts}
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _out_path(video: Path, in_root: Path, out_root: Path) -> Path:
    rel = video.name if in_root.is_file() else video.relative_to(in_root)
    return (out_root / rel).with_suffix(".h5")


def _work(args):
    video, in_root, out_root, cfg_path, calib_path, overlay = args
    video, in_root, out_root = Path(video), Path(in_root), Path(out_root)
    cfg = load_config(cfg_path)
    row = {"relative_path": str(video.relative_to(in_root) if in_root.is_dir() else video.name),
           "config_hash": cfg.extraction_hash()}
    t0 = time.time()
    try:
        from .evaluate import qc_status
        from .pipeline import file_attrs, process_video
        from .storage import read_attrs, write_h5
        from .video_io import file_sha256, probe
        out = _out_path(video, in_root, out_root)
        row["h5_path"] = str(out)
        if cfg["bulk"]["skip_existing"] and out.exists():
            try:
                at = read_attrs(out)
                sha = file_sha256(video)
                if at.get("source_sha256") == sha and at.get("config_hash") == row["config_hash"]:
                    row.update(status="skipped_existing", sha256=sha, seconds=round(time.time() - t0, 2))
                    return row
            except Exception:
                pass
        meta = probe(video)
        tree, meta = process_video(video, cfg, meta=meta)
        calib = json.loads(Path(calib_path).read_text(encoding="utf-8")) if calib_path and Path(calib_path).exists() else None
        qc = tree["qc"]["__attrs__"]
        st = qc_status(qc, calib, cfg)
        tree["qc"]["__attrs__"] = {**qc, "qc_status": st["status"], "qc_flags": st["flags"],
                                   "qc_calibrated": st["calibrated"]}
        write_h5(out, tree, file_attrs(cfg, meta))
        if overlay:
            from .overlay import render
            render(out, video, out.with_name(out.stem + "_overlay.mp4"), cfg)
        row.update(status="ok", qc_status=st["status"], qc_flags=";".join(st["flags"]),
                   n_frames=meta["n_frames"], duration_s=round(meta["duration_s"], 3),
                   width=meta["display_width"], height=meta["display_height"], rotation_cw=meta["rotation_cw"],
                   rotation_verified=meta["rotation_verified"], fps_measured=round(meta["fps_measured"], 3),
                   any_hand_detection_rate=round(qc["any_hand_detection_rate"], 4),
                   pose_detection_rate=round(qc["pose_detection_rate"], 4),
                   face_detection_rate=round(qc["face_detection_rate"], 4), sha256=meta["sha256"])
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500])
        (Path(out_root) / "errors.log").open("a", encoding="utf-8").write(
            f"\n=== {video}\n{traceback.format_exc()}")
    row["seconds"] = round(time.time() - t0, 2)
    return row


def run_bulk(input_path, cfg, out_root: Path | None = None, calibration: Path | None = None,
             workers: int | None = None, overlay: bool | None = None, log=print):
    in_root = Path(input_path).resolve()
    out_root = Path(out_root or cfg.path("output_dir") / "bulk").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if "onedrive" in str(out_root).lower():
        log("WARNING: output folder is inside OneDrive - sync can lock/corrupt HDF5 files. "
            "Set paths.output_dir outside OneDrive.")
    calibration = calibration or cfg.path("output_dir") / "pilot" / "calibration.json"
    calib_path = str(calibration) if Path(calibration).exists() else ""
    if not calib_path:
        log("note: no calibration.json found - QC uses only the hard detection-rate rule. Run the pilot first.")
    videos = list_videos(in_root, cfg["video"]["extensions"])
    log(f"{len(videos)} videos found under {in_root}")
    workers = int(workers or cfg["bulk"]["workers"])
    overlay = cfg["bulk"]["write_overlay"] if overlay is None else overlay
    jobs = [(str(v), str(in_root), str(out_root), cfg["_path"], calib_path, overlay) for v in videos]
    manifest = out_root / "manifest.csv"
    rows = {}
    if manifest.exists():
        with open(manifest, newline="", encoding="utf-8") as fh:
            rows = {r["relative_path"]: r for r in csv.DictReader(fh)}

    def save():
        tmp = manifest.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for k in sorted(rows):
                w.writerow({f: rows[k].get(f, "") for f in FIELDS})
        tmp.replace(manifest)

    def results():
        if workers <= 1:
            yield from map(_work, jobs)
        else:
            with mp.get_context("spawn").Pool(workers) as pool:
                yield from pool.imap_unordered(_work, jobs)

    it = results()
    counts = {}
    for n, row in enumerate(it, 1):
        prev = rows.get(row["relative_path"])
        if row["status"] == "skipped_existing" and prev:
            row = {**prev, "status": "skipped_existing", "seconds": row["seconds"]}
        rows[row["relative_path"]] = row
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        log(f"[{n}/{len(jobs)}] {row['status']:<16} {row.get('qc_status', '') or '':<5} "
            f"{row['relative_path']} ({row['seconds']}s)")
        save()
    save()
    log(f"done: {counts}. Manifest: {manifest}")
    return counts
