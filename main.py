"""Skeleton extraction / reconstruction / evaluation - command line entry point.

    python main.py selftest
    python main.py download-models
    python main.py download-coco
    python main.py pilot  <folder with 2-3 clips>
    python main.py extract <video file or folder>
    python main.py overlay <file.h5> <video>
    python main.py report
    python main.py inspect <file.h5>

Run `python main.py <command> -h` for options.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _cfg(args):
    from skel.config import load_config
    return load_config(args.config)


def cmd_selftest(args):
    from skel.selftest import run
    return run(_cfg(args))


def cmd_download_models(args):
    from skel.models import download_models
    download_models(_cfg(args).path("models_dir"), force=args.force)


def cmd_download_coco(args):
    from skel.models import download_coco
    cfg = _cfg(args)
    download_coco(cfg.path("coco_images_dir"), cfg.path("coco_annotations"))


def _require_models(cfg):
    from skel.models import check_models
    probs = check_models(cfg.path("models_dir"))
    if probs:
        sys.exit("\n".join(probs))


def _clips(folder, cfg):
    from skel.bulk import list_videos
    clips = list_videos(Path(folder).resolve(), cfg["video"]["extensions"])
    if not clips:
        sys.exit(f"no videos found in {folder}")
    return clips


def cmd_pilot(args):
    cfg = _cfg(args)
    _require_models(cfg)
    from skel.evaluate import evaluate_clips
    from skel.report import build_report
    out = Path(args.out).expanduser().resolve() if args.out else cfg.path("output_dir") / "pilot"
    clips = _clips(args.clips, cfg)
    t0 = time.time()
    evaluate_clips(clips, cfg, out)
    if not args.skip_coco:
        try:
            from skel.coco_eval import coco_eval
            coco_eval(cfg, out)
        except FileNotFoundError as exc:
            print(f"\nG1 (COCO-WholeBody) skipped: {exc}")
    rp = build_report(out, baseline_dir=Path(args.baseline).expanduser().resolve() if args.baseline else None)
    print(f"\nPilot finished in {(time.time() - t0) / 60:.1f} min.\nReport: {rp}")


def cmd_evaluate(args):
    cfg = _cfg(args)
    _require_models(cfg)
    from skel.evaluate import evaluate_clips
    out = Path(args.out).resolve() if args.out else cfg.path("output_dir") / "pilot"
    evaluate_clips(_clips(args.clips, cfg), cfg, out)


def cmd_coco(args):
    cfg = _cfg(args)
    _require_models(cfg)
    from skel.coco_eval import coco_eval
    out = Path(args.out).resolve() if args.out else cfg.path("output_dir") / "pilot"
    r = coco_eval(cfg, out)
    for k in ("hands", "face", "body"):
        if k in r:
            print(f"{k:>6}: NME {r[k]['nme']:.4f}  PCK@0.1 {r[k]['pck@0.1']:.3f}  det {r[k]['detection_rate']:.3f}")


def cmd_report(args):
    cfg = _cfg(args)
    from skel.report import build_report
    pilot = Path(args.pilot).expanduser().resolve() if args.pilot else cfg.path("output_dir") / "pilot"
    bl = Path(args.baseline).expanduser().resolve() if args.baseline else None
    print(f"Report: {build_report(pilot, baseline_dir=bl)}")


def cmd_extract(args):
    cfg = _cfg(args)
    _require_models(cfg)
    from skel.bulk import run_bulk
    run_bulk(args.input, cfg, Path(args.out) if args.out else None,
             Path(args.calibration) if args.calibration else None, args.workers, args.overlay or None)


def cmd_overlay(args):
    cfg = _cfg(args)
    from skel.overlay import render
    out = Path(args.out) if args.out else Path(args.h5).with_name(Path(args.h5).stem + "_overlay.mp4")
    r = render(Path(args.h5), Path(args.video), out, cfg)
    print(f"overlay: {r['overlay']}\n  frames matched {r['frames_matched']}/{r['frames_in_video']}, "
          f"timestamp err max {r['timestamp_abs_err_ms_max']:.3f} ms, "
          f"reconstruction err max {r['reconstruction_err_px_max']:.2e} px")


def cmd_inspect(args):
    import h5py
    with h5py.File(args.h5, "r") as fh:
        print(f"{args.h5}")
        for k in ("pipeline_version", "config_hash", "source_sha256", "created_utc"):
            print(f"  {k}: {fh.attrs.get(k)}")

        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  /{name:<40} {str(obj.shape):<16} {obj.dtype}")
        fh.visititems(show)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to config.yaml (default: next to main.py)")
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest", help="verify the install (no video needed)").set_defaults(fn=cmd_selftest)
    a = sp.add_parser("download-models", help="download MediaPipe models and check the lock file")
    a.add_argument("--force", action="store_true")
    a.set_defaults(fn=cmd_download_models)
    sp.add_parser("download-coco", help="download COCO val2017 images (annotation JSON: see README)") \
        .set_defaults(fn=cmd_download_coco)
    a = sp.add_parser("pilot", help="full evaluation on 2-3 clips + COCO-WholeBody + report.md")
    a.add_argument("clips")
    a.add_argument("--out")
    a.add_argument("--skip-coco", action="store_true")
    a.add_argument("--baseline", help="an earlier pilot folder: adds a before/after table to report.md")
    a.set_defaults(fn=cmd_pilot)
    a = sp.add_parser("evaluate", help="clip evaluation only (G2-G4, round trips)")
    a.add_argument("clips")
    a.add_argument("--out")
    a.set_defaults(fn=cmd_evaluate)
    a = sp.add_parser("coco-eval", help="absolute accuracy on COCO-WholeBody (G1)")
    a.add_argument("--out")
    a.set_defaults(fn=cmd_coco)
    a = sp.add_parser("report", help="rebuild report.md from existing results")
    a.add_argument("--pilot")
    a.add_argument("--baseline", help="an earlier pilot folder: adds a before/after table")
    a.set_defaults(fn=cmd_report)
    a = sp.add_parser("extract", help="extract one video or a whole folder (bulk) to HDF5")
    a.add_argument("input")
    a.add_argument("--out")
    a.add_argument("--workers", type=int)
    a.add_argument("--calibration", help="calibration.json from the pilot (default: <output_dir>/pilot)")
    a.add_argument("--overlay", action="store_true", help="also write an overlay MP4 per video")
    a.set_defaults(fn=cmd_extract)
    a = sp.add_parser("overlay", help="rebuild skeleton from an HDF5 file and draw it on the video")
    a.add_argument("h5")
    a.add_argument("video")
    a.add_argument("--out")
    a.set_defaults(fn=cmd_overlay)
    a = sp.add_parser("inspect", help="print the layout of an HDF5 file")
    a.add_argument("h5")
    a.set_defaults(fn=cmd_inspect)
    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
