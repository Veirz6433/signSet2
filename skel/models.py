"""Model download + integrity lock (same model bytes on every machine)."""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

MODEL_URLS = {
    "pose_landmarker_heavy.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}
COCO_VAL_IMAGES = "http://images.cocodataset.org/zips/val2017.zip"
LOCK_NAME = "models.lock.json"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _download(url, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 20)
    tmp.replace(dest)


def download_models(models_dir: Path, force=False):
    models_dir.mkdir(parents=True, exist_ok=True)
    lock_path = models_dir / LOCK_NAME
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    for name, url in MODEL_URLS.items():
        dest = models_dir / name
        if force or not dest.exists():
            _download(url, dest)
        digest = sha256(dest)
        if name in lock and lock[name] != digest:
            raise RuntimeError(
                f"{name}: sha256 {digest[:12]} does not match the committed lock {lock[name][:12]}.\n"
                f"Results would not be identical to the reference machine. Delete {lock_path} only if you\n"
                f"intentionally want to change models (and tell everyone).")
        lock[name] = digest
        print(f"  {name}: sha256 {digest[:16]} OK")
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"Model lock written: {lock_path}  (commit this file)")


def model_hashes(models_dir: Path) -> dict:
    out = {}
    for name in MODEL_URLS:
        p = models_dir / name
        out[name] = sha256(p) if p.exists() else None
    return out


def check_models(models_dir: Path) -> list[str]:
    """Return list of problems (empty if fine)."""
    problems = []
    lock_path = models_dir / LOCK_NAME
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    for name in MODEL_URLS:
        p = models_dir / name
        if not p.exists():
            problems.append(f"missing model {p} (run: python main.py download-models)")
        elif name in lock and sha256(p) != lock[name]:
            problems.append(f"{name} differs from models.lock.json")
    return problems


def download_coco(images_dir: Path, annotations: Path):
    if images_dir.exists() and any(images_dir.glob("*.jpg")):
        print(f"COCO val2017 images already present: {images_dir}")
    else:
        zpath = images_dir.parent / "val2017.zip"
        _download(COCO_VAL_IMAGES, zpath)
        print("  extracting ...")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(images_dir.parent)
        zpath.unlink()
    if not annotations.exists():
        print("\nCOCO-WholeBody annotation file is NOT downloadable automatically (it is hosted on\n"
              "Google Drive). Open https://github.com/jin-s13/COCO-WholeBody , download\n"
              "'coco_wholebody_val_v1.0.json' and save it as:\n"
              f"  {annotations}\n")
    else:
        print(f"COCO-WholeBody annotations present: {annotations}")
