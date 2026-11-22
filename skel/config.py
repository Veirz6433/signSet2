"""Config loading and hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Sections whose values change extracted numbers. Their hash is stored in every
# HDF5 file so outputs made with different settings are never mixed up.
EXTRACTION_SECTIONS = ("pipeline_version", "video", "quality", "detection", "cleanup")


class Config(dict):
    """dict with attribute access and resolved paths."""

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    @property
    def root(self) -> Path:
        return Path(self["_root"])

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key]).expanduser()
        return p if p.is_absolute() else (self.root / p).resolve()

    def extraction_dict(self) -> dict:
        return {k: self[k] for k in EXTRACTION_SECTIONS}

    def extraction_hash(self) -> str:
        blob = json.dumps(self.extraction_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    data["_root"] = str(path.resolve().parent)
    data["_path"] = str(path.resolve())
    return Config(data)
