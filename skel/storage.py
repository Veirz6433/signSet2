"""HDF5 storage: one file per video, float32 + gzip/shuffle, written atomically (temp + rename).
Missing values are NaN; every group/array is documented in README.md ("HDF5 layout")."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

FORMAT_VERSION = "1.0"
KEEP_F64 = {"timestamps_s", "origin_px", "scale_px", "root"}   # stored in float64


def _h5():
    import h5py
    return h5py


def _attr(v):
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if v is None:
        return "null"
    if isinstance(v, (np.generic,)):
        return v.item()
    return v


def write_h5(path, tree: dict, attrs: dict | None = None):
    """tree: nested dict; leaves are numpy arrays, or {"__attrs__": {...}} entries for groups."""
    h5py = _h5()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    def put(group, d):
        for k, v in d.items():
            if k == "__attrs__":
                for ak, av_ in v.items():
                    group.attrs[ak] = _attr(av_)
            elif isinstance(v, dict):
                put(group.require_group(k), v)
            else:
                arr = np.asarray(v)
                if arr.dtype == np.float64 and k not in KEEP_F64:
                    arr = arr.astype(np.float32)
                kw = dict(compression="gzip", compression_opts=4, shuffle=True) if arr.size > 16 else {}
                group.create_dataset(k, data=arr, **kw)

    with h5py.File(tmp, "w") as fh:
        fh.attrs["format_version"] = FORMAT_VERSION
        for k, v in (attrs or {}).items():
            fh.attrs[k] = _attr(v)
        put(fh, tree)
    os.replace(tmp, path)


def read_h5(path) -> dict:
    h5py = _h5()

    def get(g):
        out = {"__attrs__": {k: _decode(v) for k, v in g.attrs.items()}}
        for k, v in g.items():
            out[k] = get(v) if isinstance(v, h5py.Group) else v[()]
        return out

    with h5py.File(path, "r") as fh:
        return get(fh)


def read_attrs(path) -> dict:
    h5py = _h5()
    with h5py.File(path, "r") as fh:
        return {k: _decode(v) for k, v in fh.attrs.items()}


def _decode(v):
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str) and (v[:1] in ("[", "{") or v == "null"):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v.item() if isinstance(v, np.generic) else v
