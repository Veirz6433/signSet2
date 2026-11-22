"""Static figures for report.md (matplotlib, Agg)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"hands": "#2a78d6", "body": "#eb6834", "face": "#1baf7a"}
OFFLINE, LIVE = "#2a78d6", "#eb6834"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def degradation_plot(res, out: Path):
    kinds = list(res["degradations"].keys())
    if not kinds:
        return None
    n = len(kinds)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), facecolor=SURFACE, squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, kind in zip(axes.ravel(), kinds):
        _style(ax)
        lv = [r["level"] for r in res["degradations"][kind]]
        for part, col in SERIES.items():
            y = [r["vs_clean"].get(part, {}).get("pck@0.1") for r in res["degradations"][kind]]
            y = [v if v is not None else float("nan") for v in y]
            ax.plot(lv, y, color=col, linewidth=2, marker="o", markersize=4, label=part)
        ax.set_title(kind.replace("_", " "), color=INK, fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("PCK@0.1 vs clean", color=INK2, fontsize=8)
    axes.ravel()[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out


def smoothing_plot(sweep, out: Path):
    fig, ax = plt.subplots(figsize=(5.4, 3.6), facecolor=SURFACE)
    _style(ax)
    for kind, col, name in (("offline", OFFLINE, "zero-phase Gaussian (offline)"),
                            ("live", LIVE, "One Euro (live, causal)")):
        rows = [r for r in sweep["rows"] if r["kind"] == kind]
        x = [100 * (r["peak_loss"] or 0) for r in rows]
        y = [100 * (r["jitter_reduction"] or 0) for r in rows]
        ax.plot(x, y, color=col, linewidth=2, marker="o", markersize=5, label=name)
        for r, xi, yi in zip(rows, x, y):
            lab = f"{r['param']}s" if kind == "offline" else f"{r['param']}Hz"
            ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(-6, 7) if kind == "offline" else (5, -11),
                        fontsize=7, color=INK2, ha="right" if kind == "offline" else "left")
    ax.set_xlabel("peak-speed loss (%)", color=INK2, fontsize=8)
    ax.set_ylabel("jitter reduction (%)", color=INK2, fontsize=8)
    ax.set_title("Smoothing trade-off (hands)", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out


def gap_plot(gap, required, out: Path):
    rows = [r for r in gap["curve"] if r.get("n")]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 3.2), facecolor=SURFACE)
    _style(ax)
    ax.plot([r["gap_frames"] for r in rows], [r["pck"] for r in rows], color=SERIES["hands"], linewidth=2,
            marker="o", markersize=5, label="hands")
    ax.axhline(required, color=INK2, linewidth=1, linestyle="--")
    ax.annotate(f"required {required}", (rows[-1]["gap_frames"], required), textcoords="offset points",
                xytext=(-60, 4), fontsize=7, color=INK2)
    ax.set_xlabel("gap length (frames)", color=INK2, fontsize=8)
    ax.set_ylabel("PCK@0.1 of filled points", color=INK2, fontsize=8)
    ax.set_title("Gap-fill test", color=INK, fontsize=9)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out
