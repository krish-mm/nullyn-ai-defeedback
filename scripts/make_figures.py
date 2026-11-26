#!/usr/bin/env python3
"""Render README result charts from checked simulated and benchmark numbers.

Sources (no invented values):
  * Closed-loop steady protocol in the research summary.json generated
    26 November 2026, 07:46 CT (575 held-out scenarios; 1,725 matched runs
    per processor; 690 static-mic runs; 1,035 walking-performer runs).
  * Offline CPU timings already reported in the README (scripts/benchmark_cpu.py
    on the development box, 26 November 2026).

If NULYN_SUMMARY_JSON or the research-tree path points at that summary.json,
the script cross-checks the embedded numbers and refuses to plot on a mismatch.
It never invents a figure that is not in those sources.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"

# Brand
LIME = "#B6FF00"
RED = "#FF1A1A"
INK = "#0A0A0A"
GRAY = "#8A8A8A"
LIGHT = "#E6E6E6"
GRID = "#2A2A2A"

# --- SIMULATED figures from summary.json › steady (07:46 CT, 26 Nov 2026) ---
# Values are the percentages / dB already published in the README, which were
# rounded from summary.json the same way as whitepaper v0.3 (one decimal).
STEADY = {
    "source": "summary.json › steady, generated 2026-11-26 07:46 CST",
    "label": "SIMULATED. pyroomacoustics image-source shoebox rooms, 16 kHz, 4 s runs.",
    "scenarios": 575,
    "runs_matched": 1725,
    "runs_static": 690,
    "runs_walking": 1035,
    "methods": ["No processing", "Classic notch", "Nullyn AI v0\n(notch)", "Nullyn AI\nclosed loop"],
    "howl_all": [46.6, 32.2, 25.6, 22.8],
    "howl_static": [48.6, 34.5, 26.5, 0.7],
    "howl_walking": [45.3, 30.7, 25.0, 37.6],
    "headroom50_all": [1.0, 3.5, 5.0, 4.0],
    "headroom50_static": [1.0, 2.5, 4.5, 15.5],
    "headroom50_walking": [2.5, 4.5, 5.0, 2.5],
}

# Exact fractions in summary.json, used only for a cross-check when the file
# is available (rounded values above must match to one decimal).
STEADY_EXACT = {
    "howl_all": [0.46608695652173915, 0.32231884057971016, 0.25623188405797104, 0.2284057971014493],
    "howl_static": [0.4855072463768116, 0.34492753623188405, 0.26521739130434785, 0.007246376811594203],
    "howl_walking": [0.45314009661835747, 0.3072463768115942, 0.25024154589371983, 0.37584541062801935],
    "headroom50_all": [1.0, 3.5, 5.0, 4.0],
    "headroom50_static": [1.0, 2.5, 4.5, 15.5],
    "headroom50_walking": [2.5, 4.5, 5.0, 2.5],
}

# README "Minimum laptop requirements" table, 26 Nov 2026.
# The 48 kHz closed-loop row is a range of two runs; both ends are plotted.
CPU = {
    "source": "README Minimum laptop requirements; scripts/benchmark_cpu.py, 26 Nov 2026",
    "label": "Measured offline CPU time, not analog latency, not a venue result.",
    "rows": [
        {
            "name": "Notch v0\n48 kHz / 128",
            "budget_ms": 2.667,
            "median_ms": (0.005, 0.005),
            "p95_ms": (1.242, 1.242),
        },
        {
            "name": "Closed loop\n48 kHz / 192",
            "budget_ms": 4.000,
            "median_ms": (0.356, 0.394),
            "p95_ms": (3.903, 4.405),
        },
        {
            "name": "Closed loop\n16 kHz / 64",
            "budget_ms": 4.000,
            "median_ms": (0.242, 0.242),
            "p95_ms": (2.252, 2.252),
        },
    ],
}


def _apply_style():
    plt.rcParams.update({
        "figure.facecolor": INK,
        "axes.facecolor": INK,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": LIGHT,
        "axes.titlecolor": LIGHT,
        "xtick.color": LIGHT,
        "ytick.color": LIGHT,
        "text.color": LIGHT,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "savefig.facecolor": INK,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
    })


def _banner(ax, title, source_line):
    ax.set_title(title, loc="left", pad=12, fontweight="bold", color=LIGHT)
    ax.text(
        0.0, 1.01, source_line,
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=8, color=GRAY, clip_on=False,
    )


def _footnote(fig, text):
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7.5, color=GRAY)


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(length=0)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def grouped_bars(ax, labels, series, colors, ylabel):
    n = len(labels)
    k = len(series)
    x = np.arange(n)
    width = 0.34 if k == 2 else 0.7 / k
    offsets = (np.arange(k) - (k - 1) / 2) * width
    bars = []
    for off, (name, values, color) in zip(offsets, series):
        b = ax.bar(x + off, values, width=width * 0.92, color=color, label=name, zorder=3)
        bars.append(b)
        for rect, val in zip(b, values):
            ax.annotate(
                f"{val:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, color=LIGHT,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.55, n - 0.45)
    _style_axis(ax)
    return bars


def _maybe_crosscheck():
    candidates = []
    env = os.environ.get("NULLYN_SUMMARY_JSON")
    if env:
        candidates.append(Path(env))
    candidates.append(Path("/workspace/nullyn/results/closedloop/summary.json"))
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return
    s = json.loads(path.read_text())
    order = ["none", "classic", "nullyn_v0", "closedloop"]
    keys = [
        ("howl_all", lambda n: s["steady"]["processors"][n]["matched"]["howl_rate"]),
        ("howl_static", lambda n: s["steady"]["processors"][n]["static_mic"]["howl_rate_matched"]),
        ("howl_walking", lambda n: s["steady"]["processors"][n]["walking_performer"]["howl_rate_matched"]),
        ("headroom50_all", lambda n: s["steady"]["processors"][n]["headroom_db_at_le_50pct"]),
        ("headroom50_static", lambda n: s["steady"]["processors"][n]["static_mic"]["headroom_db_at_le_50pct"]),
        ("headroom50_walking", lambda n: s["steady"]["processors"][n]["walking_performer"]["headroom_db_at_le_50pct"]),
    ]
    for name, getter in keys:
        got = [getter(n) for n in order]
        exp = STEADY_EXACT[name]
        if any(abs(a - b) > 1e-9 for a, b in zip(got, exp)):
            raise SystemExit(f"summary.json mismatch for {name}: {got} != {exp}")
    gen = s.get("generated", "")
    if "2026-11-26 07:46" not in gen:
        raise SystemExit(f"unexpected summary.json generated timestamp: {gen!r}")
    print(f"cross-checked embedded numbers against {path}")


def fig_howl_static_walking():
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    grouped_bars(
        ax,
        STEADY["methods"],
        [
            ("Static mic  (690 runs)", STEADY["howl_static"], LIME),
            ("Walking performer  (1,035 runs)", STEADY["howl_walking"], RED),
        ],
        (LIME, RED),
        "Howl rate (%)",
    )
    ax.set_ylim(0, 62)
    ax.legend(frameon=False, loc="upper right", labelcolor=LIGHT)
    _banner(
        ax,
        "Howl rate by method  ·  static vs walking  ·  SIMULATED",
        STEADY["source"] + "  ·  575 held-out scenarios",
    )
    _footnote(
        fig,
        "SIMULATED. Not measured in a real venue. Closed loop: 0.7% static, 37.6% walking. "
        "Classic notch walking is 30.7%; its 34.5% rate is static, not walking.",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    path = OUT / "howl-static-vs-walking.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_headroom():
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    grouped_bars(
        ax,
        STEADY["methods"],
        [
            ("Static mic  (690 runs)", STEADY["headroom50_static"], LIME),
            ("Walking performer  (1,035 runs)", STEADY["headroom50_walking"], RED),
        ],
        (LIME, RED),
        "Fader headroom at ≤50% howl risk (dB)",
    )
    ax.set_ylim(0, 19)
    ax.legend(frameon=False, loc="upper left", labelcolor=LIGHT)
    _banner(
        ax,
        "Fader headroom  ·  static vs walking  ·  SIMULATED",
        STEADY["source"] + "  ·  575 held-out scenarios",
    )
    _footnote(
        fig,
        "SIMULATED. Not measured in a real venue. Closed-loop static headroom is +15.5 dB; "
        "walking headroom is +2.5 dB. Classic notch walking headroom is +4.5 dB.",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    path = OUT / "headroom-static-vs-walking.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_howl_all():
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    labels = STEADY["methods"]
    values = STEADY["howl_all"]
    colors = [GRAY, LIGHT, LIME, RED]
    bars = ax.bar(labels, values, color=colors, width=0.64, zorder=3)
    for rect, val in zip(bars, values):
        ax.annotate(
            f"{val:g}%",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=10, color=LIGHT,
        )
    ax.set_ylabel("Howl rate (%)")
    ax.set_ylim(0, 58)
    _style_axis(ax)
    _banner(
        ax,
        "Overall howl rate  ·  matched runs  ·  SIMULATED",
        STEADY["source"] + "  ·  1,725 matched runs / processor (static and walking combined)",
    )
    _footnote(
        fig,
        "SIMULATED. Not measured in a real venue. Combined figure; see the static-vs-walking "
        "chart for the split (closed loop 0.7% static vs 37.6% walking).",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    path = OUT / "howl-rate-all.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_cpu():
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    rows = CPU["rows"]
    x = np.arange(len(rows))
    width = 0.34
    # For ranged rows, draw the interval as a bar from low to high plus a midpoint marker.
    med_low = [r["median_ms"][0] for r in rows]
    med_high = [r["median_ms"][1] for r in rows]
    p95_low = [r["p95_ms"][0] for r in rows]
    p95_high = [r["p95_ms"][1] for r in rows]
    budgets = [r["budget_ms"] for r in rows]

    def interval_bars(xs, lows, highs, color, label, offset):
        centers = [(a + b) / 2 for a, b in zip(lows, highs)]
        yerr = np.array([
            [c - lo for c, lo in zip(centers, lows)],
            [hi - c for c, hi in zip(centers, highs)],
        ])
        # A zero-height error bar still places a marker; use a thin bar for the point estimate
        # and error bars only when the two ends differ.
        bars = ax.bar(
            xs + offset, centers, width=width, color=color, label=label, zorder=3,
            yerr=yerr, capsize=5, ecolor=LIGHT, error_kw={"linewidth": 1.1, "zorder": 4},
        )
        for xi, lo, hi, c in zip(xs + offset, lows, highs, centers):
            if abs(hi - lo) < 1e-9:
                ax.annotate(
                    f"{c:.3f}",
                    xy=(xi, c), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=LIGHT,
                )
            else:
                ax.annotate(
                    f"{lo:.3f}–{hi:.3f}",
                    xy=(xi, hi), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color=LIGHT,
                )
        return bars

    interval_bars(x, med_low, med_high, LIME, "Median", -width / 2)
    interval_bars(x, p95_low, p95_high, RED, "p95", width / 2)
    # Budget as a per-category horizontal tick mark
    for xi, b in zip(x, budgets):
        ax.plot(
            [xi - 0.42, xi + 0.42], [b, b],
            color=GRAY, lw=1.2, ls="--", zorder=5, solid_capstyle="butt",
        )
        ax.annotate(
            f"budget {b:g} ms",
            xy=(xi + 0.42, b), xytext=(4, 0), textcoords="offset points",
            ha="left", va="center", fontsize=7.5, color=GRAY,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in rows])
    ax.set_ylabel("Callback / block CPU time (ms)")
    ax.set_ylim(0, 5.6)
    _style_axis(ax)
    ax.legend(frameon=False, loc="upper left", labelcolor=LIGHT)
    _banner(
        ax,
        "Per-block CPU time  ·  measured on the development box",
        CPU["source"],
    )
    _footnote(
        fig,
        CPU["label"] + " Intel Xeon Processor, 8 logical processors, 2.40 GHz, Linux VM. "
        "48 kHz closed-loop values are the range of two runs.",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    path = OUT / "cpu-per-block.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _apply_style()
    _maybe_crosscheck()
    paths = [
        fig_howl_static_walking(),
        fig_headroom(),
        fig_howl_all(),
        fig_cpu(),
    ]
    for p in paths:
        print(f"wrote {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
