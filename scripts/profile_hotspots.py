#!/usr/bin/env python3
"""Profile the real-time path with cProfile (and per-stage timers).

    python scripts/profile_hotspots.py [--fs 48000] [--seconds 6] [--top 30]

Runs the same seeded closed-loop scenarios as ``bench_latency.py`` and
prints the top functions by cumulative and internal time, plus a per-stage
breakdown (AFC filter/adapt, LPC/LTP update, side-chain analysis + features,
notch bank) measured with lightweight wrappers.
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bench_latency as bl  # noqa: E402
from nullyn import features, filters, notch  # noqa: E402
from nullyn.closedloop import afc  # noqa: E402

STAGES = [
    (afc.ClosedLoopAFC, "_yhat", "afc.yhat (filter, P partitions)"),
    (afc.ClosedLoopAFC, "_adapt", "afc.adapt (PB-FDAF update)"),
    (afc.ClosedLoopAFC, "_update_lpc", "afc.lpc+ltp (PEM prewhitening model)"),
    (afc.ClosedLoopAFC, "push_ref", "afc.push_ref (ref FFT + prewhiten)"),
    (features.SideChain, "analyze", "sidechain.analyze (FFT + peaks + features)"),
    (features.SideChain, "_features", "  of which sidechain._features (per candidate)"),
    (notch.NullynV0, "decide", "detector.decide (MLP + persistence)"),
    (filters.NotchBank, "process", "notchbank.process (sosfilt, K biquads)"),
    (filters.NotchBank, "tick", "notchbank.tick"),
]


def instrument():
    acc = defaultdict(lambda: [0, 0.0, 0.0])
    for cls, name, label in STAGES:
        orig = getattr(cls, name)

        def wrap(*a, _o=orig, _l=label, **k):
            t0 = time.perf_counter_ns()
            r = _o(*a, **k)
            dt = (time.perf_counter_ns() - t0) * 1e-6
            s = acc[_l]
            s[0] += 1
            s[1] += dt
            s[2] = max(s[2], dt)
            return r
        setattr(cls, name, wrap)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", type=int, default=48000)
    ap.add_argument("--mode", default="closedloop")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--cprofile", action="store_true")
    a = ap.parse_args()
    if a.cprofile:
        pr = cProfile.Profile()
        pr.enable()
        for n in bl.SCENARIOS:
            bl.run_scenario(a.fs, a.mode, "nullyn_v0", n, a.seconds)
        pr.disable()
        st = pstats.Stats(pr).strip_dirs()
        st.sort_stats("tottime").print_stats(a.top)
        return
    acc = instrument()
    nblocks = 0
    total = 0.0
    for n in bl.SCENARIOS:
        r = bl.run_scenario(a.fs, a.mode, "nullyn_v0", n, a.seconds, warmup_s=0.0)
        nblocks += len(r["times"])
        total += r["times"].sum()
    print(f"{nblocks} blocks, total process() time {total:.1f} ms, mean {total / nblocks:.3f} ms/block")
    print(f"{'stage':<48}{'calls':>8}{'total ms':>10}{'%':>7}{'ms/call':>9}{'max ms':>9}{'ms/block':>10}")
    for _, _, label in STAGES:
        c, t, m = acc.get(label, [0, 0.0, 0.0])
        if c:
            print(f"{label:<48}{c:>8}{t:>10.1f}{100 * t / total:>7.1f}{t / c:>9.3f}{m:>9.3f}{t / nblocks:>10.3f}")


if __name__ == "__main__":
    main()
