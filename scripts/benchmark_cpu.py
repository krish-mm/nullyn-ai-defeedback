#!/usr/bin/env python3
"""Measure per-block CPU time of NullynEngine on this machine.

Prints a small table used by the README 'Minimum laptop requirements' section.
This is offline compute time, not analog I/O latency, and not a Windows ASIO
or macOS CoreAudio measurement.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from nullyn.engine import NullynEngine


def cpu_label():
    model = platform.processor() or ""
    # /proc/cpuinfo is more honest on Linux VMs that hide the SKU.
    name, mhz, cores = model, "", os.cpu_count()
    try:
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.startswith("model name") and not name:
                name = line.split(":", 1)[1].strip()
            if line.startswith("cpu MHz") and not mhz:
                mhz = line.split(":", 1)[1].strip()
    except OSError:
        pass
    return name or "unknown CPU", mhz, cores


def bench(fs, block, mode, detector, n_blocks=400, warmup=50):
    eng = NullynEngine(fs=fs, mode=mode, detector=detector)
    rng = np.random.default_rng(0)
    mic = rng.standard_normal(block) * 0.05
    ref = rng.standard_normal(block) * 0.05 if mode != "notch" else None
    # Use the engine's internal block if closed-loop requires it.
    B = eng.block if mode != "notch" else block
    mic = rng.standard_normal(B) * 0.05
    ref = (rng.standard_normal(B) * 0.05) if mode != "notch" else None
    for _ in range(warmup):
        eng.process(mic, ref)
    times = []
    for _ in range(n_blocks):
        t0 = time.perf_counter()
        eng.process(mic, ref)
        times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1e3
    budget = 1000.0 * B / fs
    return dict(
        fs=fs, block=B, mode=mode, detector=detector,
        budget_ms=budget,
        median_ms=float(np.median(arr)),
        mean_ms=float(arr.mean()),
        p95_ms=float(np.percentile(arr, 95)),
        max_ms=float(arr.max()),
        load=float(np.median(arr) / budget),
    )


def main():
    name, mhz, cores = cpu_label()
    print(f"CPU: {name}" + (f", {mhz} MHz" if mhz else "") + f", {cores} logical processors")
    print(f"platform: {platform.system()} {platform.release()} {platform.machine()}")
    print(f"python: {sys.version.split()[0]}")
    cases = [
        (48000, 128, "notch", "nullyn_v0"),
        (48000, 128, "closedloop", "nullyn_v0"),
        (48000, 256, "closedloop", "nullyn_v0"),
        (16000, 64, "closedloop", "nullyn_v0"),
        (48000, 128, "notch", "classic"),
    ]
    rows = [bench(*c) for c in cases]
    print(f"{'mode':<12} {'det':<10} {'fs':>6} {'block':>6} {'budget':>8} "
          f"{'median':>8} {'p95':>8} {'max':>8} {'load':>8}")
    for r in rows:
        print(f"{r['mode']:<12} {r['detector']:<10} {r['fs']:6d} {r['block']:6d} "
              f"{r['budget_ms']:7.3f}ms {r['median_ms']:7.3f}ms {r['p95_ms']:7.3f}ms "
              f"{r['max_ms']:7.3f}ms {100*r['load']:7.1f}%")
    return rows


if __name__ == "__main__":
    main()
