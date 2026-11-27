#!/usr/bin/env python3
"""Reproducible per-block latency benchmark for the Nullyn AI real-time path.

Drives ``NullynEngine`` inside a SIMULATED closed acoustic loop (speaker ->
room -> mic) so that the canceller adapts, the path-change logic fires and
notches get created, exactly as they would in a live callback. Only the time
spent inside ``engine.process(mic, ref)`` is measured; the loop simulation
itself is excluded.

Everything is seeded, so two runs on the same machine see identical audio.

Reports per configuration: median, p95, p99, max, jitter (std and p99-p50),
budget load, deadline misses, GC collections and the tracemalloc transient
peak per block (a separate, untimed pass, because tracemalloc slows Python
down).

Examples::

    python scripts/bench_latency.py                 # default suite, table to stdout
    python scripts/bench_latency.py --json out.json  # also save raw stats
    python scripts/bench_latency.py --quick          # 2 s per scenario

The numbers are offline compute time on this machine, not analog-to-analog
latency and not a Windows ASIO / macOS CoreAudio measurement.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

import nullyn
from nullyn.engine import NullynEngine

SEED = 20261126


# ----------------------------------------------------------------- loop model
def _pink(n, rng):
    w = rng.standard_normal(n)
    W = np.fft.rfft(w)
    W /= np.sqrt(np.maximum(np.arange(len(W)), 1))
    x = np.fft.irfft(W, n)
    return x / (np.std(x) + 1e-12)


def _speechlike(n, fs, rng):
    """Pink noise with a 4 Hz syllable envelope and pauses (not real speech)."""
    x = _pink(n, rng)
    t = np.arange(n) / fs
    env = 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 2 * np.pi)))
    gate = (np.sin(2 * np.pi * 0.37 * t + rng.uniform(0, 2 * np.pi)) > -0.6).astype(float)
    return 0.05 * x * env * gate


def _room_ir(fs, rng, delay_ms, rt60=0.5, length_s=0.12):
    n = int(length_s * fs)
    d = int(delay_ms * 1e-3 * fs)
    t = np.arange(n - d) / fs
    tail = rng.standard_normal(n - d) * np.exp(-6.9 * t / rt60)
    h = np.zeros(n)
    h[d] = 1.0
    h[d:] += 0.3 * tail
    return h


def _msg0_db(h, nfft=1 << 15):
    H = np.abs(np.fft.rfft(h, nfft))
    return -20 * np.log10(H.max() + 1e-12)


SCENARIOS = {
    # name: (walking, gain_db re MSG0 of the start path, delay ms, rt60)
    "static_low":   (False, -6.0, 12.0, 0.5),
    "static_hi":    (False, 4.0, 12.0, 0.8),
    "walking_low":  (True, -6.0, 12.0, 0.5),
    "walking_hi":   (True, 4.0, 9.0, 0.8),
}


def run_scenario(fs, mode, detector, name, seconds, warmup_s=0.5, track_alloc=False):
    walking, gain_db, delay_ms, rt60 = SCENARIOS[name]
    rng = np.random.default_rng(SEED + sum(map(ord, name)))      # stable across processes
    eng = NullynEngine(fs=fs, mode=mode, detector=detector, soundcheck_s=0.0)
    B = eng.block if mode != "notch" else max(8, int(round(4e-3 * fs)))
    n = int(seconds * fs) // B * B
    src = _speechlike(n, fs, rng)
    h0 = _room_ir(fs, rng, delay_ms, rt60)
    h1 = _room_ir(fs, rng, delay_ms * 0.6, rt60) * (2.0 if walking else 1.0)
    K = 10 ** ((_msg0_db(h0) + gain_db) / 20)
    L = len(h0)
    spk = np.zeros(n + L)
    times = np.empty(n // B)
    peaks = np.empty(n // B) if track_alloc else None
    warm = int(warmup_s * fs) // B
    gc0 = [s["collections"] for s in gc.get_stats()]
    for j, i in enumerate(range(0, n, B)):
        a = min(1.0, i / n * 2) if walking else 0.0          # walk toward the PA over half the run
        # feedback for this block from past speaker output (path >= delay_ms > one block)
        seg = spk[max(0, i - L + 1):i + B]
        fb0 = np.convolve(seg, h0, "full")[len(seg) - B:len(seg)] if len(seg) >= B else np.zeros(B)
        if walking:
            fb1 = np.convolve(seg, h1, "full")[len(seg) - B:len(seg)] if len(seg) >= B else np.zeros(B)
            fb = (1 - a) * fb0 + a * fb1
        else:
            fb = fb0
        mic = src[i:i + B] + fb
        ref = spk[i - B:i] if (mode != "notch" and i >= B) else (np.zeros(B) if mode != "notch" else None)
        if track_alloc:
            tracemalloc.reset_peak()
            base = tracemalloc.get_traced_memory()[0]
            y = eng.process(mic, ref)
            peaks[j] = tracemalloc.get_traced_memory()[1] - base
        else:
            t0 = time.perf_counter_ns()
            y = eng.process(mic, ref)
            times[j] = (time.perf_counter_ns() - t0) * 1e-6
        m = K * y
        spk[i:i + B] = 4.0 * np.tanh(m / 4.0)
    gcn = sum(s["collections"] for s in gc.get_stats()) - sum(gc0)
    st = eng.stats()
    out = dict(B=B, warm=warm, gc=gcn, stats={k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else v) for k, v in st.items()},
               out_rms=float(np.sqrt(np.mean(spk[:n] ** 2))))
    if track_alloc:
        out["peak_bytes"] = peaks[warm:]
    else:
        out["times"] = times[warm:]
    return out


def summarize(t, budget):
    return dict(
        n=int(len(t)),
        median_ms=float(np.median(t)),
        mean_ms=float(np.mean(t)),
        p95_ms=float(np.percentile(t, 95)),
        p99_ms=float(np.percentile(t, 99)),
        max_ms=float(np.max(t)),
        jitter_std_ms=float(np.std(t)),
        jitter_p99_p50_ms=float(np.percentile(t, 99) - np.median(t)),
        budget_ms=budget,
        load_median=float(np.median(t) / budget),
        over_budget=int(np.sum(t > budget)),
        over_half_budget=int(np.sum(t > budget / 2)),
    )


CONFIGS = [
    # (fs, mode, detector)
    (48000, "closedloop", "nullyn_v0"),
    (16000, "closedloop", "nullyn_v0"),
    (48000, "notch", "nullyn_v0"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=8.0, help="audio per scenario (default 8)")
    ap.add_argument("--quick", action="store_true", help="2 s per scenario")
    ap.add_argument("--repeats", type=int, default=1, help="repeat the whole suite and pool the timings")
    ap.add_argument("--config", default=None, help="only fs:mode, e.g. 48000:closedloop")
    ap.add_argument("--no-alloc", action="store_true", help="skip the tracemalloc pass")
    ap.add_argument("--json", default=None, help="write results JSON here")
    ap.add_argument("--label", default=None)
    ap.add_argument("--gc", choices=["on", "off", "freeze"], default="on",
                    help="garbage collector during the timed pass (default on, as in v0.2.0)")
    a = ap.parse_args(argv)
    secs = 2.0 if a.quick else a.seconds
    cfgs = CONFIGS
    if a.config:
        fs_s, mode = a.config.split(":")
        cfgs = [c for c in CONFIGS if c[0] == int(fs_s) and c[1] == mode]
    meta = dict(label=a.label or f"nullyn {nullyn.__version__}", version=nullyn.__version__,
                python=sys.version.split()[0], numpy=np.__version__,
                platform=f"{platform.system()} {platform.release()} {platform.machine()}",
                cpu=_cpu(), seed=SEED, seconds=secs, repeats=a.repeats,
                load_avg=os.getloadavg() if hasattr(os, "getloadavg") else None,
                time=time.strftime("%Y-%m-%d %H:%M %Z"))
    try:
        import scipy
        meta["scipy"] = scipy.__version__
    except ImportError:
        pass
    meta["gc"] = a.gc
    if a.gc == "freeze":
        gc.collect()
        gc.freeze()
    if a.gc in ("off", "freeze"):
        gc.disable()
    results = []
    for fs, mode, det in cfgs:
        pooled, per = [], {}
        B = None
        for _ in range(a.repeats):
            for name in SCENARIOS:
                r = run_scenario(fs, mode, det, name, secs)
                B = r["B"]
                pooled.append(r["times"])
                per.setdefault(name, dict(gc=0, stats=r["stats"], out_rms=r["out_rms"]))
                per[name]["gc"] += r["gc"]
                per[name].setdefault("times", []).append(r["times"])
        budget = 1000.0 * B / fs
        row = dict(fs=fs, mode=mode, detector=det, block=B, **summarize(np.concatenate(pooled), budget))
        row["scenarios"] = {k: dict(summarize(np.concatenate(v["times"]), budget), gc=v["gc"],
                                    out_rms=v["out_rms"], notches_created=v["stats"].get("notches_created"),
                                    n_path_changes=v["stats"].get("n_path_changes"))
                            for k, v in per.items()}
        if not a.no_alloc:
            tracemalloc.start()
            pk = np.concatenate([run_scenario(fs, mode, det, n, min(secs, 2.0), track_alloc=True)["peak_bytes"]
                                 for n in SCENARIOS])
            tracemalloc.stop()
            row["alloc_peak_kib_median"] = float(np.median(pk) / 1024)
            row["alloc_peak_kib_max"] = float(np.max(pk) / 1024)
        results.append(row)
    _print(meta, results)
    if a.json:
        Path(a.json).write_text(json.dumps(dict(meta=meta, results=results), indent=2))
    return results


def _cpu():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip() + f" ({os.cpu_count()} logical)"
    except OSError:
        pass
    return platform.processor() or "unknown"


def _print(meta, results):
    print(f"# {meta['label']}  python {meta['python']}  numpy {meta['numpy']}  scipy {meta.get('scipy')}")
    print(f"# {meta['cpu']}  {meta['platform']}  load {meta['load_avg']}")
    print(f"# seed {meta['seed']}, {meta['seconds']} s x {len(SCENARIOS)} scenarios x {meta['repeats']} repeat(s), gc {meta['gc']}")
    hdr = (f"{'mode':<11}{'fs':>6}{'B':>5}{'budget':>8}{'median':>8}{'p95':>8}{'p99':>8}"
           f"{'max':>8}{'jit_sd':>8}{'>budget':>8}{'allocKiB':>9}")
    print(hdr)
    for r in results:
        print(f"{r['mode']:<11}{r['fs']:>6}{r['block']:>5}{r['budget_ms']:>8.3f}{r['median_ms']:>8.3f}"
              f"{r['p95_ms']:>8.3f}{r['p99_ms']:>8.3f}{r['max_ms']:>8.3f}{r['jitter_std_ms']:>8.3f}"
              f"{r['over_budget']:>8d}{r.get('alloc_peak_kib_median', float('nan')):>9.1f}")


if __name__ == "__main__":
    main()
