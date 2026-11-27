#!/usr/bin/env python3
"""Quality gate for performance work: re-run held-out closed-loop scenarios.

Every speed-up must leave the audio result alone. This script replays a
slice of the SIMULATED held-out steady protocol (the one behind
``docs/results/closedloop-v0.2-steady.json``: test scenarios, 3 gains from
``rng_for(test, idx, salt=1)`` plus 2 high gains from ``salt=78``, P=64,
mu=0.02, soundcheck_s=3.0, tracking=True) with the package ``ClosedLoopAFC``
from *this* checkout, and records per run: howl flag, LSD, max excess and a
fingerprint of the speaker signal.

It needs the private research workspace (room simulator, sources, scenario
generator) and a Python with pyroomacoustics; it is not part of the public
test suite.

    # record a baseline on main, then on the branch, then compare
    python scripts/quality_gate.py --research ../nullyn --count 40 --out base.json
    python scripts/quality_gate.py --research ../nullyn --count 40 --out new.json --compare base.json

Pass criteria used by the performance issues:
  * identical howl flags on every run (no new howls, none lost by accident)
  * |delta LSD| <= 0.05 dB median, <= 0.5 dB on any run
  * bit-exact changes should show max |delta u| == 0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
CL_KW = dict(P=64, mu=0.02, soundcheck_s=3.0, tracking=True)
SOUNDCHECK_DB = -10.0
_R = {}


def _load_research(root):
    """Import the research package under the alias ``nullyn_research``."""
    root = Path(root).resolve()
    spec = importlib.util.spec_from_file_location(
        "nullyn_research", root / "nullyn" / "__init__.py",
        submodule_search_locations=[str(root / "nullyn")])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nullyn_research"] = mod
    spec.loader.exec_module(mod)
    import importlib as il
    return dict(scen=il.import_module("nullyn_research.scenarios"),
                sweep=il.import_module("nullyn_research.sweep"),
                cl=il.import_module("nullyn_research.closedloop"))


def _init(research):
    sys.path.insert(0, str(REPO_SRC))
    _R.update(_load_research(research))


def _work(idx):
    from nullyn.closedloop import ClosedLoopAFC
    sc_mod, sw, cl = _R["scen"], _R["sweep"], _R["cl"]
    sc = sc_mod.sample_scenario("test", idx)
    Fs, msg0, info, F2 = cl.build_paths_ext(sc, None)
    s = sc_mod.load_source(sc)
    r = sc_mod.rng_for("test", idx, salt=1)
    gains = list(r.uniform(*sw.GAIN_RANGE, size=sw.N_GAINS))
    seeds = [int(r.integers(1 << 31)) for _ in range(sw.N_GAINS)]
    r2 = sc_mod.rng_for("test", idx, salt=78)
    gains += list(r2.uniform(8.0, 20.0, size=2))
    seeds += [int(r2.integers(1 << 31)) for _ in range(2)]
    rows = []
    for gi, (g, seed) in enumerate(zip(gains, seeds)):
        p = ClosedLoopAFC(fs=16000, **CL_KW)
        s_pre = np.zeros(p.n_probe * p.B) if p.n_probe else None
        t0 = time.perf_counter()
        u, ref = cl.simulate_cl(s, Fs, msg0[0], p, g, seed, s_pre=s_pre, pre_db=SOUNDCHECK_DB)
        dt = time.perf_counter() - t0
        m = sw.metrics(u, ref, p)
        rows.append(dict(idx=idx, gain_idx=gi, walking=bool(sc["walking"]), howl=bool(m["howl"]),
                         lsd_db=float(m["lsd_db"]), max_excess_db=float(m["max_excess_db"]),
                         u_rms=float(np.sqrt(np.mean(u ** 2))), u_sum=float(np.sum(u)),
                         u_head=[float(v) for v in u[16000:16008]], sim_s=dt))
    return rows


def compare(base, new):
    b = {(r["idx"], r["gain_idx"]): r for r in base["runs"]}
    n = {(r["idx"], r["gain_idx"]): r for r in new["runs"]}
    keys = sorted(set(b) & set(n))
    flips = [k for k in keys if b[k]["howl"] != n[k]["howl"]]
    dl = np.array([n[k]["lsd_db"] - b[k]["lsd_db"] for k in keys if np.isfinite(b[k]["lsd_db"]) and np.isfinite(n[k]["lsd_db"])])
    du = max(max(abs(x - y) for x, y in zip(b[k]["u_head"], n[k]["u_head"])) for k in keys)
    drms = max(abs(n[k]["u_rms"] - b[k]["u_rms"]) / (b[k]["u_rms"] + 1e-20) for k in keys)
    hb = sum(b[k]["howl"] for k in keys)
    hn = sum(n[k]["howl"] for k in keys)
    res = dict(runs=len(keys), howl_base=hb, howl_new=hn, howl_flips=[list(k) for k in flips],
               lsd_delta_median_db=float(np.median(np.abs(dl))) if len(dl) else 0.0,
               lsd_delta_max_db=float(np.max(np.abs(dl))) if len(dl) else 0.0,
               u_head_max_abs_delta=float(du), u_rms_max_rel_delta=float(drms))
    res["pass"] = (not flips and res["lsd_delta_median_db"] <= 0.05 and res["lsd_delta_max_db"] <= 0.5)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--research", required=True, help="path to the private research workspace (contains nullyn/scenarios.py)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.research,)) as pool:
        runs = sum(pool.map(_work, range(a.start, a.start + a.count)), [])
    sys.path.insert(0, str(REPO_SRC))
    import nullyn
    out = dict(label="SIMULATED", version=nullyn.__version__, protocol="steady held-out, package ClosedLoopAFC",
               start=a.start, count=a.count, kw=CL_KW, elapsed_s=round(time.time() - t0, 1),
               howl_rate=float(np.mean([r["howl"] for r in runs])),
               howl_rate_static=float(np.mean([r["howl"] for r in runs if not r["walking"]] or [np.nan])),
               howl_rate_walking=float(np.mean([r["howl"] for r in runs if r["walking"]] or [np.nan])),
               runs=runs)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{len(runs)} runs in {out['elapsed_s']} s; howl rate {out['howl_rate']:.3f} "
          f"(static {out['howl_rate_static']:.3f}, walking {out['howl_rate_walking']:.3f})")
    if a.compare:
        res = compare(json.loads(Path(a.compare).read_text()), out)
        print(json.dumps(res, indent=1))
        sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
