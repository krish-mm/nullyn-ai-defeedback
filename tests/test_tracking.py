"""Tests for v0.2 moving-mic tracking controls."""

import time

import numpy as np
import pytest

from nullyn.closedloop import ClosedLoopAFC, afc_block_size
from nullyn.models import load_detector_params


def test_tracking_off_and_on_process_the_same_shapes():
    fs = 16000
    params = load_detector_params("v0")
    for tracking in (False, True):
        afc = ClosedLoopAFC(fs=fs, v0params=params, P=8, mu=0.02,
                            soundcheck_s=0.0, tracking=tracking)
        B = afc.B
        rng = np.random.default_rng(0)
        for _ in range(30):
            mic = rng.standard_normal(B) * 0.05
            y = afc.process(mic)
            afc.push_ref(y)
            assert y.shape == (B,)
            assert np.all(np.isfinite(y))
        st = afc.stats()
        assert st["tracking"] is tracking


def test_path_change_detector_trips_on_a_residual_jump():
    fs = 16000
    afc = ClosedLoopAFC(fs=fs, fallback=False, P=8, mu=0.02,
                        soundcheck_s=0.0, tracking=True, erle_drop_db=6.0)
    B = afc.B
    rng = np.random.default_rng(1)
    # Prime a "previously good" slow model (high average ERLE), then a jump.
    for _ in range(20):
        mic = rng.standard_normal(B) * 0.01
        y = afc.process(mic)
        afc.push_ref(y)
    afc.pe_avg = 1e-6
    afc.px_avg = 1e-2  # ~40 dB average ERLE
    afc.erle_peak = 40.0
    afc.change_timer = 0
    assert afc.n_path_changes == 0 or afc.pe_avg > 0
    burst = rng.standard_normal(B) * 1.0
    afc.process(burst)
    assert afc.change_timer > 0
    assert afc.n_path_changes >= 1
    assert afc.mu_fast_now >= afc.mu_fast


def test_safety_gate_reduces_gamma_when_prediction_hurts():
    fs = 16000
    afc = ClosedLoopAFC(fs=fs, fallback=False, P=8, mu=0.0,  # freeze after we seed W
                        soundcheck_s=0.0, tracking=True, adapt=False)
    B = afc.B
    # Force a large wrong prediction by stuffing W with energy, then feed
    # a quiet mic; residual will exceed the mic power and gamma should fall.
    afc.W[:] = 3.0 + 0.0j
    afc.X[0] = np.fft.rfft(np.concatenate([np.zeros(B), np.ones(B) * 0.5]))
    quiet = np.zeros(B)
    for _ in range(8):
        afc.process(quiet)
    assert afc.gamma < 0.95


def test_cpu_48k_192_stays_inside_block_budget():
    """Offline process() time at the 48 kHz / 192-sample block, with tracking on."""
    fs = 48000
    params = load_detector_params("v0")
    afc = ClosedLoopAFC(fs=fs, v0params=params, P=64, mu=0.02,
                        soundcheck_s=0.0, tracking=True)
    assert afc.B == afc_block_size(fs) == 192
    rng = np.random.default_rng(0)
    mic = rng.standard_normal(afc.B) * 0.05
    ref = rng.standard_normal(afc.B) * 0.05
    for _ in range(40):
        afc.process(mic)
        afc.push_ref(ref)
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        afc.process(mic)
        afc.push_ref(ref)
        times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1e3
    budget = 1000.0 * afc.B / fs  # 4.0 ms
    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    # The hard requirement is median cost inside the 4 ms block budget.
    # p95 on this shared VM is noisy (even the one-filter v0.1 canceller
    # sometimes spiked above the budget); report it, don't fail the suite on it.
    assert median < budget, f"median {median:.3f} ms vs budget {budget:.3f} ms"
    print(f"\nCPU 48k/192 tracking: median {median:.3f} ms, p95 {p95:.3f} ms, "
          f"max {arr.max():.3f} ms, budget {budget:.3f} ms")
