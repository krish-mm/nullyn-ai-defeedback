"""Optional numba backend (#13) must match the SciPy routines it replaces."""
import numpy as np
import pytest
from scipy import signal
from scipy.linalg import solve_toeplitz

from nullyn._accel import (
    HAS_NUMBA,
    _levinson_nb,
    _sosfilt_nb,
    levinson,
    levinson_scipy,
    sosfilt,
    sosfilt_scipy,
    warmup,
)
from nullyn.closedloop import ClosedLoopAFC
from nullyn.engine import NullynEngine
from nullyn.filters import NotchBank, peaking_eq


def test_levinson_matches_solve_toeplitz():
    rng = np.random.default_rng(13)
    for n, p in ((256, 16), (1024, 16), (64, 8)):
        w = rng.standard_normal(n) * np.hanning(n)
        r = np.array([np.dot(w[:n - k], w[k:]) for k in range(p + 1)])
        r = r * np.exp(-0.5 * (0.01 * np.arange(p + 1)) ** 2)
        r[0] *= 1.0001
        ref = solve_toeplitz(r[:-1], -r[1:])
        np.testing.assert_allclose(levinson_scipy(r), ref, rtol=0, atol=0)
        np.testing.assert_allclose(levinson(r), ref, rtol=0, atol=1e-12)
        if HAS_NUMBA:
            np.testing.assert_allclose(_levinson_nb(np.ascontiguousarray(r, dtype=np.float64)),
                                       ref, rtol=0, atol=1e-12)


def test_sosfilt_matches_scipy_and_keeps_state():
    rng = np.random.default_rng(13)
    fs = 48000.0
    for k in (1, 4, 12, 24):
        rows = [np.r_[peaking_eq(800 + 250 * i, -9.0, 20.0, fs)] for i in range(k)]
        sos = np.array(rows, dtype=np.float64)
        x = rng.standard_normal(192)
        zi0 = rng.standard_normal((k, 2)) * 0.01
        y_ref, z_ref = signal.sosfilt(sos, x, zi=zi0)
        y, z = sosfilt(sos, x, zi=zi0)
        np.testing.assert_allclose(y, y_ref, rtol=0, atol=1e-12)
        np.testing.assert_allclose(z, z_ref, rtol=0, atol=1e-12)
        # A second block must continue from the same state.
        x2 = rng.standard_normal(192)
        y2_ref, z2_ref = signal.sosfilt(sos, x2, zi=z_ref)
        y2, z2 = sosfilt(sos, x2, zi=z)
        np.testing.assert_allclose(y2, y2_ref, rtol=0, atol=1e-12)
        np.testing.assert_allclose(z2, z2_ref, rtol=0, atol=1e-12)
        if HAS_NUMBA:
            zc = np.array(zi0, dtype=np.float64, copy=True, order="C")
            yn = _sosfilt_nb(np.ascontiguousarray(sos), np.ascontiguousarray(x, dtype=np.float64), zc)
            np.testing.assert_allclose(yn, y_ref, rtol=0, atol=1e-12)
            np.testing.assert_allclose(zc, z_ref, rtol=0, atol=1e-12)


def test_warmup_compiles_before_process():
    warmup()
    if not HAS_NUMBA:
        pytest.skip("numba is not installed")
    assert _levinson_nb.nopython_signatures
    assert _sosfilt_nb.nopython_signatures
    # Construction itself warms up, so the first process() must not compile.
    sigs_lpc = list(_levinson_nb.nopython_signatures)
    sigs_sos = list(_sosfilt_nb.nopython_signatures)
    eng = NullynEngine(fs=16000, mode="closedloop", soundcheck_s=0.0)
    B = eng.block
    y = eng.process(np.zeros(B), np.zeros(B))
    assert y.shape == (B,)
    assert list(_levinson_nb.nopython_signatures) == sigs_lpc
    assert list(_sosfilt_nb.nopython_signatures) == sigs_sos
    afc = ClosedLoopAFC(fs=16000, fallback=False, P=4, soundcheck_s=0.0)
    afc.process(np.zeros(afc.B))
    assert list(_levinson_nb.nopython_signatures) == sigs_lpc
    bank = NotchBank(fs=16000)
    bank.hit(1000.0, now=0.0)
    bank.process(np.zeros(64))
    assert list(_sosfilt_nb.nopython_signatures) == sigs_sos


def test_notch_bank_matches_scipy_on_a_tone():
    fs = 16000
    f0 = 1250.0
    t = np.arange(int(0.25 * fs)) / fs
    x = np.sin(2 * np.pi * f0 * t)
    bank = NotchBank(fs=fs)
    for _ in range(7):
        bank.hit(f0, now=0.05)
    y = bank.process(x)
    # Independent SciPy cascade with the same coefficients and zero state.
    y_ref, _ = signal.sosfilt(bank.sos, x, zi=np.zeros_like(bank.zi))
    np.testing.assert_allclose(y, y_ref, rtol=0, atol=1e-12)
