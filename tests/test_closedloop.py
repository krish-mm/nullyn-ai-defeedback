import numpy as np
import pytest

from nullyn.closedloop import ClosedLoopAFC, afc_block_size
from nullyn.models import load_detector_params


def test_closedloop_is_a_separate_package():
    import nullyn.closedloop as pkg
    import nullyn.closedloop.afc as afc
    assert pkg.__name__ == "nullyn.closedloop"
    assert hasattr(afc, "ClosedLoopAFC") and hasattr(pkg, "ClosedLoopAFC")


def test_afc_block_size_scales():
    assert afc_block_size(16000) == 64
    assert afc_block_size(48000) == 192


def test_afc_process_and_push_ref_roundtrip():
    fs = 16000
    params = load_detector_params("v0")
    afc = ClosedLoopAFC(fs=fs, v0params=params, P=8, mu=0.02, soundcheck_s=0.0)
    B = afc.B
    rng = np.random.default_rng(0)
    y = None
    for _ in range(20):
        mic = rng.standard_normal(B) * 0.05
        y = afc.process(mic)
        afc.push_ref(y)
        assert y.shape == (B,)
        assert np.all(np.isfinite(y))


def test_afc_rejects_wrong_block_length():
    afc = ClosedLoopAFC(fs=16000, fallback=False, P=4, soundcheck_s=0.0)
    with pytest.raises(ValueError):
        afc.process(np.zeros(16))


def test_lpc_autocorrelation_matches_full_correlate():
    """The p+1-lag autocorrelation must equal the v0.2.0 np.correlate result."""
    afc = ClosedLoopAFC(fs=48000, fallback=False, P=4, soundcheck_s=0.0, ltp=False)
    rng = np.random.default_rng(3)
    afc.eh = rng.standard_normal(afc.win) * 0.1
    w = afc.eh * afc.lpc_win
    ref = np.correlate(w, w, "full")[len(w) - 1:len(w) + afc.ord]
    ref = ref * np.exp(-0.5 * (0.01 * np.arange(afc.ord + 1)) ** 2)
    ref[0] *= 1.0001
    from scipy.linalg import solve_toeplitz
    a_ref = np.r_[1.0, solve_toeplitz(ref[:-1], -ref[1:])]
    afc._update_lpc()
    np.testing.assert_allclose(afc.A, a_ref, rtol=0, atol=1e-12)
