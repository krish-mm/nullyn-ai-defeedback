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


def test_history_ring_matches_shift_register():
    """The O(K) history window must hold exactly what the v0.2 shift held."""
    afc = ClosedLoopAFC(fs=16000, fallback=False, P=5, soundcheck_s=0.05)
    B, P, K = afc.B, afc.P, afc.B + 1
    ref = {n: np.zeros((P, K), dtype=complex if n in ("X", "Xw") else float)
           for n in ("X", "Xw", "Xp", "Xp0")}
    rng = np.random.default_rng(7)
    for i in range(4 * P + 3):
        afc.process(rng.standard_normal(B) * 0.05)
        if afc.kt == afc.n_probe + 1 and afc.n_probe:
            ref["Xp"][:] = 0.0
            ref["Xw"][:] = 0.0
        m = rng.standard_normal(B) * 0.05
        prev, prevw, Af, mh = afc.prev.copy(), afc.prevw.copy(), afc.Af.copy(), afc.mh_w.copy()
        afc.push_ref(m)
        ref["X"][1:] = ref["X"][:-1]
        ref["X"][0] = np.fft.rfft(np.concatenate([prev, m]))
        z = np.concatenate([mh, m])
        mw = np.convolve(z[len(mh) - (len(Af) - 1):], Af, "valid")
        ref["Xw"][1:] = ref["Xw"][:-1]
        ref["Xw"][0] = np.fft.rfft(np.concatenate([prevw, mw]))
        ref["Xp"][1:] = ref["Xp"][:-1]
        ref["Xp"][0] = np.abs(ref["Xw"][0]) ** 2
        if afc.kt <= afc.n_probe:
            ref["Xp0"][1:] = ref["Xp0"][:-1]
            ref["Xp0"][0] = np.abs(ref["X"][0]) ** 2
        for n, r in ref.items():
            np.testing.assert_array_equal(getattr(afc, n), r, err_msg=f"{n} after block {i}")
            assert getattr(afc, n).flags.c_contiguous


def test_adapt_matches_v032_update():
    """Shared normaliser + rfft(n=2B) + in-place W must equal the v0.3.2 update (#11)."""
    afc = ClosedLoopAFC(fs=16000, fallback=False, P=6, soundcheck_s=0.0)
    B = afc.B
    rng = np.random.default_rng(11)
    for _ in range(3 * afc.P):
        afc.process(rng.standard_normal(B) * 0.05)
        afc.push_ref(rng.standard_normal(B) * 0.05)
    W0 = rng.standard_normal(afc.W.shape) + 1j * rng.standard_normal(afc.W.shape)
    e = rng.standard_normal(B) * 0.01
    st = afc.eh_w.copy()
    # v0.3.2 reference
    ew, _ = afc._fir(e, st.copy())
    E = np.fft.rfft(np.concatenate([np.zeros(B), ew]))
    nrm = np.sum(afc.Xp, axis=0)
    nrm = nrm + afc.delta_rel * np.mean(nrm) + 1e-12
    G = np.conj(afc.Xw) * (E / nrm)[None, :]
    gt = np.fft.irfft(G, axis=1, n=2 * B)[:, :B]
    ref = W0 + 0.02 * np.fft.rfft(np.concatenate([gt, np.zeros_like(gt)], axis=1), axis=1)
    W = W0.copy()
    out, _ = afc._adapt(W, 0.02, e, st.copy(), afc._adapt_shared())
    assert out is W
    np.testing.assert_array_equal(out, ref)
