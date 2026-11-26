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
