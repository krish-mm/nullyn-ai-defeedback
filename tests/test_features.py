import numpy as np

from nullyn.features import FEATURE_NAMES, SideChain


def test_feature_count_and_shape():
    assert len(FEATURE_NAMES) == 17
    fs = 16000
    sc = SideChain(fs)
    # Enough samples for several analysis frames.
    t = np.arange(int(0.5 * fs)) / fs
    x = 0.2 * np.sin(2 * np.pi * 1000 * t)
    frames = sc.push(x)
    assert frames, "expected at least one analysis frame"
    cands = frames[-1]
    assert cands, "a 1 kHz tone should produce a candidate peak"
    assert cands[0]["x"].shape == (17,)
    assert 900 < cands[0]["f"] < 1100


def test_sidechain_scales_with_sample_rate():
    sc16 = SideChain(16000)
    sc48 = SideChain(48000)
    assert sc16.nfft == 1024
    assert sc16.hop == 256
    assert sc48.nfft == 3072
    assert sc48.hop == 768
