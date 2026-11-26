import numpy as np

from nullyn.filters import NotchBank
from nullyn.notch import ClassicNotch, NullynV0, TinyMLP
from nullyn.models import load_classic_config, load_detector_params


def test_notch_bank_attenuates_tone():
    fs = 16000
    f0 = 1250.0
    t = np.arange(int(0.25 * fs)) / fs
    x = np.sin(2 * np.pi * f0 * t)
    bank = NotchBank(fs=fs)
    bank.hit(f0, now=0.0, strength=1.0)
    # Deepen the notch a few times, as the live suppressor would.
    for _ in range(6):
        bank.hit(f0, now=0.05, strength=1.0)
    y = bank.process(x)
    # Compare steady-state tail energy.
    n = fs // 20
    in_e = np.mean(x[-n:] ** 2)
    out_e = np.mean(y[-n:] ** 2)
    assert out_e < 0.25 * in_e


def test_classic_config_and_detector_flags_howl_tone():
    cfg = load_classic_config()
    assert set(cfg) >= {"pnpr", "phpr", "papr", "need"}
    fs = 16000
    proc = ClassicNotch(fs=fs)
    t = np.arange(int(0.8 * fs)) / fs
    x = 0.3 * np.sin(2 * np.pi * 1500 * t)
    _ = proc.process(x)
    assert proc.bank.n_created >= 1
    freqs = [n["f"] for n in proc.bank.notches]
    assert any(abs(f - 1500) < 40 for f in freqs)


def test_mlp_loads_and_scores_howl_like_vector():
    params = load_detector_params("v0")
    assert params["nparams"] == 1121
    assert params["W"][0].shape == (17, 32)
    model = TinyMLP(params)
    # A synthetic howl-like feature vector: high PAPR/PNPR/PHPR, high IPMP,
    # low frequency wobble, growing slope. Column order matches FEATURE_NAMES.
    x = np.array([[
        25.0, 20.0, 25.0, 20.0, 1.0, 2.0, 2.0, 0.05, 0.2,
        20.0, 1.0, 2.0, 0.0, 15.0, 1.0, 1.0, 0.0,
    ]])
    p = model(x)
    assert p.shape == (1,)
    assert 0.0 <= p[0] <= 1.0
    # The kept v0 model should treat a clean, growing, isolated peak as howl.
    assert p[0] >= 0.5


def test_nullyn_v0_places_a_notch_on_a_howl():
    fs = 16000
    proc = NullynV0(model="v0", fs=fs)
    t = np.arange(int(1.0 * fs)) / fs
    x = 0.3 * np.sin(2 * np.pi * 1800 * t)
    y = proc.process(x)
    assert y.shape == x.shape
    assert proc.bank.n_created >= 1
