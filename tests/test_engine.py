import numpy as np
import soundfile as sf

from nullyn import NullynEngine
from nullyn.cli import main as cli_main


def _tone(fs, seconds, hz, amp=0.2):
    t = np.arange(int(seconds * fs)) / fs
    return amp * np.sin(2 * np.pi * hz * t)


def test_engine_notch_mode_same_length():
    fs = 16000
    eng = NullynEngine(fs=fs, mode="notch", detector="nullyn_v0")
    x = _tone(fs, 0.5, 1200)
    y = eng.process(x)
    assert y.shape == x.shape
    assert eng.stats()["mode"] == "notch"


def test_engine_auto_without_ref_is_notch_only():
    fs = 16000
    eng = NullynEngine(fs=fs, mode="auto")
    y = eng.process(_tone(fs, 0.3, 900))
    assert len(y) == int(0.3 * fs)
    assert eng.stats()["mode"] == "notch"


def test_engine_with_ref_uses_closed_loop_and_matches_length():
    fs = 16000
    eng = NullynEngine(fs=fs, mode="auto", detector="classic")
    n = 16000  # 1 s, a multiple of the 64-sample AFC block
    mic = _tone(fs, 1.0, 1100)
    ref = 0.3 * _tone(fs, 1.0, 1100)
    y = eng.process(mic, ref)
    assert y.shape == mic.shape
    assert eng.stats()["mode"] == "closedloop"


def test_engine_blockwise_matches_one_shot_notch():
    fs = 16000
    x = _tone(fs, 0.8, 1400)
    a = NullynEngine(fs=fs, mode="notch")
    b = NullynEngine(fs=fs, mode="notch")
    y1 = a.process(x)
    parts = [b.process(x[i:i + 64]) for i in range(0, len(x), 64)]
    y2 = np.concatenate(parts)
    np.testing.assert_allclose(y1, y2, atol=1e-12)


def test_offline_cli_wav(tmp_path):
    fs = 16000
    mic = _tone(fs, 0.4, 1000)
    path_in = tmp_path / "mic.wav"
    path_out = tmp_path / "out.wav"
    sf.write(path_in, mic, fs)
    rc = cli_main(["process", str(path_in), "-o", str(path_out), "--mode", "notch"])
    assert rc == 0
    y, fs_out = sf.read(path_out)
    assert fs_out == fs
    assert len(y) == len(mic)
