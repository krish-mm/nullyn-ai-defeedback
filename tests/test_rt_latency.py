"""Closed-loop FIFO delay and the real-time CLI defaults (issue #8)."""
import numpy as np
import pytest

from nullyn.cli import build_parser, parse_latency, resolve_blocksize
from nullyn.engine import NullynEngine, fifo_delay_samples


def _impulse_delay(engine, n, total=9600, at=1000):
    x = np.zeros(total)
    x[at] = 1.0
    outs = [engine.process(x[i:i + n], np.zeros(n)) for i in range(0, total - total % n, n)]
    return int(np.argmax(np.abs(np.concatenate(outs)))) - at


@pytest.mark.parametrize("n", [64, 128, 192, 256, 384, 480])
def test_fifo_delay_formula_matches_impulse(n):
    eng = NullynEngine(fs=48000, mode="closedloop", soundcheck_s=0.0, fallback=False)
    assert eng.block == 192
    assert _impulse_delay(eng, n) == fifo_delay_samples(n, eng.block)


def test_multiples_of_engine_block_add_no_delay():
    for m in (1, 2, 3):
        assert fifo_delay_samples(192 * m, 192) == 0
    assert fifo_delay_samples(128, 192) == 128  # the v0.2 CLI default


def test_rt_defaults_have_zero_fifo_delay_with_reference():
    args = build_parser().parse_args(["rt", "--ref-channel", "2"])
    assert args.blocksize is None and args.latency == "low"
    eng = NullynEngine(fs=48000, mode="auto", soundcheck_s=0.0)
    bs, delay = resolve_blocksize(eng, args.blocksize, use_ref=True)
    assert (bs, delay) == (192, 0)
    assert _impulse_delay(NullynEngine(fs=48000, mode="closedloop", soundcheck_s=0.0, fallback=False), bs) == 0


def test_notch_only_keeps_small_block_and_no_delay():
    eng = NullynEngine(fs=48000, mode="notch")
    assert resolve_blocksize(eng, None, use_ref=False) == (128, 0)


def test_explicit_blocksize_reports_delay():
    eng = NullynEngine(fs=48000, mode="auto", soundcheck_s=0.0)
    assert resolve_blocksize(eng, 128, use_ref=True) == (128, 128)


def test_parse_latency():
    assert parse_latency("low") == "low"
    assert parse_latency("HIGH") == "high"
    assert parse_latency("0.005") == 0.005
    with pytest.raises(SystemExit):
        parse_latency("fast")


class _FakeSD:
    """Minimal stand-in for sounddevice so cmd_rt can run without hardware."""

    def __init__(self):
        self.kw = None
        outer = self

        class _Default:
            device = (0, 0)
        self.default = _Default()

        class Stream:
            latency = (0.003, 0.004)

            def __init__(self, **kw):
                outer.kw = kw

            def __enter__(self):
                n = outer.kw["blocksize"]
                ind = np.zeros((n, outer.kw["channels"][0]), dtype=np.float32)
                outd = np.zeros((n, outer.kw["channels"][1]), dtype=np.float32)
                for _ in range(4):
                    outer.kw["callback"](ind, outd, n, None, None)
                return self

            def __exit__(self, *a):
                return False
        self.Stream = Stream

    def query_devices(self, i=None):
        return {"name": "fake", "max_input_channels": 2, "max_output_channels": 2}


def test_cmd_rt_opens_stream_with_engine_block_and_low_latency(monkeypatch, capsys):
    import nullyn.cli as cli
    fake = _FakeSD()
    monkeypatch.setattr(cli, "_require_sounddevice", lambda: fake)
    monkeypatch.setattr(cli.time, "sleep", lambda s: (_ for _ in ()).throw(KeyboardInterrupt()))
    args = cli.build_parser().parse_args(["rt", "--ref-channel", "2"])
    assert cli.cmd_rt(args) == 0
    assert fake.kw["blocksize"] == 192
    assert fake.kw["latency"] == "low"
    out = capsys.readouterr().out
    assert "engine FIFO delay: 0 samples" in out
