from nullyn.demo import run_demo


def test_demo_suppresses_howl():
    result = run_demo(seconds=3.0, fs=16000, howl_hz=1250.0, gain_db=6.0,
                      mode="notch", detector="nullyn_v0", quiet=True)
    assert result["none"]["howl"], "the unprocessed delay-line loop should howl"
    assert result["ok"], "Nullyn AI should suppress the demo howl relative to no processing"
    assert result["processed"]["max_excess_db"] < result["none"]["max_excess_db"]
