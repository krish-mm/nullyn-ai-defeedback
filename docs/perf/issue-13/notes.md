# Issue 13 measurements

Numba 0.68.1, quality gate run with `/workspace/nullyn/.venv/bin/python` and
numba visible on `PYTHONPATH`. Baseline is v0.2.0 (`qg/base_v020.json`).

## Adopted: numba DF2T notch cascade

`quality-gate.json`: 2,875 runs, 0 howl flips, 0 dB LSD, identical output
fingerprint. Bit-exact against scipy.signal.sosfilt.

Stage profiler (48 kHz closed loop, 6,000 blocks, two interleaved runs):
notchbank.process 0.078–0.080 ms → 0.032–0.034 ms per call.

Micro-benchmark, 192-sample block: K=4 19.23 → 3.37 µs (5.70×); K=12 20.89 →
7.88 µs (2.65×); K=24 25.62 → 13.23 µs (1.94×).

## Not adopted: numba Levinson-Durbin in the audio path

The kernel is in `nullyn._accel` and tested (typical frames within ~2e-16 of
`solve_toeplitz`; micro-benchmark 15.22 → 0.83 µs, 18×). Wiring it into
`_update_lpc` gave `quality-gate-levinson-numba.json`: still 0 howl flips
(1,184 → 1,184), but LSD on already-howling runs moved by up to 7.53 dB
(median |ΔLSD| 6.9e-14). On the 1,691 runs that do not howl in either
version, max |ΔLSD| was 3.5e-10 and max |Δ u_head| was 1.8e-11. Those 1-ulp
coefficient differences diverge once a run is already unstable, so the
official gate (`pass` requires max |ΔLSD| ≤ 0.5 dB on every run, including
howling ones) fails. `_update_lpc` therefore still uses `solve_toeplitz`.
