# Changelog

All notable changes to the Nullyn AI package are documented here.
Version numbers follow [pyproject.toml](pyproject.toml): patch bumps for
fixes, minor bumps for features.

## 0.3.1 — 2026-11-26

Faster side-chain features (issue #9). No change in behaviour.

- `SideChain._features` fits its two level slopes in closed form instead of with `np.polyfit`, reads the 8-frame history with one fancy-indexed slice instead of Python list comprehensions, and uses a small exact `np.std` replacement. Per candidate it went from 0.151 ms to 0.050 ms (3.0×), and the whole side-chain frame from 0.944 ms to 0.457 ms at 48 kHz.
- Benchmark (same seed, interleaved with 0.3.0): 48 kHz closed-loop p95 2.04–2.13 ms → 1.52–1.61 ms, p99 2.39–2.47 ms → 1.76–1.84 ms; 16 kHz p95 1.454 → 0.918 ms; notch-only 48 kHz p95 1.153 → 0.581 ms.
- Features match the old code to within 4e-14 (only the two slopes differ at all). Quality gate on all 575 held-out steady scenarios (SIMULATED): 0 howl flips, 0 dB LSD change, identical output fingerprint.

## 0.3.0 — 2026-11-26

Real-time CLI defaults that add no engine delay (issue #8). Minor bump: the `nullyn rt` default changes.

- With a reference channel, `nullyn rt` and `scripts/record_session.py` now default the callback block to the engine's 4 ms block (192 samples at 48 kHz) instead of 128. At 48 kHz a 128-sample callback made the engine FIFO add **128 samples (2.67 ms)** of delay; the default now adds **0**. Notch-only keeps 128.
- New `--latency` option (default `low`; `high` or seconds also work). It is passed to PortAudio, which defaulted to `high` before. The real-world effect depends on the driver and was not measured on hardware.
- The CLI prints the engine FIFO delay and warns when a `--blocksize` is not a multiple of the engine block. New `NullynEngine.fifo_delay_samples(n)` and `nullyn.engine.fifo_delay_samples(n, block)`.
- No DSP change: quality gate on all 575 held-out steady scenarios (SIMULATED) is identical to 0.2.1 (0 howl flips, 0 dB LSD change).

## 0.2.2 — 2026-11-26

Tagline update (GitHub issue #28).

- README now leads with the tagline “More gain. No feedback pain.”
- Package description in `pyproject.toml` includes the tagline. No algorithm or API changes.

## 0.2.1 — 2026-11-26

Faster LPC update in the closed-loop canceller (issue #7). No change in behaviour.

- `ClosedLoopAFC._update_lpc` now computes only the 17 autocorrelation lags it uses instead of all 6,143 (`np.correlate(..., "full")`): O(N_w·p) instead of O(N_w²). The autocorrelation step is 52× faster at 48 kHz, and the whole LPC/LTP update went from 1.02 ms to 0.27 ms per call.
- 48 kHz closed-loop benchmark (same seed, same box, interleaved runs): p95 per block 2.96–3.16 ms → 2.26–2.35 ms, p99 3.43–3.63 ms → 2.49–2.82 ms, blocks over the 4 ms budget 32–48 → 3–14 of 7,500. Median unchanged (~0.77 ms).
- Bit-exact: the quality gate on all 575 held-out steady scenarios (2,875 runs, SIMULATED) shows 0 howl flips, 0 dB LSD change and an identical output fingerprint (RMS and sampled values) on every run.

## Unreleased — performance tooling (issue #6)

No change to the `nullyn` package; tooling and docs only, so no version bump.

- `scripts/bench_latency.py`: seeded closed-loop latency benchmark at 48 kHz and 16 kHz (median, p95, p99, max, jitter, deadline misses, GC, transient allocation peak).
- `scripts/profile_hotspots.py`: per-stage timers and a cProfile mode.
- `scripts/quality_gate.py`: replays the SIMULATED held-out steady protocol with the package canceller and compares howl flags, LSD and output against a baseline. On v0.2.0 it reproduces the published counts exactly (331 / 1,725 matched runs, 4 / 690 static, 327 / 1,035 walking).
- `docs/performance.md`: per-block pipeline, time and space complexity per stage, and the v0.2.0 latency baseline. Tracking issue #27 lists the 20 performance steps in order.

## 0.2.0 — 2026-11-26

Closed-loop tracking for a moving microphone (GitHub issue #1).

- The canceller now runs a slow/accurate filter beside a faster one, detects a path change from a drop in ERLE, raises the fast step size, and backs off (plus a more aggressive notch fallback) when a stale prediction would make the residual louder than the mic. `tracking=False` restores the v0.1 canceller.
- SIMULATED, same 575 held-out scenarios, gains, and seeds as `summary.json` (07:46 CT). New closed-loop run 15:11 CT, America/Chicago. Walking howl fell from 37.6% to 31.6% (327 / 1,035 runs) but is still just above the classic notch's 30.7%. Static howl stayed at 0.6% (4 / 690; was 0.7%) with the same +15.5 dB headroom at ≤50% risk. Walking headroom rose from +2.5 dB to +3.0 dB. Overall matched howl fell from 22.8% to 19.2%; overall headroom rose from +4.0 dB to +6.0 dB.
- Per-block CPU at 48 kHz / 192 is still inside the 4.000 ms median budget (0.684–0.694 ms median, about 17% load, vs 0.356–0.394 ms in 0.1.0). The two-filter path is the reason for the increase. p95 on this shared VM can still meet or exceed the budget.
- README charts regenerated from the new closed-loop numbers (labelled SIMULATED, 15:11 CT) and the new CPU timings.

## 0.1.0 — 2026-11-26

Initial public release, maintained by Krish Moodbidri.

- Installable `nullyn` package with a streaming `process(mic_block, ref_block=None)` API, offline and real-time CLIs, a synthetic feedback demo, and tests.
- Notch layer steered by Nullyn AI v0 or a classic detector; optional closed-loop canceller in `nullyn.closedloop`.
- Algorithm paper in `paper/`.
- Released under the MIT License.
- README result charts of the simulated closed-loop steady protocol (`summary.json`, 07:46 CT, 26 Nov 2026) and the measured per-block CPU timings, plus a link to the project website at [nullyn.ai](https://nullyn.ai).
- Project status, collaborator help-wanted areas, and the v1 / v2 / v3 roadmap (no dates).
