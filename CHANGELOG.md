# Changelog

All notable changes to the Nullyn AI package are documented here.
Version numbers follow [pyproject.toml](pyproject.toml): patch bumps for
fixes, minor bumps for features.

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
