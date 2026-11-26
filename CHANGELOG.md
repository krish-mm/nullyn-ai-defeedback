# Changelog

All notable changes to the Nullyn AI package are documented here.
Version numbers follow [pyproject.toml](pyproject.toml): patch bumps for
fixes, minor bumps for features.

## 0.1.0 — 2026-11-26

Initial public release, maintained by Krish Moodbidri.

- Installable `nullyn` package with a streaming `process(mic_block, ref_block=None)` API, offline and real-time CLIs, a synthetic feedback demo, and tests.
- Notch layer steered by Nullyn AI v0 or a classic detector; optional closed-loop canceller in `nullyn.closedloop`.
- Algorithm paper in `paper/`.
- Released under the MIT License.
- README result charts of the simulated closed-loop steady protocol (`summary.json`, 07:46 CT, 26 Nov 2026) and the measured per-block CPU timings, plus a link to the project website at [nullyn.ai](https://nullyn.ai).
- Project status, collaborator help-wanted areas, and the v1 / v2 / v3 roadmap (no dates).
