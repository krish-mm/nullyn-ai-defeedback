# Contributing to Nullyn AI

Thank you for wanting to help. This is an ongoing project, and collaborators
are welcome. This repository is the Python implementation of the Nullyn AI
feedback-suppression algorithm, maintained by Krish Moodbidri. Room
validation is done with working FOH engineers; please do not submit venue
recordings or operator names unless you have their permission and Krish has
asked for them.

High-priority areas are listed under [Collaborators welcome](README.md#collaborators-welcome)
in the README: moving-mic tracking, Windows/macOS latency tests, real-room
logs, and embedded/Daisy ports. Issues tagged `help wanted` or
`good first issue` are a good place to start.

## Before you start

- Read the [README](README.md) status section. Published performance numbers
  in this repo are **simulated**. Real-room results are pending.
- The closed-loop canceller lives in `src/nullyn/closedloop/` so that it can
  be left out of a public release if needed. If you change that package, say
  so clearly in the pull request.
- This project is released under the MIT License. See `LICENSE`.

## Development workflow

Every feature or fix starts with a GitHub issue.

1. Open an issue that describes the work (use the [issue templates](.github/ISSUE_TEMPLATE/) when one fits).
2. Create a branch from `main` named `feat/<issue#>-short-name` or `fix/<issue#>-short-name`.
3. Open a pull request against `main` and link it to the issue with `Closes #N` in the PR body. The PR body should explain the change in plain terms.
4. Bump the version in `pyproject.toml` and `src/nullyn/__init__.py` (`__version__`) when it is warranted: a patch bump for fixes, a minor bump for features.
5. Only merge to `main` after Krish Moodbidri approves.

## Set up

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m nullyn demo
```

Real-time I/O (USB interface) needs the optional extra:

```bash
pip install -e ".[realtime]"
nullyn rt --list-devices
```

## What to work on

Good first contributions:

- tests, documentation, and clarity in the notch / detector path
- laptop I/O robustness (`nullyn rt`, `scripts/record_session.py`)
- reproducible room-test logs (see the Real-time table in the README)

Please open an issue first for algorithm changes, especially anything in
`src/nullyn/closedloop/`.

## Pull requests

Follow the [Development workflow](#development-workflow): one issue, one branch, a PR that says `Closes #N` and explains the change in plain terms, and no merge to `main` until Krish Moodbidri approves.

- Keep changes focused. One concern per PR when you can.
- Run `pytest` and `python -m nullyn demo` and paste the output in the PR.
- Do not add simulator sweep dumps, unpublished closed-loop numbers, or
  large audio files.
- Do not invent performance figures. If you report a measurement, include
  the machine, sample rate, block size, and the command you ran.
- Commit messages in the imperative, for example: `Add device listing to rt CLI`.

## Issues

Use the issue templates:

- **Bug report** for incorrect behaviour or crashes
- **Feature request** for new capability
- **Room test** for real-venue logs produced with `scripts/record_session.py`

## Code style

Match the surrounding Python. Prefer small, explicit modules. Public names
say **Nullyn AI**. The streaming API is `NullynEngine.process(mic_block, ref_block=None)`.

## Safety and disclosure

This is live-audio software. A bad change can howl a PA. Flag anything that
weakens the fallback notch layer, removes a safety clip, or changes default
step sizes, and test it against the synthetic demo before a room test.
