# Contributing to Nullyn AI

Thank you for wanting to help. This repository is the Python implementation
of the Nullyn AI feedback-suppression algorithm, maintained by Krish
Moodbidri. Room validation is done with working FOH engineers; please do
not submit venue recordings or operator names unless you have their
permission and Krish has asked for them.

## Before you start

- Read the [README](README.md) status section. Published performance numbers
  in this repo are **simulated**. Real-room results are pending.
- The closed-loop canceller lives in `src/nullyn/closedloop/` so that it can
  be left out of a public release if needed. If you change that package, say
  so clearly in the pull request.
- A license has not been chosen yet (GPL vs MIT is still pending). Do not
  add a LICENSE text that picks one, and do not assume a license in new
  files.

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
