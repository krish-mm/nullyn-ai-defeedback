---
name: Room test
about: File a real-venue session log for the Real-time results table
title: "[room-test] "
labels: room-test
---

Use this template after a session with `scripts/record_session.py`. Real-room
results are pending; this is how they will be added. Do not invent numbers.
Do not attach recordings of people without their permission.

**Venue type** (for example: rehearsal room, club, house of worship — no need to name it)

**Who ran the test** (you, or a working FOH engineer; no need for a public name)

**Hardware**
- Laptop (CPU, OS):
- USB audio interface:
- Mic and speakers, in brief:
- `nullyn rt` / `record_session.py` flags (sample rate, block size, mode, devices):

**Log file**
Attach the CSV from `scripts/record_session.py`.

**Summary (from the log, not estimated)**
- Measured callback latency (median / p95 / max):
- PortAudio reported round-trip, if available:
- Gain-before-feedback steps tried (dB):
- Howl events (operator and/or detector):
- Mode used (notch / closed-loop):

**Notes**
What went well, what failed, and anything that would change a later run.
