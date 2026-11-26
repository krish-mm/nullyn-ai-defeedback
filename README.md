# Nullyn AI

Python implementation of the Nullyn AI acoustic feedback suppressor, for the
v1 laptop prototype and for offline experiments.

Nullyn AI sits on a microphone channel as an insert. A side-chain detector
decides which spectral peaks are howl and steers a bank of per-sample IIR
notch filters. The detector never delays the audio. When a speaker-feed
reference is available, a closed-loop canceller subtracts predicted feedback
first and the notch layer acts as fallback. The canceller lives in
`src/nullyn/closedloop/` so that package can be left out of a public release
if needed.

Maintainer: Krish Moodbidri. Room validation is planned with working FOH
engineers.

This repository has not been published yet. The closed-loop method should
not be pushed to a public host until a provisional patent is filed and
Krish Moodbidri approves.

## Status

**Every performance number in this repository is simulated**, unless a
section is explicitly labelled as a real-room measurement. Nothing here has
been measured in a real venue, on a real microphone or loudspeaker, or on
the v1 USB-interface laptop path. The closed-loop simulation covers less
than a fifth of its planned held-out scenarios. Moving microphones are the
open problem: in simulation the closed loop almost eliminates howl when the
mic is still, and does worse than a notch filter when the performer walks.
See [Results](#results) and `paper/nullyn-algorithm.pdf`.

## Install

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
# optional, for the real-time USB interface path:
pip install -e ".[realtime]"
```

## Quickstart

Synthetic delay-line demo (not a venue measurement):

```bash
nullyn demo
# or
python -m nullyn demo -o /tmp/nullyn-demo.wav
```

Offline files. Microphone only (notch layer). Microphone plus a speaker-feed
reference (closed loop, if the `closedloop` package is present):

```bash
nullyn process mic.wav -o out.wav --mode notch
nullyn process mic.wav --ref speaker_ref.wav -o out.wav --mode auto
```

Streaming API:

```python
from nullyn import NullynEngine
import soundfile as sf

mic, fs = sf.read("mic.wav")
engine = NullynEngine(fs=fs, mode="auto")          # auto: closed loop if ref is passed
out = engine.process(mic)                          # notch only
# out = engine.process(mic, speaker_ref)           # closed loop
sf.write("out.wav", out, fs)
```

## v1 laptop setup

The v1 prototype runs the same algorithm in real time on a laptop. Conversion
is done by a USB audio interface, not by this software:

**microphone → interface XLR in → laptop (Nullyn AI) → interface line out → console**

For the closed loop, also feed a line-level copy of what the speakers play
into a second input on the interface (a console aux or main send, or a
speaker-level tap passed through a pad — the software expects a line-level
reference):

**speaker feed → interface input 2 → Nullyn AI reference**

Input and output can be different devices (`--input-device` and
`--output-device`). Using two unsynchronised interfaces for input and output
is possible in the CLI but is a poor idea for a feedback loop; prefer one
interface for both.

```bash
nullyn rt --list-devices
nullyn rt \
  --input-device "Scarlett" \
  --output-device "Scarlett" \
  --mic-channel 1 \
  --ref-channel 2 \
  --output-channel 1 \
  --samplerate 48000 \
  --blocksize 128
```

The command prints the callback block size in milliseconds, PortAudio's
reported input and output latency if the driver provides it, and running
callback CPU time (median / p95 / max). Those figures are not an
analog-to-analog measurement. `--soundcheck 3` runs three seconds of
closed-loop path identification (probe noise) before passing audio.

To log a room test for the Real-time results table, use
`scripts/record_session.py` (see [Results](#results)).

## Hardware to try it

You need an interface with **at least one XLR microphone input** and a
**line-level output** back to the console. A second input is required for
the closed-loop reference. Prices below are as shown on the linked page
when checked on **26 November 2026, America/Chicago**. They will change.
Dead links were dropped (the former official PreSonus AudioBox USB 96
black page returned 404 and is not listed). This is not a purchase, and
no order will be placed unless Krish Moodbidri approves a specific item
and price.

| Interface | Why it fits | Price as shown | Source (checked 26 Nov 2026, CT) |
|---|---|---|---|
| Focusrite Scarlett Solo, 4th Gen | 1 mic preamp, 1 line/instrument input, 2 balanced line outputs, USB-C. The second input is line/instrument, not a second XLR. | **$129.99** sale, **$159.99** regular. The maker page also showed a sold-out message when checked. | [us.focusrite.com/products/scarlett-solo](https://us.focusrite.com/products/scarlett-solo) |
| Focusrite Scarlett 2i2, 4th Gen | 2 combo XLR/TRS inputs and 2 balanced line outputs. The better closed-loop fit of the two Scarletts, because the reference can use a second combo jack. | **$189.99** sale, **$224.99** regular. The maker page also showed a sold-out message when checked. | [us.focusrite.com/products/scarlett-2i2](https://us.focusrite.com/products/scarlett-2i2) |
| Behringer U-Phoria UMC202HD | 2 combo XLR/TRS inputs and line outputs, 24-bit / 192 kHz. The official Behringer product page lists no price, so the retailer page is the price source. | **$86.90**, in stock | [sweetwater.com/store/detail/UMC202HD](https://www.sweetwater.com/store/detail/UMC202HD--behringer-u-phoria-umc202hd-usb-audio-interface) |
| PreSonus Studio 24c | 2 combo XLR/TRS inputs, 2 balanced ¼-inch TRS line outputs, USB-C. | **$157.99** regular; the maker page showed the unit as sold out. | [presonus.com/products/studio-24c](https://www.presonus.com/products/studio-24c) |
| M-Audio M-Track Solo | 1 combo XLR input plus a ¼-inch instrument/line input, and line outputs. The official maker page ([m-audio.com](https://www.m-audio.com/audio-midi-interfaces/m-track-solo.html)) lists RCA stereo line outputs and does not show a price, so the retailer page is the price source. Only one XLR: the reference would use the ¼-inch input. | **$49.00**, in stock | [sweetwater.com/store/detail/MTrackSolo](https://www.sweetwater.com/store/detail/MTrackSolo--m-audio-m-track-solo-usb-audio-interface) |

## Minimum laptop requirements

These figures were measured by running `scripts/benchmark_cpu.py` on the
development box, **26 November 2026**. That machine reports itself as an
**Intel(R) Xeon(R) Processor, 8 logical processors, 2.40 GHz**, running
Linux 6.12 on x86_64, Python 3.13. The box is a virtual machine and does
not expose a more specific CPU SKU (CPUID family 6, model 207). The
benchmark calls `NullynEngine.process` in a tight loop. It is **offline
CPU time**, not analog I/O latency, and not a measurement of Windows ASIO
or macOS CoreAudio.

| Mode | Rate / internal block | Budget | Median | p95 | Max | Median load |
|---|---|---|---|---|---|---|
| Notch, Nullyn AI v0 | 48 kHz, 128 samples | 2.667 ms | 0.005 ms | 1.242 ms | 2.844 ms | 0.2% |
| Closed loop, Nullyn AI v0 | 48 kHz, 192 samples (4 ms) | 4.000 ms | 0.356–0.394 ms | 3.903–4.405 ms | 8.4–10.9 ms | 8.9–9.8% |
| Closed loop, Nullyn AI v0 | 16 kHz, 64 samples | 4.000 ms | 0.242 ms | 2.252 ms | 9.446 ms | 6.1% |

Median load is modest, including for the closed loop. On this shared VM,
though, p95 callback time approached or exceeded the 4 ms closed-loop
budget, and the worst block clearly exceeded it. Real-time audio has to
meet the budget almost every block, not on average.

**Practical reading, with headroom.** A 64-bit laptop that can run Python
3.10+ should handle the notch path with a large median margin on a machine
in this class. For closed-loop v1, start at 48 kHz with a 128–256 sample
callback (about 2.7–5.3 ms of buffer latency), close other heavy
applications, and watch the callback times that `nullyn rt` prints. If p95
exceeds the buffer, raise `--blocksize`. Plan for several times the median
load we measured — the p95/max on a noisy host is the number that matters,
and we did not measure a quiet personal laptop.

**Unknown, and not claimed here:** Windows ASIO performance, macOS
CoreAudio performance, USB analog-to-analog round-trip, thermal
throttling, and driver quality on any particular interface. Those have to
be measured on the laptop and interface you actually use.

## Results

### Simulated

Labelled **SIMULATED**. pyroomacoustics image-source shoebox rooms, 16 kHz,
4 s runs. Not measured in real venues or on hardware. Source:
`summary.json` from the research simulator, generated 26 November 2026,
07:46 CT, and the unpublished Nullyn AI whitepaper v0.3. The closed-loop
protocols cover less than a fifth of the planned 3,900 held-out scenarios.

**Steady protocol** — 575 held-out scenarios, 1,725 matched runs per
processor (690 static microphone, 1,035 walking performer):

| Metric | No processing | Classic notch | Nullyn AI v0 (notch) | Nullyn AI closed loop |
|---|---|---|---|---|
| Howl rate, all | 46.6% | 32.2% | 25.6% | 22.8% |
| Static mic (690 runs) | 48.6% | 34.5% | 26.5% | **0.7%** |
| Walking performer (1,035 runs) | 45.3% | 30.7% | **25.0%** | 37.6% |
| Fader headroom ≤50% (all / static / walking) | +1.0 / +1.0 / +2.5 dB | +3.5 / +2.5 / +4.5 dB | +5.0 / +4.5 / +5.0 dB | +4.0 / **+15.5** / +2.5 dB |

Moving mics are the open problem. Details, operator and ablation tables,
run counts and limitations are in `paper/nullyn-algorithm.pdf` (and
`paper/nullyn-algorithm.md`).

This repository does not include simulator sweep dumps.

### Real-time

Pending: Krish's room tests. No real-room numbers are reported here.

Use `scripts/record_session.py` to log callback latency, gain-before-feedback
steps and howl events to CSV. After a session, file a [Room test](.github/ISSUE_TEMPLATE/room_test.md)
issue (or hand Krish the CSV) and fill the table below from the log — do
not estimate the numbers.

```bash
nullyn rt --list-devices
python scripts/record_session.py -o session_log.csv \
  --input-device "Scarlett" --output-device "Scarlett" \
  --mic-channel 1 --ref-channel 2 \
  --blocksize 128 --samplerate 48000
```

During the session: `g 3` logs a +3 dB gain step, `h` logs that a howl was
heard, `c` logs that it cleared, `m` adds a note, `q` quits.

| Date (CT) | Venue type | Laptop / OS | Interface | Mode | Sample rate / block | Callback median / p95 / max (ms) | Reported RT (ms) | GBF steps (dB) | Howl events | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| — | pending | — | — | — | — | — | — | — | — | Krish's room tests have not been run |

## Package layout

```
src/nullyn/              installable package
  features.py            side-chain peak features
  filters.py, notch.py   IIR notch bank; classic and Nullyn AI v0 detectors
  engine.py              process(mic_block, ref_block=None)
  closedloop/            PEM-prewhitened adaptive canceller (optional)
  data/                  packed v0 / v0.1 weights and classic thresholds
scripts/record_session.py
scripts/benchmark_cpu.py
paper/                   algorithm note (Markdown + PDF)
```

Tests: `pytest`. The packaged weights are the small trained models already
used in the research simulator (`nullyn_v0`, kept after the first learning
loop; `nullyn_v0_1` is shipped for comparison and was not adopted).

## License

The license is **pending**. Krish Moodbidri has not yet chosen between the
GNU GPL and the MIT License. See `LICENSE`. Do not assume either license
applies.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please run `pytest` and
`python -m nullyn demo` before opening a pull request.
