#!/usr/bin/env python3
"""Log a real-room v1 test session to CSV.

This script is the capture tool for the README "Real-time" results table,
which is pending Krish's room tests. It does not invent results.

CSV columns:
  timestamp_iso, elapsed_s, event, latency_ms, block_budget_ms, gain_step_db,
  howl_freq_hz, howl_score, note

Events:
  start, stop, callback, xrun, gbf_step, howl_on, howl_off, mark

Keyboard (press enter after the letter if stdin is line-buffered):
  g   mark a gain-before-feedback step (optionally type a dB value first, e.g. 3)
  h   mark that a howl was heard (operator judgment)
  c   mark that the howl cleared
  m   free-text mark (rest of the line is the note)
  q   quit
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Make `python scripts/record_session.py` work without an install when the
# package lives at ../src (editable installs also work).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from nullyn.cli import _resolve_device, _require_sounddevice
from nullyn.engine import NullynEngine
from nullyn.features import SideChain
from nullyn.notch import NullynV0


def _iso_now():
    # Store UTC and let the reader convert; the session header records local zone.
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def main(argv=None):
    p = argparse.ArgumentParser(description="Record a Nullyn AI room-test session to CSV.")
    p.add_argument("-o", "--output", default="session_log.csv")
    p.add_argument("--input-device")
    p.add_argument("--output-device")
    p.add_argument("--mic-channel", type=int, default=1)
    p.add_argument("--ref-channel", type=int, default=None)
    p.add_argument("--output-channel", type=int, default=1)
    p.add_argument("--blocksize", type=int, default=128)
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--mode", choices=("auto", "notch", "closedloop"), default="auto")
    p.add_argument("--detector", default="nullyn_v0",
                   choices=("nullyn_v0", "nullyn_v0.1", "classic"))
    p.add_argument("--soundcheck", type=float, default=0.0)
    p.add_argument("--howl-threshold", type=float, default=0.7,
                   help="detector score at which a side-chain howl_on event is logged")
    args = p.parse_args(argv)

    sd = _require_sounddevice()
    in_dev = _resolve_device(sd, args.input_device, "input")
    out_dev = _resolve_device(sd, args.output_device, "output")
    fs = int(args.samplerate)
    blocksize = int(args.blocksize)
    mic_ch = int(args.mic_channel)
    ref_ch = args.ref_channel
    out_ch = int(args.output_channel)
    budget_ms = 1000.0 * blocksize / fs

    mode = args.mode
    if mode == "auto" and ref_ch is None:
        mode = "notch"

    engine = NullynEngine(fs=fs, mode=mode, detector=args.detector,
                          soundcheck_s=args.soundcheck or 0.0)
    # A second side-chain + v0 model, used only for logging (does not change audio).
    logger_sc = SideChain(fs)
    logger_det = NullynV0(model="v0" if args.detector != "nullyn_v0.1" else "v0.1", fs=fs)
    howl_flag = {"on": False}

    out_path = Path(args.output)
    fieldnames = [
        "timestamp_iso", "elapsed_s", "event", "latency_ms", "block_budget_ms",
        "gain_step_db", "howl_freq_hz", "howl_score", "note",
    ]
    fh = out_path.open("w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    t0 = time.perf_counter()
    lock = threading.Lock()

    def log(**kw):
        row = {k: "" for k in fieldnames}
        row["timestamp_iso"] = _iso_now()
        row["elapsed_s"] = f"{time.perf_counter() - t0:.4f}"
        row["block_budget_ms"] = f"{budget_ms:.3f}"
        row.update({k: v for k, v in kw.items() if v is not None and v != ""})
        with lock:
            writer.writerow(row)
            fh.flush()

    in_info = sd.query_devices(in_dev if in_dev is not None else sd.default.device[0])
    out_info = sd.query_devices(out_dev if out_dev is not None else sd.default.device[1])
    n_in = max(mic_ch, ref_ch or 1)
    n_out = max(out_ch, 2 if out_info["max_output_channels"] >= 2 else 1)

    def callback(indata, outdata, frames, time_info, status):
        started = time.perf_counter()
        if status:
            log(event="xrun", note=str(status))
        mic = indata[:, mic_ch - 1].astype(np.float64, copy=False)
        ref = None if ref_ch is None else indata[:, ref_ch - 1].astype(np.float64, copy=False)
        y = engine.process(mic, ref)
        outdata[:] = 0
        outdata[:len(y), out_ch - 1] = y[:frames]
        cpu_ms = (time.perf_counter() - started) * 1e3
        # Log every callback would swamp the file; keep one in 50, plus overruns.
        n = getattr(callback, "n", 0) + 1
        callback.n = n
        if cpu_ms > 0.9 * budget_ms or n % 50 == 0:
            log(event="callback", latency_ms=f"{cpu_ms:.3f}")
        # Detector log from the *output* residual / mic, for later labelling.
        for cands in logger_sc.push(mic):
            if not cands:
                continue
            X = np.array([c["x"] for c in cands])
            p = logger_det.model(X)
            i = int(np.argmax(p))
            if p[i] >= args.howl_threshold and not howl_flag["on"]:
                howl_flag["on"] = True
                log(event="howl_on", howl_freq_hz=f"{cands[i]['f']:.1f}",
                    howl_score=f"{p[i]:.3f}", note="side-chain detector")
            elif p[i] < args.howl_threshold * 0.5 and howl_flag["on"]:
                howl_flag["on"] = False
                log(event="howl_off", howl_freq_hz=f"{cands[i]['f']:.1f}",
                    howl_score=f"{p[i]:.3f}", note="side-chain detector")

    log(event="start", note=(
        f"input={in_info['name']!s}; output={out_info['name']!s}; "
        f"fs={fs}; block={blocksize}; mode={mode}; detector={args.detector}; "
        f"zone=America/Chicago; this file is a log, not a published result"
    ))
    stream = sd.Stream(
        samplerate=fs,
        blocksize=blocksize,
        device=(in_dev, out_dev),
        channels=(n_in, n_out),
        dtype="float32",
        callback=callback,
    )
    print(f"recording session log -> {out_path}")
    print(f"  in:  {in_info['name']}  mic ch {mic_ch}"
          + (f"  ref ch {ref_ch}" if ref_ch else "  (no reference)"))
    print(f"  out: {out_info['name']}  ch {out_ch}")
    print("  keys: g [dB] = gain step,  h = howl heard,  c = howl cleared,  m note,  q = quit")
    with stream:
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                cmd, *rest = line.split(None, 1)
                cmd = cmd.lower()
                note = rest[0] if rest else ""
                if cmd in {"q", "quit"}:
                    break
                if cmd in {"g", "gain"}:
                    db = ""
                    if note:
                        try:
                            db = f"{float(note.split()[0]):.2f}"
                            note = " ".join(note.split()[1:])
                        except ValueError:
                            pass
                    log(event="gbf_step", gain_step_db=db, note=note or "operator gain step")
                    print(f"  logged gbf_step {db}")
                elif cmd in {"h", "howl"}:
                    log(event="howl_on", note=note or "operator heard howl")
                    print("  logged howl_on (operator)")
                elif cmd in {"c", "clear"}:
                    log(event="howl_off", note=note or "operator heard howl clear")
                    print("  logged howl_off (operator)")
                elif cmd in {"m", "mark"}:
                    log(event="mark", note=note)
                    print("  logged mark")
                else:
                    print("  unknown command; use g, h, c, m, or q")
        except KeyboardInterrupt:
            pass
    log(event="stop")
    fh.close()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
