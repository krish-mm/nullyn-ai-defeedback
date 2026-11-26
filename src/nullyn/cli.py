"""Command-line interface: offline WAV processing, real-time laptop I/O, demo."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import __version__
from .engine import NullynEngine


def _engine_from_args(args, fs):
    return NullynEngine(
        fs=fs,
        mode=getattr(args, "mode", "auto"),
        detector=getattr(args, "detector", "nullyn_v0"),
        soundcheck_s=getattr(args, "soundcheck", 0.0) or 0.0,
    )


def cmd_process(args):
    import soundfile as sf

    mic, fs = sf.read(args.mic, always_2d=True, dtype="float64")
    mic = mic[:, 0]
    ref = None
    if args.ref:
        r, fs_r = sf.read(args.ref, always_2d=True, dtype="float64")
        if fs_r != fs:
            raise SystemExit(f"reference sample rate {fs_r} does not match mic rate {fs}")
        ref = r[:, 0]
        n = min(len(mic), len(ref))
        mic, ref = mic[:n], ref[:n]
    engine = _engine_from_args(args, fs)
    # Process in engine-sized blocks so closed-loop stays aligned.
    B = engine.block
    out = np.zeros_like(mic)
    for i in range(0, len(mic), B):
        sl = slice(i, min(i + B, len(mic)))
        block = mic[sl]
        if len(block) < B:
            pad = B - len(block)
            mb = np.concatenate([block, np.zeros(pad)])
            rb = None if ref is None else np.concatenate([ref[sl], np.zeros(pad)])
            y = engine.process(mb, rb)[:len(block)]
        else:
            y = engine.process(block, None if ref is None else ref[sl])
        out[sl] = y
    sf.write(args.output, out, fs)
    stats = engine.stats()
    print(f"wrote {args.output}  ({len(out)} samples at {fs} Hz, mode={stats.get('mode')})")
    if "notches_created" in stats:
        print(f"  notches created: {stats['notches_created']}, "
              f"active: {stats['notches_active']}, "
              f"depth sum: {stats['depth_sum_db']:.1f} dB")
    return 0


def _require_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(
            "sounddevice is required for real-time I/O. Install it with:\n"
            "  pip install 'nullyn[realtime]'"
        ) from exc
    return sd


def _device_table(sd):
    rows = []
    default = sd.default.device
    for i, d in enumerate(sd.query_devices()):
        mark = []
        if i == default[0]:
            mark.append("default in")
        if i == default[1]:
            mark.append("default out")
        rows.append({
            "id": i,
            "name": d["name"],
            "in": int(d["max_input_channels"]),
            "out": int(d["max_output_channels"]),
            "default_sr": d.get("default_samplerate"),
            "host": d.get("hostapi"),
            "mark": ", ".join(mark),
        })
    return rows


def cmd_list_devices(args=None):
    sd = _require_sounddevice()
    print("audio devices (PortAudio via sounddevice)")
    print(f"{'id':>4}  {'in':>3}  {'out':>3}  name")
    for r in _device_table(sd):
        extra = f"  [{r['mark']}]" if r["mark"] else ""
        print(f"{r['id']:4d}  {r['in']:3d}  {r['out']:3d}  {r['name']}{extra}")
    return 0


def _resolve_device(sd, spec, kind):
    if spec is None:
        return None
    # Allow a numeric index or a case-insensitive substring of the device name.
    try:
        return int(spec)
    except (TypeError, ValueError):
        pass
    name = str(spec).lower()
    hits = []
    for i, d in enumerate(sd.query_devices()):
        if kind == "input" and d["max_input_channels"] <= 0:
            continue
        if kind == "output" and d["max_output_channels"] <= 0:
            continue
        if name in d["name"].lower():
            hits.append(i)
    if not hits:
        raise SystemExit(f"no {kind} device matching {spec!r}. Use: nullyn rt --list-devices")
    if len(hits) > 1:
        listing = ", ".join(f"{i}: {sd.query_devices(i)['name']}" for i in hits)
        raise SystemExit(f"ambiguous {kind} device {spec!r}: {listing}")
    return hits[0]


def cmd_rt(args):
    sd = _require_sounddevice()
    if args.list_devices:
        return cmd_list_devices()

    in_dev = _resolve_device(sd, args.input_device, "input")
    out_dev = _resolve_device(sd, args.output_device, "output")
    fs = int(args.samplerate)
    blocksize = int(args.blocksize)
    mic_ch = int(args.mic_channel)
    ref_ch = None if args.ref_channel is None else int(args.ref_channel)
    out_ch = int(args.output_channel)

    in_info = sd.query_devices(in_dev if in_dev is not None else sd.default.device[0])
    n_in_needed = max(mic_ch, ref_ch or 1)
    if in_info["max_input_channels"] < n_in_needed:
        raise SystemExit(
            f"input device {in_info['name']!r} has {in_info['max_input_channels']} "
            f"input channels; need at least {n_in_needed}"
        )
    out_info = sd.query_devices(out_dev if out_dev is not None else sd.default.device[1])
    if out_info["max_output_channels"] < out_ch:
        raise SystemExit(
            f"output device {out_info['name']!r} has {out_info['max_output_channels']} "
            f"output channels; need at least {out_ch}"
        )

    mode = args.mode
    if mode == "auto" and ref_ch is None:
        mode = "notch"
    if mode == "closedloop" and ref_ch is None:
        raise SystemExit("closed-loop mode needs --ref-channel pointing at the speaker-feed input")

    engine = NullynEngine(
        fs=fs,
        mode=mode,
        detector=args.detector,
        soundcheck_s=args.soundcheck or 0.0,
    )

    in_channels = n_in_needed
    out_channels = max(out_ch, 2 if out_info["max_output_channels"] >= 2 else 1)
    times = []
    xruns = {"i": 0, "o": 0}
    t0 = time.perf_counter()
    printed = {"n": 0}

    def callback(indata, outdata, frames, time_info, status):
        started = time.perf_counter()
        if status:
            if status.input_overflow or status.input_underflow:
                xruns["i"] += 1
            if status.output_overflow or status.output_underflow:
                xruns["o"] += 1
        mic = indata[:, mic_ch - 1].astype(np.float64, copy=False)
        ref = None if ref_ch is None else indata[:, ref_ch - 1].astype(np.float64, copy=False)
        y = engine.process(mic, ref)
        outdata[:] = 0
        n = min(len(y), frames)
        outdata[:n, out_ch - 1] = y[:n]
        times.append(time.perf_counter() - started)

    device = (in_dev, out_dev)
    stream = sd.Stream(
        samplerate=fs,
        blocksize=blocksize,
        device=device,
        channels=(in_channels, out_channels),
        dtype="float32",
        callback=callback,
    )
    buf_ms = 1000.0 * blocksize / fs
    print("Nullyn AI real-time (v1 laptop prototype)")
    print(f"  input device:  {in_info['name']}  (mic ch {mic_ch}"
          + (f", ref ch {ref_ch}" if ref_ch else ", no reference") + ")")
    print(f"  output device: {out_info['name']}  (out ch {out_ch})")
    print(f"  sample rate: {fs} Hz, callback block: {blocksize} samples ({buf_ms:.2f} ms)")
    print(f"  engine AFC/notch block: {engine.block} samples ({1000.0 * engine.block / fs:.2f} ms)")
    print(f"  mode: {mode}, detector: {args.detector}")
    print("  Ctrl-C to stop.")
    with stream:
        # Sounddevice reports stream.latency as seconds (in, out) when available.
        try:
            lat = stream.latency
            if isinstance(lat, (tuple, list)) and len(lat) == 2:
                print(f"  PortAudio reported latency: in {lat[0]*1e3:.2f} ms, "
                      f"out {lat[1]*1e3:.2f} ms, "
                      f"round-trip (reported) {(lat[0]+lat[1])*1e3:.2f} ms")
            else:
                print(f"  PortAudio reported latency: {lat}")
        except Exception as exc:
            print(f"  PortAudio reported latency: unavailable ({exc})")
        print(f"  callback buffer latency (blocksize/fs): {buf_ms:.2f} ms  "
              "[this is not a measured analog-to-analog figure]")
        try:
            while True:
                time.sleep(1.0)
                if times:
                    arr = np.array(times[-200:]) * 1e3
                    print(
                        f"  callback CPU: median {np.median(arr):.3f} ms, "
                        f"p95 {np.percentile(arr, 95):.3f} ms, "
                        f"max {arr.max():.3f} ms  / {buf_ms:.2f} ms budget; "
                        f"xruns in/out {xruns['i']}/{xruns['o']}"
                    )
        except KeyboardInterrupt:
            print("\nstopped.")
    if times:
        arr = np.array(times) * 1e3
        print(
            f"measured callback processing time over {len(arr)} blocks, "
            f"{time.perf_counter()-t0:.1f} s: "
            f"median {np.median(arr):.3f} ms, mean {arr.mean():.3f} ms, "
            f"p95 {np.percentile(arr, 95):.3f} ms, max {arr.max():.3f} ms"
        )
        print("this is callback CPU time, not analog round-trip latency.")
    return 0


def cmd_demo(args):
    from .demo import run_demo
    result = run_demo(
        seconds=args.seconds,
        fs=args.samplerate,
        howl_hz=args.howl_hz,
        gain_db=args.gain_db,
        out_wav=args.output,
        mode=args.mode,
        detector=args.detector,
        quiet=args.quiet,
    )
    return 0 if result["ok"] else 1


def build_parser():
    p = argparse.ArgumentParser(
        prog="nullyn",
        description="Nullyn AI acoustic feedback suppression.",
    )
    p.add_argument("--version", action="version", version=f"nullyn {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--mode", choices=("auto", "notch", "closedloop"), default="auto",
                        help="auto uses the closed loop when a reference is provided")
        sp.add_argument("--detector", default="nullyn_v0",
                        choices=("nullyn_v0", "nullyn_v0.1", "classic"),
                        help="howl detector that steers the notch bank (default: nullyn_v0)")
        return sp

    sp = add_common(sub.add_parser("process", help="process a microphone WAV (optional speaker-ref WAV)"))
    sp.add_argument("mic", help="microphone (or channel-insert) WAV")
    sp.add_argument("-r", "--ref", help="speaker-feed reference WAV, same sample rate")
    sp.add_argument("-o", "--output", required=True, help="output WAV")
    sp.set_defaults(func=cmd_process)

    sp = add_common(sub.add_parser("rt", help="real-time v1 laptop prototype (USB interface in/out)"))
    sp.add_argument("--list-devices", action="store_true",
                    help="list input and output devices and exit")
    sp.add_argument("--input-device", help="input device index or name substring (default: system default)")
    sp.add_argument("--output-device", help="output device index or name substring (may differ from input)")
    sp.add_argument("--mic-channel", type=int, default=1, help="1-based input channel for the microphone")
    sp.add_argument("--ref-channel", type=int, default=None,
                    help="1-based input channel for the speaker-feed reference (enables closed loop)")
    sp.add_argument("--output-channel", type=int, default=1, help="1-based output channel")
    sp.add_argument("--blocksize", type=int, default=128, help="callback block size in samples")
    sp.add_argument("--samplerate", type=int, default=48000)
    sp.add_argument("--soundcheck", type=float, default=0.0,
                    help="seconds of closed-loop probe identification before passing audio")
    sp.set_defaults(func=cmd_rt)

    # Alias so `nullyn --list-devices` is also easy to find via `nullyn rt --list-devices`.
    sp = sub.add_parser("list-devices", help="list audio input and output devices")
    sp.set_defaults(func=cmd_list_devices)

    sp = add_common(sub.add_parser("demo", help="simulate a feedback loop and show howl being suppressed"))
    sp.add_argument("--seconds", type=float, default=4.0)
    sp.add_argument("--samplerate", type=int, default=16000)
    sp.add_argument("--howl-hz", type=float, default=1250.0)
    sp.add_argument("--gain-db", type=float, default=6.0,
                    help="loop gain in dB relative to the delay-line stability limit")
    sp.add_argument("-o", "--output", help="optional WAV of the processed speaker signal")
    sp.add_argument("-q", "--quiet", action="store_true")
    sp.set_defaults(func=cmd_demo)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
