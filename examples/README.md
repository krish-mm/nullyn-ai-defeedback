# Examples

The synthetic feedback-loop demo lives in the package:

```bash
python -m nullyn demo
# or
nullyn demo -o /tmp/nullyn-demo.wav
```

That demo is a delay-line illustration, not a venue measurement.

For a real-room session log (latency, gain-before-feedback steps, howl events),
list devices first, then start a session:

```bash
nullyn rt --list-devices
python scripts/record_session.py -o session_log.csv \
  --input-device "Scarlett" --output-device "Scarlett" \
  --mic-channel 1 --ref-channel 2 --samplerate 48000
```
