# Nullyn AI: The Feedback-Suppression Algorithm

**Krish Moodbidri**  
Nullyn AI  
26 November 2026  

*Confidential draft. This note describes the algorithm implemented in the `nullyn-ai-defeedback` repository. It is not for public release until a provisional patent is filed and Krish Moodbidri approves. All performance numbers below are **SIMULATED**. None have been measured in a real room or on hardware.*

## 1. The problem

In live sound, a microphone hears the loudspeakers that are amplifying it. When the loop gain around some frequency exceeds unity and the phase lines up, that frequency rings and then howls. Front-of-house engineers have lived with this for decades. The usual remedies are well known: ring the room out by hand with a graphic or parametric equalizer; cut automatic notches as howl appears; or shift the pitch of the channel by a few hertz so the loop cannot stay coherent [1], [2], [4].

Those tools work well enough in many rooms, and the market is not empty. Hardware boxes, mixer firmware and plugins already sell into this job. The remaining complaint, and the reason Nullyn AI exists, is quality. Notch filters thin the sound as more of them go in. Frequency shifters smear vocals and are harsh on instruments. A lot of working FOH engineers therefore turn automatic suppressors off and ring the room by hand.

Nullyn AI is an attempt to get more gain before feedback without making the sound worse, including when a performer walks with the microphone. This paper explains how the algorithm does that: the signal path, the detector, the notch layer, the closed-loop canceller, the learning loop and the latency budget. It then reports the simulated results, states the open problem of moving microphones, and sketches the v1 / v2 / v3 roadmap. It does not set dates.

A longer research write-up (Nullyn AI whitepaper v0.3, unpublished) contains the full evaluation protocol, competitive context and hardware notes. The numbers here are taken from that paper and from the closed-loop summary file it uses (`summary.json`, generated 26 November 2026, 07:46 CT).

## 2. Signal path

Nullyn AI sits on a microphone channel as an insert. It has two inputs and one output.

- **Input 1 (microphone).** The performer's microphone or line signal, before the mixer.
- **Input 2 (reference).** A line-level feed of the mixer output that drives the loudspeakers the microphone can hear (mains or a monitor send).
- **Output.** The processed microphone signal, returned to the mixer channel.

The first product prototype (v1) does this on a laptop. An XLR-to-USB audio interface converts the microphone — and, for the closed loop, the speaker feed — to digital audio; Nullyn AI processes the stream; the interface converts the output back and returns it to the console. Conversion, USB transport and the operating system's audio buffers all add delay. That delay has not been measured yet.

The algorithm itself is built as two layers on two time scales.

| Layer | Role | Time scale | In the audio path? |
|---|---|---|---|
| Closed-loop canceller | Subtracts predicted feedback using the speaker-feed reference (Section 5) | Per sample, from an estimate computed on earlier reference blocks | Yes; adds no samples of its own (Section 7) |
| Notch filters | Per-sample IIR notches at frequencies the detector flags (Section 4) | Per sample | Yes; about one sample of processing |
| Detector | Decides which spectral peaks are howl (Section 3) | 64 ms windows every 16 ms | No; it only updates notch settings |
| Learning loop | Retrains the detector from failures (Section 6) | Offline | No |

The audio is never delayed by the neural network. The network, or the classic rule set, only changes filter coefficients. That split is the same idea used in some noise-cancelling products, where a fast filter path is steered by slower control [14].

## 3. Detection

The detector never sits in the audio path. A causal side chain (`nullyn.features.SideChain`) looks at the microphone signal — or, when the canceller is running, at the residual after cancellation.

At a 16 kHz research rate the side chain takes a 1,024-point Hann window (64 ms) every 256 samples (16 ms) and computes the power spectrum in dB. Window and hop lengths scale with sample rate so that those time intervals are kept at 48 kHz or 96 kHz. It keeps about one second of history: 64 frames of spectra, a peak map, sub-bin peak positions, spectral flux and a leaky average. Each frame it lists local spectral maxima between 80 Hz and 7.8 kHz, keeps the five strongest candidates that stand at least 6 dB above the frame mean, and refines each frequency with parabolic interpolation.

Each candidate is described by 17 features, chosen to capture what the literature and our experiments suggest separates a growing feedback tone from a sung or played note [2], [3]:

- **Prominence.** Peak-to-average (PAPR), peak-to-neighbour (PNPR), level versus the frame maximum, and level versus a one-second average. Howl stands out from what is around it.
- **Harmonicity.** Peak-to-harmonic (PHPR, 2nd–4th harmonics) and peak-to-subharmonic. A feedback tone is usually a lone sinusoid; voice and instruments come with harmonics.
- **Persistence.** Inter-frame peak magnitude persistence (IPMP over five frames) and run length as a peak. Howl stays put.
- **Growth.** Level slope over four and eight frames. Howl grows steadily; notes decay or vary.
- **Stability.** Frequency spread and amplitude spread over recent frames. Vibrato and playing wobble; feedback does not.
- **Shape and context.** −6 dB width, log frequency, spectral flux at birth and in the current frame, and a count of strong peaks.

Two detectors share this side chain and the same notch bank.

**Classic rules.** A peak is flagged when PNPR, PHPR, PAPR and IPMP all clear their thresholds, and the flag holds for two consecutive frames. The thresholds used here were tuned on training rooms (`classic_tuned.json`: PNPR 10 dB, PHPR 20 dB, PAPR 15 dB, two-frame persistence). This is the van Waterschoot and Moonen style of automatic notch suppressor [2], [3], and it is the baseline the learned detector has to beat.

**Nullyn AI v0.** A small multilayer perceptron (17–32–16–1, 1,121 parameters) maps the same 17 features to a howl probability. The default threshold is 0.7. The same two-frame persistence rule is applied after the network. The forward pass is plain NumPy; there is no neural-network runtime on the audio path. Weights shipped with the package were trained on simulated rooms and are a few tens of kilobytes.

Detection therefore costs a spectrum every 16 ms and a 1,121-parameter evaluation on at most five candidates. Median detection delay in the main held-out sweep was 96 ms for Nullyn AI v0 and 112 ms for the classic suppressor (whitepaper v0.3, Table 3; 3,900 scenarios, 11,700 runs per processor). That delay is in the control path. It does not delay the audio.

## 4. Notch suppression

The audio-path filter is a bank of per-sample IIR peaking notches (RBJ cookbook biquads [the same topology used as a cut]). Up to 24 notches may be active. A new howl starts at −6 dB; each later hit at a nearby frequency deepens the cut by 3 dB, down to −30 dB. After 8 s without a hit the notch releases at 1.5 dB per second and is removed once it is shallower than −1 dB. Nearby hits are merged rather than stacked, so the bank does not fill up with duplicates.

Coefficient updates take effect on subsequent samples. The bank itself adds no look-ahead and no frame of algorithmic delay. An IIR notch does add frequency-dependent group delay, largest next to the notch frequency; that is a property of the filter, not of the detector.

This layer is the fallback as well as a complete suppressor on its own. When no speaker-feed reference is available, Nullyn AI runs the notch layer only. That is also how the v1 laptop prototype should start in a room that cannot provide a clean reference.

## 5. Closed-loop cancellation

When a reference of what the loudspeakers are playing is available, Nullyn AI first subtracts a prediction of the feedback and then runs the same notch layer on the residual.

Write \(e[n] = x[n] - \hat{y}[n]\), where \(x[n]\) is the microphone and \(\hat{y}[n]\) is an adaptive estimate of the speaker-to-microphone path applied to the reference. The estimate is a partitioned-block frequency-domain NLMS filter (overlap-save, gradient-constrained) with 64 partitions. At 16 kHz the block is 64 samples (4 ms) and the filter covers 256 ms (4,096 taps). Those are the settings used in the closed-loop simulations (`P = 64`, step size \(\mu = 0.02\), 3 s sound check).

\(\hat{y}\) for the current block is formed only from earlier reference blocks. That is possible because the true acoustic path, plus any mixer and USB delay, is itself delayed: the first 4 ms of the path are treated as zero. Under that condition the canceller adds **no samples** of audio latency. If the external loop delay is shorter than one AFC block plus reference-path converter and buffer delay, the condition fails and the canceller must not be used in this form. On a laptop, USB buffer delay on the reference is part of that budget and is still unmeasured (Section 7).

A well-known bias appears when the source signal (voice or music) is correlated with the loudspeaker signal, which it always is in this problem [2], [11]. Nullyn AI follows the prediction-error method (PEM): the error and the reference are prewhitened by a short-term LPC (order 16, updated every four blocks, with a lag window) and an optional long-term (pitch) predictor before the NLMS update [11], [2]. Without prewhitening, every ablation run howled (Section 8).

**Sound check.** Before a show, the box can identify the path with the talker silent. For three seconds it plays white probe noise on its own mixer channel, at −10 dB relative to broadband maximum stable gain in the simulation, and adapts with plain NLMS (a white probe does not need PEM). Then it switches to show mode: live cancellation with PEM, plus the notch fallback.

The canceller code lives in a separate package, `nullyn.closedloop`, so that it can be omitted from a public release if needed.

## 6. The learning loop

Nullyn AI is designed to improve from its own failures.

1. Run the current detector on training-split scenarios. A failure is a run where a howl was not recovered, or where three or more false notches were placed.
2. Re-simulate those failures with the microphone recorded, and label every side-chain candidate with the simulation oracle (a narrow band of the loudspeaker sitting well above the dry, feedback-free signal).
3. Retrain the classifier on the old and new data, with failures weighted more heavily.
4. Re-threshold on a held-out slice of the training split. The objective used in the first loop was howl rate + 0.02 × false notches per run.
5. Accept or reject the new model on held-out scenarios.

In the first loop, v0 ran on 8,832 training-split runs. Of its 6,076 failures, 900 runs were relabelled, adding 878,911 samples (21.5% of them howl). The retrained model, v0.1, used threshold 0.9. On an acceptance test of 2,000 held-out scenarios (6,000 runs per processor; whitepaper v0.3, Table 2) v0.1 traded a 1.3-point higher howl rate (27.3% versus 26.0%) for 22% fewer false notches (3.50 versus 4.49 per run) and slightly better tone. That was not a clear win, so **v0 was kept**. The loop works mechanically. The weak points are the objective and the labels. The data that would matter most are real-venue failures logged during prototype use, labelled with the help of working FOH engineers.

The same loop is planned for the closed-loop layer: log cases where the canceller diverges or tracks too slowly, and use them to tune adaptation control. That has not been run yet.

## 7. Latency

Performers notice delay. In a study of 19 musicians, in-ear vocalists rated up to about 1 ms of added latency as “Good” [15]. The design is budgeted against that, with an honest split between what is known and what is not.

**Notch path.** The filters run one sample at a time. Processing adds about one sample period: 10.4 µs at 96 kHz (1/\(f_s\); our arithmetic). In the device model, 24 settled notches at 96 kHz had a median group delay of 678.5 µs across 100 Hz–10 kHz, and up to 8,769 µs right beside a notch (whitepaper v0.3, Section 8.1).

**Converters, in a future hardware box.** Analog-to-digital and digital-to-analog filters dominate a dedicated device. Using datasheet group delays rounded up to whole samples, the device model gives analog-in to analog-out figures of 104.17 µs for a low-latency converter pair, 406.25 µs for a PCM3060, and 427.08 µs for a typical pair, all at 96 kHz, of which 10.42 µs is the DSP share (whitepaper v0.3, Section 8.2). Those are model figures, not hardware measurements. Apart from the PCM3060 datasheet [16], the whitepaper notes that the other datasheets were not re-checked for that draft.

**Closed-loop canceller.** As in Section 5, the canceller adds no audio-path samples provided the external loop delay exceeds one 4 ms block plus reference-path delay.

**v1 laptop prototype: unknown until measured.** A USB interface and the operating system's audio buffers add round-trip latency that we have not measured. It may well exceed the 1 ms “Good” level for in-ear vocals [15]. That is acceptable for a prototype whose job is to validate the algorithm, but it also tightens the canceller's external-delay condition. Measuring both — callback time, reported stream latency, and analog-to-analog round trip — is a first task for v1. The `nullyn rt` command prints callback CPU time, the buffer length, and PortAudio's reported latency. It does not print a measured analog figure, because we do not have one.

A compute estimate for a future embedded circuit (a 480 MHz Cortex-M7 class part) is given in whitepaper v0.3, Section 8.5. It is an estimate from operation counts, not a measurement.

## 8. Simulated results

Every number in this section is **SIMULATED** in pyroomacoustics image-source shoebox rooms [12], [13]. The rooms are synthetic venue layouts, 16 kHz, 4 s runs. Nothing here was measured in a real venue or on hardware. The closed-loop protocols cover less than a fifth of their planned 3,900 held-out scenarios (575 and 475 scenarios in `summary.json`). Results should be read as early. `summary.json` records 14,875 new closed-loop simulations, with no failed scenarios, generated at 07:46 CT on 26 November 2026.

**Howl** means some band of the speaker signal sits more than 12 dB above the dry (feedback-free) reference for at least 0.5 s after \(t = 1\) s. **Fader headroom** at ≤50% (respectively ≤10%) is the highest fader gain, in dB relative to broadband maximum stable gain at the start position, such that every 0.5 dB gain bin up to it has a howl rate of at most 50% (10%). An **oracle isolation** column is included as an upper-bound reference. It is our own idealised construction, not a model of any commercial product: the simulator hands it the true dry microphone signal and it leaks the non-voice residual at −10 dB, with 4.5 ms of delay.

### 8.1 Notch layer, main held-out sweep

Whitepaper v0.3, Table 3, reports the larger notch-only sweep: 3,900 held-out scenarios, 11,700 runs per processor. Nullyn AI v0 howled in 26.1% of runs, against 33.8% for the tuned classic suppressor and 47.3% with no processing. It placed 4.51 false notches per run, against 7.83 for classic, and gave +5.0 dB of fader headroom at ≤50% risk, against +3.5 dB for classic and +1.5 dB with no processing. About a quarter of runs still howled. Very few howls recovered within the 4 s run (5.0% for v0, 4.2% for classic).

### 8.2 Closed loop, steady protocol

The table below is the closed-loop steady protocol: 575 held-out scenarios, 1,725 matched runs per processor, of which 690 are static microphones and 1,035 are walking performers (`summary.json › steady`).

| Metric | No processing | Classic notch | Nullyn AI v0 (notch) | **Nullyn AI closed loop** | Oracle isolation (upper bound) |
|---|---|---|---|---|---|
| Howl rate, all | 46.6% | 32.2% | 25.6% | **22.8%** | 2.1% |
| Voice | 44.7% | 28.7% | **20.6%** | 22.6% | 1.2% |
| Music | 50.3% | 39.1% | 35.2% | **23.4%** | 3.9% |
| **Static mic** (690 runs) | 48.6% | 34.5% | 26.5% | **0.7%** | 0.3% |
| **Walking performer** (1,035 runs) | 45.3% | 30.7% | **25.0%** | 37.6% | 3.4% |
| Mean howl time per run | 1.44 s | 1.08 s | 0.94 s | **0.87 s** | 0.10 s |
| False notches per run | 0 | 7.86 | 4.46 | **4.25** | n/a |
| LSD when stable (voice / music) | 5.89 / 2.99 dB | 6.35 / 5.30 dB | 6.68 / 3.95 dB | 6.42 / 3.33 dB | 4.84 / 3.06 dB |
| Fader headroom ≤50% | +1.0 dB | +3.5 dB | **+5.0 dB** | +4.0 dB | +12.0 dB |
| Fader headroom ≤10% | −2.5 dB | 0.0 dB | +0.5 dB | **+1.0 dB** | +6.0 dB |

The overall figure hides two very different behaviours.

**Static microphones are a step change.** The closed loop howled in 0.7% of static runs, against 34.5% for the classic suppressor and 26.5% for the notch layer alone. Its fader headroom at ≤50% risk was **+15.5 dB** (≤10%: +10.5 dB), against +2.5 dB for classic and +4.5 dB for v0. Even at extra-high gains (+8 to +20 dB), it howled in 28.9% of static runs (460 runs).

**Walking performers are worse than notch filters.** The closed loop howled in **37.6%** of walking runs, worse than the classic notch filter (30.7%) and the notch layer alone (25.0%). Its walking headroom at ≤50% risk was +2.5 dB, against +4.5 dB for classic and +5.0 dB for v0. At the extra-high gains, 99.7% of walking runs howled (690 runs). Section 9 takes this up.

Music is where the closed loop helps most overall: 23.4% howl against 39.1% for classic. That is consistent with a canceller that does not have to tell a sustained note from howl. Tone is mixed. On stable runs, closed-loop log-spectral distance (5.37 dB) sits between no processing (4.97 dB) and the notch suppressors (6.02 and 5.87 dB). On music it is much closer to unprocessed sound (3.33 dB, against 5.30 dB for classic).

### 8.3 Operator changes after the device

Because the reference is taken after the mixer, a fader push or EQ boost shows up in the reference rather than as an unexplained change. On the operator protocol — 475 held-out scenarios, 1,425 runs per processor (`summary.json › operator`) — the closed loop had the lowest howl rate of the real processors in every condition: 40.4% of all runs, against 48.7% for Nullyn AI v0, 55.0% for classic and 67.7% with no processing. On runs with no event and a single microphone (515 runs) the rates were 23.5%, 25.8%, 31.1% and 48.4% respectively. A fader-only push (351 runs) went to 50.7% closed loop, 65.2% v0, 71.5% classic. A fader push plus an EQ boost (320 runs) went to 63.1%, 77.8% and 84.7%. A second open microphone that the box does not process (441 runs) went to 48.8%, 55.1% and 61.2%. The movement weakness carries over: in this protocol the closed loop howled in 11.1% of static runs but 59.3% of walking runs (classic: 58.6% and 52.6%).

### 8.4 Ablation

`summary.json › ablation` reports 80 scenarios and 400 runs per variant. Without PEM prewhitening, **every run howled** (100%). That is the bias problem the literature predicts [2], [11]. A canceller with no notch fallback howled in 33.8% of matched runs (47.5% across all gains in that file). Freezing the model after the sound check howled in 24.2% of matched runs. A cold start with a faster step size (\(\mu = 0.05\), no sound check) howled in 20.0% of matched runs, but its tone on stable runs was worse (LSD 6.66 dB, against 4.72 dB when the model was frozen and 4.49 dB for the canceller alone). Decorrelation is essential. The fallback matters. The sound check, in this slice, looks more like a tone result than a safety result. Those comparisons should not be mixed naïvely with the 575-scenario steady table; they are a smaller, separate protocol.

## 9. The open problem: moving microphones

The closed loop's results split along one variable: whether the microphone moves. In the 575-scenario steady table, static howl is 0.7% and walking howl is 37.6%. A later snapshot reported in whitepaper v0.3, Table 7 (750 scenarios, 2,250 matched runs per processor, calculated from run shards) shows the same split and adds that even the slowest walks (0.3–0.6 m/s, 405 runs) howled in 32.6% of closed-loop runs, against 0.6% for a static microphone. Faster walking made it only somewhat worse. Notch layers degrade gracefully under motion; the canceller does not. When the walk made the feedback path at least 3 dB worse, every real processor howled in 43–67% of runs.

We think this happens for reasons that are not yet tested. The step size (\(\mu = 0.02\)) was tuned for low steady-state error, in the range the survey gives for speech [2]. In the simulator the true path changes with each 0.5 s waypoint. A 4,096-tap NLMS filter at that step size probably cannot re-converge between changes. A stale model does not just fail to cancel: it subtracts the wrong signal, and the residual can be larger than with no canceller at all. The notch fallback only sees that residual after detection (median about 100 ms). And the simulator may exaggerate the problem: paths are recomputed every 0.5 s and crossfaded, so motion appears as a sequence of path switches. Real movement is continuous. Whether that makes tracking harder or easier than in a real room is unknown.

Candidate fixes — a foreground and a background filter, adaptation control, a variable step size — are directions, not results [2]. None has been simulated yet. We make no performance claims for them.

## 10. Limitations

- **Simulated shoebox rooms only.** Image-source order is capped, so late reverb in large venues is incomplete. Crowds are a change in RT60. Moving talkers are a crossfade every 0.5 s.
- **Sweeps are partly done.** The closed-loop protocols cover less than a fifth of their planned scenarios.
- **The oracle isolation reference is a construction.** Its −10 dB leak and 4.5 ms delay are our assumptions. It bounds an idealised approach and says nothing about any product.
- **Laptop latency is unmeasured.** Converter delays in the hardware model come from datasheets; some were not re-checked for whitepaper v0.3. The embedded compute budget is an estimate.
- **Reference availability.** The closed loop assumes a clean line-level feed of what the speakers play. Setups with several speaker zones, speaker processing that changes over time, or no accessible send have not been simulated.
- **No real-room numbers yet.** Validation with working FOH engineers is planned for v1 and has not been done.

## 11. Roadmap

The roadmap has three stages. We give no dates.

| Stage | What it is | What it must prove |
|---|---|---|
| **v1: laptop prototype** | The Nullyn AI algorithms running in real time on a laptop, with a USB audio interface providing the microphone input, the optional reference input and the output | That the simulated behaviour holds in real rooms. Measure round-trip latency and the external loop-delay condition. Collect real failures for the learning loop, with working FOH engineers. |
| **v2: our own circuit** | A dedicated hardware device running the same algorithms | That the latency and compute budgets hold on embedded hardware |
| **v3: deploy at scale** | Production and wide deployment | Reliability across many venues and operators |

v1 will be judged on questions this paper cannot answer. Does the closed loop's static-microphone result carry over to real rooms? Can tracking fixes close the walking gap? How often does a real operator create the conditions of Section 8.3? What real-world label source can replace the simulation oracle?

The closed loop is a potential solution, not a proven product. The static-mic simulation is the first real sign that a speaker-feed approach can beat a notch box. Moving microphones are the problem to crack, and the simulation is only partly done. We will not treat a laptop plugin as the product: that slot is already occupied. The laptop is the test rig. The claim has to be earned in a room, against the incumbents, including on a walking microphone.

## 12. Conclusion

Nullyn AI attacks feedback with two layers. A per-sample notch bank, steered by a small learned detector (or by classic rules), is always available and never waits on a network. When a speaker-feed reference is present, a PEM-prewhitened adaptive canceller subtracts predicted feedback first, with the notch layer as fallback, and does so without adding samples of audio latency under a stated delay condition.

In simulation, that closed loop almost eliminates howl when the microphone is still (0.7% of 690 runs) and gives more than 15 dB of extra fader headroom. When the performer walks, it currently does worse than a notch filter. Those are simulated results on an unfinished sweep. The next proof is a laptop in a real room.

## References

1. G. Sigismondi, *Selection and Operation of Audio Signal Processors*, Shure Inc., 2016. https://content-files.shure.com/Pubs/asp-selection-and-operation-of-audio-signal-processors/us_pro_audiosignalprocessor_ea.pdf
2. T. van Waterschoot and M. Moonen, "Fifty Years of Acoustic Feedback Control: State of the Art and Future Challenges," *Proc. IEEE*, vol. 99, no. 2, pp. 288–327, 2011. https://doi.org/10.1109/JPROC.2010.2090998
3. T. van Waterschoot and M. Moonen, "Comparative Evaluation of Howling Detection Criteria in Notch-Filter-Based Howling Suppression," *J. Audio Eng. Soc.*, vol. 58, no. 11, pp. 923–940, 2010.
4. M. R. Schroeder, "Improvement of Acoustic-Feedback Stability by Frequency Shifting," *J. Acoust. Soc. Am.*, vol. 36, no. 9, pp. 1718–1724, 1964. https://doi.org/10.1121/1.1919270 (Cited as reported in [1].)
11. A. Spriet, I. Proudler, M. Moonen and J. Wouters, "Adaptive Feedback Cancellation in Hearing Aids With Linear Prediction of the Desired Signal," *IEEE Trans. Signal Process.*, vol. 53, no. 10, pp. 3749–3763, 2005. https://doi.org/10.1109/TSP.2005.855108
12. J. B. Allen and D. A. Berkley, "Image Method for Efficiently Simulating Small-Room Acoustics," *J. Acoust. Soc. Am.*, vol. 65, pp. 943–950, 1979. https://doi.org/10.1121/1.382599
13. R. Scheibler, E. Bezzam and I. Dokmanić, "Pyroomacoustics: A Python Package for Audio Room Simulation and Array Processing Algorithms," *Proc. IEEE ICASSP*, pp. 351–355, 2018. https://doi.org/10.1109/ICASSP.2018.8461310
14. R. F. Carreras, M. P. O'Connell, C. J. Mulhearn et al. (Bose Corporation), "Computational architecture for active noise reduction device," US Patent 12,148,413 B2, 2024. https://patents.google.com/patent/US12148413B2/en
15. M. Lester and J. Boley, "The Effects of Latency on Live Sound Monitoring," AES Convention 123, paper 7198, 2007. https://aes.org/publications/elibrary/elibrary-page/?id=14256
16. Texas Instruments, *PCM3060 24-Bit, 96/192 kHz Asynchronous Stereo Audio Codec* datasheet (Rev. B). https://www.ti.com/lit/ds/symlink/pcm3060.pdf
17. Nullyn AI, "Nullyn AI: How We Detect and Suppress Acoustic Feedback in Live Sound," whitepaper v0.3, unpublished confidential draft, 26 November 2026. Closed-loop figures in this note are from `summary.json`, generated 26 November 2026, 07:46 CT.
