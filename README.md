# keymask — acoustic keystroke masking on Raspberry Pi + AIY Voice HAT

A standalone box that defeats acoustic keystroke side-channel attacks (a
microphone near the keyboard, or a call/meeting recording). While you type it
plays **decoy keystrokes recorded from your own keyboard** at random
(Poisson) times, so an attacker cannot segment real keystrokes from fake
ones or recover inter-key timing. It listens with its own mics to notice when
typing starts, and it is structurally unable to be triggered by its own
decoys (schedule-based self-blanking, not echo cancellation).

This is a **defensive** tool. Nothing in this repository recovers which keys
were pressed; the evaluation only counts onsets and clusters them.

## Contents

```
keymask/            runtime package (one process, systemd service)
  audio.py          full-duplex engine: sample clock, ring buffer, decoy mixer, backends
  detector.py       keystroke onset detector + self-blanking
  scheduler.py      Poisson decoy scheduler + IDLE/ACTIVE/TAPER gate
  decoys.py         decoy library (load, normalise, pick)
  calibrate.py      output->input loop latency + level measurement
  hat.py            HAT button + LED
  main.py           wiring, CLI, systemd entry point
  config.py, dsp.py, jsonlog.py, rng.py
tools/
  record_decoys.py  record + segment your real keystrokes into decoys/
  keylog.py         ground-truth key-press timestamps (no key identities)
  evaluate.py       offline attacker-side evaluation (dev machine)
scripts/bringup.sh  hardware bring-up on the Pi
scripts/install.sh  systemd install
systemd/keymask.service
tests/              pytest, no hardware needed
config.yaml         all tunables (human units; converted to samples at load)
CHECKLIST.md        manual hardware test checklist, one section per milestone
```

## Hardware

* Raspberry Pi (older 32-bit models fine) + Google AIY Voice HAT v1.
  * Mics: 2× ICS-43432 I2S MEMS on the mic board (stereo capture; we use the left channel).
  * Speaker: mono through the MAX98357A class-D amp.
  * Button on GPIO 23, LED on GPIO 25.
* Capture and playback are **one ALSA card sharing one clock**. keymask relies
  on this: the speaker→mic delay is a constant number of samples, measured
  once, and every decoy is blanked in the detector at exactly that offset.

### Placement (matters)

```
   [keyboard] ~~~ [mic board]           [speaker] --> room
      5-15 cm away, same desk         50-100 cm away, pointed AWAY from the mic,
      cable to the Pi                 ideally toward where an attacker mic /
                                      laptop would be
```

* Mic board close to the keyboard so real keystrokes are loud at the mic.
* Speaker further away and not facing the mic board, so decoys are *quieter*
  at our own mic than real keystrokes. `calibrate --typing` measures both and
  warns if the decoys are louder (then the detector threshold has less
  headroom and blanking must do all the work).
* Keep the speaker and mic off resonant surfaces; a mousepad under the speaker helps.
* For the attacker, decoys and real keystrokes should be comparably loud. If
  your phone-at-attacker-position recording shows decoys much quieter than
  keys, raise `audio.output_gain` or move the speaker closer to the attacker position.

## Install on the Pi

### 1. Hardware bring-up (do this first)

```bash
bash scripts/bringup.sh
```

It checks `dtoverlay=googlevoicehat-soundcard` in `/boot/config.txt` or
`/boot/firmware/config.txt` (and warns if the on-board `dtparam=audio=on`
is also enabled), finds the card via `arecord -l` / `aplay -l`, dumps the
supported hw params for both directions, picks a sample format both sides
support (the HAT's ICS-43432 mics are 32-bit only, so expect `S32_LE` →
`audio.dtype: "int32"`), runs a 5 s `arecord | aplay` full-duplex loopback
in that format, and probes the Python backends. Copy the device string,
rate and dtype it prints into `config.yaml`. **If the loopback fails with "Device or resource busy", the
driver cannot do simultaneous capture+playback; stop and report — do not work
around it.**

Then find the smallest xrun-free block size:

```bash
python3 -m keymask duplex-test
```

and put it in `audio.blocksize`. Record what you measured in the table at the end of this README.

### 2. Dependencies

```bash
sudo apt install python3-numpy python3-yaml python3-gpiozero python3-alsaaudio libportaudio2 python3-pip
pip3 install sounddevice      # add --break-system-packages on Bookworm
```

Python ≥ 3.7 is enough; no scipy, no ML libraries on the Pi.

### Audio backend

`audio.backend: auto` tries **sounddevice (PortAudio) with one full-duplex
callback** first, runs it for a second, and falls back to **pyalsaaudio with
two threads** only if PortAudio cannot open the card or xruns during the
probe. The choice is made once at start and logged (`"event":"backend"`);
write the outcome into the table below after bring-up and then pin it with
`audio.backend: sounddevice` or `alsa` so it does not change under you. Both
backends feed the same `AudioCore`, whose sample counter is the only clock.

### 3. Record your decoy library

```bash
python3 tools/record_decoys.py --seconds 120
```

Type for two minutes as instructed (all keys, varied force, space/enter/
backspace, slow key-ups). It runs the same onset detector as the live
system, cuts −5 … +90 ms around each onset, fades, rejects overlapping or
clipped segments, and writes `decoys/kNNNN.wav`. Repeat until ≥ 300 clips —
variety is a security property; a small library can be fingerprinted and
discarded by the attacker. Audition a few: `python3 -m keymask play 10`.

### 4. Calibrate

```bash
python3 -m keymask calibrate --typing
```

Quiet room. Plays a clip 10×, cross-correlates the mic signal against it,
stores the median delay as `calibration.loop_latency_samples`, the decoy
level at the mic, and (with `--typing`) your keystroke level at the mic. It
exits non-zero if the ten measurements spread by more than one block or the
correlation peak is weak. Calibration also runs automatically on first boot
if the value is missing, and on a 3 s button hold.

### 5. Verify the detector on hardware

```bash
python3 -m keymask detect            # type, talk, clap: keys and claps print 'onset', speech must not
python3 -m keymask detect --decoys   # plays decoys continuously; must end with 'PASS' (zero unblanked onsets)
```

### 6. Run / install the service

```bash
python3 -m keymask run               # foreground, JSON log to stdout
sudo bash scripts/install.sh         # /opt/keymask + systemd unit, enabled at boot
sudo systemctl start keymask && journalctl -u keymask -f
```

## How it works

### Time base
`AudioCore.t` counts input frames since the stream started. Output block *k*
is produced in the same callback as input block *k*, so output and input
share the index space up to the constant `loop_latency_samples`. Detector,
scheduler, blanking and logging all use this counter; wall-clock appears only
in log lines for humans.

### Detector (`detector.py`)
33-tap FIR band-pass 2–10 kHz → 2.5 ms frame energy in dB → slow background
(exponential, τ = 0.5 s, frozen during refractory and blank windows) →
trigger when a frame is ≥ 15 dB above background **and** rose ≥ 15 dB within
5 ms. The fast-rise condition is what rejects speech, HVAC starting up, and
other slow-attack sounds; there is no plain level threshold. Refractory 30 ms.
Algorithmic latency ≤ one frame + 0.3 ms FIR delay; block latency on top.
Triggers are hard-ignored for `settle_s` after start while the background settles.

### Self-blanking
Every decoy is registered as `(start_output_sample, length)`. The detector
ignores (but still logs, as `onset_blanked`) any trigger in
`[start + latency − 5 ms, start + latency + length + blank_tail_ms]`. Blanked
time is reported in every `stats` log line as `blank_frac`.

**Expected blanked fraction while ACTIVE.** Windows are 5 ms + clip (50–120 ms)
+ 80 ms tail ≈ 165–200 ms; at 7 decoys/s the Poisson coverage is
1 − e^(−7·0.17) ≈ **68 %** with the defaults. That is a little above the ~60 %
design target; it is fine for gating (a typist at 5 keys/s still gets ≈ 1.5
unblanked onsets per second, and the hold timer only needs one every 2–5 s).
If you want it under 60 % in a dry room, set `blank_tail_ms: 40`
(≈ 55 %) and confirm `detect --decoys` still passes. Do not reduce it below
what the room needs: a self-trigger makes the masker self-sustain.

### Scheduler (`scheduler.py`) — independence by construction
One homogeneous Poisson **candidate** stream at `active_rate_hz` runs for
the life of the process, from a CSPRNG-seeded generator (`secrets` →
`numpy.random.default_rng`). The gate only sets the *acceptance probability*
of each candidate (thinning): 1 in ACTIVE, `idle/active` in IDLE, a linear
ramp during TAPER. A real onset therefore does exactly two things — it may
flip the state, and it resets the hold timer to U(2, 5) s — and never
schedules, delays or advances a decoy. The first decoy after IDLE→ACTIVE is
Exp(active_rate) from the onset (memoryless), not a fixed response delay.
`tests/test_scheduler.py` checks the delay-since-last-onset distribution is
flat (χ² against uniform) and that inter-arrivals are exponential.

States: IDLE (background `idle_rate_hz`, default 0.1/s; set 0 to disable)
→ onset → ACTIVE (7/s) → hold expires → TAPER (ramp to idle over U(2,4) s)
→ IDLE. Onset in TAPER → ACTIVE. `active_rate_hz` must be ≥ your typing rate
(fast typing is 5–8 keys/s; the service warns below 5).

### Decoys (`decoys.py`)
Clips normalised to a common peak (`decoys.peak_norm`), then per play: random
clip (never the same twice in a row) and random gain in ±4 dB. Mixer is
clip-safe (hard limit, counted as `clipped` in stats).

### HAT (`hat.py`)
LED: off = IDLE, on = ACTIVE, slow blink = TAPER, fast blink = error /
calibrating. Button: short press toggles the masker (double-blink confirms);
3 s hold runs calibration.

## Logging

JSON lines, one object per line, to `logging.path`
(`/var/log/keymask/keymask.jsonl`, falls back to `./keymask.jsonl`) and
stdout. `t` is the audio sample index. Events: `start`, `backend`, `audio`,
`decoys`, `calibration`, `calib_sample`, `state` (old/new), `onset`,
`onset_blanked`, `decoy` (t, clip, gain_db, blank window), `masker`
(enabled), `hat`, `stats` every 10 s (`cpu_core_frac`, `blank_frac`, `xruns`,
`voices`, `clipped`, state, counters), `warning`, `error`, `stop`.

```bash
# useful one-liners
jq -c 'select(.event=="state")' /var/log/keymask/keymask.jsonl
jq -c 'select(.event=="stats") | {t,cpu_core_frac,blank_frac,xruns}' /var/log/keymask/keymask.jsonl
```

## Tests (dev machine, no hardware)

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Covers: FIR/correlation helpers; detector on synthetic clicks vs. noise,
speech-like and HVAC-like signals; blanking windows and blanked fraction;
scheduler state machine, Poisson inter-arrivals, **decoy/onset independence**,
memoryless activation, disabled mode; mixer sample accuracy; calibration
against a simulated acoustic loop; decoy library rules; the evaluator; and a
simulated end-to-end run (type → activates within two keystrokes → tapers →
IDLE; 60 s of quiet stays IDLE; forced decoys never self-sustain).
Hardware tests are manual: see `CHECKLIST.md`.

## Evaluation procedure (does the mask work?)

1. Put a phone or recorder where an attacker would be (e.g. the laptop's
   mic position, or 1 m away on the desk). Record at ≥ 44.1 kHz WAV.
2. On the typing computer run `python3 tools/keylog.py --out keys_off.txt`,
   start the recording, press SPACE three times, pause, type the shown text,
   press ESC. Save the recording as `attacker_off.wav`. Masker OFF.
3. Repeat with the masker ON → `attacker_on.wav`, `keys_on.txt`.
4. On the dev machine:

   ```bash
   python tools/evaluate.py --off attacker_off.wav --off-keys keys_off.txt \
                            --on  attacker_on.wav  --on-keys  keys_on.txt --report eval.json
   ```

   It runs a standard spectral-flux onset detector on both recordings, aligns
   the keylog to each recording by maximising matches (no clock sync needed),
   reports detected vs. true counts and precision/recall at ±15 ms, clusters
   13 MFCCs of the first 40 ms of every detected onset with k-means (k = 2)
   and reports the adjusted Rand index against real/decoy labels, and prints
   inter-onset-interval histograms.

**Pass criteria (mask ON):** recall of real keystrokes stays high (≥ 0.7 — the
mask buries keystrokes, it does not hide them), **precision < 0.5**, and
**real/decoy clustering ARI < 0.1**.

### Results

Fill in after the hardware run (milestone 7):

| | mask OFF | mask ON |
|---|---|---|
| true keystrokes | | |
| detected onsets | | |
| precision | | |
| recall | | |
| k-means ARI (real vs decoy) | | |

### Bring-up record

| item | value |
|---|---|
| Pi model / OS / Python | |
| ALSA card string | |
| rate / format | |
| smallest stable blocksize (xruns in 5 s) | |
| backend chosen (sounddevice / alsa) and why | |
| loop_latency_samples (spread) | |
| decoy level at mic / keystroke level at mic (dBFS) | |
| CPU (`cpu_core_frac` while ACTIVE) | |
| 30 min quiet-room run: state stayed IDLE? | |

## Known limitations

* **Bandwidth.** The HAT captures and plays at 48 kHz, so nothing above
  ~20 kHz is covered. Mechanical key switches emit energy above 20 kHz; an
  attacker with an ultrasonic-capable microphone is out of scope for this hardware.
* **First decoy after activation.** The gate reveals *that* typing started
  (the first decoy follows the first keystroke with an exponential delay). It
  does not reveal subsequent keystroke timing. A non-zero `idle_rate_hz`
  blurs even this a little.
* **Blanked keystrokes.** Real keystrokes inside a blank window are not
  used for gating (they are still masked for the attacker). At the defaults
  ~30 % of keystrokes gate the state machine, which is plenty. Occasionally
  more than `hold_range_s[0]` seconds of keystrokes all land in blank
  windows and the gate dips to TAPER mid-sentence; decoys continue at
  near-active rate and the next unblanked keystroke restores ACTIVE. Widen
  `hold_range_s` (e.g. `[3.0, 6.0]`) if you see this often in the log.
* **No echo canceller.** Schedule-based blanking is the primary and only
  anti-feedback mechanism. AEC could be added later as a secondary layer.
* **Not a substitute for a quiet speaker placement.** If decoys at our mic
  are louder than keystrokes, the detector has less headroom; calibration warns.

## Things this deliberately does not do

* React to a keystroke with a decoy at a fixed delay.
* Use broadband noise instead of recorded keystrokes (call-software noise
  suppression removes noise and keeps transients).
* Use wall-clock time for any audio decision.
* Install ML dependencies on the Pi (evaluation runs on the dev machine).
* Build any classifier that recovers key identities.
