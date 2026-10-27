# keymask hardware checklist

Manual tests, in milestone order. Do not move to the next milestone until the
current one passes on the real hardware. Record numbers in the README tables.

## M1 — Bring-up and full-duplex proof

- [ ] `dtoverlay=googlevoicehat-soundcard` present in `/boot/config.txt` or `/boot/firmware/config.txt`; rebooted after change.
- [ ] `bash scripts/bringup.sh` finds the card in both `arecord -l` and `aplay -l`. Card string: `hw:__,0`
- [ ] hw params dumped for capture and playback; chosen rate ______ and format ______ written to `config.yaml`.
- [ ] 5 s `arecord | aplay` loopback runs with no "busy"/error (tap the mic, hear it). **If it fails: stop and report.**
- [ ] `python3 -m keymask duplex-test` run; smallest blocksize with 0 xruns in 5 s: ______ → `audio.blocksize`.
- [ ] Backend chosen by `auto` (see `"event":"backend"` log line): ______; pinned in `config.yaml`.

## M2 — audio.py + calibrate.py

- [ ] `python3 -m keymask calibrate` (quiet room) completes 10 runs.
- [ ] `loop_latency_samples` = ______ (= ______ ms), spread ≤ 1 block, `reliable: true`.
- [ ] Rerun once; value within one block of the first run.
- [ ] `calibration.loop_latency_samples` now written into `config.yaml`.
- [ ] Decoy level at mic ______ dBFS.

## M3 — decoys.py + tools/record_decoys.py

- [ ] `python3 tools/record_decoys.py --seconds 120`; onsets detected ≈ keys typed (within ±20 %).
- [ ] Rejected counts (overlap / clipped / edge / quiet) are small; if "clipped" is high, lower mic gain or type softer.
- [ ] Library ≥ 300 clips after N passes. N = ____, total = ____.
- [ ] `python3 -m keymask play 10`: clips sound like single keystrokes, no clicks at clip edges, no cut-off tails.
- [ ] `python3 -m keymask info` shows clip duration range inside 50–120 ms.

## M4 — detector.py standalone

- [ ] `python3 -m keymask detect --seconds 60`:
  - [ ] every keystroke prints an `onset` line (count 20 keys → ______ onsets).
  - [ ] talking normally for 20 s prints **no** onsets.
  - [ ] a single hand clap prints exactly one onset.
  - [ ] HVAC / fan running: no onsets.
- [ ] `python3 -m keymask calibrate --typing`: keystroke level ______ dBFS > decoy level ______ dBFS. If not, move the speaker further away / turn it away from the mic.
- [ ] `python3 -m keymask detect --decoys --seconds 60` ends with `PASS` (0 unblanked onsets while decoys play continuously). If FAIL: raise `blank_tail_ms` in 20 ms steps and retest; note the final value: ______.

## M5 — scheduler.py + main.py full loop

- [ ] `python3 -m keymask run` in the foreground.
- [ ] Type: `state IDLE→ACTIVE` appears within one or two keystrokes; decoys audible immediately.
- [ ] Keep typing 30 s: state stays ACTIVE (no flapping to TAPER).
- [ ] Stop typing: `ACTIVE→TAPER` after 2–5 s, `TAPER→IDLE` 2–4 s later; decoys thin out, then stop (apart from the idle rate).
- [ ] Decoys never restart on their own after IDLE (listen 2 min).
- [ ] `stats` lines: `blank_frac` while ACTIVE ______ (expect ≈ 0.6–0.7 at defaults); `xruns` stays 0; `clipped` stays small.
- [ ] `cpu_core_frac` while ACTIVE ______ (must be < 0.5). If higher: increase blocksize, or check that numpy uses the system BLAS.
- [ ] **30-minute quiet-room soak**: leave running, nobody typing. Log must show no `state` transitions and `onsets` counter unchanged. Onsets at end: ______.
- [ ] Short button press: `masker enabled:false`, LED double-blink, decoys stop even while typing. Press again: resumes.

## M6 — hat.py, systemd, README

- [ ] LED: off in IDLE, on in ACTIVE, slow blink in TAPER, fast blink during calibration.
- [ ] 3 s button hold triggers calibration (LED fast blink, then back to state pattern); new value saved to `config.yaml`.
- [ ] `sudo bash scripts/install.sh`; `systemctl status keymask` active; survives reboot (`systemctl is-enabled keymask`).
- [ ] With `loop_latency_samples: null`, first boot calibrates automatically (watch `journalctl -u keymask -f`).
- [ ] Log file at `/var/log/keymask/keymask.jsonl` growing; rotates or is pruned by your policy (not provided).
- [ ] README bring-up table filled in.

## M7 — Evaluation with an attacker mic

- [ ] Phone/recorder placed at attacker position (photo it; note the distance ______ cm).
- [ ] `keys_off.txt` + `attacker_off.wav` recorded (masker OFF); `keys_on.txt` + `attacker_on.wav` (masker ON). Same text, same placement.
- [ ] `python tools/evaluate.py ...` on the dev machine; `keylog_offset_s` sensible (recording started after the keylog → negative offset of a few seconds, or vice-versa).
- [ ] Mask OFF: precision and recall both > 0.9 (sanity: detector and alignment work).
- [ ] Mask ON: recall ≥ 0.7, **precision < 0.5**, **ARI < 0.1**. Numbers copied into README.
- [ ] If precision is too high: decoys too quiet at the attacker position (raise `output_gain`, move speaker) or library too small / too uniform.
- [ ] If ARI ≥ 0.1: decoys are distinguishable by timbre — re-record the library in the same placement, include key-ups, widen `gain_jitter_db`, and check the speaker is not colouring the sound (try a different speaker position).
