#!/usr/bin/env python3
"""Record and segment the user's real keystrokes into the decoy library.

Run on the Pi (uses the HAT mics):
    python3 tools/record_decoys.py --seconds 120
Or segment an existing recording (dev machine, no hardware):
    python3 tools/record_decoys.py --input typing.wav

Runs the SAME onset detector as the live system, cuts -5 ms .. +90 ms around
each onset, fades in/out, rejects overlapping or clipped segments, and writes
numbered 16-bit mono WAVs into decoys/. Repeat passes until >= 300 clips.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from keymask import config as cfgmod  # noqa: E402
from keymask import dsp  # noqa: E402
from keymask.decoys import read_wav, write_wav  # noqa: E402
from keymask.detector import detect_offline  # noqa: E402

INSTRUCTIONS = """
Decoy recording pass ({seconds:.0f} s). The library must cover everything you
actually do on this keyboard, because an attacker can discard decoys that
do not look like your real keystrokes.

  * Type real sentences at your normal speed for the first half.
  * Then cover: every letter row, number row, SPACE (lots), ENTER, BACKSPACE,
    SHIFT, TAB, arrow keys, modifier combos.
  * Vary force: some gentle, some hard, some held and released slowly
    (key-UP clicks matter too).
  * Keep the room otherwise quiet. Do not talk.

Recording starts in 3 seconds...
"""


def segment(x: np.ndarray, rate: int, p, pre_ms: float = 5.0, post_ms: float = 90.0,
            clip_thresh: float = 0.99, fade_in_ms: float = 1.0, fade_out_ms: float = 5.0):
    """Return (clips, rejected_reasons)."""
    onsets = [o.t for o in detect_offline(x, p)]
    pre, post = int(rate * pre_ms / 1000), int(rate * post_ms / 1000)
    fi, fo = int(rate * fade_in_ms / 1000), int(rate * fade_out_ms / 1000)
    clips, reasons = [], {"overlap": 0, "clipped": 0, "edge": 0, "quiet": 0}
    for i, t in enumerate(onsets):
        a, b = t - pre, t + post
        if a < 0 or b > len(x):
            reasons["edge"] += 1
            continue
        prev_ok = i == 0 or onsets[i - 1] < a
        next_ok = i == len(onsets) - 1 or onsets[i + 1] > b
        if not (prev_ok and next_ok):
            reasons["overlap"] += 1
            continue
        seg = x[a:b]
        if np.max(np.abs(seg)) >= clip_thresh:
            reasons["clipped"] += 1
            continue
        if dsp.rms_dbfs(seg) < -60:
            reasons["quiet"] += 1
            continue
        clips.append(dsp.fade(seg, fi, fo))
    return clips, reasons, len(onsets)


def record_live(cfg, seconds: float) -> np.ndarray:
    from keymask.audio import AudioCore, open_backend

    a = cfg["audio"]
    core = AudioCore(rate=a["rate"], blocksize=a["blocksize"], input_channel=a["input_channel"], ring_seconds=0.5)
    chunks = []
    core.block_handlers.append(lambda mono, t0: chunks.append(mono.copy()))
    be = open_backend(core, a["device"], a["dtype"], backend=a["backend"])
    try:
        print(INSTRUCTIONS.format(seconds=seconds))
        time.sleep(3.0)
        chunks.clear()
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            remaining = t_end - time.monotonic()
            print(f"\r  recording... {remaining:5.0f} s left ", end="", flush=True)
            time.sleep(1.0)
        print("\r  done.                         ")
    finally:
        be.stop()
    return np.concatenate(chunks)


def next_index(out_dir: str) -> int:
    n = 0
    for f in os.listdir(out_dir):
        if f.startswith("k") and f.endswith(".wav"):
            try:
                n = max(n, int(f[1:-4]) + 1)
            except ValueError:
                pass
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", default=None)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--input", help="segment this WAV instead of recording")
    ap.add_argument("--out", default=None, help="output directory (default: decoys.dir from config)")
    ap.add_argument("--save-raw", help="also save the raw recording to this WAV")
    ap.add_argument("--target", type=int, default=300)
    args = ap.parse_args()

    cfg = cfgmod.load_raw(args.config)
    out_dir = args.out or cfg["decoys"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    rate = int(cfg["audio"]["rate"])

    if args.input:
        x, r = read_wav(args.input)
        if r != rate:
            print(f"warning: {args.input} is {r} Hz but config rate is {rate}; clips will be written at {r} Hz")
            rate = r
            cfg["audio"]["rate"] = r
    else:
        x = record_live(cfg, args.seconds)
        if args.save_raw:
            write_wav(args.save_raw, x, rate)

    p = cfgmod.detector_params(cfg)
    p.loop_latency_samples = None
    clips, reasons, n_onsets = segment(x, rate, p)
    start = next_index(out_dir)
    for i, c in enumerate(clips):
        write_wav(os.path.join(out_dir, f"k{start + i:04d}.wav"), c, rate)
    total = len([f for f in os.listdir(out_dir) if f.endswith(".wav")])
    print(f"\nonsets detected: {n_onsets}; clips written: {len(clips)}; rejected: {reasons}")
    print(f"library now has {total} clips in {out_dir}")
    if total < args.target:
        print(f"That is fewer than {args.target}. Run another pass (same command) and cover the keys you missed.")
        return 1
    print("Library is large enough. Listen to a few: python3 -m keymask play 10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
