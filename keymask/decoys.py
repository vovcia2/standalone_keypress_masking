"""Decoy sample library.

Clips are mono 16-bit WAVs, one keystroke each, 50-120 ms, recorded from the
user's own keyboard by tools/record_decoys.py. Variety is a security property:
a small library lets an attacker fingerprint and discard the decoys, so we
warn below `min_clips_warn` and never play the same clip twice in a row.
"""
from __future__ import annotations

import os
import struct
import wave
from typing import List, Optional, Tuple

import numpy as np


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Return (float32 mono samples in [-1,1], rate). stdlib only."""
    with wave.open(path, "rb") as w:
        ch, sw, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        x = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int8).astype(np.int32) << 16)).astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"{path}: unsupported sample width {sw}")
    if ch > 1:
        x = x.reshape(-1, ch)[:, 0]
    return x, rate


def write_wav(path: str, x: np.ndarray, rate: int) -> None:
    """Write float32 [-1,1] mono as 16-bit PCM."""
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm.tobytes())


class DecoyLibrary:
    def __init__(self, clips: List[np.ndarray], rate: int, names: Optional[List[str]] = None, peak_norm: float = 0.5) -> None:
        if not clips:
            raise ValueError("decoy library is empty")
        self.rate = int(rate)
        self.peak_norm = float(peak_norm)
        self.names = names or [f"clip{i}" for i in range(len(clips))]
        self.clips: List[np.ndarray] = []
        for c in clips:
            c = np.asarray(c, dtype=np.float32)
            peak = float(np.max(np.abs(c))) if len(c) else 0.0
            if peak > 0:
                c = c * (self.peak_norm / peak)
            self.clips.append(c)
        self._last = -1
        self.max_len = max(len(c) for c in self.clips)

    @classmethod
    def load(cls, directory: str, rate: int, peak_norm: float = 0.5, min_ms: float = 20.0, max_ms: float = 200.0) -> "DecoyLibrary":
        clips, names, skipped = [], [], []
        for fn in sorted(os.listdir(directory)):
            if not fn.lower().endswith(".wav"):
                continue
            p = os.path.join(directory, fn)
            x, r = read_wav(p)
            if r != rate:
                skipped.append((fn, f"rate {r} != {rate}"))
                continue
            dur_ms = 1000.0 * len(x) / r
            if not (min_ms <= dur_ms <= max_ms):
                skipped.append((fn, f"duration {dur_ms:.0f} ms"))
                continue
            clips.append(x)
            names.append(fn)
        lib = cls(clips, rate, names, peak_norm)
        lib.skipped = skipped  # type: ignore[attr-defined]
        return lib

    def __len__(self) -> int:
        return len(self.clips)

    def pick(self, rng: np.random.Generator, gain_jitter_db: float = 4.0) -> Tuple[int, np.ndarray, float]:
        """Random clip (never the same twice in a row) and random gain in
        +/- gain_jitter_db. Returns (clip_id, samples, linear_gain)."""
        n = len(self.clips)
        if n == 1:
            idx = 0
        else:
            idx = int(rng.integers(0, n - 1))
            if idx >= self._last:
                idx += 1  # skip the previous clip without biasing the others
        self._last = idx
        db = float(rng.uniform(-gain_jitter_db, gain_jitter_db))
        return idx, self.clips[idx], 10.0 ** (db / 20.0)

    def level_dbfs(self) -> float:
        """Median RMS level of the normalised clips, in dBFS (for README/calibration)."""
        rms = [float(np.sqrt(np.mean(c * c))) for c in self.clips if len(c)]
        return 20.0 * np.log10(max(np.median(rms), 1e-9))


def synth_click(rate: int, dur_ms: float = 80.0, rng: Optional[np.random.Generator] = None, f0: float = 4000.0) -> np.ndarray:
    """A synthetic keystroke-like transient for tests and for calibration when
    no library exists yet: sharp attack, exponentially decaying band-limited burst."""
    rng = rng or np.random.default_rng(0)
    n = int(rate * dur_ms / 1000.0)
    t = np.arange(n) / rate
    env = np.exp(-t / 0.008)
    noise = rng.standard_normal(n)
    tone = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * f0 * 1.7 * t)
    x = env * (0.6 * tone + 0.4 * noise)
    x[: max(1, n // 200)] *= np.linspace(0, 1, max(1, n // 200))
    return (x / np.max(np.abs(x))).astype(np.float32) * 0.5
