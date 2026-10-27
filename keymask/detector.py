"""Keystroke onset detector with self-blanking.

Method
------
1. Band-limit the mic signal to ~2-10 kHz (33-tap linear-phase FIR; keystroke
   transients live there, speech and HVAC mostly below).
2. Short-frame energy (default 2.5 ms frames) in dB.
3. Slow background estimate: exponential tracker in the dB domain with a
   ~0.5 s time constant. Frozen while a trigger's refractory period or a
   self-blank window is active, so neither real nor decoy keystrokes pull it up.
4. Trigger when a frame is `onset_db` above background AND rose by at least
   `onset_db` within the last `rise_ms` — the fast-rise requirement is what
   rejects speech, HVAC and door slams with slow attacks. A plain level
   threshold is deliberately not used.
5. Refractory period after each trigger.

Self-blanking
-------------
The scheduler registers every decoy it emits as (start_output_sample, length).
The detector ignores triggers inside
    [start + latency - margin,  start + latency + length + tail]
in input-sample time, where `latency` is `calibration.loop_latency_samples`.
This is the primary anti-feedback mechanism; it works because capture and
playback share one clock, so the offset is a constant.

Latency: decision is made at the end of the first frame whose energy exceeds
the threshold, so algorithmic latency is <= frame_ms + FIR group delay
(0.33 ms) — well under 20 ms. Block latency (blocksize / rate) adds on top
and is a property of the audio path, not the detector.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np

from . import dsp
from .config import DetectorParams


class Onset:
    __slots__ = ("t", "level_db", "background_db", "blanked")

    def __init__(self, t: int, level_db: float, background_db: float, blanked: bool) -> None:
        self.t = t
        self.level_db = level_db
        self.background_db = background_db
        self.blanked = blanked

    def __repr__(self) -> str:
        return f"Onset(t={self.t}, level={self.level_db:.1f}dB, bg={self.background_db:.1f}dB, blanked={self.blanked})"


class OnsetDetector:
    def __init__(self, p: DetectorParams, ntaps: int = 33) -> None:
        self.p = p
        self.fir = dsp.StreamingFIR(dsp.fir_bandpass(p.rate, p.band_lo_hz, p.band_hi_hz, ntaps))
        self.frame = p.frame_samples
        self._carry = np.zeros(0, dtype=np.float32)
        self._carry_t = 0  # input-sample index of _carry[0] (post-FIR, already delay-corrected)
        self.alpha = min(1.0, self.frame / max(1, p.background_tau_samples))
        self.background: Optional[float] = None
        self._hist: Deque[float] = deque(maxlen=max(1, p.rise_frames))
        self.last_trigger_t = -10 ** 12
        self.latency: Optional[int] = p.loop_latency_samples
        self._blanks: List[Tuple[int, int]] = []  # (lo, hi) in input time, sorted by insertion
        self.on_onset: List[Callable[[Onset], None]] = []
        self.on_blanked: List[Callable[[Onset], None]] = []
        # stats
        self.frames_total = 0
        self.frames_blanked = 0
        self.n_onsets = 0
        self.n_blanked_onsets = 0
        self.settled = False

    # ---- self-blanking ---------------------------------------------------
    def set_latency(self, latency_samples: int) -> None:
        self.latency = int(latency_samples)

    def register_blank(self, start_out_sample: int, length: int) -> Tuple[int, int]:
        if self.latency is None:
            raise RuntimeError("loop latency unknown; run calibration before enabling decoys")
        lo = start_out_sample + self.latency - self.p.blank_margin_samples
        hi = start_out_sample + self.latency + length + self.p.blank_tail_samples
        self._blanks.append((lo, hi))
        return lo, hi

    def is_blanked(self, t: int) -> bool:
        # prune expired windows (they are roughly time-ordered)
        if self._blanks:
            self._blanks = [w for w in self._blanks if w[1] >= t]
            for lo, hi in self._blanks:
                if lo <= t <= hi:
                    return True
        return False

    def blank_fraction(self, reset: bool = True) -> float:
        f = self.frames_blanked / self.frames_total if self.frames_total else 0.0
        if reset:
            self.frames_total = self.frames_blanked = 0
        return f

    # ---- processing --------------------------------------------------------
    def process(self, mono: np.ndarray, t0: int) -> List[Onset]:
        """Process one input block starting at input sample t0. Returns the
        onsets that fired in this block (blanked ones included, flagged)."""
        y = self.fir.process(mono)
        if len(self._carry) == 0:
            self._carry_t = t0 - self.fir.delay
        buf = np.concatenate((self._carry, y)) if len(self._carry) else y
        nframes = len(buf) // self.frame
        used = nframes * self.frame
        frames_t0 = self._carry_t
        self._carry = buf[used:].copy()
        self._carry_t = frames_t0 + used
        if nframes == 0:
            return []
        e = dsp.frame_energy_db(buf[:used], self.frame)
        out: List[Onset] = []
        p = self.p
        for i in range(nframes):
            t = frames_t0 + i * self.frame
            ei = float(e[i])
            self.frames_total += 1
            if self.background is None:
                self.background = ei
            if t < p.settle_samples:
                self.background += self.alpha * (ei - self.background)
                self._hist.append(ei)
                continue
            self.settled = True
            blanked = self.is_blanked(t)
            if blanked:
                self.frames_blanked += 1
            in_refractory = t < self.last_trigger_t + p.refractory_samples
            fired = False
            if not in_refractory and len(self._hist) >= 1:
                rise = ei - min(self._hist)
                above = ei - self.background
                if above >= p.onset_db and rise >= p.onset_db:
                    fired = True
                    self.last_trigger_t = t
                    o = Onset(t, ei, float(self.background), blanked)
                    out.append(o)
                    if blanked:
                        self.n_blanked_onsets += 1
                        for cb in self.on_blanked:
                            cb(o)
                    else:
                        self.n_onsets += 1
                        for cb in self.on_onset:
                            cb(o)
            if not (fired or in_refractory or blanked):
                self.background += self.alpha * (ei - self.background)
            self._hist.append(ei)
        return out

    def reset(self) -> None:
        self.fir.reset()
        self._carry = np.zeros(0, dtype=np.float32)
        self.background = None
        self._hist.clear()
        self.last_trigger_t = -10 ** 12
        self._blanks = []
        self.settled = False


def detect_offline(x: np.ndarray, p: DetectorParams, blocksize: int = 512) -> List[Onset]:
    """Run the streaming detector over a whole array (tools use this so the
    offline and online paths are identical)."""
    det = OnsetDetector(p)
    onsets: List[Onset] = []
    for i in range(0, len(x), blocksize):
        onsets.extend(det.process(np.asarray(x[i : i + blocksize], dtype=np.float32), i))
    return onsets
