"""Output -> input loop latency calibration.

Plays a known clip at a known output sample index, records the microphone,
cross-correlates the (band-limited) recording against the clip and reads
off the delay in samples. Repeated `repeats` times; the median is stored in
config.yaml as `calibration.loop_latency_samples`.

Also measures:
* decoy level at the mic (dBFS) — should sit below real keystroke level.
* optionally the user's keystroke level at the mic (dBFS) if asked to type.

Run with the room quiet.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from . import dsp
from .audio import AudioCore
from .decoys import DecoyLibrary, synth_click


class _Recorder:
    """Block handler that captures input between two sample indices."""

    def __init__(self) -> None:
        self.chunks: List[np.ndarray] = []
        self.t_start: Optional[int] = None
        self.t_end: Optional[int] = None
        self.armed = False

    def arm(self, t_start: int, t_end: int) -> None:
        self.chunks = []
        self.t_start, self.t_end = t_start, t_end
        self.armed = True

    def __call__(self, mono: np.ndarray, t0: int) -> None:
        if not self.armed:
            return
        t1 = t0 + len(mono)
        if t1 <= self.t_start:
            return
        if t0 >= self.t_end:
            self.armed = False
            return
        a = max(0, self.t_start - t0)
        b = min(len(mono), self.t_end - t0)
        self.chunks.append(mono[a:b].copy())

    def wait(self, core: AudioCore, timeout_s: float = 5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while self.armed and time.monotonic() < deadline:
            time.sleep(0.01)
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.float32)


def measure_latency(core: AudioCore, clip: np.ndarray, repeats: int = 10, max_latency_s: float = 0.5,
                    band=(2000.0, 10000.0), gap_s: float = 0.4, log=None) -> Dict[str, object]:
    """Returns dict with median latency, all measurements, confidence ratios and
    decoy level at the mic. `core` must already be running on a backend."""
    rate = core.rate
    taps = dsp.fir_bandpass(rate, band[0], band[1])
    tpl = np.convolve(clip, taps, mode="same")
    rec = _Recorder()
    core.block_handlers.append(rec)
    lags, ratios, levels = [], [], []
    try:
        max_lag = int(max_latency_s * rate)
        for i in range(repeats):
            # start a few blocks ahead so the recorder is armed before the clip plays
            start = core.t + 4 * core.blocksize
            rec.arm(start - core.blocksize, start + max_lag + len(clip) + core.blocksize)
            actual = core.play(clip, gain=1.0, at=start, clip_id=-2)
            x = rec.wait(core, timeout_s=max_latency_s + 3.0)
            if len(x) < len(clip) + max_lag:
                if log:
                    log.log("calib_short_record", got=len(x))
                continue
            xf = np.convolve(x, taps, mode="same")
            lag, peak, ratio = dsp.xcorr_lag(xf, tpl, max_lag=max_lag + core.blocksize)
            latency = (rec.t_start + lag) - actual
            arrival = x[lag : lag + len(clip)]
            lvl = dsp.rms_dbfs(arrival)
            lags.append(int(latency))
            ratios.append(float(ratio))
            levels.append(lvl)
            if log:
                log.log("calib_sample", t_samples=actual, latency=int(latency), ratio=round(ratio, 1), level_dbfs=round(lvl, 1))
            else:
                print(f"  run {i + 1}/{repeats}: latency {latency} samples ({1000.0 * latency / rate:.2f} ms), peak/median {ratio:.0f}, level {lvl:.1f} dBFS")
            time.sleep(gap_s)
    finally:
        core.block_handlers.remove(rec)
    if not lags:
        raise RuntimeError("calibration captured nothing; is the stream running?")
    med = int(np.median(lags))
    spread = int(np.max(lags) - np.min(lags))
    return {
        "loop_latency_samples": med,
        "loop_latency_ms": 1000.0 * med / rate,
        "all": lags,
        "spread_samples": spread,
        "min_peak_ratio": float(np.min(ratios)),
        "decoy_level_at_mic_dbfs": float(np.median(levels)),
        "reliable": spread <= core.blocksize and float(np.min(ratios)) > 8.0,
    }


def measure_keystroke_level(core: AudioCore, seconds: float = 6.0, band=(2000.0, 10000.0), frame_ms: float = 2.5) -> float:
    """Ask the user to type; return the 95th-percentile band-limited frame
    level in dBFS (robust 'typical keystroke peak' estimate)."""
    rate = core.rate
    rec = _Recorder()
    core.block_handlers.append(rec)
    try:
        start = core.t + 2 * core.blocksize
        rec.arm(start, start + int(seconds * rate))
        print(f"  Type normally for {seconds:.0f} seconds...")
        x = rec.wait(core, timeout_s=seconds + 3.0)
    finally:
        core.block_handlers.remove(rec)
    taps = dsp.fir_bandpass(rate, band[0], band[1])
    xf = np.convolve(x, taps, mode="same")
    e = dsp.frame_energy_db(xf, max(1, int(rate * frame_ms / 1000.0)))
    return float(np.percentile(e, 95)) if len(e) else -120.0


def pick_calibration_clip(lib: Optional[DecoyLibrary], rate: int) -> np.ndarray:
    if lib is not None and len(lib) > 0:
        # longest clip gives the sharpest correlation peak
        return max(lib.clips, key=len)
    return synth_click(rate)
