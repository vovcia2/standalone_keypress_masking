"""Small numpy-only DSP helpers shared by detector, calibration and tools."""
from __future__ import annotations

from typing import Optional

import numpy as np


def fir_bandpass(rate: int, lo_hz: float, hi_hz: float, ntaps: int = 33) -> np.ndarray:
    """Windowed-sinc (Hamming) linear-phase bandpass. Odd `ntaps`."""
    if ntaps % 2 == 0:
        ntaps += 1
    n = np.arange(ntaps) - (ntaps - 1) / 2.0
    hi = min(hi_hz, rate / 2.0 * 0.98)

    def lowpass(fc: float) -> np.ndarray:
        wc = 2.0 * fc / rate
        h = wc * np.sinc(wc * n)
        return h

    h = lowpass(hi) - lowpass(lo_hz)
    h *= np.hamming(ntaps)
    # normalise passband gain to ~1 at band centre
    fc = 0.5 * (lo_hz + hi)
    g = np.abs(np.sum(h * np.exp(-2j * np.pi * fc / rate * np.arange(ntaps))))
    return (h / max(g, 1e-9)).astype(np.float32)


class StreamingFIR:
    """Block-wise FIR with carried state (`same` alignment: output sample k
    corresponds to input sample k minus (ntaps-1)/2 group delay)."""

    def __init__(self, taps: np.ndarray) -> None:
        self.taps = np.asarray(taps, dtype=np.float32)
        self._hist = np.zeros(len(self.taps) - 1, dtype=np.float32)

    @property
    def delay(self) -> int:
        return (len(self.taps) - 1) // 2

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        buf = np.concatenate((self._hist, x))
        y = np.convolve(buf, self.taps, mode="valid")
        if len(self.taps) > 1:
            self._hist = buf[-(len(self.taps) - 1) :]
        return y.astype(np.float32, copy=False)

    def reset(self) -> None:
        self._hist[:] = 0


def frame_energy_db(x: np.ndarray, frame: int) -> np.ndarray:
    """Mean-square energy per non-overlapping frame, in dB (floor -120)."""
    n = (len(x) // frame) * frame
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    f = x[:n].reshape(-1, frame)
    e = np.mean(f * f, axis=1)
    return (10.0 * np.log10(np.maximum(e, 1e-12))).astype(np.float32)


def xcorr_lag(recording: np.ndarray, template: np.ndarray, max_lag: Optional[int] = None) -> tuple:
    """Lag (in samples) at which `template` best matches `recording`, via FFT
    cross-correlation. Returns (lag, peak_value, peak_to_median_ratio)."""
    rec = np.asarray(recording, dtype=np.float64)
    tpl = np.asarray(template, dtype=np.float64)
    n = len(rec) + len(tpl) - 1
    nfft = 1 << (n - 1).bit_length()
    R = np.fft.rfft(rec, nfft)
    T = np.fft.rfft(tpl, nfft)
    cc = np.fft.irfft(R * np.conj(T), nfft)[: len(rec)]
    if max_lag is not None:
        cc = cc[: max_lag + 1]
    lag = int(np.argmax(cc))
    peak = float(cc[lag])
    med = float(np.median(np.abs(cc))) + 1e-12
    return lag, peak, peak / med


def rms_dbfs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return -120.0
    return float(20.0 * np.log10(max(np.sqrt(np.mean(x * x)), 1e-6)))


def fade(x: np.ndarray, fade_in: int, fade_out: int) -> np.ndarray:
    y = np.array(x, dtype=np.float32, copy=True)
    fi = min(fade_in, len(y))
    fo = min(fade_out, len(y))
    if fi > 0:
        y[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 0:
        y[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return y
