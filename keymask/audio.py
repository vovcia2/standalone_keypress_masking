"""Full-duplex audio engine.

Design
------
`AudioCore` is pure numpy and hardware-free. It owns the single time base
(`core.t`, a monotonically increasing sample counter), a ring buffer of recent
input, and the output mixer. Backends (`SoundDeviceBackend`, `AlsaBackend`,
`FakeBackend`) only move blocks between the hardware and `core.process()`.

Everything downstream (detector, scheduler, calibration) is driven from
`core.process()` through `core.block_handlers` and talks exclusively in
sample indices. There is no wall-clock anywhere in the audio path.

Sample index convention
-----------------------
* Input sample index i is the i-th frame captured since start.
* Output sample index i is the i-th frame played since start, where a block
  of output returned from `process()` is aligned with the input block passed
  into the same call. The fixed hardware/driver offset between the two is
  what `calibrate.py` measures as `loop_latency_samples`.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

BlockHandler = Callable[[np.ndarray, int], None]  # (mono float32 block, t0)


class _Voice:
    __slots__ = ("clip", "gain", "start", "clip_id")

    def __init__(self, clip: np.ndarray, gain: float, start: int, clip_id: int) -> None:
        self.clip = clip
        self.gain = gain
        self.start = start
        self.clip_id = clip_id


class AudioCore:
    """Hardware-independent mixer + ring buffer + sample clock."""

    def __init__(
        self,
        rate: int,
        blocksize: int,
        in_channels: int = 2,
        out_channels: int = 2,
        input_channel: int = 0,
        ring_seconds: float = 1.0,
        output_gain: float = 0.7,
    ) -> None:
        self.rate = int(rate)
        self.blocksize = int(blocksize)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.input_channel = int(input_channel)
        self.output_gain = float(output_gain)

        self.t = 0  # index of the first sample of the NEXT input block
        self.ring_len = max(self.blocksize * 2, int(ring_seconds * self.rate))
        self._ring = np.zeros(self.ring_len, dtype=np.float32)
        self._ring_pos = 0

        self._voices: List[_Voice] = []
        self._lock = threading.Lock()
        self.block_handlers: List[BlockHandler] = []

        # stats
        self.blocks = 0
        self.xruns = 0
        self.max_voices = 0
        self.clipped_samples = 0

    # ---- time base -------------------------------------------------------
    @property
    def now(self) -> int:
        return self.t

    # ---- input side ------------------------------------------------------
    def _push_ring(self, mono: np.ndarray) -> None:
        n = len(mono)
        if n >= self.ring_len:
            self._ring[:] = mono[-self.ring_len :]
            self._ring_pos = 0
            return
        end = self._ring_pos + n
        if end <= self.ring_len:
            self._ring[self._ring_pos : end] = mono
        else:
            k = self.ring_len - self._ring_pos
            self._ring[self._ring_pos :] = mono[:k]
            self._ring[: n - k] = mono[k:]
        self._ring_pos = end % self.ring_len

    def recent(self, n: int) -> np.ndarray:
        """Last n input samples, oldest first."""
        n = min(int(n), self.ring_len)
        start = (self._ring_pos - n) % self.ring_len
        if start + n <= self.ring_len:
            return self._ring[start : start + n].copy()
        return np.concatenate((self._ring[start:], self._ring[: (start + n) % self.ring_len]))

    # ---- output side -----------------------------------------------------
    def play(self, clip: np.ndarray, gain: float = 1.0, at: Optional[int] = None, clip_id: int = -1) -> int:
        """Queue a mono float32 clip to start at output sample `at`.

        `at` defaults to the start of the next block. Returns the actual start
        sample so the caller can register a self-blank window.
        """
        with self._lock:
            start = self.t if at is None else max(int(at), self.t)
            self._voices.append(_Voice(np.asarray(clip, dtype=np.float32), float(gain), start, clip_id))
            self.max_voices = max(self.max_voices, len(self._voices))
        return start

    def active_voices(self) -> int:
        with self._lock:
            return len(self._voices)

    def _mix(self, t0: int, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        with self._lock:
            keep: List[_Voice] = []
            for v in self._voices:
                if v.start >= t0 + n:
                    keep.append(v)  # not started yet
                    continue
                off = max(0, v.start - t0)
                src = max(0, t0 - v.start)
                length = min(n - off, len(v.clip) - src)
                if length > 0:
                    out[off : off + length] += v.clip[src : src + length] * v.gain
                if src + length < len(v.clip):
                    keep.append(v)
            self._voices = keep
        out *= self.output_gain
        over = np.abs(out) > 1.0
        if over.any():
            self.clipped_samples += int(over.sum())
            np.clip(out, -1.0, 1.0, out=out)
        return out

    # ---- the one place where time advances ---------------------------------
    def process(self, in_block: np.ndarray) -> np.ndarray:
        """Consume one input block (frames x channels, float32 in [-1,1]) and
        return the aligned output block (frames x out_channels)."""
        if in_block.ndim == 1:
            mono = in_block.astype(np.float32, copy=False)
        else:
            mono = in_block[:, self.input_channel].astype(np.float32, copy=False)
        n = len(mono)
        t0 = self.t
        self._push_ring(mono)
        for h in self.block_handlers:
            h(mono, t0)
        out_mono = self._mix(t0, n)
        self.t = t0 + n
        self.blocks += 1
        if self.out_channels == 1:
            return out_mono[:, None]
        return np.repeat(out_mono[:, None], self.out_channels, axis=1)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

_DTYPE_SCALE = {"int16": 32768.0, "int32": 2147483648.0, "float32": 1.0}


class BackendError(RuntimeError):
    pass


class FakeBackend:
    """Drives an AudioCore from arrays. Used by tests and offline tools."""

    def __init__(self, core: AudioCore) -> None:
        self.core = core
        self.output: List[np.ndarray] = []

    def run(self, signal: np.ndarray) -> np.ndarray:
        """Feed `signal` (frames or frames x ch) block by block; return the
        concatenated mono output."""
        if signal.ndim == 1:
            signal = signal[:, None]
        n = self.core.blocksize
        outs = []
        for i in range(0, len(signal), n):
            blk = signal[i : i + n]
            if len(blk) < n:
                blk = np.concatenate((blk, np.zeros((n - len(blk), blk.shape[1]), dtype=blk.dtype)))
            outs.append(self.core.process(blk.astype(np.float32))[:, 0])
        return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)

    def run_silence(self, n_blocks: int) -> np.ndarray:
        z = np.zeros((n_blocks * self.core.blocksize, self.core.in_channels), dtype=np.float32)
        return self.run(z)


class SoundDeviceBackend:
    """PortAudio full-duplex stream, one callback for both directions."""

    def __init__(self, core: AudioCore, device: str, dtype: str = "int16") -> None:
        import sounddevice as sd  # noqa: F401 (import error surfaces here)

        self.sd = sd
        self.core = core
        self.device = self._resolve(device)
        self.dtype = dtype
        self.scale = _DTYPE_SCALE[dtype]
        self.stream = None

    def _resolve(self, device: str):
        sd = self.sd
        if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
            return int(device)
        for i, d in enumerate(sd.query_devices()):
            name = d["name"]
            if f"({device})" in name or device == name:
                return i
        for i, d in enumerate(sd.query_devices()):
            if "googlevoi" in d["name"].lower() or "voicehat" in d["name"].lower():
                return i
        # last resort: let PortAudio try the string itself
        return device

    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            self.core.xruns += 1
        x = indata.astype(np.float32) * (1.0 / self.scale)
        out = self.core.process(x)
        if self.dtype == "float32":
            outdata[:] = out
        else:
            np.multiply(out, self.scale - 1.0, out=out)
            outdata[:] = out.astype(outdata.dtype)

    def start(self) -> None:
        sd = self.sd
        self.stream = sd.Stream(
            device=(self.device, self.device),
            samplerate=self.core.rate,
            blocksize=self.core.blocksize,
            channels=(self.core.in_channels, self.core.out_channels),
            dtype=self.dtype,
            latency="low",
            callback=self._callback,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


class AlsaBackend:
    """pyalsaaudio fallback: capture thread drives the clock, playback thread
    writes what `process()` produced. Both PCMs are on the same card, so they
    share a hardware clock; the constant offset is measured by calibrate."""

    PREFILL_BLOCKS = 2

    def __init__(self, core: AudioCore, device: str, dtype: str = "int16") -> None:
        import alsaaudio

        self.alsa = alsaaudio
        self.core = core
        self.device = device
        self.dtype = dtype
        self.scale = _DTYPE_SCALE[dtype]
        self.np_dtype = {"int16": np.int16, "int32": np.int32}[dtype]
        self.fmt = {"int16": alsaaudio.PCM_FORMAT_S16_LE, "int32": alsaaudio.PCM_FORMAT_S32_LE}[dtype]
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self.cap = None
        self.pb = None

    def start(self) -> None:
        a = self.alsa
        self.cap = a.PCM(
            type=a.PCM_CAPTURE, mode=a.PCM_NORMAL, device=self.device,
            channels=self.core.in_channels, rate=self.core.rate, format=self.fmt,
            periodsize=self.core.blocksize,
        )
        self.pb = a.PCM(
            type=a.PCM_PLAYBACK, mode=a.PCM_NORMAL, device=self.device,
            channels=self.core.out_channels, rate=self.core.rate, format=self.fmt,
            periodsize=self.core.blocksize,
        )
        self._stop.clear()
        zeros = np.zeros((self.core.blocksize, self.core.out_channels), dtype=self.np_dtype)
        for _ in range(self.PREFILL_BLOCKS):
            self.pb.write(zeros.tobytes())
        tc = threading.Thread(target=self._capture_loop, name="alsa-capture", daemon=True)
        tp = threading.Thread(target=self._playback_loop, name="alsa-playback", daemon=True)
        self._threads = [tc, tp]
        tp.start()
        tc.start()

    def _capture_loop(self) -> None:
        n = self.core.blocksize
        ch = self.core.in_channels
        while not self._stop.is_set():
            length, data = self.cap.read()
            if length <= 0:
                if length < 0:
                    self.core.xruns += 1
                continue
            x = np.frombuffer(data, dtype=self.np_dtype)
            frames = len(x) // ch
            x = x.reshape(frames, ch).astype(np.float32) * (1.0 / self.scale)
            if frames != n:
                # driver returned an odd period; pad/trim so the clock stays block-aligned
                if frames < n:
                    x = np.concatenate((x, np.zeros((n - frames, ch), dtype=np.float32)))
                else:
                    x = x[:n]
            out = self.core.process(x)
            pcm = (out * (self.scale - 1.0)).astype(self.np_dtype)
            try:
                self._q.put_nowait(pcm)
            except queue.Full:
                self.core.xruns += 1

    def _playback_loop(self) -> None:
        zeros = np.zeros((self.core.blocksize, self.core.out_channels), dtype=self.np_dtype)
        while not self._stop.is_set():
            try:
                blk = self._q.get(timeout=0.05)
            except queue.Empty:
                blk = zeros
                self.core.xruns += 1
            self.pb.write(blk.tobytes())

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        for p in (self.cap, self.pb):
            if p is not None:
                try:
                    p.close()
                except Exception:
                    pass
        self.cap = self.pb = None


def open_backend(core: AudioCore, device: str, dtype: str = "int16", backend: str = "auto", probe_seconds: float = 1.0, log=None):
    """Pick a backend once, based on a real test, and return it started.

    auto: try sounddevice; if it cannot open or produces xruns during the
    probe, fall back to pyalsaaudio. The decision is logged so it can be
    recorded in the README.
    """
    def _log(msg: str, **kw) -> None:
        if log:
            log.log("backend", msg=msg, **kw)
        else:
            print("[backend]", msg, kw if kw else "")

    order = {"auto": ["sounddevice", "alsa"], "sounddevice": ["sounddevice"], "alsa": ["alsa"]}[backend]
    errors = []
    for name in order:
        try:
            if name == "sounddevice":
                be = SoundDeviceBackend(core, device, dtype)
            else:
                be = AlsaBackend(core, device, dtype)
            xr0, b0 = core.xruns, core.blocks
            be.start()
            time.sleep(probe_seconds)
            blocks = core.blocks - b0
            xruns = core.xruns - xr0
            expected = probe_seconds * core.rate / core.blocksize
            if blocks < expected * 0.5:
                be.stop()
                raise BackendError(f"{name}: only {blocks} blocks in {probe_seconds}s (expected ~{expected:.0f})")
            if backend == "auto" and xruns > 2:
                be.stop()
                raise BackendError(f"{name}: {xruns} xruns during probe")
            _log(f"using {name}", blocks=blocks, xruns=xruns, blocksize=core.blocksize)
            return be
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
            _log(f"{name} rejected", error=str(e))
    raise BackendError("no usable audio backend: " + "; ".join(errors))


def duplex_test(device: str, rate: int, dtype: str, blocksizes: Tuple[int, ...] = (128, 256, 512, 1024), seconds: float = 5.0, backend: str = "auto") -> List[dict]:
    """Measure xruns per blocksize. Used by `python -m keymask duplex-test`."""
    results = []
    for bs in blocksizes:
        core = AudioCore(rate=rate, blocksize=bs)
        try:
            be = open_backend(core, device, dtype, backend=backend, probe_seconds=0.5)
        except BackendError as e:
            results.append({"blocksize": bs, "ok": False, "error": str(e)})
            continue
        xr0, b0 = core.xruns, core.blocks
        time.sleep(seconds)
        res = {"blocksize": bs, "ok": True, "blocks": core.blocks - b0, "xruns": core.xruns - xr0,
               "backend": type(be).__name__}
        be.stop()
        results.append(res)
        print(res)
    return results
