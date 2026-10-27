import threading
import time

import numpy as np

from keymask.audio import AudioCore
from keymask.calibrate import measure_latency
from keymask.decoys import synth_click
from tests.conftest import RATE


class SimulatedLoop:
    """Acoustic path simulator: the core's output reaches its input after
    `latency` samples, attenuated and with noise. Runs in a thread, faster
    than real time."""

    def __init__(self, core, latency, atten=0.3, noise=0.003, seed=0):
        self.core, self.latency, self.atten, self.noise = core, latency, atten, noise
        self.rng = np.random.default_rng(seed)
        self.line = np.zeros(latency + core.blocksize * 4, dtype=np.float32)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        n = self.core.blocksize
        while not self._stop.is_set():
            inp = self.line[:n] + self.rng.standard_normal(n).astype(np.float32) * self.noise
            out = self.core.process(inp[:, None])[:, 0] * self.atten
            # output sample s (absolute index = line[0] + s) arrives at input s + latency
            self.line[self.latency : self.latency + n] += out
            self.line = np.concatenate((self.line[n:], np.zeros(n, dtype=np.float32)))
            time.sleep(0.0005)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.thread.join(1.0)


def test_measure_latency_simulated():
    core = AudioCore(RATE, 512, in_channels=1, out_channels=1, output_gain=1.0)
    L = 1234
    clip = synth_click(RATE)
    with SimulatedLoop(core, L):
        res = measure_latency(core, clip, repeats=5, gap_s=0.01)
    assert res["loop_latency_samples"] == L, res
    assert res["reliable"]
    assert res["spread_samples"] == 0
    assert -40 < res["decoy_level_at_mic_dbfs"] < 0
