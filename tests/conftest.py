import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from keymask import config as cfgmod  # noqa: E402
from keymask.decoys import synth_click  # noqa: E402

RATE = 48000


@pytest.fixture
def cfg():
    c = cfgmod.load_raw(os.path.join(ROOT, "config.yaml"))
    c["audio"]["rate"] = RATE
    c["audio"]["blocksize"] = 512
    return c


@pytest.fixture
def det_params(cfg):
    p = cfgmod.detector_params(cfg)
    p.loop_latency_samples = 1000
    return p


@pytest.fixture
def sched_params(cfg):
    return cfgmod.scheduler_params(cfg)


def make_clicks(rate, times_s, dur_s, amp=0.3, noise_rms=0.002, seed=1, click_kw=None):
    """White-noise floor plus synthetic keystroke transients at `times_s`."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(dur_s * rate)).astype(np.float32) * noise_rms
    for i, t in enumerate(times_s):
        c = synth_click(rate, rng=np.random.default_rng(100 + i), **(click_kw or {}))
        c = c / np.max(np.abs(c)) * amp
        s = int(t * rate)
        n = min(len(c), len(x) - s)
        if n > 0:
            x[s : s + n] += c[:n]
    return x
