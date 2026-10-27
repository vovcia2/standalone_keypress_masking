import numpy as np

from keymask import dsp
from tests.conftest import RATE


def _gain_at(h, f, rate):
    n = np.arange(len(h))
    return abs(np.sum(h * np.exp(-2j * np.pi * f / rate * n)))


def test_fir_bandpass_shape():
    h = dsp.fir_bandpass(RATE, 2000, 10000)
    assert abs(_gain_at(h, 5000, RATE) - 1.0) < 0.15
    assert _gain_at(h, 200, RATE) < 0.1
    assert _gain_at(h, 20000, RATE) < 0.2


def test_streaming_fir_matches_convolve():
    h = dsp.fir_bandpass(RATE, 2000, 10000)
    x = np.random.default_rng(0).standard_normal(5000).astype(np.float32)
    f = dsp.StreamingFIR(h)
    y = np.concatenate([f.process(x[i : i + 512]) for i in range(0, len(x), 512)])
    ref = np.convolve(np.concatenate((np.zeros(len(h) - 1, dtype=np.float32), x)), h, mode="valid")
    assert np.allclose(y, ref, atol=1e-4)


def test_xcorr_lag_recovers_delay():
    rng = np.random.default_rng(2)
    tpl = rng.standard_normal(2000)
    rec = np.zeros(20000)
    rec[7331 : 7331 + 2000] = tpl
    rec += 0.1 * rng.standard_normal(len(rec))
    lag, _, ratio = dsp.xcorr_lag(rec, tpl)
    assert lag == 7331
    assert ratio > 20


def test_frame_energy_db():
    e = dsp.frame_energy_db(np.ones(240, dtype=np.float32) * 0.1, 120)
    assert len(e) == 2
    assert np.allclose(e, -20.0, atol=0.01)
