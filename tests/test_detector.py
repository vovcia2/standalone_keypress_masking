import numpy as np

from keymask.detector import OnsetDetector, detect_offline
from tests.conftest import RATE, make_clicks


def test_detects_synthetic_clicks(det_params):
    times = [2.0, 2.3, 2.45, 2.9, 3.6, 4.1]
    x = make_clicks(RATE, times, 5.0)
    onsets = detect_offline(x, det_params)
    assert len(onsets) == len(times)
    for o, t in zip(onsets, times):
        assert abs(o.t / RATE - t) < 0.005, (o, t)
        assert not o.blanked


def test_ignores_noise_and_slow_rises(det_params):
    rng = np.random.default_rng(5)
    dur = 6.0
    n = int(dur * RATE)
    t = np.arange(n) / RATE
    noise = rng.standard_normal(n).astype(np.float32) * 0.002
    # speech-like: broadband burst with 30 ms attack, 4 Hz syllable AM, 20 dB above floor
    env = 0.5 * (1 + np.sin(2 * np.pi * 4 * t)) * (t > 2.0)
    env = np.convolve(env, np.ones(int(0.03 * RATE)) / int(0.03 * RATE), mode="same")
    speech = rng.standard_normal(n).astype(np.float32) * 0.02 * env
    # HVAC: loud low-frequency hum that switches on with a 200 ms ramp
    hum = 0.2 * np.sin(2 * np.pi * 120 * t) * np.clip((t - 4.0) / 0.2, 0, 1)
    x = (noise + speech + hum).astype(np.float32)
    onsets = detect_offline(x, det_params)
    assert onsets == [], onsets


def test_refractory_merges_double_hits(det_params):
    # two transients 10 ms apart -> one onset
    x = make_clicks(RATE, [2.0, 2.010], 3.0)
    assert len(detect_offline(x, det_params)) == 1


def test_background_settle_blocks_early_triggers(det_params):
    x = make_clicks(RATE, [0.5], 3.0)  # before settle_s = 1.5
    assert detect_offline(x, det_params) == []


def test_blanking_window(det_params):
    det = OnsetDetector(det_params)
    lat = det_params.loop_latency_samples
    clip_len = int(0.08 * RATE)
    # decoy emitted at output sample s0 -> expected at input s0+lat
    s0 = int(2.5 * RATE)
    lo, hi = det.register_blank(s0, clip_len)
    assert lo == s0 + lat - det_params.blank_margin_samples
    assert hi == s0 + lat + clip_len + det_params.blank_tail_samples
    inside = (s0 + lat + 10) / RATE
    tail = (s0 + lat + clip_len + det_params.blank_tail_samples - 50) / RATE
    outside = (s0 + lat + clip_len + det_params.blank_tail_samples + int(0.1 * RATE)) / RATE
    x = make_clicks(RATE, [2.0, inside, tail, outside], 4.0)
    onsets = []
    for i in range(0, len(x), 512):
        onsets.extend(det.process(x[i : i + 512], i))
    flags = [o.blanked for o in onsets]
    assert len(onsets) == 4
    assert flags == [False, True, True, False]
    assert det.n_onsets == 2 and det.n_blanked_onsets == 2
    assert 0 < det.blank_fraction(reset=False) < 0.2


def test_blank_fraction_at_configured_rate(det_params):
    """At 7 decoys/s with 80 ms clips, 5 ms margin and 80 ms tail each window
    is 165 ms, so Poisson coverage is 1 - exp(-7 * 0.165) = 68 % while ACTIVE.
    The README documents this and the knob (blank_tail_ms) to bring it under 60 %."""
    det = OnsetDetector(det_params)
    clip_len = int(0.08 * RATE)
    rng = np.random.default_rng(0)
    t = 2 * RATE
    end = 62 * RATE
    while t < end:
        det.register_blank(t, clip_len)
        t += int(rng.exponential(RATE / 7.0))
    x = np.zeros(end, dtype=np.float32) + 1e-4
    for i in range(0, len(x), 512):
        det.process(x[i : i + 512], i)
    frac = det.blank_fraction()
    assert 0.6 < frac < 0.75, frac
    # with the shorter tail recommended in the README the figure drops below 60 %
    det_params.blank_tail_samples = int(0.040 * RATE)
    det2 = OnsetDetector(det_params)
    rng = np.random.default_rng(0)
    t = 2 * RATE
    while t < end:
        det2.register_blank(t, clip_len)
        t += int(rng.exponential(RATE / 7.0))
    for i in range(0, len(x), 512):
        det2.process(x[i : i + 512], i)
    assert det2.blank_fraction() < 0.6


def test_register_blank_requires_latency(det_params):
    det_params.loop_latency_samples = None
    det = OnsetDetector(det_params)
    try:
        det.register_blank(0, 100)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError without calibration")
