import os

import numpy as np

from keymask.decoys import synth_click, write_wav
from tests.conftest import RATE
from tools import evaluate as ev


def _scene(rate, real_times, decoy_times, seed=0):
    rng = np.random.default_rng(seed)
    dur = max(list(real_times) + list(decoy_times)) + 1.0
    x = rng.standard_normal(int(dur * rate)).astype(np.float32) * 0.003
    for i, t in enumerate(real_times):
        c = synth_click(rate, rng=np.random.default_rng(10 + i), f0=4000.0) * 0.6
        s = int(t * rate)
        x[s : s + len(c)] += c
    for i, t in enumerate(decoy_times):
        # decoys share the keystroke-like envelope but have a different spectral tilt
        c = synth_click(rate, rng=np.random.default_rng(500 + i), f0=4000.0 + 300 * rng.standard_normal()) * 0.6
        s = int(t * rate)
        x[s : s + len(c)] += c
    return x


def test_ari_and_kmeans_sanity():
    assert np.isclose(ev.adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]), 1.0)
    assert ev.adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) < 0  # anti-correlated split
    # reference value from the scikit-learn adjusted_rand_score docs
    assert np.isclose(ev.adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 2]), 0.5714, atol=1e-3)
    rng = np.random.default_rng(0)
    assert abs(ev.adjusted_rand_index(rng.integers(0, 2, 4000), rng.integers(0, 2, 4000))) < 0.03
    X = np.concatenate((np.random.default_rng(0).normal(0, 0.1, (30, 3)), np.random.default_rng(1).normal(5, 0.1, (30, 3))))
    lab = ev.kmeans2(X)
    assert np.isclose(ev.adjusted_rand_index(lab, [0] * 30 + [1] * 30), 1.0)


def test_spectral_flux_finds_clicks():
    times = [1.0, 1.25, 1.6, 2.0, 2.2]
    x = _scene(RATE, times, [])
    det = ev.spectral_flux_onsets(x, RATE)
    assert len(det) == len(times)
    assert np.all(np.abs(det - np.array(times)) < 0.015)


def test_match_and_align():
    det = np.array([1.0, 1.5, 2.0, 2.7])
    keys = np.array([101.0, 101.5, 102.0]) + 0.004
    off = ev.align(det, keys, 0.015)
    assert abs(off + 100.004) < 0.002
    m, n = ev.match(det, keys + off, 0.015)
    assert n == 3 and list(m) == [True, True, True, False]


def test_evaluate_off_vs_on(tmp_path):
    rng = np.random.default_rng(3)
    real = np.cumsum(rng.uniform(0.15, 0.3, 40)) + 1.0
    off_wav = os.path.join(tmp_path, "off.wav")
    on_wav = os.path.join(tmp_path, "on.wav")
    write_wav(off_wav, _scene(RATE, real, []), RATE)
    decoys = np.cumsum(rng.exponential(1 / 7.0, 100)) + 0.5
    decoys = decoys[decoys < real[-1] + 0.5]
    # drop decoys that collide with a real key (would merge in any detector)
    decoys = np.array([d for d in decoys if np.min(np.abs(real - d)) > 0.03])
    write_wav(on_wav, _scene(RATE, real, decoys), RATE)
    keys = real + 1234.5  # unsynchronised clock
    kp = os.path.join(tmp_path, "keys.txt")
    with open(kp, "w") as f:
        f.write("# header\n" + "\n".join(f"{k:.6f}" for k in keys))
    r_off = ev.evaluate_recording(off_wav, kp, 15.0, "off")
    r_on = ev.evaluate_recording(on_wav, kp, 15.0, "on")
    assert r_off["precision"] > 0.95 and r_off["recall"] > 0.95
    assert r_on["recall"] > 0.8
    assert r_on["precision"] < 0.5
    assert r_on["detected_onsets"] > 2 * r_on["true_keystrokes"]
    assert r_on["cluster_ari"] is not None
    assert sum(r_on["ioi_histogram_ms"].values()) == r_on["detected_onsets"] - 1
