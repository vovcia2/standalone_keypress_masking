import os

import numpy as np

from keymask.decoys import DecoyLibrary, read_wav, synth_click, write_wav
from keymask.rng import make_rng
from tests.conftest import RATE


def test_wav_roundtrip(tmp_path):
    x = synth_click(RATE)
    p = os.path.join(tmp_path, "a.wav")
    write_wav(p, x, RATE)
    y, r = read_wav(p)
    assert r == RATE and len(y) == len(x)
    assert np.max(np.abs(x - y)) < 1e-3


def test_library_load_and_filter(tmp_path):
    for i in range(5):
        write_wav(os.path.join(tmp_path, f"k{i:04d}.wav"), synth_click(RATE, dur_ms=60 + 10 * i), RATE)
    write_wav(os.path.join(tmp_path, "long.wav"), np.zeros(RATE), RATE)  # 1 s -> rejected
    write_wav(os.path.join(tmp_path, "rate.wav"), synth_click(44100), 44100)  # wrong rate
    lib = DecoyLibrary.load(str(tmp_path), RATE, peak_norm=0.5)
    assert len(lib) == 5 and len(lib.skipped) == 2
    for c in lib.clips:
        assert np.isclose(np.max(np.abs(c)), 0.5)


def test_pick_never_repeats_and_gain_in_range():
    lib = DecoyLibrary([synth_click(RATE, rng=np.random.default_rng(i)) for i in range(6)], RATE)
    rng = make_rng(0)
    last = None
    counts = np.zeros(6)
    for _ in range(3000):
        cid, clip, g = lib.pick(rng, 4.0)
        assert cid != last
        assert 10 ** (-4 / 20) <= g <= 10 ** (4 / 20)
        counts[cid] += 1
        last = cid
    # roughly uniform usage
    assert counts.min() > 0.8 * counts.max()
