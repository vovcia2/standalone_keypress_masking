import numpy as np

from keymask.audio import AudioCore, FakeBackend
from tests.conftest import RATE


def test_sample_counter_and_ring():
    core = AudioCore(RATE, 512, ring_seconds=0.1)
    fb = FakeBackend(core)
    x = np.arange(4096, dtype=np.float32) / 4096
    fb.run(x)
    assert core.t == 4096
    assert np.allclose(core.recent(300), x[-300:])


def test_mix_at_exact_sample_offset_and_overlap():
    core = AudioCore(RATE, 256, output_gain=1.0, out_channels=1)
    fb = FakeBackend(core)
    clip = np.ones(100, dtype=np.float32) * 0.25
    core.play(clip, gain=1.0, at=300)
    core.play(clip, gain=2.0, at=350)
    out = fb.run_silence(4)
    assert out[299] == 0 and out[300] == 0.25
    assert np.isclose(out[349], 0.25) and np.isclose(out[350], 0.75)
    assert np.isclose(out[399], 0.75) and np.isclose(out[400], 0.5)
    assert out[450] == 0
    assert core.active_voices() == 0


def test_play_in_the_past_starts_now():
    core = AudioCore(RATE, 256, output_gain=1.0)
    fb = FakeBackend(core)
    fb.run_silence(2)
    start = core.play(np.ones(10, dtype=np.float32), at=5)
    assert start == 512


def test_clip_safe():
    core = AudioCore(RATE, 256, output_gain=1.0, out_channels=1)
    fb = FakeBackend(core)
    for _ in range(5):
        core.play(np.ones(256, dtype=np.float32) * 0.9, at=0)
    out = fb.run_silence(1)
    assert np.max(np.abs(out)) <= 1.0 and core.clipped_samples == 256


def test_block_handlers_see_mono_and_t0():
    core = AudioCore(RATE, 128, in_channels=2, input_channel=1)
    seen = []
    core.block_handlers.append(lambda mono, t0: seen.append((t0, float(mono[0]))))
    x = np.zeros((256, 2), dtype=np.float32)
    x[:, 1] = 0.5
    FakeBackend(core).run(x)
    assert seen == [(0, 0.5), (128, 0.5)]
