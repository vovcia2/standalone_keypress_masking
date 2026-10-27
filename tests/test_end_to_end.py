"""Simulated milestone 5: full loop with a modelled acoustic path.

The core's output is fed back into its input after `LAT` samples (speaker ->
mic), attenuated. Typing bursts are injected as synthetic clicks. The masker
must activate within a couple of keystrokes, emit decoys, never trigger on
its own decoys, taper after typing stops, and stay IDLE in silence.
"""
import numpy as np

from keymask.decoys import DecoyLibrary, synth_click
from keymask.jsonlog import JsonLog
from keymask.main import Masker
from keymask.scheduler import ACTIVE, IDLE
from tests.conftest import RATE, make_clicks

LAT = 1500
BS = 512


def build(cfg, decoy_gain_at_mic=0.5):
    cfg["calibration"]["loop_latency_samples"] = LAT
    cfg["scheduler"]["idle_rate_hz"] = 0.0
    log = JsonLog(None, stdout=False)
    m = Masker(cfg, None, log, rng=np.random.default_rng(2024))  # deterministic for the test only
    m.core.in_channels = 1
    m.lib = DecoyLibrary([synth_click(RATE, dur_ms=60 + 5 * i, rng=np.random.default_rng(i)) for i in range(12)], RATE)
    m.det.set_latency(LAT)
    m.core.block_handlers.append(m._block)
    m.enable_decoys(True)
    return m


def simulate(m, external, decoy_gain_at_mic=0.5):
    """Feed `external` (room sound at the mic) plus the fed-back decoys."""
    line = np.zeros(len(external) + LAT + BS, dtype=np.float32)
    states = []
    for i in range(0, len(external), BS):
        blk = external[i : i + BS]
        if len(blk) < BS:
            blk = np.concatenate((blk, np.zeros(BS - len(blk), dtype=np.float32)))
        inp = blk + line[i : i + BS]
        out = m.core.process(inp[:, None])[:, 0]
        line[i + LAT : i + LAT + BS] += out * decoy_gain_at_mic
        states.append(m.sched.state)
    return states


def test_full_loop_activates_tapers_and_never_self_triggers(cfg):
    m = build(cfg)
    # 4 s silence, 6 s typing at ~5 keys/s, 15 s silence
    rng = np.random.default_rng(9)
    typing = np.cumsum(rng.uniform(0.15, 0.25, 30)) + 4.0
    typing = typing[typing < 10.0]
    x = make_clicks(RATE, typing, 25.0, amp=0.4)
    real_onsets = []
    m.det.on_onset.append(lambda o: real_onsets.append(o.t / RATE))
    states = simulate(m, x)
    t_axis = np.arange(len(states)) * BS / RATE
    first_active = t_axis[states.index(ACTIVE)]
    # activated within two keystrokes of the first one
    assert typing[0] - BS / RATE <= first_active <= typing[1] + 0.05, (first_active, typing[:3])
    # masking never stopped during the burst: no IDLE, and ACTIVE most of the time.
    # (A brief dip to TAPER is possible when > hold_min seconds of keystrokes all
    # land inside blank windows; decoys keep playing at near-active rate then.)
    burst = [s for s, t in zip(states, t_axis) if typing[0] + 0.5 < t < typing[-1]]
    assert IDLE not in burst
    assert sum(s == ACTIVE for s in burst) >= 0.7 * len(burst)
    # back to IDLE within hold(5) + taper(4) + slack after the last key
    assert states[-1] == IDLE
    last_active = t_axis[len(states) - 1 - states[::-1].index(ACTIVE)]
    assert last_active - typing[-1] <= 5.2
    # decoys were emitted at roughly the active rate during the burst
    burst_s = typing[-1] - typing[0]
    assert m.sched.n_emitted > 0.5 * cfg["scheduler"]["active_rate_hz"] * burst_s
    # SECURITY/feedback: every unblanked onset is a real keystroke (within 10 ms of one)
    assert m.det.n_onsets <= len(typing)
    for t in real_onsets:
        assert np.min(np.abs(typing - t)) < 0.010, t
    # enough real keystrokes survive the blank windows to keep the gate alive
    # (~30% expected at 7 decoys/s with 170 ms blank windows; see README)
    assert m.det.n_onsets >= 4, m.det.n_onsets
    # decoys were loud enough at the mic to have been seen if not blanked
    assert m.det.n_blanked_onsets > 0


def test_quiet_room_stays_idle(cfg):
    m = build(cfg)
    rng = np.random.default_rng(11)
    x = rng.standard_normal(60 * RATE).astype(np.float32) * 0.003
    states = simulate(m, x)
    assert set(states) == {IDLE}
    assert m.sched.n_emitted == 0 and m.det.n_onsets == 0


def test_decoys_alone_do_not_self_sustain(cfg):
    """Force ACTIVE with no typing: decoys play, loop back, and must not
    produce unblanked onsets (which would keep the masker alive forever)."""
    m = build(cfg, decoy_gain_at_mic=0.9)
    m.sched.on_onset(RATE * 2)
    m.sched.hold_until = 4 * RATE
    x = np.random.default_rng(12).standard_normal(20 * RATE).astype(np.float32) * 0.003
    states = simulate(m, x, decoy_gain_at_mic=0.9)
    assert m.sched.n_emitted > 5
    assert m.det.n_onsets == 0, m.det.n_onsets
    assert states[-1] == IDLE
