import numpy as np

from keymask.rng import make_rng
from keymask.scheduler import ACTIVE, IDLE, TAPER, Scheduler
from tests.conftest import RATE

BS = 512


def run(sched, n_samples, onsets=()):
    """Drive the scheduler block by block; deliver onsets at their sample times."""
    emits = []
    onsets = sorted(onsets)
    k = 0
    for t0 in range(0, n_samples, BS):
        while k < len(onsets) and onsets[k] < t0 + BS:
            sched.on_onset(onsets[k])
            k += 1
        emits.extend(sched.advance(t0, t0 + BS))
    return np.array(emits)


def test_state_machine_cycle(sched_params):
    s = Scheduler(sched_params, make_rng(1))
    trans = []
    s.on_transition.append(lambda a, b, t: trans.append((a, b, t)))
    assert s.state == IDLE
    s.on_onset(RATE)
    assert s.state == ACTIVE
    hold = s.hold_until
    assert sched_params.hold_range_samples[0] <= hold - RATE <= sched_params.hold_range_samples[1]
    # another onset extends hold
    s.on_onset(RATE + 100)
    assert s.hold_until >= hold - sched_params.hold_range_samples[1] + sched_params.hold_range_samples[0]
    run(s, 12 * RATE)
    states = [b for _, b, _ in trans]
    assert states == [ACTIVE, TAPER, IDLE]
    taper_len = trans[2][2] - trans[1][2]
    lo, hi = sched_params.taper_range_samples
    assert lo - BS <= taper_len <= hi + BS


def test_onset_during_taper_reactivates(sched_params):
    s = Scheduler(sched_params, make_rng(2))
    s.on_onset(0)
    s.hold_until = 2 * RATE
    run(s, 3 * RATE)
    assert s.state == TAPER
    s.on_onset(3 * RATE)
    assert s.state == ACTIVE and s.taper_end is None


def test_poisson_interarrival_in_active(sched_params):
    s = Scheduler(sched_params, make_rng(3))
    s.on_onset(0)
    s.hold_until = 10 ** 12  # stay ACTIVE
    emits = run(s, 600 * RATE)
    gaps = np.diff(emits) / RATE
    mean = 1.0 / sched_params.active_rate_hz
    assert abs(gaps.mean() - mean) / mean < 0.05, gaps.mean()
    # exponential: std == mean (CV = 1); also check the survival function
    assert abs(gaps.std() / gaps.mean() - 1.0) < 0.08
    for q in (0.5, 1.0, 2.0):
        frac = np.mean(gaps > q * mean)
        assert abs(frac - np.exp(-q)) < 0.03, (q, frac)
    # no quantisation to the block grid
    assert len(np.unique(emits % BS)) > 100


def test_idle_rate(sched_params):
    s = Scheduler(sched_params, make_rng(4))
    emits = run(s, 2000 * RATE)
    rate = len(emits) / 2000.0
    assert abs(rate - sched_params.idle_rate_hz) / sched_params.idle_rate_hz < 0.15, rate


def test_idle_rate_zero_emits_nothing(sched_params):
    sched_params.idle_rate_hz = 0.0
    s = Scheduler(sched_params, make_rng(5))
    assert len(run(s, 300 * RATE)) == 0


def test_decoys_independent_of_onsets(sched_params):
    """SECURITY: within ACTIVE, the delay from each decoy back to the most
    recent real onset must be uniform over the inter-onset interval, i.e. no
    peak at any particular delay. Chi-square against uniform."""
    rng = make_rng(6)
    s = Scheduler(sched_params, rng)
    # regular typing at 5 keys/s with jitter, for 10 minutes
    ioi = RATE / 5.0
    onsets = np.cumsum(rng.uniform(0.8 * ioi, 1.2 * ioi, size=3000)).astype(int) + RATE
    emits = run(s, int(onsets[-1] + RATE), onsets)
    # keep decoys that fall strictly between two onsets, express as phase in [0,1)
    idx = np.searchsorted(onsets, emits, side="right") - 1
    ok = (idx >= 0) & (idx < len(onsets) - 1)
    prev = onsets[idx[ok]]
    nxt = onsets[idx[ok] + 1]
    phase = (emits[ok] - prev) / (nxt - prev)
    assert len(phase) > 2000
    nb = 20
    h, _ = np.histogram(phase, bins=nb, range=(0, 1))
    expected = len(phase) / nb
    chi2 = float(((h - expected) ** 2 / expected).sum())
    # 19 dof: 99.9% quantile is 43.8
    assert chi2 < 43.8, (chi2, h)
    # and no bin is a spike: max/min ratio modest
    assert h.max() / h.min() < 1.6, h


def test_first_decoy_after_activation_is_memoryless(sched_params):
    """The first decoy after IDLE->ACTIVE must be Exp(active_rate) from the
    onset: no fixed/narrow response delay. Repeat many activations."""
    sched_params.idle_rate_hz = 0.0
    rng = make_rng(7)
    delays = []
    for _ in range(600):
        s = Scheduler(sched_params, rng)
        t_on = 3 * RATE + int(rng.integers(0, BS))
        emits = run(s, 6 * RATE, [t_on])
        after = emits[emits >= t_on]
        if len(after):
            delays.append((after[0] - t_on) / RATE)
    d = np.array(delays)
    mean = 1.0 / sched_params.active_rate_hz
    assert abs(d.mean() - mean) / mean < 0.1, d.mean()
    assert abs(d.std() / d.mean() - 1.0) < 0.12
    # the smallest delays are not clustered at one value
    assert np.std(d[d < 0.05]) > 0.005


def test_disabled_emits_nothing_and_ignores_onsets(sched_params):
    s = Scheduler(sched_params, make_rng(8))
    s.set_enabled(False, 0)
    emits = run(s, 30 * RATE, [RATE, 2 * RATE])
    assert len(emits) == 0 and s.state == IDLE


def test_rng_is_not_default_seeded():
    a = make_rng().random(8)
    b = make_rng().random(8)
    assert not np.array_equal(a, b)
