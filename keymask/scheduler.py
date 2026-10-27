"""Poisson decoy scheduler + gate state machine.

Security design
---------------
Decoy emission times must be statistically independent of real keystroke
times. We achieve this structurally rather than by care:

* One homogeneous Poisson *candidate* process runs at `active_rate_hz` for
  the whole lifetime of the scheduler, drawn from a CSPRNG-seeded generator.
  Its arrival times never depend on anything the detector reports.
* The gate state only sets the *acceptance probability* of each candidate
  (thinning): 1.0 in ACTIVE, `idle_rate/active_rate` in IDLE, and a linear
  ramp between the two during TAPER. Thinning a Poisson process with a
  rate function that is independent of the process gives a Poisson process
  with that rate function.
* A real onset therefore does exactly two things: it may change the gate
  state, and it resets the hold timer. It never schedules, delays or
  advances a decoy. The first decoy after an IDLE->ACTIVE transition is
  Exp(active_rate) distributed from the transition (memoryless), the same
  as if the process had been running all along — there is no fixed or
  narrowly distributed "response delay".

States
------
IDLE   : background decoys at idle_rate (0 disables). Onset -> ACTIVE.
ACTIVE : decoys at active_rate. Every onset resets hold timer to U(hold_range).
         Hold expiry -> TAPER.
TAPER  : rate ramps linearly to idle_rate over U(taper_range), then IDLE.
         Onset -> ACTIVE.

All times are audio sample indices.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from .config import SchedulerParams

IDLE, ACTIVE, TAPER = "IDLE", "ACTIVE", "TAPER"


class Scheduler:
    def __init__(self, p: SchedulerParams, rng: np.random.Generator, log=None) -> None:
        if p.active_rate_hz <= 0:
            raise ValueError("active_rate_hz must be > 0")
        if p.idle_rate_hz > p.active_rate_hz:
            raise ValueError("idle_rate_hz must not exceed active_rate_hz")
        self.p = p
        self.rng = rng
        self.log = log
        self.state = IDLE
        self.enabled = True
        self.hold_until: Optional[int] = None
        self.taper_start: Optional[int] = None
        self.taper_end: Optional[int] = None
        self._next_candidate: int = self._draw_gap(0)
        self.on_transition: List[Callable[[str, str, int], None]] = []
        self.n_emitted = 0
        self.n_candidates = 0

    # ---- randomness --------------------------------------------------------
    def _draw_gap(self, t: int) -> int:
        """Next candidate time: t + Exp(mean = rate/active_rate samples)."""
        mean = self.p.rate / self.p.active_rate_hz
        return t + max(1, int(round(self.rng.exponential(mean))))

    def _draw_range(self, lo_hi: Tuple[int, int]) -> int:
        lo, hi = lo_hi
        return int(self.rng.integers(lo, max(lo + 1, hi + 1)))

    # ---- state -----------------------------------------------------------
    def _set_state(self, new: str, t: int) -> None:
        if new == self.state:
            return
        old = self.state
        self.state = new
        if self.log:
            self.log.log("state", t_samples=t, old=old, new=new,
                         hold_until=self.hold_until, taper_end=self.taper_end)
        for cb in self.on_transition:
            cb(old, new, t)

    def on_onset(self, t: int) -> None:
        """A real (non-blanked) keystroke onset at input sample t."""
        if not self.enabled:
            return
        self.hold_until = t + self._draw_range(self.p.hold_range_samples)
        if self.state != ACTIVE:
            self.taper_start = self.taper_end = None
            self._set_state(ACTIVE, t)

    def set_enabled(self, enabled: bool, t: int) -> None:
        self.enabled = bool(enabled)
        if not enabled:
            self.hold_until = None
            self.taper_start = self.taper_end = None
            self._set_state(IDLE, t)

    def rate_at(self, t: int) -> float:
        """Instantaneous decoy rate (Hz) at sample t given the current state."""
        if not self.enabled:
            return 0.0
        if self.state == ACTIVE:
            return self.p.active_rate_hz
        if self.state == TAPER and self.taper_start is not None and self.taper_end is not None:
            span = max(1, self.taper_end - self.taper_start)
            frac = min(1.0, max(0.0, (t - self.taper_start) / span))
            return self.p.active_rate_hz + (self.p.idle_rate_hz - self.p.active_rate_hz) * frac
        return self.p.idle_rate_hz

    def _update_timers(self, t: int) -> None:
        if self.state == ACTIVE and self.hold_until is not None and t >= self.hold_until:
            self.taper_start = self.hold_until
            self.taper_end = self.taper_start + self._draw_range(self.p.taper_range_samples)
            self.hold_until = None
            self._set_state(TAPER, t)
        if self.state == TAPER and self.taper_end is not None and t >= self.taper_end:
            self.taper_start = self.taper_end = None
            self._set_state(IDLE, t)

    # ---- the per-block step ------------------------------------------------
    def advance(self, t0: int, t1: int) -> List[int]:
        """Return the sample times in [t0, t1) at which to emit decoys."""
        emits: List[int] = []
        self._update_timers(t0)
        while self._next_candidate < t1:
            tc = self._next_candidate
            self.n_candidates += 1
            self._update_timers(tc)
            accept_p = self.rate_at(tc) / self.p.active_rate_hz
            if accept_p >= 1.0 or (accept_p > 0.0 and self.rng.random() < accept_p):
                emits.append(max(tc, t0))
                self.n_emitted += 1
            self._next_candidate = self._draw_gap(tc)
        return emits

    def status(self) -> dict:
        return {"state": self.state, "enabled": self.enabled, "hold_until": self.hold_until,
                "taper_end": self.taper_end, "emitted": self.n_emitted}
