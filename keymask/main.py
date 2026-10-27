"""Wiring, CLI and systemd entry point.

    python -m keymask run            full masker (default; what the service runs)
    python -m keymask calibrate      measure loop latency (+ optional typing level)
    python -m keymask duplex-test    find the smallest stable blocksize
    python -m keymask detect         print onsets while you type (no decoys)
    python -m keymask detect --decoys   also play decoys; expect zero unblanked onsets
    python -m keymask play N         audition N random decoy clips
    python -m keymask info           library / config summary
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from . import __version__, calibrate, config as cfgmod
from .audio import AudioCore, BackendError, duplex_test, open_backend
from .decoys import DecoyLibrary
from .detector import OnsetDetector
from .hat import Hat
from .jsonlog import JsonLog
from .rng import make_rng
from .scheduler import Scheduler


class Masker:
    """Everything wired together around one AudioCore."""

    def __init__(self, cfg: Dict[str, Any], config_path: Optional[str], log: JsonLog,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.log = log
        a = cfg["audio"]
        self.core = AudioCore(
            rate=a["rate"], blocksize=a["blocksize"], in_channels=2, out_channels=2,
            input_channel=a["input_channel"], ring_seconds=a["ring_seconds"], output_gain=a["output_gain"],
        )
        # production always uses the CSPRNG-seeded generator; tests may inject a seeded one
        self.rng = rng if rng is not None else make_rng()
        self.lib: Optional[DecoyLibrary] = None
        self.det = OnsetDetector(cfgmod.detector_params(cfg))
        self.sched = Scheduler(cfgmod.scheduler_params(cfg), self.rng, log=log)
        self.backend = None
        self.hat: Optional[Hat] = None
        self._stop = threading.Event()
        self._decoys_on = False
        self.gain_jitter_db = float(cfg["scheduler"]["gain_jitter_db"])

    # ---- setup -------------------------------------------------------------
    def load_library(self, required: bool = True) -> None:
        d = self.cfg["decoys"]
        path = d["dir"]
        if not os.path.isdir(path) or not any(f.lower().endswith(".wav") for f in os.listdir(path)):
            if required:
                raise SystemExit(f"no decoy clips in {path!r}; run tools/record_decoys.py first")
            self.log.log("decoys", msg="no library", dir=path)
            return
        self.lib = DecoyLibrary.load(path, self.cfg["audio"]["rate"], peak_norm=d["peak_norm"])
        n = len(self.lib)
        self.log.log("decoys", count=n, skipped=len(self.lib.skipped), level_dbfs=round(self.lib.level_dbfs(), 1))
        if n < d["min_clips_warn"]:
            self.log.log("warning", msg=f"only {n} decoy clips (< {d['min_clips_warn']}); small libraries can be fingerprinted")

    def open_audio(self) -> None:
        a = self.cfg["audio"]
        self.backend = open_backend(self.core, a["device"], a["dtype"], backend=a["backend"], log=self.log)
        self.log.log("audio", device=a["device"], rate=a["rate"], blocksize=a["blocksize"],
                     backend=type(self.backend).__name__)

    def close_audio(self) -> None:
        if self.backend is not None:
            self.backend.stop()
            self.backend = None

    # ---- the per-block hook (runs in the audio thread) ---------------------
    def _block(self, mono: np.ndarray, t0: int) -> None:
        onsets = self.det.process(mono, t0)
        for o in onsets:
            if o.blanked:
                self.log.log("onset_blanked", t_samples=o.t, level=round(o.level_db, 1), bg=round(o.background_db, 1))
            else:
                self.log.log("onset", t_samples=o.t, level=round(o.level_db, 1), bg=round(o.background_db, 1), state=self.sched.state)
                self.sched.on_onset(o.t)
        if not self._decoys_on or self.lib is None:
            return
        for te in self.sched.advance(t0, t0 + len(mono)):
            clip_id, clip, gain = self.lib.pick(self.rng, self.gain_jitter_db)
            start = self.core.play(clip, gain, at=te, clip_id=clip_id)
            lo, hi = self.det.register_blank(start, len(clip))
            self.log.log("decoy", t_samples=start, clip=clip_id, gain_db=round(20 * np.log10(gain), 2), blank=[lo, hi])

    def enable_decoys(self, on: bool) -> None:
        if on and self.det.latency is None:
            raise RuntimeError("cannot enable decoys without loop latency calibration")
        self._decoys_on = on
        self.sched.set_enabled(on, self.core.t)
        self.log.log("masker", enabled=on, t_samples=self.core.t)

    # ---- calibration -------------------------------------------------------
    def run_calibration(self, measure_typing: bool = False, repeats: int = 10) -> Dict[str, object]:
        was_on = self._decoys_on
        self._decoys_on = False
        try:
            clip = calibrate.pick_calibration_clip(self.lib, self.core.rate)
            res = calibrate.measure_latency(self.core, clip, repeats=repeats,
                                            band=tuple(self.cfg["detector"]["band_hz"]), log=self.log)
            self.log.log("calibration", **{k: v for k, v in res.items() if k != "all"})
            if not res["reliable"]:
                self.log.log("warning", msg="calibration unreliable (spread > 1 block or weak correlation); check speaker level / room noise")
            self.det.set_latency(int(res["loop_latency_samples"]))
            self.cfg["calibration"]["loop_latency_samples"] = int(res["loop_latency_samples"])
            self.cfg["calibration"]["decoy_level_at_mic_dbfs"] = round(float(res["decoy_level_at_mic_dbfs"]), 1)
            values = {
                "loop_latency_samples": int(res["loop_latency_samples"]),
                "decoy_level_at_mic_dbfs": round(float(res["decoy_level_at_mic_dbfs"]), 1),
            }
            if measure_typing:
                lvl = calibrate.measure_keystroke_level(self.core, band=tuple(self.cfg["detector"]["band_hz"]),
                                                        frame_ms=self.cfg["detector"]["frame_ms"])
                values["keystroke_level_at_mic_dbfs"] = round(lvl, 1)
                self.cfg["calibration"]["keystroke_level_at_mic_dbfs"] = round(lvl, 1)
                self.log.log("calibration_typing", keystroke_level_at_mic_dbfs=round(lvl, 1))
            self._check_levels()
            if self.config_path:
                cfgmod.save_calibration(self.config_path, **values)
                self.log.log("config_saved", path=self.config_path, **values)
            return res
        finally:
            self._decoys_on = was_on

    def _check_levels(self) -> None:
        c = self.cfg["calibration"]
        d, k = c.get("decoy_level_at_mic_dbfs"), c.get("keystroke_level_at_mic_dbfs")
        if d is not None and k is not None and d > k:
            self.log.log("warning", msg=f"decoys at the mic ({d} dBFS) are louder than keystrokes ({k} dBFS): "
                                        "mic is too close to the speaker; see README placement")

    # ---- housekeeping thread -----------------------------------------------
    def _stats_loop(self) -> None:
        interval = float(self.cfg["logging"]["cpu_interval_s"])
        cpu0, wall0 = time.process_time(), time.monotonic()
        while not self._stop.wait(interval):
            cpu1, wall1 = time.process_time(), time.monotonic()
            load = (cpu1 - cpu0) / max(1e-6, wall1 - wall0)
            cpu0, wall0 = cpu1, wall1
            self.log.log("stats", t_samples=self.core.t, cpu_core_frac=round(load, 3),
                         blank_frac=round(self.det.blank_fraction(reset=True), 3),
                         xruns=self.core.xruns, voices=self.core.active_voices(),
                         max_voices=self.core.max_voices, clipped=self.core.clipped_samples,
                         state=self.sched.state, onsets=self.det.n_onsets,
                         blanked_onsets=self.det.n_blanked_onsets, decoys=self.sched.n_emitted,
                         bg_db=None if self.det.background is None else round(self.det.background, 1))
            if load > 0.5:
                self.log.log("warning", msg=f"CPU {load:.0%} of one core (> 50%)")
            if self.det.blank_fraction(reset=False) > 0.6 and self.sched.state == "ACTIVE":
                pass  # already visible in stats; threshold documented in README

    # ---- full run ----------------------------------------------------------
    def run(self) -> int:
        self.log.log("start", version=__version__, pid=os.getpid(), python=sys.version.split()[0])
        self.load_library(required=True)
        self.open_audio()
        self.core.block_handlers.append(self._block)
        h = self.cfg["hat"]
        self.hat = Hat(h["button_gpio"], h["led_gpio"], h["long_press_s"],
                       on_short_press=self._toggle, on_long_press=self._long_press)
        self.log.log("hat", available=self.hat.available)
        self.sched.on_transition.append(lambda old, new, t: self.hat.for_state(new) if self.hat else None)
        try:
            if self.cfg["calibration"].get("loop_latency_samples") is None:
                self.log.log("calibration", msg="no loop_latency_samples in config; calibrating now (keep the room quiet)")
                self.hat.set_pattern("fast")
                self.run_calibration()
            else:
                self.det.set_latency(int(self.cfg["calibration"]["loop_latency_samples"]))
            self._check_levels()
            self.hat.for_state(self.sched.state)
            if self.cfg["scheduler"]["active_rate_hz"] < 5.0:
                self.log.log("warning", msg="active_rate_hz < 5; fast typists exceed this, raise it")
            self.enable_decoys(True)
            threading.Thread(target=self._stats_loop, name="stats", daemon=True).start()
            while not self._stop.is_set():
                self._stop.wait(1.0)
                if self.backend is None:
                    break
        except Exception as e:  # noqa: BLE001
            self.log.log("error", msg=str(e), type=type(e).__name__)
            if self.hat:
                self.hat.set_pattern("fast")
                time.sleep(2.0)
            return 1
        finally:
            self.close_audio()
            if self.hat:
                self.hat.close()
            self.log.log("stop", t_samples=self.core.t)
        return 0

    def stop(self) -> None:
        self._stop.set()

    def _toggle(self) -> None:
        on = not self._decoys_on
        try:
            self.enable_decoys(on)
        except RuntimeError as e:
            self.log.log("error", msg=str(e))
            return
        if self.hat:
            self.hat.confirm()
            self.hat.for_state(self.sched.state if on else "IDLE")

    def _long_press(self) -> None:
        self.log.log("hat", msg="long press: calibrating")
        if self.hat:
            self.hat.set_pattern("fast")
        try:
            self.run_calibration()
        except Exception as e:  # noqa: BLE001
            self.log.log("error", msg=f"calibration failed: {e}")
        if self.hat:
            self.hat.for_state(self.sched.state)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make(args) -> Masker:
    path = cfgmod.find_config_path(args.config)
    cfg = cfgmod.load_raw(args.config)
    lg = cfg["logging"]
    log = JsonLog(None if args.no_logfile else lg["path"], stdout=lg["stdout"])
    if path is None:
        log.log("warning", msg="no config.yaml found; using built-in defaults (calibration cannot be saved)")
    return Masker(cfg, path, log)


def cmd_run(args) -> int:
    m = _make(args)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: m.stop())
    return m.run()


def cmd_calibrate(args) -> int:
    m = _make(args)
    m.load_library(required=False)
    m.open_audio()
    try:
        res = m.run_calibration(measure_typing=args.typing, repeats=args.repeats)
    finally:
        m.close_audio()
    print(f"\nloop latency: {res['loop_latency_samples']} samples = {res['loop_latency_ms']:.2f} ms "
          f"(spread {res['spread_samples']} samples, reliable={res['reliable']})")
    print(f"decoy level at mic: {res['decoy_level_at_mic_dbfs']:.1f} dBFS")
    k = m.cfg["calibration"].get("keystroke_level_at_mic_dbfs")
    if k is not None:
        print(f"keystroke level at mic: {k:.1f} dBFS")
    return 0 if res["reliable"] else 2


def cmd_duplex_test(args) -> int:
    cfg = cfgmod.load_raw(args.config)
    a = cfg["audio"]
    res = duplex_test(a["device"], a["rate"], a["dtype"], tuple(args.blocksizes), seconds=args.seconds, backend=a["backend"])
    ok = [r for r in res if r.get("ok") and r.get("xruns", 1) == 0]
    if ok:
        print(f"\nsmallest stable blocksize: {ok[0]['blocksize']}  -> set audio.blocksize in config.yaml")
        return 0
    print("\nno xrun-free blocksize found; try larger values or the alsa backend")
    return 2


def cmd_detect(args) -> int:
    m = _make(args)
    m.load_library(required=args.decoys)
    m.open_audio()
    m.core.block_handlers.append(m._block)
    try:
        lat = m.cfg["calibration"].get("loop_latency_samples")
        if args.decoys:
            if lat is None:
                print("no calibration; run `python -m keymask calibrate` first")
                return 2
            m.det.set_latency(int(lat))
            m.enable_decoys(True)
            m.sched.set_enabled(True, m.core.t)
            m.sched.state = "ACTIVE"
            m.sched.hold_until = None  # stay ACTIVE for the whole test
            print("Playing decoys continuously. Stay silent. Any 'onset' line below is a self-trigger (bad).")
        else:
            print("Type, talk, clap. Keystrokes and claps should print 'onset'; speech should not.")
        t_end = time.monotonic() + args.seconds
        while time.monotonic() < t_end:
            time.sleep(0.2)
        print(f"\nunblanked onsets: {m.det.n_onsets}   blanked onsets: {m.det.n_blanked_onsets}   decoys: {m.sched.n_emitted}")
        if args.decoys:
            print("PASS" if m.det.n_onsets == 0 else "FAIL: self-triggering; increase blank_tail_ms or move the mic")
            return 0 if m.det.n_onsets == 0 else 1
    finally:
        m.close_audio()
    return 0


def cmd_play(args) -> int:
    m = _make(args)
    m.load_library(required=True)
    m.open_audio()
    try:
        for i in range(args.n):
            cid, clip, gain = m.lib.pick(m.rng, m.gain_jitter_db)
            print(f"  {m.lib.names[cid]}  gain {20 * np.log10(gain):+.1f} dB  {1000 * len(clip) / m.core.rate:.0f} ms")
            m.core.play(clip, gain)
            time.sleep(0.6)
        time.sleep(0.5)
    finally:
        m.close_audio()
    return 0


def cmd_info(args) -> int:
    m = _make(args)
    m.load_library(required=False)
    print(f"keymask {__version__}; config: {m.config_path}")
    print(f"audio: {m.cfg['audio']}")
    print(f"calibration: {m.cfg['calibration']}")
    if m.lib:
        lens = [len(c) for c in m.lib.clips]
        print(f"decoys: {len(m.lib)} clips, {min(lens) * 1000 / m.lib.rate:.0f}-{max(lens) * 1000 / m.lib.rate:.0f} ms, "
              f"median level {m.lib.level_dbfs():.1f} dBFS; skipped {len(m.lib.skipped)}")
    p = cfgmod.detector_params(m.cfg)
    print(f"detector: frame {p.frame_samples} samples, rise {p.rise_frames} frames, refractory {p.refractory_samples}, "
          f"blank margin {p.blank_margin_samples} tail {p.blank_tail_samples}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="keymask", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", default=None, help="path to config.yaml")
    ap.add_argument("--no-logfile", action="store_true", help="log to stdout only")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("run").set_defaults(fn=cmd_run)
    c = sub.add_parser("calibrate")
    c.add_argument("--typing", action="store_true", help="also measure keystroke level (you will be asked to type)")
    c.add_argument("--repeats", type=int, default=10)
    c.set_defaults(fn=cmd_calibrate)
    d = sub.add_parser("duplex-test")
    d.add_argument("--blocksizes", type=int, nargs="+", default=[128, 256, 512, 1024])
    d.add_argument("--seconds", type=float, default=5.0)
    d.set_defaults(fn=cmd_duplex_test)
    e = sub.add_parser("detect")
    e.add_argument("--decoys", action="store_true", help="play decoys and verify zero self-triggers")
    e.add_argument("--seconds", type=float, default=30.0)
    e.set_defaults(fn=cmd_detect)
    pl = sub.add_parser("play")
    pl.add_argument("n", type=int, nargs="?", default=10)
    pl.set_defaults(fn=cmd_play)
    sub.add_parser("info").set_defaults(fn=cmd_info)
    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        args.fn = cmd_run
    try:
        return args.fn(args)
    except BackendError as e:
        print(f"audio backend error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
