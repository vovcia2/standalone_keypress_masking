"""Configuration loading.

The YAML file holds human units (ms, s, Hz). This module converts them to
sample counts once, so every other module works purely in samples.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_CONFIG_PATHS = [
    os.environ.get("KEYMASK_CONFIG", ""),
    "config.yaml",
    "/etc/keymask/config.yaml",
]

DEFAULTS: Dict[str, Any] = {
    "audio": {
        "device": "hw:1,0",
        "rate": 48000,
        "blocksize": 512,
        "input_channel": 0,
        "dtype": "int32",  # AIY Voice HAT mics are S32_LE only
        "backend": "auto",
        "ring_seconds": 1.0,
        "output_gain": 0.7,
    },
    "detector": {
        "band_hz": [2000, 10000],
        "frame_ms": 2.5,
        "onset_db": 15,
        "rise_ms": 5,
        "refractory_ms": 30,
        "background_tau_s": 0.5,
        "settle_s": 1.5,
        "blank_margin_ms": 5,
        "blank_tail_ms": 80,
    },
    "scheduler": {
        "idle_rate_hz": 0.1,
        "active_rate_hz": 7.0,
        "hold_range_s": [2.0, 5.0],
        "taper_range_s": [2.0, 4.0],
        "gain_jitter_db": 4.0,
    },
    "decoys": {"dir": "decoys", "min_clips_warn": 300, "peak_norm": 0.5},
    "calibration": {
        "loop_latency_samples": None,
        "decoy_level_at_mic_dbfs": None,
        "keystroke_level_at_mic_dbfs": None,
    },
    "hat": {"button_gpio": 23, "led_gpio": 25, "long_press_s": 3.0},
    "logging": {
        "path": "/var/log/keymask/keymask.jsonl",
        "cpu_interval_s": 10,
        "stdout": True,
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def find_config_path(explicit: Optional[str] = None) -> Optional[str]:
    candidates = [explicit] if explicit else DEFAULT_CONFIG_PATHS
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def load_raw(path: Optional[str] = None) -> Dict[str, Any]:
    found = find_config_path(path)
    data: Dict[str, Any] = {}
    if found:
        with open(found, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    return _deep_update(DEFAULTS, data)


def save_calibration(path: str, **values: Any) -> None:
    """Rewrite only the calibration block of the YAML file (preserves comments
    elsewhere as best as a line-based edit can)."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    out: List[str] = []
    in_cal = False
    seen = set()
    for line in lines:
        stripped = line.rstrip("\n")
        if stripped.startswith("calibration:"):
            in_cal = True
            out.append(line)
            continue
        if in_cal:
            if stripped and not stripped.startswith(" ") and not stripped.startswith("#"):
                # leaving the block: emit any keys not yet written
                for k, v in values.items():
                    if k not in seen:
                        out.append(f"  {k}: {_yaml_scalar(v)}\n")
                in_cal = False
                out.append(line)
                continue
            key = stripped.strip().split(":")[0]
            if key in values:
                out.append(f"  {key}: {_yaml_scalar(values[key])}\n")
                seen.add(key)
                continue
        out.append(line)
    if in_cal:
        for k, v in values.items():
            if k not in seen:
                out.append(f"  {k}: {_yaml_scalar(v)}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)


def _yaml_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


@dataclass
class DetectorParams:
    rate: int
    band_lo_hz: float
    band_hi_hz: float
    frame_samples: int
    onset_db: float
    rise_frames: int
    refractory_samples: int
    background_tau_samples: int
    settle_samples: int
    blank_margin_samples: int
    blank_tail_samples: int
    loop_latency_samples: Optional[int]


@dataclass
class SchedulerParams:
    rate: int
    idle_rate_hz: float
    active_rate_hz: float
    hold_range_samples: tuple
    taper_range_samples: tuple
    gain_jitter_db: float


def detector_params(cfg: Dict[str, Any]) -> DetectorParams:
    rate = int(cfg["audio"]["rate"])
    d = cfg["detector"]
    ms = rate / 1000.0
    frame = max(1, int(round(d["frame_ms"] * ms)))
    return DetectorParams(
        rate=rate,
        band_lo_hz=float(d["band_hz"][0]),
        band_hi_hz=float(d["band_hz"][1]),
        frame_samples=frame,
        onset_db=float(d["onset_db"]),
        rise_frames=max(1, int(round(d["rise_ms"] * ms / frame))),
        refractory_samples=int(round(d["refractory_ms"] * ms)),
        background_tau_samples=int(round(d["background_tau_s"] * rate)),
        settle_samples=int(round(d["settle_s"] * rate)),
        blank_margin_samples=int(round(d["blank_margin_ms"] * ms)),
        blank_tail_samples=int(round(d["blank_tail_ms"] * ms)),
        loop_latency_samples=cfg["calibration"].get("loop_latency_samples"),
    )


def scheduler_params(cfg: Dict[str, Any]) -> SchedulerParams:
    rate = int(cfg["audio"]["rate"])
    s = cfg["scheduler"]
    return SchedulerParams(
        rate=rate,
        idle_rate_hz=float(s["idle_rate_hz"]),
        active_rate_hz=float(s["active_rate_hz"]),
        hold_range_samples=(int(s["hold_range_s"][0] * rate), int(s["hold_range_s"][1] * rate)),
        taper_range_samples=(int(s["taper_range_s"][0] * rate), int(s["taper_range_s"][1] * rate)),
        gain_jitter_db=float(s["gain_jitter_db"]),
    )
