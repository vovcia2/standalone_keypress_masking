"""Structured JSON-lines logger.

Every record carries the audio sample counter (`t_samples`) when one is
available so events can be correlated precisely with the audio stream.
Wall-clock is included for humans only; nothing reads it back.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Optional, TextIO


class JsonLog:
    def __init__(self, path: Optional[str], stdout: bool = True) -> None:
        self._lock = threading.Lock()
        self._fh: Optional[TextIO] = None
        self._stdout = stdout
        self.path = None
        if path:
            try:
                d = os.path.dirname(path)
                if d:
                    os.makedirs(d, exist_ok=True)
                self._fh = open(path, "a", encoding="utf-8")
                self.path = path
            except OSError:
                fallback = os.path.join(os.getcwd(), "keymask.jsonl")
                try:
                    self._fh = open(fallback, "a", encoding="utf-8")
                    self.path = fallback
                except OSError:
                    self._fh = None

    def log(self, event: str, t_samples: Optional[int] = None, **fields: Any) -> None:
        rec = {"ts": round(time.time(), 3), "event": event}
        if t_samples is not None:
            rec["t"] = int(t_samples)
        rec.update(fields)
        line = json.dumps(rec, separators=(",", ":"), default=_default)
        with self._lock:
            if self._fh:
                self._fh.write(line + "\n")
                self._fh.flush()
            if self._stdout:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None


def _default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
    except ImportError:
        pass
    return str(o)
