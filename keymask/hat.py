"""AIY Voice HAT v1 button (GPIO 23) and LED (GPIO 25).

LED:   off = idle, on = active, slow blink = taper, fast blink = error,
       double-blink = confirmation of a button toggle.
Button: short press toggles the masker on/off; long press (3 s) runs
        calibration.

Uses gpiozero when available. On a machine without GPIO (dev box, tests)
`Hat` degrades to a no-op that still records the requested LED pattern.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional


class Hat:
    def __init__(self, button_gpio: int = 23, led_gpio: int = 25, long_press_s: float = 3.0,
                 on_short_press: Optional[Callable[[], None]] = None,
                 on_long_press: Optional[Callable[[], None]] = None) -> None:
        self.on_short_press = on_short_press
        self.on_long_press = on_long_press
        self.pattern = "off"
        self._held = False
        self._lock = threading.Lock()
        self.available = False
        try:
            from gpiozero import LED, Button  # type: ignore

            self.led = LED(led_gpio)
            self.button = Button(button_gpio, pull_up=True, hold_time=long_press_s, bounce_time=0.05)
            self.button.when_held = self._on_held
            self.button.when_released = self._on_released
            self.available = True
        except Exception:  # noqa: BLE001 - no GPIO on this machine
            self.led = None
            self.button = None

    # ---- button -----------------------------------------------------------
    def _on_held(self) -> None:
        self._held = True
        if self.on_long_press:
            threading.Thread(target=self.on_long_press, name="hat-long-press", daemon=True).start()

    def _on_released(self) -> None:
        if self._held:
            self._held = False
            return
        if self.on_short_press:
            threading.Thread(target=self.on_short_press, name="hat-short-press", daemon=True).start()

    # ---- LED ---------------------------------------------------------------
    def set_pattern(self, pattern: str) -> None:
        """pattern in {off, on, slow, fast}."""
        with self._lock:
            self.pattern = pattern
            if not self.led:
                return
            if pattern == "off":
                self.led.off()
            elif pattern == "on":
                self.led.on()
            elif pattern == "slow":
                self.led.blink(on_time=0.5, off_time=0.5, background=True)
            elif pattern == "fast":
                self.led.blink(on_time=0.1, off_time=0.1, background=True)

    def for_state(self, state: str) -> None:
        self.set_pattern({"IDLE": "off", "ACTIVE": "on", "TAPER": "slow"}.get(state, "fast"))

    def confirm(self) -> None:
        """Double blink, then restore the previous pattern."""
        prev = self.pattern
        if not self.led:
            return

        def _run() -> None:
            with self._lock:
                self.led.blink(on_time=0.1, off_time=0.1, n=2, background=False)
            self.set_pattern(prev)

        threading.Thread(target=_run, name="hat-confirm", daemon=True).start()

    def close(self) -> None:
        if self.led:
            try:
                self.led.off()
                self.led.close()
            except Exception:  # noqa: BLE001
                pass
        if self.button:
            try:
                self.button.close()
            except Exception:  # noqa: BLE001
                pass
        time.sleep(0)
