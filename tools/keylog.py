#!/usr/bin/env python3
"""Ground-truth keystroke TIMESTAMP logger for the evaluation.

Run this on the typing computer, type the evaluation text INTO THIS TERMINAL,
and it writes one timestamp (seconds, high-resolution monotonic) per key
press. Key identities are deliberately NOT recorded. It does not install a
global hook; it only sees keys typed into its own terminal window.

    python3 tools/keylog.py --out keys_on.txt
    (type the known text; press ESC to finish)

Alignment with the attacker recording is done by tools/evaluate.py by
searching the offset that maximises matches, so no clock sync is needed.
For a robust lock, start by pressing SPACE three times quickly, pause one
second, then type the text.
"""
from __future__ import annotations

import argparse
import sys
import time

TEXT = ("the quick brown fox jumps over the lazy dog. "
        "pack my box with five dozen liquor jugs. "
        "how vexingly quick daft zebras jump! "
        "sphinx of black quartz, judge my vow. ")


def _getch_factory():
    if sys.platform.startswith("win"):
        import msvcrt

        def getch() -> str:
            c = msvcrt.getwch()
            if c in ("\x00", "\xe0"):  # arrow/function key prefix: consume second code
                msvcrt.getwch()
            return c

        return getch, lambda: None, lambda: None
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    def enter() -> None:
        tty.setcbreak(fd)

    def leave() -> None:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def getch() -> str:
        return sys.stdin.read(1)

    return getch, enter, leave


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default=TEXT)
    args = ap.parse_args()
    getch, enter, leave = _getch_factory()
    print("Type the following text (ESC when done). Only timestamps are recorded.\n")
    print(args.text + "\n")
    stamps = []
    enter()
    try:
        while True:
            c = getch()
            t = time.perf_counter()
            if c == "\x1b":
                break
            stamps.append(t)
            sys.stdout.write("." if c not in ("\r", "\n") else "\n")
            sys.stdout.flush()
    finally:
        leave()
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# keymask keylog: one key-press timestamp (s, perf_counter) per line; no key identities\n")
        for t in stamps:
            f.write(f"{t:.6f}\n")
    print(f"\n{len(stamps)} key presses written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
