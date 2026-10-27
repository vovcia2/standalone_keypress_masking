"""keymask: acoustic keystroke masking for Raspberry Pi + AIY Voice HAT.

Defensive tool. Plays decoy keystrokes (recorded from the user's own
keyboard) on a Poisson schedule while the user types so that an acoustic
side-channel attacker cannot segment real keystrokes or recover timing.
"""

__version__ = "0.1.0"
