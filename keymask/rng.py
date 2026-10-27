"""Cryptographically seeded random source shared by scheduler and decoys.

Security requirement: decoy timing and clip choice must not be predictable.
We seed numpy's Generator from os.urandom (via secrets) rather than relying
on `random` with a default or reproducible seed. Tests may pass an explicit
seed for reproducibility; production code never does.
"""
from __future__ import annotations

import secrets
from typing import Optional

import numpy as np


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    if seed is None:
        seed = secrets.randbits(128)
    return np.random.default_rng(seed)
