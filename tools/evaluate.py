#!/usr/bin/env python3
"""Offline evaluation: can an attacker still count / segment real keystrokes?

Runs on the DEV MACHINE (numpy only). Inputs are recordings made from an
attacker-position microphone (phone/recorder) while the user typed a known
text with the masker OFF and ON, plus the ground-truth timestamp logs from
tools/keylog.py.

    python tools/evaluate.py --off attacker_off.wav --off-keys keys_off.txt \
                             --on  attacker_on.wav  --on-keys  keys_on.txt  \
                             --report eval.json

Per recording it reports
  1. detected onsets (standard spectral-flux onset detector) vs true count;
     precision / recall of matching detections to true keystrokes within
     +/- 15 ms (after aligning the keylog to the recording by maximising
     matches, since the two clocks are not synchronised),
  2. k-means (k=2) over 13 MFCCs of the first 40 ms of each detected onset,
     and the adjusted Rand index of the clusters vs real/decoy labels,
  3. the inter-onset-interval histogram.

Pass criteria (README): with the mask ON, recall of real keystrokes stays
high (the mask does not hide them, it buries them) but precision < 0.5 and
real/decoy clustering ARI < 0.1.

No classifier that recovers key identities is built here, by design.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from keymask.decoys import read_wav  # noqa: E402

# ---------------------------------------------------------------------------
# Standard onset detector: spectral flux + adaptive peak picking
# ---------------------------------------------------------------------------

def stft_mag(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    n = 1 + max(0, (len(x) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(win)[None, :]
    return np.abs(np.fft.rfft(frames, axis=1))


def spectral_flux_onsets(x: np.ndarray, rate: int, win_ms: float = 10.0, hop_ms: float = 2.0,
                         lo_hz: float = 1000.0, delta: float = 0.08, median_ms: float = 100.0,
                         min_gap_ms: float = 20.0) -> np.ndarray:
    """Onset times (seconds). Classic Dixon-style: half-wave-rectified log
    spectral flux, normalised, picked where it exceeds a local median plus
    delta and is a local maximum."""
    win = int(rate * win_ms / 1000)
    hop = int(rate * hop_ms / 1000)
    mag = stft_mag(x, win, hop)
    freqs = np.fft.rfftfreq(win, 1.0 / rate)
    mag = mag[:, freqs >= lo_hz]
    lm = np.log1p(100.0 * mag)
    flux = np.sum(np.maximum(lm[1:] - lm[:-1], 0.0), axis=1)
    flux = np.concatenate(([0.0], flux))
    if flux.max() > 0:
        flux = flux / flux.max()
    m = max(1, int(median_ms / hop_ms))
    pad = np.pad(flux, m, mode="edge")
    med = np.array([np.median(pad[i : i + 2 * m + 1]) for i in range(len(flux))])
    thr = med + delta
    gap = max(1, int(min_gap_ms / hop_ms))
    onsets = []
    last = -gap
    for i in range(1, len(flux) - 1):
        if flux[i] > thr[i] and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1] and i - last >= gap:
            onsets.append(i)
            last = i
    return (np.array(onsets, dtype=float) * hop + win / 2.0) / rate


# ---------------------------------------------------------------------------
# Matching and alignment
# ---------------------------------------------------------------------------

def match(det: np.ndarray, truth: np.ndarray, tol: float) -> Tuple[np.ndarray, int]:
    """Greedy one-to-one matching. Returns (bool mask over det: matched, n_truth_matched)."""
    matched_det = np.zeros(len(det), dtype=bool)
    used = np.zeros(len(truth), dtype=bool)
    j = 0
    for i, t in enumerate(det):
        while j < len(truth) and truth[j] < t - tol:
            j += 1
        k = j
        while k < len(truth) and truth[k] <= t + tol:
            if not used[k]:
                used[k] = True
                matched_det[i] = True
                break
            k += 1
    return matched_det, int(used.sum())


def align(det: np.ndarray, keys: np.ndarray, tol: float, search_s: Optional[float] = None, step: float = 0.001) -> float:
    """Offset o such that keys + o best matches det (coarse-to-fine). The
    keylog clock (perf_counter on the typing computer) and the recording are
    unrelated, so the offset can be anything; `search_s` optionally bounds it."""
    if len(det) == 0 or len(keys) == 0:
        return 0.0
    # candidate offsets: every det minus every key is expensive; use the
    # first/last few keys against all detections to seed, then refine.
    seeds = set()
    for k in np.concatenate((keys[:8], keys[-8:])):
        for d in det:
            seeds.add(round(float(d - k), 3))
    if search_s is not None:
        seeds = [s for s in seeds if abs(s) <= search_s]
    best, best_n = 0.0, -1
    for o in seeds:
        _, n = match(det, keys + o, tol)
        if n > best_n:
            best, best_n = o, n
    # fine refinement
    for o in np.arange(best - 0.02, best + 0.02, step):
        _, n = match(det, keys + o, tol)
        if n > best_n:
            best, best_n = float(o), n
    return best


# ---------------------------------------------------------------------------
# Features + clustering
# ---------------------------------------------------------------------------

def mel_filterbank(rate: int, nfft: int, nbands: int = 26, lo: float = 300.0, hi: Optional[float] = None) -> np.ndarray:
    hi = hi or rate / 2.0
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
    imel = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)  # noqa: E731
    pts = imel(np.linspace(mel(lo), mel(hi), nbands + 2))
    bins = np.floor((nfft + 1) * pts / rate).astype(int)
    fb = np.zeros((nbands, nfft // 2 + 1))
    for b in range(nbands):
        l, c, r = bins[b], bins[b + 1], bins[b + 2]
        if c == l:
            c += 1
        if r == c:
            r += 1
        fb[b, l:c] = (np.arange(l, c) - l) / (c - l)
        fb[b, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


def mfcc13(seg: np.ndarray, rate: int, nfft: int = 1024, win_ms: float = 10.0, hop_ms: float = 5.0) -> np.ndarray:
    """Mean of 13 MFCCs over short frames of the segment (first 40 ms)."""
    win = int(rate * win_ms / 1000)
    hop = int(rate * hop_ms / 1000)
    if len(seg) < win:
        seg = np.pad(seg, (0, win - len(seg)))
    mag = stft_mag(seg, win, hop)
    spec = np.zeros((mag.shape[0], nfft // 2 + 1))
    spec[:, : mag.shape[1]] = mag ** 2
    fb = mel_filterbank(rate, nfft)
    logmel = np.log(spec @ fb.T + 1e-10)
    n = logmel.shape[1]
    k = np.arange(13)[:, None]
    dct = np.cos(np.pi * k * (np.arange(n)[None, :] + 0.5) / n)
    return (logmel @ dct.T).mean(axis=0)


def kmeans2(X: np.ndarray, restarts: int = 10, iters: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-9)
    best_lab, best_inertia = None, np.inf
    for _ in range(restarts):
        c = Xn[rng.choice(len(Xn), 2, replace=False)]
        for _ in range(iters):
            d = ((Xn[:, None, :] - c[None, :, :]) ** 2).sum(-1)
            lab = d.argmin(1)
            new = np.array([Xn[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(2)])
            if np.allclose(new, c):
                break
            c = new
        inertia = ((Xn - c[lab]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia, best_lab = inertia, lab
    return best_lab


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    ua, ub = np.unique(a), np.unique(b)
    M = np.array([[np.sum((a == x) & (b == y)) for y in ub] for x in ua], dtype=float)
    comb = lambda n: n * (n - 1) / 2.0  # noqa: E731
    sum_ij = comb(M).sum()
    sum_a = comb(M.sum(1)).sum()
    sum_b = comb(M.sum(0)).sum()
    n = comb(M.sum())
    expected = sum_a * sum_b / n if n else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    return float((sum_ij - expected) / denom) if denom else 0.0


# ---------------------------------------------------------------------------

def ioi_histogram(onsets_s: np.ndarray, bins_ms=(0, 50, 100, 150, 200, 300, 400, 600, 1000, 2000)) -> Dict[str, int]:
    if len(onsets_s) < 2:
        return {}
    ioi = np.diff(onsets_s) * 1000.0
    h, _ = np.histogram(ioi, bins=list(bins_ms) + [np.inf])
    labels = [f"{bins_ms[i]}-{bins_ms[i + 1]}" for i in range(len(bins_ms) - 1)] + [f">{bins_ms[-1]}"]
    return dict(zip(labels, [int(v) for v in h]))


def load_keys(path: str) -> np.ndarray:
    ts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ts.append(float(line))
    return np.array(ts)


def evaluate_recording(wav: str, keys_path: str, tol_ms: float, label: str, offset: Optional[float] = None) -> Dict[str, object]:
    x, rate = read_wav(wav)
    x = x - x.mean()
    keys = load_keys(keys_path)
    det = spectral_flux_onsets(x, rate)
    tol = tol_ms / 1000.0
    off = align(det, keys, tol) if offset is None else offset
    truth = keys + off
    matched, n_truth = match(det, truth, tol)
    precision = float(matched.sum() / len(det)) if len(det) else 0.0
    recall = float(n_truth / len(truth)) if len(truth) else 0.0
    # features for clustering: first 40 ms after each detected onset
    seglen = int(0.040 * rate)
    feats, labs = [], []
    for t, m in zip(det, matched):
        i = int(t * rate)
        if i + seglen <= len(x):
            feats.append(mfcc13(x[i : i + seglen], rate))
            labs.append(1 if m else 0)
    ari = None
    if len(feats) >= 4 and len(set(labs)) == 2:
        X = np.array(feats)
        cl = kmeans2(X)
        ari = adjusted_rand_index(np.array(labs), cl)
    return {
        "label": label, "file": wav, "rate": rate, "duration_s": round(len(x) / rate, 1),
        "true_keystrokes": int(len(truth)), "detected_onsets": int(len(det)),
        "keylog_offset_s": round(float(off), 4), "tolerance_ms": tol_ms,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "cluster_ari": None if ari is None else round(float(ari), 3),
        "ioi_histogram_ms": ioi_histogram(det),
    }


def print_report(r: Dict[str, object]) -> None:
    print(f"\n== {r['label']}  ({r['file']}, {r['duration_s']} s @ {r['rate']} Hz)")
    print(f"   true keystrokes: {r['true_keystrokes']}   detected onsets: {r['detected_onsets']}   keylog offset {r['keylog_offset_s']} s")
    print(f"   precision {r['precision']:.3f}   recall {r['recall']:.3f}   (tolerance +/-{r['tolerance_ms']} ms)")
    print(f"   real/decoy k-means ARI: {r['cluster_ari']}")
    print("   inter-onset intervals (ms):")
    h = r["ioi_histogram_ms"]
    mx = max(h.values()) if h else 1
    for k, v in h.items():
        print(f"     {k:>10} | {'#' * int(40 * v / mx):<40} {v}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--off", help="attacker recording, masker OFF")
    ap.add_argument("--off-keys")
    ap.add_argument("--on", help="attacker recording, masker ON")
    ap.add_argument("--on-keys")
    ap.add_argument("--tol-ms", type=float, default=15.0)
    ap.add_argument("--offset", type=float, default=None, help="manual keylog offset (s) instead of auto-align")
    ap.add_argument("--report", help="write JSON report here")
    args = ap.parse_args()
    results = []
    if args.off and args.off_keys:
        results.append(evaluate_recording(args.off, args.off_keys, args.tol_ms, "mask OFF", args.offset))
    if args.on and args.on_keys:
        results.append(evaluate_recording(args.on, args.on_keys, args.tol_ms, "mask ON", args.offset))
    if not results:
        ap.error("give at least one of --off/--off-keys or --on/--on-keys")
    for r in results:
        print_report(r)
    on = next((r for r in results if r["label"] == "mask ON"), None)
    verdict = None
    if on:
        ok_recall = on["recall"] >= 0.7
        ok_prec = on["precision"] < 0.5
        ok_ari = on["cluster_ari"] is not None and on["cluster_ari"] < 0.1
        verdict = ok_recall and ok_prec and ok_ari
        print(f"\nPASS CRITERIA (mask ON): recall >= 0.7: {ok_recall}   precision < 0.5: {ok_prec}   ARI < 0.1: {ok_ari}")
        print("RESULT:", "PASS" if verdict else "FAIL")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"results": results, "pass": verdict}, f, indent=2)
        print(f"report written to {args.report}")
    return 0 if verdict in (None, True) else 1


if __name__ == "__main__":
    sys.exit(main())
