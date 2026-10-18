"""Data-quality diagnostics for resampled magnetometer data.

The AMDA product used here is 'ulys_fgm_rtn_b_rtn' resampled to an exact
1 s grid.  The source data are samples 'at about one-second intervals', so
the resampling has to shift each sample by a fraction ``a`` of a step.  If
it interpolates linearly, it applies the low-pass transfer function

    |H_a(f)|**2 = 1 - 2 a (1 - a) (1 - cos(2 pi f dt)),

with ``a`` drifting as the two clocks slip.  The drift is reset at data gaps.
This module measures the symptoms:

* ``hf_ratio_series``: power in a high band relative to a reference band in
  short windows.  A physical spectrum gives a slowly varying ratio.  A
  drifting resampling phase gives large, quasi-periodic swings.
* ``gap_jump_test``: the change of that ratio across each data gap compared
  with the change across random cut points.
* ``interpolation_transfer``: |H|**2 for a given ``a``, or its average over a
  uniformly distributed ``a``, ``1 - (1 - cos 2 pi f dt) / 3``, which brackets
  the possible bias.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .io import nan_runs


def _band_ratio(seg, dt, nperseg, hf_band, ref_band):
    f, P = signal.welch(seg, fs=1 / dt, nperseg=nperseg, detrend="linear", axis=0)
    Pt = P.sum(axis=1)
    hf = Pt[(f >= hf_band[0]) & (f <= hf_band[1])].mean()
    ref = Pt[(f >= ref_band[0]) & (f <= ref_band[1])].mean()
    return hf / ref


def hf_ratio_series(B, dt: float, window: int = 600, nperseg: int = 120,
                    hf_band=(0.35, 0.5), ref_band=(0.1, 0.15)):
    """Return (window-centre times [s], ratio) for gap-free windows."""
    B = np.asarray(B, float)
    t, r = [], []
    for i in range(len(B) // window):
        seg = B[i * window:(i + 1) * window]
        if np.isnan(seg).any():
            continue
        t.append((i + 0.5) * window * dt)
        r.append(_band_ratio(seg, dt, nperseg, hf_band, ref_band))
    return np.array(t), np.array(r)


def gap_jump_test(B, dt: float, window: int = 900, nperseg: int = 120, n_control: int = 400,
                  hf_band=(0.35, 0.5), ref_band=(0.1, 0.15), seed: int = 1) -> dict:
    """|log10| change of the band ratio across gaps vs. across random cuts."""
    B = np.asarray(B, float)
    bad = np.isnan(B).any(axis=1)
    starts, lengths = nan_runs(bad)
    gaps = []
    for s, ln in zip(starts, lengths):
        a, b = B[max(0, s - window):s], B[s + ln:s + ln + window]
        if len(a) < window or len(b) < window or np.isnan(a).any() or np.isnan(b).any():
            continue
        jump = np.log10(_band_ratio(b, dt, nperseg, hf_band, ref_band)
                        / _band_ratio(a, dt, nperseg, hf_band, ref_band))
        gaps.append((s * dt, ln * dt, jump))
    rng = np.random.default_rng(seed)
    ctrl = []
    for c in rng.integers(window, len(B) - window, n_control):
        a, b = B[c - window:c], B[c:c + window]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        ctrl.append(np.log10(_band_ratio(b, dt, nperseg, hf_band, ref_band)
                             / _band_ratio(a, dt, nperseg, hf_band, ref_band)))
    gaps = np.array(gaps).reshape(-1, 3)
    ctrl = np.abs(np.array(ctrl))
    return {"gap_time_s": gaps[:, 0], "gap_length_s": gaps[:, 1], "gap_dlog10": gaps[:, 2],
            "control_abs_dlog10": ctrl,
            "median_abs_gap": float(np.median(np.abs(gaps[:, 2]))) if len(gaps) else np.nan,
            "median_abs_control": float(np.median(ctrl)),
            "p90_abs_control": float(np.percentile(ctrl, 90))}


def interpolation_transfer(f, dt: float = 1.0, a: float | None = None) -> np.ndarray:
    """|H|^2 of linear interpolation at fractional offset ``a``; ``a=None``
    averages over a uniformly distributed offset."""
    c = 1.0 - np.cos(2 * np.pi * np.asarray(f) * dt)
    if a is None:
        return 1.0 - c / 3.0
    return 1.0 - 2.0 * a * (1.0 - a) * c
