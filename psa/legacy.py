"""The repository's original algorithm, kept as a function for comparison.

It reproduces the earlier README script step by step: raw FFT periodogram of
one component normalised by n, a 50-point moving average in linear
frequency, and a straight-line fit in log-log space over 1-10 mHz.  Its
weaknesses are listed in the README ('What was wrong with the original
script').
"""

from __future__ import annotations

import numpy as np


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def legacy_slope(x, window_size: int = 50, fit_range=(0.001, 0.01)):
    """Return (slope, sma_frequencies, sma_power) exactly as the old script.

    Like the original, it assumes 1-s sampling: ``fftfreq(n, 1/n) / n`` is
    the frequency in Hz only when dt = 1 s.  NaNs propagate into the FFT and
    make the result undefined, which is what the old script did on gappy data.
    """
    x = np.asarray(x, float)
    n = len(x)
    P = (np.abs(np.fft.fft(x)) ** 2 / n)[: n // 2]
    freqs = np.fft.fftfreq(n, 1 / n)[: n // 2]
    pfactor = ((1 / n) * freqs)[: n // 2]
    sma_P = moving_average(P, window_size)
    sma_f = moving_average(pfactor, window_size)
    r = (sma_f >= fit_range[0]) & (sma_f <= fit_range[1])
    coeffs = np.polyfit(np.log(sma_f[r]), np.log(sma_P[r]), 1)
    return float(coeffs[0]), sma_f, sma_P
