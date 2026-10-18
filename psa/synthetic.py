"""Synthetic Gaussian time series with prescribed one-sided PSD (for tests)."""

from __future__ import annotations

import numpy as np


def colored_noise(n: int, dt: float, psd_fn, m: int = 1, rng=None) -> np.ndarray:
    """Real Gaussian series of length n, m columns, whose one-sided PSD has
    expectation ``psd_fn(f)`` at the Fourier frequencies (periodic series)."""
    rng = np.random.default_rng() if rng is None else rng
    f = np.fft.rfftfreq(n, dt)
    S = np.zeros_like(f)
    S[1:] = psd_fn(f[1:])
    amp = np.sqrt(S * n / (2 * dt))
    X = amp[:, None] * (rng.standard_normal((len(f), m)) + 1j * rng.standard_normal((len(f), m))) / np.sqrt(2)
    if n % 2 == 0:  # the Nyquist coefficient is real and takes the full one-sided power
        X[-1] = np.sqrt(S[-1] * n / dt) * rng.standard_normal(m)
    x = np.fft.irfft(X, n=n, axis=0)
    return x[:, 0] if m == 1 else x


def power_law(alpha: float, amp: float = 1.0, f0: float = 1.0):
    return lambda f: amp * (f / f0) ** alpha


def broken_power_law(fb: float, alpha1: float, alpha2: float, amp: float = 1.0):
    def fn(f):
        x = f / fb
        return amp * np.where(x < 1, x ** alpha1, x ** alpha2)
    return fn
