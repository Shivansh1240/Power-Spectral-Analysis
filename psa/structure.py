"""Structure functions, flatness and normalised fluctuation amplitude.

For a vector field, S_p(tau) = < |B(t + tau) - B(t)|**p >.  If the trace
PSD is a power law f**alpha with -3 < alpha < -1, then S_2 ∝ tau**(-alpha-1)
(e.g. Monin & Yaglom 1975), so the S_2 slope is an independent check of the
spectral index.  The flatness F = <dB_i**4> / <dB_i**2>**2 of each component
is 3 for Gaussian increments.  Growth of F toward small tau is the standard
signature of intermittency (Frisch 1995, Turbulence, CUP; Bruno & Carbone
2013, doi:10.12942/lrsp-2013-2).
"""

from __future__ import annotations

import numpy as np


def log_lags(max_lag: int, n: int = 60, min_lag: int = 1) -> np.ndarray:
    """Unique integer lags, approximately log-spaced."""
    return np.unique(np.round(np.logspace(np.log10(min_lag), np.log10(max_lag), n)).astype(int))


def _increments(B, lag):
    d = B[lag:] - B[:-lag]
    return d[~np.isnan(d).any(axis=1)]


def structure_functions(B, lags, orders=(2, 4)) -> dict:
    """Vector structure functions S_p(lag) of an (N, m) array (NaNs skipped)."""
    B = np.asarray(B, float)
    if B.ndim == 1:
        B = B[:, None]
    out = {p: np.empty(len(lags)) for p in orders}
    for i, lag in enumerate(lags):
        mag = np.linalg.norm(_increments(B, lag), axis=1)
        for p in orders:
            out[p][i] = np.mean(mag ** p)
    return out


def flatness(B, lags) -> np.ndarray:
    """Component flatness <dB_i^4>/<dB_i^2>^2, shape (n_lags, m)."""
    B = np.asarray(B, float)
    if B.ndim == 1:
        B = B[:, None]
    out = np.empty((len(lags), B.shape[1]))
    for i, lag in enumerate(lags):
        d = _increments(B, lag)
        d = d - d.mean(axis=0)
        out[i] = np.mean(d ** 4, axis=0) / np.mean(d ** 2, axis=0) ** 2
    return out
