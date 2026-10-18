"""Power spectral density estimation (Welch's method) with gap handling.

Conventions
-----------
* One-sided PSD in units of x**2 / Hz, normalised so that
  ``sum(P) * df == variance`` (Parseval).  For a window ``w`` of length M and
  sampling interval ``dt`` the estimate of one segment is

      P(f_k) = 2 dt |sum_n w_n x_n exp(-2 pi i k n / M)|**2 / sum_n w_n**2,

  without the factor 2 at f = 0 and at the Nyquist frequency.  This is the
  modified periodogram of Welch (1967, doi:10.1109/TAU.1967.1161901).
* The averaged estimate is approximately ``P * chi2_nu / nu`` distributed.
  The equivalent degrees of freedom ``nu`` account for segment overlap,
  ``nu = 2 K**2 / (K + 2 sum_{i<j} rho_ij**2)``, where ``rho_ij`` is the
  window overlap correlation between segments i and j (Welch 1967;
  Percival & Walden 1993, Spectral Analysis for Physical Applications, §6.17).
* Adjacent frequency ordinates of a tapered periodogram are correlated.  The
  variance inflation factor for sums over neighbouring ordinates,
  ``c_w = 1 + 2 sum_l |rho_l|**2`` with ``rho_l`` taken from the DFT of
  ``w**2``, is stored as ``Spectrum.ordinate_corr`` (Hann: c_w = 1.944) and is
  used to inflate formal fit errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal, stats

from .io import interpolate_gaps


@dataclass
class Spectrum:
    f: np.ndarray                 # (nf,) Hz, zero frequency excluded
    psd: np.ndarray               # (nf, m) one-sided PSD
    dof: float                    # equivalent chi-square dof of each ordinate
    n_segments: int
    nperseg: int
    dt: float
    window: str
    ordinate_corr: float          # c_w, see module docstring
    segment_starts: np.ndarray
    csd: np.ndarray | None = None          # (nf, m, m) one-sided cross-spectral matrix
    segment_psd: np.ndarray | None = None  # (K, nf, m) per-segment estimates

    @property
    def trace(self) -> np.ndarray:
        return self.psd.sum(axis=1)

    def confidence(self, y=None, level: float = 0.95):
        """Chi-square confidence band ``(lo, hi)`` for ``y`` (default: trace)."""
        y = self.trace if y is None else y
        a = 1.0 - level
        lo = y * self.dof / stats.chi2.ppf(1 - a / 2, self.dof)
        hi = y * self.dof / stats.chi2.ppf(a / 2, self.dof)
        return lo, hi


def window_ordinate_correlation(window: np.ndarray, max_lag: int = 16) -> float:
    """c_w = 1 + 2 sum_{l>=1} rho_l**2 for a data taper (see module docstring)."""
    w2 = np.asarray(window, float) ** 2
    W = np.fft.fft(w2)
    rho = np.abs(W[1:max_lag + 1]) / np.abs(W[0])
    return float(1.0 + 2.0 * np.sum(rho ** 2))


def welch_dof(window: np.ndarray, starts: np.ndarray) -> float:
    """Equivalent degrees of freedom of a Welch average over segments that
    start at ``starts`` (which need not be evenly spaced)."""
    w = np.asarray(window, float)
    M = len(w)
    K = len(starts)
    if K == 0:
        return 0.0
    norm = np.sum(w ** 2)
    s = np.sort(np.asarray(starts))
    total = 0.0
    for i in range(K - 1):
        lags = s[i + 1:] - s[i]
        lags = lags[lags < M]
        for lag in lags:
            rho = np.sum(w[:M - lag] * w[lag:]) / norm
            total += rho ** 2
    return float(2.0 * K ** 2 / (K + 2.0 * total))


def welch(x, dt: float, nperseg: int, overlap: float = 0.5, window: str = "hann",
          detrend: str | None = "linear", max_gap_fraction: float = 0.0,
          cross: bool = False, keep_segments: bool = False) -> Spectrum:
    """Gap-aware Welch PSD of a (N,) or (N, m) series with NaNs marking gaps.

    Segments whose fraction of missing samples exceeds ``max_gap_fraction`` are
    discarded; NaNs inside retained segments are linearly interpolated.  Set
    ``max_gap_fraction=1`` to keep every segment (gaps then always filled).
    """
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    N, m = x.shape
    if nperseg > N:
        raise ValueError("nperseg (%d) longer than the series (%d)" % (nperseg, N))
    step = max(1, int(round(nperseg * (1.0 - overlap))))
    w = signal.get_window(window, nperseg, fftbins=True)
    norm = 2.0 * dt / np.sum(w ** 2)
    bad = np.isnan(x).any(axis=1)

    nfreq = nperseg // 2 + 1
    acc = np.zeros((nfreq, m, m), complex) if cross else np.zeros((nfreq, m))
    segs = []
    used = []
    for s0 in range(0, N - nperseg + 1, step):
        sl = slice(s0, s0 + nperseg)
        gap_frac = bad[sl].mean()
        if gap_frac > max_gap_fraction or gap_frac == 1.0:
            continue
        seg = x[sl]
        if gap_frac > 0:
            seg = interpolate_gaps(seg)
        if detrend:
            seg = signal.detrend(seg, axis=0, type=detrend)
        X = np.fft.rfft(seg * w[:, None], axis=0)
        if cross:
            S = np.einsum("ki,kj->kij", X, X.conj()) * norm
            acc += S
            if keep_segments:
                segs.append(np.real(np.einsum("kii->ki", S)))
        else:
            P = np.abs(X) ** 2 * norm
            acc += P
            if keep_segments:
                segs.append(P)
        used.append(s0)
    K = len(used)
    if K == 0:
        raise ValueError("no segment satisfies max_gap_fraction=%g" % max_gap_fraction)
    acc /= K
    # the factor 2 does not apply at f=0 and (even nperseg) at Nyquist
    acc[0] /= 2.0
    if nperseg % 2 == 0:
        acc[-1] /= 2.0
    f = np.fft.rfftfreq(nperseg, dt)
    if cross:
        psd = np.real(np.einsum("kii->ki", acc))
        csd = acc[1:]
    else:
        psd, csd = acc, None
    seg_arr = None
    if keep_segments:
        seg_arr = np.array(segs)[:, 1:]
        if nperseg % 2 == 0:
            seg_arr[:, -1] /= 2.0
    return Spectrum(f=f[1:], psd=psd[1:], dof=welch_dof(w, np.array(used)), n_segments=K,
                    nperseg=nperseg, dt=dt, window=window,
                    ordinate_corr=window_ordinate_correlation(w),
                    segment_starts=np.array(used), csd=csd, segment_psd=seg_arr)


def reduced_magnetic_helicity(spec: Spectrum, i: int = 1, j: int = 2) -> np.ndarray:
    """Normalised reduced magnetic helicity from the cross-spectral matrix,

        sigma_m(f) = 2 Im S_ij(f) / (S_ii(f) + S_jj(f)),

    for components i, j perpendicular to the sampling direction (RTN with the
    flow along +R: i = T, j = N).  With the FFT sign convention used here a
    spatially positive (right-handed, curl B = +k B) helix advected past the
    spacecraft gives sigma_m = +1 (Matthaeus, Goldstein & Smith 1982,
    doi:10.1103/PhysRevLett.48.1256; tested in tests/test_helicity.py).
    """
    if spec.csd is None:
        raise ValueError("spectrum computed without cross=True")
    S = spec.csd
    return 2.0 * np.imag(S[:, i, j]) / (np.real(S[:, i, i]) + np.real(S[:, j, j]))


@dataclass
class Binned:
    f: np.ndarray        # geometric-mean frequency of the ordinates in each bin
    y: np.ndarray        # arithmetic mean of the ordinates
    n: np.ndarray        # number of ordinates per bin
    edges: np.ndarray


def log_bin(f, y, bins_per_decade: int = 10, fmin: float | None = None,
            fmax: float | None = None, min_count: int = 1) -> Binned:
    """Average spectral ordinates in logarithmically spaced frequency bins."""
    f = np.asarray(f, float)
    y = np.asarray(y, float)
    fmin = f[f > 0].min() if fmin is None else fmin
    fmax = f.max() if fmax is None else fmax
    lo = np.floor(np.log10(fmin) * bins_per_decade) / bins_per_decade
    hi = np.ceil(np.log10(fmax) * bins_per_decade) / bins_per_decade
    edges = 10 ** np.arange(lo, hi + 0.5 / bins_per_decade, 1.0 / bins_per_decade)
    sel = (f >= fmin) & (f <= fmax)
    k = np.digitize(f[sel], edges) - 1
    nb = len(edges) - 1
    fc, yc, nc = [], [], []
    for b in range(nb):
        s = k == b
        if s.sum() >= min_count:
            fc.append(np.exp(np.mean(np.log(f[sel][s]))))
            yc.append(np.mean(y[sel][s], axis=0))
            nc.append(s.sum())
    return Binned(np.array(fc), np.array(yc), np.array(nc), edges)


def magnitude(B: np.ndarray) -> np.ndarray:
    """|B| row-wise, NaN-propagating."""
    return np.sqrt(np.sum(np.asarray(B, float) ** 2, axis=1))
