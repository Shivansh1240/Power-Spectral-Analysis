"""Maximum-likelihood power-law fits to spectral estimates.

A Welch estimate averaged over segments is distributed as
``P_hat = S * chi2_nu / nu`` about the true spectrum ``S`` (a Gamma variable of
shape ``nu/2``).  Its negative log-likelihood, up to terms independent of the
model, is

    NLL = (nu / 2) * sum_k [ ln S_k + P_hat_k / S_k ],

the Whittle likelihood generalised to averaged spectra (Whittle 1953;
Vaughan 2010, doi:10.1111/j.1365-2966.2009.15868.x; Barret & Vaughan 2012,
doi:10.1088/0004-637X/746/2/131).  For models that are linear in ln S
(power laws, continuous broken power laws at fixed break) the NLL is convex,
so Newton's method converges to the unique optimum.

Least squares on log10(P) treats the scatter as Gaussian, which it is not
(chi-square), and weights equal-width log bins by hand.  The MLE uses every
ordinate with its correct distribution.  Because neighbouring ordinates of a
tapered periodogram are correlated, the Fisher covariance is multiplied by the
window factor ``c_w`` (``Spectrum.ordinate_corr``).  These formal errors
describe statistical noise only.  Non-stationarity and the choice of fit
range are usually larger, so the analysis script also reports bootstrap and
sub-interval spreads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


def _nll(theta, X, P, nu):
    u = X @ theta
    return 0.5 * nu * np.sum(u + P * np.exp(-u))


def _gamma_mle(X, P, nu, max_iter: int = 200):
    """Newton iterations for theta minimising the Gamma NLL of ln S = X theta."""
    theta = np.linalg.lstsq(X, np.log(P), rcond=None)[0]
    f0 = _nll(theta, X, P, nu)
    for _ in range(max_iter):
        u = X @ theta
        r = P * np.exp(-u)
        g = 0.5 * nu * X.T @ (1.0 - r)
        H = 0.5 * nu * X.T @ (r[:, None] * X)
        step = np.linalg.solve(H, g)
        t = 1.0
        while True:
            new = theta - t * step
            f1 = _nll(new, X, P, nu)
            if f1 <= f0 + 1e-12 * abs(f0) or t < 1e-8:
                break
            t *= 0.5
        theta, f_prev, f0 = new, f0, f1
        if np.max(np.abs(t * step)) < 1e-10 or abs(f_prev - f0) < 1e-13 * abs(f0):
            break
    fisher = 0.5 * nu * X.T @ X
    return theta, f0, np.linalg.inv(fisher)


def _select(f, P, fmin, fmax):
    f = np.asarray(f, float)
    P = np.asarray(P, float)
    s = (f >= fmin) & (f <= fmax) & np.isfinite(P) & (P > 0)
    if s.sum() < 3:
        raise ValueError("fewer than 3 ordinates in [%g, %g] Hz" % (fmin, fmax))
    return f[s], P[s]


@dataclass
class PowerLawFit:
    alpha: float          # spectral index: S ∝ f**alpha
    alpha_err: float      # 1-sigma statistical error (Fisher x c_w)
    amp: float            # S(f0)
    f0: float
    fmin: float
    fmax: float
    n: int

    def model(self, f):
        return self.amp * (np.asarray(f) / self.f0) ** self.alpha


def fit_power_law(f, P, fmin: float, fmax: float, dof: float,
                  corr: float = 1.0) -> PowerLawFit:
    """Gamma-likelihood fit of S = A (f/f0)**alpha over [fmin, fmax]."""
    fs, Ps = _select(f, P, fmin, fmax)
    f0 = np.sqrt(fmin * fmax)
    X = np.column_stack([np.ones_like(fs), np.log(fs / f0)])
    theta, _, cov = _gamma_mle(X, Ps, dof)
    return PowerLawFit(alpha=float(theta[1]), alpha_err=float(np.sqrt(corr * cov[1, 1])),
                       amp=float(np.exp(theta[0])), f0=float(f0), fmin=fmin, fmax=fmax, n=len(fs))


@dataclass
class BrokenPowerLawFit:
    fb: float                     # break frequency (Hz)
    fb_ci: tuple[float, float]    # ~68% profile-likelihood interval
    alpha1: float                 # index below the break
    alpha1_err: float
    alpha2: float                 # index above the break
    alpha2_err: float
    amp: float                    # S(fb)
    fmin: float
    fmax: float
    profile_fb: np.ndarray
    profile_dnll: np.ndarray      # (NLL - min) / c_w along the grid

    def model(self, f):
        x = np.log(np.asarray(f) / self.fb)
        return self.amp * np.exp(np.where(x < 0, self.alpha1 * x, self.alpha2 * x))


def _broken_design(fs, fb):
    x = np.log(fs / fb)
    return np.column_stack([np.ones_like(x), np.minimum(x, 0.0), np.maximum(x, 0.0)])


def fit_broken_power_law(f, P, fmin: float, fmax: float, dof: float, corr: float = 1.0,
                         margin_dec: float = 0.2, n_grid: int = 241) -> BrokenPowerLawFit:
    """Continuous two-segment power law, break located by profile likelihood.

    The break is searched on a log grid that keeps at least ``margin_dec``
    decades of data on each side, then refined by a bounded scalar search.
    The interval ``fb_ci`` is where (NLL - NLL_min) / c_w <= 0.5, the
    likelihood-ratio 68% interval with the effective log-likelihood deflated
    for correlated ordinates.
    """
    fs, Ps = _select(f, P, fmin, fmax)
    lo, hi = np.log10(fmin) + margin_dec, np.log10(fmax) - margin_dec
    if hi <= lo:
        raise ValueError("fit range too narrow for a break search")
    grid = np.logspace(lo, hi, n_grid)

    def profile(fb):
        return _gamma_mle(_broken_design(fs, fb), Ps, dof)[1]

    nll = np.array([profile(fb) for fb in grid])
    k = int(np.argmin(nll))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, n_grid - 1)]
    res = optimize.minimize_scalar(lambda lf: profile(np.exp(lf)), bounds=(np.log(a), np.log(b)),
                                   method="bounded", options={"xatol": 1e-6})
    fb = float(np.exp(res.x)) if res.fun <= nll[k] else float(grid[k])
    theta, nll_min, cov = _gamma_mle(_broken_design(fs, fb), Ps, dof)
    dnll = (nll - nll_min) / corr
    inside = grid[dnll <= 0.5]
    ci = (float(inside.min()), float(inside.max())) if inside.size else (fb, fb)
    return BrokenPowerLawFit(fb=fb, fb_ci=ci, alpha1=float(theta[1]),
                             alpha1_err=float(np.sqrt(corr * cov[1, 1])), alpha2=float(theta[2]),
                             alpha2_err=float(np.sqrt(corr * cov[2, 2])), amp=float(np.exp(theta[0])),
                             fmin=fmin, fmax=fmax, profile_fb=grid, profile_dnll=dnll)


def fit_power_law_lsq(fb, yb, fmin: float, fmax: float) -> tuple[float, float]:
    """Ordinary least squares slope of log10(y) vs log10(f) on log-binned data
    (equal weight per bin).  Returns (slope, standard error).  Used only as a
    robustness check against the MLE, which weights by ordinate count."""
    fb = np.asarray(fb)
    yb = np.asarray(yb)
    s = (fb >= fmin) & (fb <= fmax)
    x, y = np.log10(fb[s]), np.log10(yb[s])
    A = np.column_stack([np.ones_like(x), x])
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    dofr = len(x) - 2
    s2 = float(res[0]) / dofr if (res.size and dofr > 0) else np.nan
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(coef[1]), float(np.sqrt(cov[1, 1]))


def local_slope(f, P, dof: float, corr: float = 1.0, width_dec: float = 0.3,
                step_dec: float = 0.05, fmin: float | None = None, fmax: float | None = None):
    """Spectral index in a sliding window of ``width_dec`` decades.

    Returns (f_center, alpha, alpha_err).  A local index reveals curvature
    and breaks that a single global fit hides.
    """
    f = np.asarray(f)
    fmin = f[0] if fmin is None else fmin
    fmax = f[-1] if fmax is None else fmax
    centers = 10 ** np.arange(np.log10(fmin) + width_dec / 2,
                              np.log10(fmax) - width_dec / 2 + 1e-9, step_dec)
    out = []
    for c in centers:
        lo, hi = c * 10 ** (-width_dec / 2), c * 10 ** (width_dec / 2)
        try:
            fit = fit_power_law(f, P, lo, hi, dof, corr)
            out.append((c, fit.alpha, fit.alpha_err))
        except ValueError:
            continue
    return tuple(np.array(v) for v in zip(*out)) if out else (np.array([]),) * 3


def non_overlapping(starts: np.ndarray, nperseg: int) -> np.ndarray:
    """Indices of a maximal greedy subset of segments that do not overlap."""
    keep, last = [], -np.inf
    for i, s in enumerate(np.asarray(starts)):
        if s >= last + nperseg:
            keep.append(i)
            last = s
    return np.array(keep, int)


def bootstrap_segments(segment_psd: np.ndarray, statistic, n_boot: int = 1000,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Bootstrap a statistic of the segment-averaged spectrum.

    ``segment_psd`` is (K, nf) for K (approximately independent) segments;
    ``statistic`` maps an averaged (nf,) spectrum to a scalar or vector.
    Resampling whole segments carries both chi-square noise and
    segment-to-segment non-stationarity into the spread.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    K = segment_psd.shape[0]
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, K, K)
        out.append(statistic(segment_psd[idx].mean(axis=0)))
    return np.array(out)
