"""Morlet wavelet spectra, local mean field anisotropy and magnetic helicity.

Transform
---------
Continuous wavelet transform with the Morlet wavelet (omega0 = 6) computed by
FFT, normalised to unit energy at every scale as in Torrence & Compo (1998,
doi:10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2, eqs. 4-6):

    W_n(s) = sum_k xhat_k psihat*(s omega_k) exp(i omega_k n dt),
    psihat(s omega) = sqrt(2 pi s / dt) pi**-0.25 H(omega) exp(-(s omega - omega0)**2 / 2).

With this normalisation E|W_n(s)|**2 = sigma**2 for white noise of variance
sigma**2, whose one-sided PSD is 2 sigma**2 dt.  The PSD estimate at the
equivalent Fourier frequency f = 1 / (1.033 s) is therefore

    P(f) = 2 dt <|W_n(s)|**2>_n,

which ``tests/test_wavelets.py`` checks against white and power-law noise.

Gaps and edges
--------------
Gaps are linearly interpolated before the transform.  A coefficient is
discarded if (a) it lies within the e-folding time sqrt(2) s of either end of
the series (the cone of influence of Torrence & Compo 1998), or (b) more than
``max_gap_weight`` of its Gaussian envelope (sigma = s) falls on interpolated
samples.

Local mean field
----------------
Following the scale-dependent local mean field idea of Horbury, Forman &
Oughton (2008, doi:10.1103/PhysRevLett.101.175005) and Podesta (2009,
doi:10.1088/0004-637X/698/2/986), B0(s, t) is B convolved with a normalised
Gaussian of width s (the Morlet envelope).  theta_VB is the angle between B0
and the local flow direction (the sampling direction under Taylor's
hypothesis).  The field-parallel coefficient is W_par = W . B0 / |B0|.

Reduced magnetic helicity
-------------------------
sigma_m(s, theta) = sum 2 Im(W_T W_N*) / sum(|W_T|**2 + |W_N|**2), summed over
coefficients in each angle bin (Matthaeus, Goldstein & Smith 1982;
He et al. 2011, doi:10.1088/0004-637X/731/2/85; Podesta & Gary 2011,
doi:10.1088/0004-637X/734/1/15).  +1 means a spatially right-handed
(positive helicity) structure advected along +R; see ``spectra.reduced_magnetic_helicity``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .io import interpolate_gaps

OMEGA0 = 6.0


def fourier_factor(omega0: float = OMEGA0) -> float:
    """Fourier period / scale for the Morlet wavelet (TC98 Table 1): 1.033 for omega0=6."""
    return 4 * np.pi / (omega0 + np.sqrt(2 + omega0 ** 2))


def power_law_bias(alpha: float, omega0: float = OMEGA0) -> float:
    """Ratio E[P_wavelet(f)] / S(f) for a power-law spectrum S ∝ f**alpha.

    The Morlet filter averages S over a band of relative width ~1/omega0.
    For a curved (in linear f) spectrum this shifts the level but not the
    slope.  The ratio is 1 for white noise and about 1.055 for alpha = -5/3.
    Computed as int (u/u_s)**alpha exp(-(u - omega0)**2) du / int exp(-(u - omega0)**2) du
    with u = 2 pi s f and u_s = 2 pi / fourier_factor.
    """
    u = np.linspace(max(omega0 - 12, 1e-6), omega0 + 12, 200001)
    w = np.exp(-(u - omega0) ** 2)
    us = 2 * np.pi / fourier_factor(omega0)
    return float(np.trapezoid((u / us) ** alpha * w, u) / np.trapezoid(w, u))


def scales_from_frequencies(fmin: float, fmax: float, dj: float = 0.125,
                            omega0: float = OMEGA0) -> np.ndarray:
    """Scales s_j = s0 2**(j dj) spanning the requested Fourier frequencies."""
    ff = fourier_factor(omega0)
    smin, smax = 1.0 / (ff * fmax), 1.0 / (ff * fmin)
    J = int(np.ceil(np.log2(smax / smin) / dj))
    return smin * 2.0 ** (dj * np.arange(J + 1))


class _Transformer:
    """Holds the padded FFTs so that each scale costs only inverse FFTs."""

    def __init__(self, n: int, dt: float, smax: float):
        self.n, self.dt = n, dt
        pad = int(np.ceil(6 * smax / dt))
        self.L = int(2 ** np.ceil(np.log2(n + pad)))
        self.omega = 2 * np.pi * np.fft.fftfreq(self.L, dt)

    def fft(self, x):
        y = np.zeros(self.L)
        y[:self.n] = x
        return np.fft.fft(y)

    def morlet(self, xhat, s, omega0=OMEGA0):
        w = self.omega
        psi = np.sqrt(2 * np.pi * s / self.dt) * np.pi ** -0.25 * np.exp(-0.5 * (s * w - omega0) ** 2)
        psi[w <= 0] = 0.0
        return np.fft.ifft(xhat * psi)[:self.n]

    def gaussian_smooth(self, xhat, sigma):
        g = np.exp(-0.5 * (sigma * self.omega) ** 2)   # FT of a unit-area Gaussian
        return np.real(np.fft.ifft(xhat * g))[:self.n]


def cwt(x, dt: float, scales, omega0: float = OMEGA0) -> np.ndarray:
    """Morlet CWT of a gap-free 1-D series; returns (n_scales, N) complex."""
    x = np.asarray(x, float)
    tr = _Transformer(len(x), dt, max(scales))
    xh = tr.fft(x - x.mean())
    return np.array([tr.morlet(xh, s, omega0) for s in scales])


@dataclass
class WaveletResult:
    f: np.ndarray                 # (ns,) equivalent Fourier frequency
    scales: np.ndarray
    dt: float
    psd_trace: np.ndarray         # (ns,) global, valid coefficients only
    psd_par: np.ndarray           # (ns,) local-mean-field-parallel component
    psd_mag: np.ndarray           # (ns,) PSD of |B|
    valid_fraction: np.ndarray    # (ns,)
    theta_edges: np.ndarray       # folded angle bins, degrees in [0, 90]
    theta_count: np.ndarray       # (nblk, ns, nth)
    theta_power: np.ndarray       # (nblk, ns, nth) summed trace |W|^2
    theta_par: np.ndarray         # (nblk, ns, nth) summed parallel |W|^2
    hel_edges: np.ndarray         # unfolded angle bins in [0, 180]
    hel_num: np.ndarray           # (nblk, ns, nh) sum 2 Im(W_T W_N*)
    hel_den: np.ndarray           # (nblk, ns, nh) sum |W_T|^2 + |W_N|^2
    hel_count: np.ndarray
    block_seconds: float
    scalogram: np.ndarray         # (ns, nt) PSD averaged in time bins (NaN = invalid)
    scalogram_t: np.ndarray       # (nt,) seconds, bin centres

    def theta_psd(self, blocks=None):
        b = slice(None) if blocks is None else blocks
        with np.errstate(invalid="ignore", divide="ignore"):
            return 2 * self.dt * self.theta_power[b].sum(0) / self.theta_count[b].sum(0)

    def theta_par_fraction(self, blocks=None):
        b = slice(None) if blocks is None else blocks
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.theta_par[b].sum(0) / self.theta_power[b].sum(0)

    def helicity(self, blocks=None, rectify: bool = False):
        """sigma_m(s, theta_VB) on the unfolded [0, 180] grid, or, with
        ``rectify``, folded to [0, 90] with the sign of cos(theta_VB) applied
        so that every coefficient is expressed as if B0 pointed along +V
        (the 'outward sector' convention of He et al. 2011)."""
        b = slice(None) if blocks is None else blocks
        num, den = self.hel_num[b].sum(0), self.hel_den[b].sum(0)
        if rectify:
            num = num - num[:, ::-1]
            den = den + den[:, ::-1]
            h = num.shape[1] // 2
            num, den = num[:, :h], den[:, :h]
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den

    @property
    def n_blocks(self) -> int:
        return self.theta_count.shape[0]


def analyse(B, V, dt: float, fmin: float, fmax: float, dj: float = 0.125,
            theta_bin_deg: float = 10.0, block_seconds: float = 6 * 3600.0,
            scalogram_seconds: float = 120.0, max_gap_weight: float = 0.02,
            omega0: float = OMEGA0) -> WaveletResult:
    """Wavelet analysis of B (N, 3) in RTN with flow V (N, 3), NaN = gap.

    Accumulates, per scale, the global trace/parallel/|B| spectra, the trace
    and parallel power binned by the local field angle theta_VB (folded to
    0-90 degrees), the reduced helicity terms on the unfolded 0-180 grid, and
    a time-binned scalogram.  Angle-binned sums are kept per time block of
    ``block_seconds`` so that jackknife errors can be computed.
    """
    B = np.asarray(B, float)
    N = len(B)
    gap = np.isnan(B).any(axis=1)
    Bf = interpolate_gaps(B)
    Vf = interpolate_gaps(np.asarray(V, float))
    vhat = Vf / np.linalg.norm(Vf, axis=1, keepdims=True)
    Bmag = np.linalg.norm(Bf, axis=1)

    scales = scales_from_frequencies(fmin, fmax, dj, omega0)
    tr = _Transformer(N, dt, scales.max())
    Bh = [tr.fft(Bf[:, k] - Bf[:, k].mean()) for k in range(3)]
    Bh_raw = [tr.fft(Bf[:, k]) for k in range(3)]
    ones_h = tr.fft(np.ones(N))
    gap_h = tr.fft(gap.astype(float))
    Mh = tr.fft(Bmag - Bmag.mean())

    t = np.arange(N) * dt
    blk = np.minimum((t // block_seconds).astype(int), int(np.ceil(N * dt / block_seconds)) - 1)
    nblk = blk.max() + 1
    th_edges = np.arange(0, 90 + theta_bin_deg / 2, theta_bin_deg)
    he_edges = np.arange(0, 180 + theta_bin_deg / 2, theta_bin_deg)
    nth, nh = len(th_edges) - 1, len(he_edges) - 1
    sbin = (t // scalogram_seconds).astype(int)
    nsb = sbin.max() + 1

    ns = len(scales)
    out = dict(
        psd_trace=np.full(ns, np.nan), psd_par=np.full(ns, np.nan), psd_mag=np.full(ns, np.nan),
        valid_fraction=np.zeros(ns),
        theta_count=np.zeros((nblk, ns, nth)), theta_power=np.zeros((nblk, ns, nth)),
        theta_par=np.zeros((nblk, ns, nth)),
        hel_num=np.zeros((nblk, ns, nh)), hel_den=np.zeros((nblk, ns, nh)),
        hel_count=np.zeros((nblk, ns, nh)),
        scalogram=np.full((ns, nsb), np.nan),
    )
    for j, s in enumerate(scales):
        W = [tr.morlet(Bh[k], s, omega0) for k in range(3)]
        Wm = tr.morlet(Mh, s, omega0)
        # local mean field; divide by the smoothed indicator so that the
        # zero padding does not bias it near the ends
        wsum = tr.gaussian_smooth(ones_h, s)
        B0 = np.column_stack([tr.gaussian_smooth(Bh_raw[k], s) / wsum for k in range(3)])
        gap_w = tr.gaussian_smooth(gap_h, s) / wsum
        coi = np.sqrt(2) * s
        valid = (t >= coi) & (t <= t[-1] - coi) & (gap_w <= max_gap_weight)
        out["valid_fraction"][j] = valid.mean()
        if not valid.any():
            continue
        p_tr = sum(np.abs(w) ** 2 for w in W)
        b0n = np.linalg.norm(B0, axis=1)
        e0 = B0 / b0n[:, None]
        Wpar = W[0] * e0[:, 0] + W[1] * e0[:, 1] + W[2] * e0[:, 2]
        p_par = np.abs(Wpar) ** 2
        out["psd_trace"][j] = 2 * dt * p_tr[valid].mean()
        out["psd_par"][j] = 2 * dt * p_par[valid].mean()
        out["psd_mag"][j] = 2 * dt * (np.abs(Wm[valid]) ** 2).mean()

        cos_vb = np.clip(np.sum(e0 * vhat, axis=1), -1, 1)
        theta = np.degrees(np.arccos(cos_vb))                  # 0..180
        folded = np.minimum(theta, 180 - theta)                 # 0..90
        ib = np.clip(np.digitize(folded, th_edges) - 1, 0, nth - 1)
        ih = np.clip(np.digitize(theta, he_edges) - 1, 0, nh - 1)
        v = valid
        key = blk[v] * nth + ib[v]
        out["theta_count"][:, j, :] = np.bincount(key, minlength=nblk * nth).reshape(nblk, nth)
        out["theta_power"][:, j, :] = np.bincount(key, p_tr[v], nblk * nth).reshape(nblk, nth)
        out["theta_par"][:, j, :] = np.bincount(key, p_par[v], nblk * nth).reshape(nblk, nth)
        num = 2 * np.imag(W[1] * np.conj(W[2]))
        den = np.abs(W[1]) ** 2 + np.abs(W[2]) ** 2
        keyh = blk[v] * nh + ih[v]
        out["hel_num"][:, j, :] = np.bincount(keyh, num[v], nblk * nh).reshape(nblk, nh)
        out["hel_den"][:, j, :] = np.bincount(keyh, den[v], nblk * nh).reshape(nblk, nh)
        out["hel_count"][:, j, :] = np.bincount(keyh, minlength=nblk * nh).reshape(nblk, nh)
        cnt = np.bincount(sbin[v], minlength=nsb)
        pw = np.bincount(sbin[v], p_tr[v], nsb)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["scalogram"][j] = np.where(cnt > 0, 2 * dt * pw / cnt, np.nan)

    return WaveletResult(f=1.0 / (fourier_factor(omega0) * scales), scales=scales, dt=dt,
                         theta_edges=th_edges, hel_edges=he_edges, block_seconds=block_seconds,
                         scalogram_t=(np.arange(nsb) + 0.5) * scalogram_seconds, **out)


def jackknife(estimator, n_blocks: int):
    """Delete-one-block jackknife: returns (full estimate, standard error).

    ``estimator(mask)`` must accept a boolean block mask and return an array.
    """
    full = np.asarray(estimator(np.ones(n_blocks, bool)))
    reps = []
    for i in range(n_blocks):
        m = np.ones(n_blocks, bool)
        m[i] = False
        reps.append(estimator(m))
    reps = np.array(reps, float)
    n = n_blocks
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        se = np.sqrt((n - 1) / n * np.nansum((reps - np.nanmean(reps, axis=0)) ** 2, axis=0))
    se[~np.isfinite(full)] = np.nan
    return full, se
