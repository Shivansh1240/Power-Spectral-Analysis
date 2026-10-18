"""Elsasser variables, cross helicity and residual energy spectra.

z± = dv ∓ sign(B0_R) db with db = dB / sqrt(mu0 m_p n_p) in velocity units,
so that z+ is the anti-sunward-propagating Elsasser field whatever the sector
polarity.  The normalised cross helicity and residual energy are

    sigma_c(f) = (E+ - E-) / (E+ + E-),   sigma_r(f) = (E_v - E_b) / (E_v + E_b)

(Matthaeus & Goldstein 1982, doi:10.1029/JA087iA08p06011; Bruno & Carbone
2013).  Alpha particles are neglected in the mass density.  Including a
typical 4% alpha abundance (by number) would reduce |db| by ~7%.  That
raises sigma_r by up to ~0.07 and leaves sigma_c almost unchanged.

Plasma moments in the Ulysses/SWOOPS dataset used here are 4- or 8-minute
samples that AMDA linearly interpolated to 1 s.  Linear interpolation is
convolution with a triangle of half-width equal to the sample spacing, so B is
smoothed with the same triangle before the two are combined
(``matched_smooth``) and results are only trusted well below the plasma
Nyquist frequency.  The density used for b is a running median
(``running_median``) because single-sample dropouts in n_p would otherwise
appear as spikes in b.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .io import interpolate_gaps
from .plasma import M_PROTON, MU0
from .spectra import welch


def matched_smooth(x, half_width: int) -> np.ndarray:
    """NaN-aware convolution with a unit-area triangle of half-width
    ``half_width`` samples (the impulse response of linear interpolation)."""
    x = np.asarray(x, float)
    one_d = x.ndim == 1
    if one_d:
        x = x[:, None]
    k = 1.0 - np.abs(np.arange(-half_width + 1, half_width)) / half_width
    good = ~np.isnan(x)
    num = signal.fftconvolve(np.where(good, x, 0.0), k[:, None], mode="same", axes=0)
    den = signal.fftconvolve(good.astype(float), k[:, None], mode="same", axes=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(den > 0.5 * k.sum(), num / den, np.nan)
    return y[:, 0] if one_d else y


def running_median(x, size: int) -> np.ndarray:
    """Running median of a 1-D series (NaNs interpolated first).

    Used for the density in b = B / sqrt(mu0 m_p n): this keeps isolated
    dropouts in the moments (e.g. one 0.004 cm^-3 sample in a 0.25 cm^-3
    stream) from turning into spikes in b."""
    from scipy.ndimage import median_filter
    return median_filter(interpolate_gaps(np.asarray(x, float)), size=size, mode="nearest")


def to_alfven_units(B_nT, n_cm3) -> np.ndarray:
    """B / sqrt(mu0 m_p n) in km/s."""
    rho = np.asarray(n_cm3)[..., None] * 1e6 * M_PROTON
    return np.asarray(B_nT) * 1e-9 / np.sqrt(MU0 * rho) / 1e3


def alfvenicity_spectra(V, B, n, dt: float, nperseg: int, polarity: float,
                        overlap: float = 0.5) -> dict:
    """Trace spectra of v, b, z+ and z-, and sigma_c(f), sigma_r(f).

    V (km/s), B (nT, already smoothed to the plasma resolution) and n (cm^-3)
    are (N, 3), (N, 3) and (N,), NaN = gap.  ``polarity`` is sign(<B_R>).
    """
    b = to_alfven_units(B, n)
    zp = V - polarity * b
    zm = V + polarity * b
    stack = np.column_stack([V, b, zp, zm])
    bad = np.isnan(stack).any(axis=1)
    stack[bad] = np.nan
    spec = welch(interpolate_gaps(stack), dt, nperseg, overlap=overlap, detrend="linear",
                 max_gap_fraction=1.0)
    P = spec.psd
    Ev, Eb, Ep, Em = (P[:, 3 * i:3 * i + 3].sum(axis=1) for i in range(4))
    return {"f": spec.f, "E_v": Ev, "E_b": Eb, "E_plus": Ep, "E_minus": Em,
            "sigma_c": (Ep - Em) / (Ep + Em), "sigma_r": (Ev - Eb) / (Ev + Eb),
            "dof": spec.dof, "ordinate_corr": spec.ordinate_corr, "n_segments": spec.n_segments,
            "gap_fraction": float(bad.mean())}
