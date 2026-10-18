import numpy as np
import pytest

from psa import spectra
from psa.synthetic import colored_noise, power_law


def test_hann_ordinate_correlation():
    from scipy.signal import get_window
    # rho_1 = 2/3, rho_2 = 1/6 for the periodic Hann window
    assert spectra.window_ordinate_correlation(get_window("hann", 4096)) == pytest.approx(1 + 2 * (4 / 9 + 1 / 36), rel=1e-6)


def test_white_noise_level_and_parseval():
    rng = np.random.default_rng(1)
    sigma, dt = 2.0, 0.5
    x = sigma * rng.standard_normal(2 ** 18)
    s = spectra.welch(x, dt, 1024, detrend="constant")
    df = s.f[1] - s.f[0]
    assert s.psd[:-1, 0].mean() == pytest.approx(2 * sigma ** 2 * dt, rel=0.01)
    # f=0 is excluded and carries ~1/nperseg of the variance
    assert s.psd[:, 0].sum() * df == pytest.approx(x.var(), rel=0.01)


def test_sinusoid_power():
    dt, n, nper = 1.0, 2 ** 16, 2 ** 12
    f0 = 100 / (nper * dt)      # exactly on a Fourier bin
    A = 3.0
    t = np.arange(n) * dt
    s = spectra.welch(A * np.sin(2 * np.pi * f0 * t), dt, nper, detrend=None)
    df = s.f[1] - s.f[0]
    k = np.argmin(abs(s.f - f0))
    assert s.psd[k - 3:k + 4, 0].sum() * df == pytest.approx(A ** 2 / 2, rel=1e-6)


def test_dof_matches_monte_carlo():
    rng = np.random.default_rng(2)
    nper, n = 256, 256 * 16
    est = []
    for _ in range(400):
        s = spectra.welch(rng.standard_normal(n), 1.0, nper, overlap=0.5, detrend="constant")
        est.append(s.psd[10:100, 0])
    est = np.array(est)
    rel_var = (est.var(axis=0) / est.mean(axis=0) ** 2).mean()
    assert rel_var == pytest.approx(2 / s.dof, rel=0.08)


def test_gappy_segments_are_dropped_or_filled():
    rng = np.random.default_rng(3)
    x = colored_noise(2 ** 15, 1.0, power_law(-5 / 3), rng=rng)
    x[5000:5100] = np.nan
    strict = spectra.welch(x, 1.0, 2048, max_gap_fraction=0.0)
    loose = spectra.welch(x, 1.0, 2048, max_gap_fraction=1.0)
    assert strict.n_segments == loose.n_segments - 2   # gap at 5000-5100 lies in the segments starting at 3072 and 4096
    assert np.all(np.isfinite(loose.psd))
    assert strict.dof < loose.dof


def test_cross_spectrum_diagonal_equals_psd():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2 ** 14, 3))
    a = spectra.welch(x, 1.0, 1024, cross=True)
    b = spectra.welch(x, 1.0, 1024)
    np.testing.assert_allclose(a.psd, b.psd, rtol=1e-12)


def test_log_bin_conserves_mean():
    f = np.arange(1, 10001) * 1e-4
    y = np.ones_like(f) * 5
    b = spectra.log_bin(f, y, 10)
    np.testing.assert_allclose(b.y, 5)
    assert b.n.sum() == len(f)
