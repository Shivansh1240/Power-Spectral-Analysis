import numpy as np
import pytest

from psa import fitting, spectra
from psa.synthetic import broken_power_law, colored_noise, power_law


def test_power_law_recovered():
    rng = np.random.default_rng(10)
    x = colored_noise(2 ** 18, 1.0, power_law(-5 / 3, 1e-2), rng=rng)
    s = spectra.welch(x, 1.0, 4096)
    fit = fitting.fit_power_law(s.f, s.trace, 1e-3, 1e-1, s.dof, s.ordinate_corr)
    assert abs(fit.alpha + 5 / 3) < 4 * fit.alpha_err
    assert abs(fit.alpha + 5 / 3) < 0.02


def test_fisher_error_is_calibrated():
    rng = np.random.default_rng(11)
    alphas, errs = [], []
    for _ in range(150):
        x = colored_noise(2 ** 14, 1.0, power_law(-1.5), rng=rng)
        s = spectra.welch(x, 1.0, 1024)
        fit = fitting.fit_power_law(s.f, s.trace, 5e-3, 2e-1, s.dof, s.ordinate_corr)
        alphas.append(fit.alpha)
        errs.append(fit.alpha_err)
    ratio = np.std(alphas) / np.mean(errs)
    assert 0.75 < ratio < 1.33
    assert abs(np.mean(alphas) + 1.5) < 3 * np.std(alphas) / np.sqrt(len(alphas))


def test_broken_power_law_recovered():
    rng = np.random.default_rng(12)
    truth = dict(fb=5e-3, alpha1=-1.0, alpha2=-5 / 3)
    x = colored_noise(2 ** 18, 1.0, broken_power_law(truth["fb"], truth["alpha1"], truth["alpha2"]), rng=rng)
    s = spectra.welch(x, 1.0, 2 ** 14)
    fit = fitting.fit_broken_power_law(s.f, s.trace, 2e-4, 1e-1, s.dof, s.ordinate_corr)
    assert fit.fb_ci[0] <= truth["fb"] <= fit.fb_ci[1] or abs(np.log10(fit.fb / truth["fb"])) < 0.1
    assert abs(fit.alpha1 - truth["alpha1"]) < 4 * fit.alpha1_err + 0.02
    assert abs(fit.alpha2 - truth["alpha2"]) < 4 * fit.alpha2_err + 0.02


def test_lsq_slope_exact_on_noise_free_data():
    f = np.logspace(-3, -1, 20)
    slope, err = fitting.fit_power_law_lsq(f, 2 * f ** -1.7, 1e-3, 1e-1)
    assert slope == pytest.approx(-1.7, abs=1e-10)


def test_local_slope_flat_for_pure_power_law():
    rng = np.random.default_rng(13)
    x = colored_noise(2 ** 17, 1.0, power_law(-2.0), rng=rng)
    s = spectra.welch(x, 1.0, 4096)
    fc, a, e = fitting.local_slope(s.f, s.trace, s.dof, s.ordinate_corr, fmin=1e-2, fmax=0.4)
    assert np.all(np.abs(a + 2.0) < 4 * e + 0.02)


def test_non_overlapping_selection():
    idx = fitting.non_overlapping(np.array([0, 512, 1024, 1536, 4096]), 1024)
    np.testing.assert_array_equal(idx, [0, 2, 4])
