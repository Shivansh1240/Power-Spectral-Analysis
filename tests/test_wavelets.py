import numpy as np
import pytest

from psa import spectra, wavelets
from psa.synthetic import colored_noise, power_law


def test_fourier_factor():
    assert wavelets.fourier_factor(6.0) == pytest.approx(1.0330, abs=1e-4)


def test_white_noise_psd_normalisation():
    rng = np.random.default_rng(20)
    sigma, dt = 1.5, 1.0
    x = sigma * rng.standard_normal(2 ** 18)
    scales = wavelets.scales_from_frequencies(0.01, 0.2, dj=0.25)
    W = wavelets.cwt(x, dt, scales)
    psd = 2 * dt * (np.abs(W[:, 2000:-2000]) ** 2).mean(axis=1)
    np.testing.assert_allclose(psd, 2 * sigma ** 2 * dt, rtol=0.05)


def test_power_law_slope_and_level_match_welch():
    rng = np.random.default_rng(21)
    alpha = -5 / 3
    B = colored_noise(2 ** 17, 1.0, power_law(alpha, 1e-2), m=3, rng=rng)
    V = np.tile([750.0, 0.0, 0.0], (len(B), 1))
    res = wavelets.analyse(B + np.array([1.0, 0, 0]), V, 1.0, 2e-3, 0.1, dj=0.25)
    slope = np.polyfit(np.log(res.f), np.log(res.psd_trace), 1)[0]
    assert slope == pytest.approx(alpha, abs=0.05)
    expected = 3 * 1e-2 * res.f ** alpha * wavelets.power_law_bias(alpha)
    ratio = res.psd_trace / expected
    assert np.median(ratio) == pytest.approx(1.0, abs=0.02)
    np.testing.assert_allclose(ratio, 1.0, atol=0.1)   # largest scales have few independent samples


def test_power_law_bias_values():
    assert wavelets.power_law_bias(0.0) == pytest.approx(1.0, abs=1e-12)
    assert 1.03 < wavelets.power_law_bias(-5 / 3) < 1.08


def test_isotropic_noise_has_flat_angle_dependence_and_one_third_parallel():
    rng = np.random.default_rng(22)
    B = colored_noise(2 ** 16, 1.0, power_law(-5 / 3, 1e-5), m=3, rng=rng) + np.array([0.0, 2.0, 0.0])
    V = np.tile([750.0, 0.0, 0.0], (len(B), 1))
    res = wavelets.analyse(B, V, 1.0, 5e-3, 0.1, dj=0.5)
    # strong steady field along T: theta_VB ~ 90 deg at every scale
    counts = res.theta_count.sum(axis=0)
    assert counts[:, -2:].sum() / counts.sum() > 0.99
    np.testing.assert_allclose(res.psd_par / res.psd_trace, 1 / 3, atol=0.05)


def test_coi_and_gap_mask():
    rng = np.random.default_rng(23)
    B = rng.standard_normal((2 ** 14, 3)) + np.array([1.0, 0, 0])
    B[8000:8200] = np.nan
    V = np.tile([700.0, 0.0, 0.0], (len(B), 1))
    res = wavelets.analyse(B, V, 1.0, 0.01, 0.2, dj=0.5)
    assert np.all(res.valid_fraction < 1.0)
    assert np.all(np.isfinite(res.psd_trace))
