import numpy as np
import pytest

from psa import alfvenicity, structure
from psa.synthetic import colored_noise, power_law


def test_s2_slope_matches_spectral_index():
    rng = np.random.default_rng(40)
    B = colored_noise(2 ** 18, 1.0, power_law(-5 / 3), m=3, rng=rng)
    lags = structure.log_lags(2000, 30, 20)
    S = structure.structure_functions(B, lags)
    zeta = np.polyfit(np.log(lags), np.log(S[2]), 1)[0]
    assert zeta == pytest.approx(2 / 3, abs=0.04)


def test_gaussian_flatness_is_three():
    rng = np.random.default_rng(41)
    B = colored_noise(2 ** 17, 1.0, power_law(-5 / 3), m=3, rng=rng)
    F = structure.flatness(B, np.array([10, 100]))
    np.testing.assert_allclose(F, 3.0, atol=0.15)


@pytest.mark.parametrize("polarity", [+1.0, -1.0])
def test_outward_alfvenic_fluctuations(polarity):
    rng = np.random.default_rng(42)
    n = np.full(2 ** 14, 0.3)
    dB = colored_noise(2 ** 14, 60.0, power_law(-1.5), m=3, rng=rng)
    B = np.array([polarity * 1.0, 0, 0]) + dB
    b = alfvenicity.to_alfven_units(B, n)
    V = np.array([780.0, 0, 0]) - polarity * (b - b.mean(axis=0))   # z- = 0
    out = alfvenicity.alfvenicity_spectra(V, B, n, 60.0, 1024, polarity)
    np.testing.assert_allclose(out["sigma_c"], 1.0, atol=1e-6)
    np.testing.assert_allclose(out["sigma_r"], 0.0, atol=1e-6)


def test_matched_smooth_preserves_constant_and_handles_nan():
    x = np.ones(1000)
    x[400:410] = np.nan
    y = alfvenicity.matched_smooth(x, 50)
    np.testing.assert_allclose(y[np.isfinite(y)], 1.0)


def test_running_median_removes_single_dropout():
    x = np.full(200, 0.25)
    x[100] = 0.004
    np.testing.assert_allclose(alfvenicity.running_median(x, 11), 0.25)


def test_legacy_returns_nan_on_gappy_input():
    from psa import legacy
    x = np.sin(np.arange(20000) / 50.0) + 0.01 * np.random.default_rng(0).standard_normal(20000)
    x[500] = np.nan
    with np.errstate(all="ignore"):
        slope = legacy.legacy_slope(x)[0]
    assert np.isnan(slope)
