import pytest

from psa import plasma


def test_reference_values():
    assert plasma.proton_gyrofrequency(1.0) == pytest.approx(0.0152452, rel=1e-5)
    assert plasma.proton_inertial_length(1.0) == pytest.approx(227.71, rel=1e-4)
    assert plasma.alfven_speed(1.0, 1.0) == pytest.approx(21.812, rel=1e-4)


def test_di_equals_va_over_omega():
    import numpy as np
    B, n = 0.9, 0.26
    di = plasma.alfven_speed(B, n) / (2 * np.pi * plasma.proton_gyrofrequency(B))
    assert di == pytest.approx(plasma.proton_inertial_length(n), rel=1e-6)
