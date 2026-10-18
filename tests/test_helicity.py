"""Sign convention of the reduced magnetic helicity.

B(x) = (0, cos kx, -sin kx) satisfies curl B = +k B (positive magnetic
helicity).  The plasma flows past the spacecraft along +R at speed V, so the
spacecraft samples x = -V t.
"""
import numpy as np
import pytest

from psa import spectra, wavelets

DT, N, V, K = 1.0, 2 ** 14, 700.0, 2 * np.pi / 14000.0   # 20-s period in the spacecraft frame


def _helix(handedness: int):
    x = -V * np.arange(N) * DT
    rng = np.random.default_rng(30)
    B = np.column_stack([np.full(N, -1.0), np.cos(K * x), -handedness * np.sin(K * x)])
    return B + 0.01 * rng.standard_normal(B.shape)


@pytest.mark.parametrize("hand", [+1, -1])
def test_fourier_helicity_sign(hand):
    s = spectra.welch(_helix(hand), DT, 2048, cross=True)
    sig = spectra.reduced_magnetic_helicity(s)
    k = np.argmax(s.psd[:, 1])
    assert sig[k] == pytest.approx(hand, abs=0.01)


@pytest.mark.parametrize("hand", [+1, -1])
def test_wavelet_helicity_sign(hand):
    Vv = np.tile([V, 0.0, 0.0], (N, 1))
    res = wavelets.analyse(_helix(hand), Vv, DT, 0.02, 0.1, dj=0.125, block_seconds=N)
    j = np.argmin(abs(res.f - V * K / (2 * np.pi)))
    num, den = res.hel_num.sum(axis=(0, 2)), res.hel_den.sum(axis=(0, 2))
    assert num[j] / den[j] == pytest.approx(hand, abs=0.02)
