"""Characteristic plasma scales and their Doppler-shifted frequencies.

Physical constants are CODATA 2018.  Under Taylor's frozen-in hypothesis
(Taylor 1938, doi:10.1098/rspa.1938.0032), a spatial scale L advected past
the spacecraft at speed V appears at f = V / (2 pi L), i.e. k = 2 pi f / V and
k L = 1.  The hypothesis needs the fluctuation phase speeds (~ V_A for
Alfvénic turbulence) to be much smaller than V.
"""

from __future__ import annotations

import numpy as np

E_CHARGE = 1.602176634e-19       # C (exact)
M_PROTON = 1.67262192369e-27     # kg
MU0 = 1.25663706212e-6           # N A^-2
EPS0 = 8.8541878128e-12          # F m^-1
C_LIGHT = 299792458.0            # m s^-1 (exact)
K_BOLTZMANN = 1.380649e-23       # J K^-1 (exact)


def alfven_speed(B_nT, n_cm3):
    """Proton Alfvén speed B / sqrt(mu0 n m_p) in km/s (alphas neglected)."""
    return np.asarray(B_nT) * 1e-9 / np.sqrt(MU0 * np.asarray(n_cm3) * 1e6 * M_PROTON) / 1e3


def proton_inertial_length(n_cm3):
    """d_i = c / omega_pi in km, omega_pi = sqrt(n e^2 / (eps0 m_p))."""
    wpi = np.sqrt(np.asarray(n_cm3) * 1e6 * E_CHARGE ** 2 / (EPS0 * M_PROTON))
    return C_LIGHT / wpi / 1e3


def proton_gyrofrequency(B_nT):
    """f_ci = e B / (2 pi m_p) in Hz."""
    return E_CHARGE * np.asarray(B_nT) * 1e-9 / (2 * np.pi * M_PROTON)


def proton_gyroradius(B_nT, T_K):
    """rho_i = v_th / Omega_i in km with v_th = sqrt(2 k_B T / m_p)."""
    vth = np.sqrt(2 * K_BOLTZMANN * np.asarray(T_K) / M_PROTON)
    return vth / (2 * np.pi * proton_gyrofrequency(B_nT)) / 1e3


def taylor_frequency(L_km, V_kms):
    """Spacecraft-frame frequency f = V / (2 pi L) of a scale L."""
    return np.asarray(V_kms) / (2 * np.pi * np.asarray(L_km))


def characteristic_scales(B_nT: float, n_cm3: float, V_kms: float) -> dict:
    """Scales available from |B|, n_p and V (no temperature in the dataset,
    so the proton gyroradius and plasma beta cannot be evaluated)."""
    vA = float(alfven_speed(B_nT, n_cm3))
    di = float(proton_inertial_length(n_cm3))
    fci = float(proton_gyrofrequency(B_nT))
    return {
        "B_nT": float(B_nT), "n_cm3": float(n_cm3), "V_kms": float(V_kms),
        "V_A_kms": vA,
        "V_A_over_V": vA / float(V_kms),
        "d_i_km": di,
        "f_ci_Hz": fci,
        # k d_i = 1 is also the cold-plasma cyclotron-resonance wavenumber
        # Omega_i / V_A, since d_i = V_A / Omega_i.  The warm resonance
        # Omega_i / (V_A + v_th) (Bruno & Trenchi 2014) needs T_p.
        "f_di_Hz": float(taylor_frequency(di, V_kms)),
    }
