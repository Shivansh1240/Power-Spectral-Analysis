"""psa: power spectral analysis of solar wind magnetic field turbulence."""

from . import alfvenicity, fitting, io, legacy, plasma, qc, spectra, structure, synthetic, wavelets

__all__ = ["alfvenicity", "fitting", "io", "legacy", "plasma", "qc", "spectra", "structure",
           "synthetic", "wavelets"]
__version__ = "1.0.0"
