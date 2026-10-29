"""Instrument package for LensForge.

Provides telescope Point Spread Function (PSF) and detector noise modeling.
"""

from lensforge.instrument.noise import add_observation_noise
from lensforge.instrument.psf import BasePSF, GaussianPSF, PixelatedPSF

__all__ = [
    "BasePSF",
    "GaussianPSF",
    "PixelatedPSF",
    "add_observation_noise",
]
