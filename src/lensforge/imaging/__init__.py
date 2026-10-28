"""Imaging and source modeling package for LensForge.

Modules:
    source_models: Surface-brightness profiles for astrophysical background galaxies.
"""

from lensforge.imaging.source_models import (
    BaseSourceModel,
    GaussianSource,
    simulate_lensed_image,
)

__all__ = [
    "BaseSourceModel",
    "GaussianSource",
    "simulate_lensed_image",
]
