"""LensForge: Computational Astrophysics Research Repository.

LensForge provides numerical methods, forward models, and inversion frameworks
for gravitational lens modeling and galaxy source reconstruction.
"""

__version__ = "0.3.0"
__author__ = "LensForge Research Team"

from lensforge.constants import (
    C_LIGHT,
    G_NEWTON,
    M_SUN,
    PARSEC,
    MEGAPARSEC,
    rad2arcsec,
    arcsec2rad,
    deg2rad,
    rad2deg,
    arcsec2deg,
    deg2arcsec,
)
from lensforge.physics.models import (
    BaseLensModel,
    CompositeLens,
    ExternalShear,
    PointMassLens,
    SingularIsothermalEllipsoid,
    SingularIsothermalSphere,
)
from lensforge.instrument import (
    BasePSF,
    GaussianPSF,
    PixelatedPSF,
    add_observation_noise,
)
from lensforge.inversion import (
    ReconstructionResult,
    SourceReconstructor,
    build_lens_mapping_matrix,
    build_regularization_matrix,
)

__all__ = [
    "__version__",
    "C_LIGHT",
    "G_NEWTON",
    "M_SUN",
    "PARSEC",
    "MEGAPARSEC",
    "rad2arcsec",
    "arcsec2rad",
    "deg2rad",
    "rad2deg",
    "arcsec2deg",
    "deg2arcsec",
    "BaseLensModel",
    "PointMassLens",
    "SingularIsothermalSphere",
    "SingularIsothermalEllipsoid",
    "ExternalShear",
    "CompositeLens",
    "BasePSF",
    "GaussianPSF",
    "PixelatedPSF",
    "add_observation_noise",
    "SourceReconstructor",
    "ReconstructionResult",
    "build_lens_mapping_matrix",
    "build_regularization_matrix",
]
