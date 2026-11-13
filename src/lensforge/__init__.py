"""LensForge: Computational Astrophysics Research Repository.

LensForge provides numerical methods, forward models, and inversion frameworks
for gravitational lens modeling and galaxy source reconstruction.
"""

__version__ = "0.5.0"
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
    EllipticalPowerLaw,
    ExternalShear,
    NFWHalo,
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
from lensforge.cosmology import (
    FlatLambdaCDMCosmology,
    PLANCK18,
    fermat_potential,
    infer_h0_from_time_delay,
    time_delay_days,
    time_delay_seconds,
)
from lensforge.data import (
    AstroObservation,
    create_annular_mask,
    estimate_sky_background,
    load_fits_observation,
    save_fits_observation,
)
from lensforge.optimization import (
    MacroLensFitResult,
    fit_lens_parameters,
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
    "EllipticalPowerLaw",
    "NFWHalo",
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
    "FlatLambdaCDMCosmology",
    "PLANCK18",
    "fermat_potential",
    "time_delay_seconds",
    "time_delay_days",
    "infer_h0_from_time_delay",
    "AstroObservation",
    "load_fits_observation",
    "save_fits_observation",
    "estimate_sky_background",
    "create_annular_mask",
    "MacroLensFitResult",
    "fit_lens_parameters",
]
