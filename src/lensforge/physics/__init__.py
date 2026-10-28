"""Physics package for LensForge.

Modules:
    coordinates: Representation, grids, and transformations for angular coordinates.
    distances: Angular diameter distance utilities and Einstein radius calculations.
    lens_equation: Ray-tracing and deflection mapping.
    lens_potential: Effective gravitational lens potential and derivative evaluations.
    magnification: Magnification tensor, point-mass magnifications, and parity.
    models: Parametric lens models (PointMassLens, SIS, SIE, ExternalShear, CompositeLens).
"""

from lensforge.physics.coordinates import (
    CoordinateGrid,
    make_coordinate_grid,
    validate_coordinates,
)
from lensforge.physics.distances import (
    einstein_radius,
    einstein_radius_arcsec,
    einstein_radius_sis,
    einstein_radius_sis_arcsec,
    validate_physical_distances,
    validate_sis_distances,
)
from lensforge.physics.lens_equation import (
    lens_equation,
    ray_trace_image_to_source,
)
from lensforge.physics.lens_potential import (
    SingularityError,
    gradient_finite_difference,
    point_mass_potential,
)
from lensforge.physics.magnification import (
    point_mass_absolute_magnification,
    point_mass_magnification,
    point_mass_total_magnification,
)
from lensforge.physics.models import (
    BaseLensModel,
    CompositeLens,
    ExternalShear,
    PointMassLens,
    SingularIsothermalEllipsoid,
    SingularIsothermalSphere,
)

__all__ = [
    "CoordinateGrid",
    "make_coordinate_grid",
    "validate_coordinates",
    "einstein_radius",
    "einstein_radius_arcsec",
    "einstein_radius_sis",
    "einstein_radius_sis_arcsec",
    "validate_physical_distances",
    "validate_sis_distances",
    "lens_equation",
    "ray_trace_image_to_source",
    "SingularityError",
    "point_mass_potential",
    "gradient_finite_difference",
    "point_mass_magnification",
    "point_mass_absolute_magnification",
    "point_mass_total_magnification",
    "BaseLensModel",
    "PointMassLens",
    "SingularIsothermalSphere",
    "SingularIsothermalEllipsoid",
    "ExternalShear",
    "CompositeLens",
]
