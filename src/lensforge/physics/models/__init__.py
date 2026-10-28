"""Lens models package for LensForge.

Contains parametric mass distributions:
    BaseLensModel: Abstract base class for all lens models.
    PointMassLens: Schwarzschild point-mass lens model.
    SingularIsothermalSphere: Singular Isothermal Sphere (SIS) galaxy model.
    SingularIsothermalEllipsoid: Singular Isothermal Ellipsoid (SIE) galaxy model.
    ExternalShear: Tidal gravitational shear model.
    CompositeLens: Composite multi-component lens model.
"""

from lensforge.physics.models.base import BaseLensModel
from lensforge.physics.models.composite import CompositeLens
from lensforge.physics.models.point_mass import PointMassLens
from lensforge.physics.models.shear import ExternalShear
from lensforge.physics.models.sie import SingularIsothermalEllipsoid
from lensforge.physics.models.sis import SingularIsothermalSphere

__all__ = [
    "BaseLensModel",
    "PointMassLens",
    "SingularIsothermalSphere",
    "SingularIsothermalEllipsoid",
    "ExternalShear",
    "CompositeLens",
]
