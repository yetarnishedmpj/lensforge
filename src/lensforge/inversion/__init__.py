"""Inversion and galaxy source reconstruction package for LensForge.

Provides:
- SourceReconstructor: Semilinear inversion engine (Warren & Dye 2003 / Suyu et al. 2006).
- ReconstructionResult: Output container with source, model image, residuals, chi2, and evidence.
- build_lens_mapping_matrix: Sparse ray-tracing bilinear interpolation operator L.
- build_regularization_matrix: Spatial prior operator R (curvature, gradient, zeroth).
"""

from lensforge.inversion.mapping import build_lens_mapping_matrix
from lensforge.inversion.reconstructor import (
    ReconstructionResult,
    SourceReconstructor,
)
from lensforge.inversion.regularization import build_regularization_matrix

__all__ = [
    "SourceReconstructor",
    "ReconstructionResult",
    "build_lens_mapping_matrix",
    "build_regularization_matrix",
]

