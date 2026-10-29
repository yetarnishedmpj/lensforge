"""Tests for regularized galaxy source inversion (Warren & Dye semilinear method)."""

import numpy as np
import pytest

from lensforge.imaging.source_models import GaussianSource
from lensforge.instrument.psf import GaussianPSF
from lensforge.inversion.mapping import build_lens_mapping_matrix
from lensforge.inversion.reconstructor import SourceReconstructor
from lensforge.inversion.regularization import build_regularization_matrix
from lensforge.physics.coordinates import make_coordinate_grid
from lensforge.physics.models.sie import SingularIsothermalEllipsoid


def test_regularization_matrices_symmetry_and_rank():
    """Verify symmetry and positive-definiteness of regularization matrices."""
    nx, ny = 12, 12
    n_tot = nx * ny

    for method in ["zeroth", "gradient", "curvature"]:
        r = build_regularization_matrix(nx, ny, method=method)
        assert r.shape == (n_tot, n_tot)

        # Symmetry: R == R^T
        assert np.allclose(r, r.T, atol=1e-12)

        # Positive semi-definite / positive-definite: eigenvalues >= 0
        eigenvalues = np.linalg.eigvalsh(r)
        assert np.all(eigenvalues >= 0.0)


def test_lens_mapping_matrix_sparsity_and_weights():
    """Verify that mapping matrix L rows sum to <= 1.0 (bilinear interpolation weights)."""
    img_grid = make_coordinate_grid(num_pix=30, field_of_view=3.0, unit="arcsec")
    src_grid = make_coordinate_grid(num_pix=20, field_of_view=2.0, unit="arcsec")
    sie = SingularIsothermalEllipsoid.from_degrees(theta_e=1.0, axis_ratio=0.8, pos_angle_deg=20.0)

    l_matrix = build_lens_mapping_matrix(img_grid, src_grid, sie)
    assert l_matrix.shape == (30 * 30, 20 * 20)

    # For any ray that landed fully inside the source grid, row sum must be 1.0
    row_sums = np.array(l_matrix.sum(axis=1)).ravel()
    # Bilinear weights must sum to <= 1.0 (1.0 inside, 0.0 if ray missed or hit singularity)
    assert np.all(row_sums <= 1.0000001)
    assert np.any(np.isclose(row_sums, 1.0, atol=1e-5))


def test_end_to_end_source_reconstruction_with_psf():
    """Verify that SourceReconstructor accurately inverts a lensed image and recovers source flux."""
    # 1. Setup grids
    img_grid = make_coordinate_grid(num_pix=40, field_of_view=3.2, unit="arcsec")
    src_grid = make_coordinate_grid(num_pix=25, field_of_view=1.6, unit="arcsec")

    # 2. Setup lens model: SIE
    sie = SingularIsothermalEllipsoid.from_degrees(
        theta_e=0.9,
        axis_ratio=0.75,
        pos_angle_deg=30.0,
        center=(0.0, 0.0),
    )

    # 3. Setup ground truth compact source galaxy
    src_pos = (0.12, 0.08)
    src_model = GaussianSource.from_degrees(
        center=src_pos,
        sigma=0.08,
        amp=5.0,
        axis_ratio=0.85,
        pos_angle_deg=20.0,
    )
    true_source = src_model.evaluate(src_grid.x, src_grid.y)

    # 4. Generate lensed image
    bx, by = sie.ray_trace(img_grid.x, img_grid.y, mask_singular=True)
    noiseless_lensed = np.nan_to_num(src_model.evaluate(bx, by), nan=0.0)

    # 5. Add PSF blurring
    psf = GaussianPSF(fwhm=0.10, pixel_scale=img_grid.pixel_scale, num_pix=11)
    blurred_lensed = psf.convolve(noiseless_lensed)

    # 6. Add known Gaussian noise
    sigma_noise = 0.05
    rng = np.random.default_rng(123)
    noisy_data = blurred_lensed + rng.normal(0.0, sigma_noise, size=blurred_lensed.shape)

    # 7. Initialize reconstructor and solve inverse problem
    reconstructor = SourceReconstructor(
        image_grid=img_grid,
        source_grid=src_grid,
        lens_model=sie,
        psf=psf,
        regularization_method="curvature",
    )

    result = reconstructor.reconstruct(
        data_image=noisy_data,
        noise_sigma=sigma_noise,
        regularization_lambda=5.0,
    )

    # Verifications:
    # A. Residuals should be consistent with noise: reduced chi2 ~ 1.0 +/- 0.4
    assert 0.6 < result.reduced_chi2 < 1.6

    # B. Peak of reconstructed source should align with true source centroid
    peak_idx = np.unravel_index(np.argmax(result.reconstructed_source), result.reconstructed_source.shape)
    peak_x = src_grid.x[peak_idx]
    peak_y = src_grid.y[peak_idx]

    dist_to_true = np.hypot(peak_x - src_pos[0], peak_y - src_pos[1])
    assert dist_to_true < 0.15  # within 2 source pixels!

    # C. Model prediction in image plane should closely match the data
    rms_image_diff = np.sqrt(np.mean((result.model_image - blurred_lensed) ** 2))
    assert rms_image_diff < 0.10
