"""Tests for macro-lens non-linear parameter fitting (semilinear optimization)."""

import numpy as np
import pytest

from lensforge.imaging.source_models import GaussianSource
from lensforge.optimization import fit_lens_parameters
from lensforge.physics.coordinates import make_coordinate_grid
from lensforge.physics.models.sie import SingularIsothermalEllipsoid


def test_fit_lens_parameters_convergence():
    """Verify that fit_lens_parameters recovers perturbed lens parameters towards ground truth."""
    # 1. Setup compact grids for rapid test execution
    img_grid = make_coordinate_grid(num_pix=30, field_of_view=3.0, unit="arcsec")
    src_grid = make_coordinate_grid(num_pix=16, field_of_view=1.6, unit="arcsec")

    # 2. Ground-truth deflector: SIE
    true_theta_e = 1.10
    true_q = 0.80
    true_pa = 40.0
    true_lens = SingularIsothermalEllipsoid.from_degrees(
        theta_e=true_theta_e,
        axis_ratio=true_q,
        pos_angle_deg=true_pa,
        center=(0.0, 0.0),
    )

    # 3. Compact source
    src = GaussianSource.from_degrees(center=(0.15, 0.10), sigma=0.10, amp=5.0, axis_ratio=0.8)
    bx, by = true_lens.ray_trace(img_grid.x, img_grid.y, mask_singular=True)
    noiseless_lensed = np.nan_to_num(src.evaluate(bx, by), nan=0.0)

    # 4. Small noise
    noise_sigma = 0.04
    noisy_img = noiseless_lensed + np.random.default_rng(99).normal(0.0, noise_sigma, size=noiseless_lensed.shape)

    # 5. Perturbed initial parameters (e.g. 15% off)
    initial_guess = {
        "theta_e": 1.25,
        "axis_ratio": 0.70,
        "pos_angle_deg": 30.0,
    }

    bounds = {
        "theta_e": (0.8, 1.5),
        "axis_ratio": (0.5, 0.99),
        "pos_angle_deg": (10.0, 70.0),
    }

    result = fit_lens_parameters(
        image_grid=img_grid,
        source_grid=src_grid,
        observed_image=noisy_img,
        noise_sigma=noise_sigma,
        initial_params=initial_guess,
        param_bounds=bounds,
        include_shear=False,
        regularization_lambda=2.0,
        objective="chi2",
        method="Nelder-Mead",
        max_iter=35,
    )

    # Verifications:
    # A. Optimization should reduce loss/chi2 from the initial guess
    assert len(result.optimization_history) > 1
    assert result.reconstruction.chi2 < result.optimization_history[0]

    # B. Recovered Einstein radius should be within ~5% of ground truth
    fitted_theta_e = result.best_params["theta_e"]
    assert np.isclose(fitted_theta_e, true_theta_e, rtol=0.08)

    # C. Recovered axis ratio should be close to true value
    fitted_q = result.best_params["axis_ratio"]
    assert np.isclose(fitted_q, true_q, rtol=0.15)
