"""Tests for point-mass gravitational lens model, Einstein radius, deflection, and potential."""

import numpy as np
import pytest

from lensforge.constants import (
    C_LIGHT,
    G_NEWTON,
    M_SUN,
    MEGAPARSEC,
    rad2arcsec,
)
from lensforge.physics.distances import (
    einstein_radius,
    einstein_radius_arcsec,
    validate_physical_distances,
)
from lensforge.physics.lens_equation import lens_equation
from lensforge.physics.lens_potential import (
    SingularityError,
    gradient_finite_difference,
    point_mass_potential,
)
from lensforge.physics.models.point_mass import PointMassLens


def test_einstein_radius_analytic():
    """Verify Einstein radius calculation against hand-calculated analytical value."""
    # Given: M = 1e11 M_sun, D_l = 1000 Mpc, D_s = 1800 Mpc, D_ls = 1100 Mpc
    mass = 1.0e11 * M_SUN
    d_l = 1000.0 * MEGAPARSEC
    d_s = 1800.0 * MEGAPARSEC
    d_ls = 1100.0 * MEGAPARSEC

    theta_e_rad = einstein_radius(mass, d_l, d_s, d_ls)

    # Analytic expression: sqrt( (4 * G * M / c^2) * (D_ls / (D_l * D_s)) )
    expected_rad = np.sqrt((4.0 * G_NEWTON * mass / (C_LIGHT**2)) * (d_ls / (d_l * d_s)))
    assert np.isclose(theta_e_rad, expected_rad, rtol=1e-15)

    # In arcsec: for ~10^11 M_sun at cosmological distances, theta_E is typically ~ 0.5 - 1.5 arcsec
    theta_e_arcsec = einstein_radius_arcsec(mass, d_l, d_s, d_ls)
    assert 0.3 < theta_e_arcsec < 2.0
    assert np.isclose(theta_e_arcsec, rad2arcsec(theta_e_rad), rtol=1e-15)


def test_invalid_physical_inputs():
    """Verify that invalid masses or distances raise ValueError."""
    valid_mass = 1.0e11 * M_SUN
    valid_dl = 1000.0 * MEGAPARSEC
    valid_ds = 1800.0 * MEGAPARSEC
    valid_dls = 1100.0 * MEGAPARSEC

    # Negative / zero mass
    with pytest.raises(ValueError, match="Lens mass must be strictly positive"):
        einstein_radius(-1.0, valid_dl, valid_ds, valid_dls)
    with pytest.raises(ValueError, match="Lens mass must be strictly positive"):
        einstein_radius(0.0, valid_dl, valid_ds, valid_dls)

    # Negative / zero distances
    with pytest.raises(ValueError, match="Observer-lens distance d_l must be strictly positive"):
        einstein_radius(valid_mass, -100.0, valid_ds, valid_dls)
    with pytest.raises(ValueError, match="Observer-source distance d_s must be strictly positive"):
        einstein_radius(valid_mass, valid_dl, 0.0, valid_dls)
    with pytest.raises(ValueError, match="Lens-source distance d_ls must be strictly positive"):
        einstein_radius(valid_mass, valid_dl, valid_ds, -5.0)

    # Cosmological ordering violation: d_s <= d_l
    with pytest.raises(ValueError, match="must be strictly greater than lens distance"):
        einstein_radius(valid_mass, valid_ds, valid_dl, valid_dls)

    # Cosmological ordering violation: d_ls >= d_s
    with pytest.raises(ValueError, match="cannot exceed or equal observer-to-source"):
        einstein_radius(valid_mass, valid_dl, valid_ds, valid_ds + 10.0)


def test_point_mass_lens_deflection_values():
    """Verify point mass deflection angle magnitude and direction."""
    theta_e = 1.2
    lens = PointMassLens(theta_e=theta_e, center=(0.0, 0.0))

    # Point along x axis at theta_x = theta_E
    ax, ay = lens.deflection(theta_e, 0.0)
    # alpha = theta_E^2 * theta / |theta|^2 -> at |theta| = theta_E, |alpha| = theta_E
    assert np.isclose(ax, theta_e, rtol=1e-14)
    assert np.isclose(ay, 0.0, atol=1e-14)

    # Point along y axis at theta_y = 2 * theta_E
    ax, ay = lens.deflection(0.0, 2.0 * theta_e)
    # alpha_y = theta_E^2 / (2 * theta_E) = 0.5 * theta_E
    assert np.isclose(ax, 0.0, atol=1e-14)
    assert np.isclose(ay, 0.5 * theta_e, rtol=1e-14)

    # With non-zero lens center
    center = (0.5, -0.3)
    lens_off = PointMassLens(theta_e=theta_e, center=center)
    ax_off, ay_off = lens_off.deflection(center[0] + theta_e, center[1])
    assert np.isclose(ax_off, theta_e, rtol=1e-14)
    assert np.isclose(ay_off, 0.0, atol=1e-14)


def test_lens_equation_mapping():
    """Verify lens equation beta = theta - alpha."""
    theta_x = np.array([1.5, 2.0, -1.0])
    theta_y = np.array([0.5, -1.0, 1.0])
    theta_e = 1.0
    lens = PointMassLens(theta_e=theta_e)

    alpha_x, alpha_y = lens.deflection(theta_x, theta_y)
    beta_x, beta_y = lens.ray_trace(theta_x, theta_y)

    assert np.allclose(beta_x, theta_x - alpha_x)
    assert np.allclose(beta_y, theta_y - alpha_y)


def test_singularity_handling():
    """Verify explicit singularity handling at lens center."""
    center = (0.2, -0.4)
    lens = PointMassLens(theta_e=1.0, center=center)

    # Exactly at singularity without mask_singular -> must raise SingularityError
    with pytest.raises(SingularityError, match="Deflection field is singular at lens center"):
        lens.deflection(center[0], center[1], mask_singular=False)

    with pytest.raises(SingularityError, match="Evaluation coordinate coincides exactly with point-mass"):
        lens.potential(center[0], center[1], mask_singular=False)

    # With mask_singular=True -> must return NaN
    ax, ay = lens.deflection(center[0], center[1], mask_singular=True)
    assert np.isnan(ax)
    assert np.isnan(ay)

    psi = lens.potential(center[0], center[1], mask_singular=True)
    assert np.isneginf(psi)


def test_gradient_potential_agrees_with_deflection():
    """Test agreement between numerical gradient of potential and analytical deflection field.

    In gravitational lensing: alpha(theta) = grad_theta psi(theta).
    This must hold everywhere away from the central singularity.
    """
    theta_e = 1.35
    center = (0.25, -0.15)
    lens = PointMassLens(theta_e=theta_e, center=center)

    # Test grid of points safely away from singularity
    x_test = np.linspace(-3.0, 3.0, 15)
    y_test = np.linspace(-3.0, 3.0, 15)
    xx, yy = np.meshgrid(x_test, y_test)

    # Filter out points too close to center to avoid numerical gradient discretization error
    dist = np.hypot(xx - center[0], yy - center[1])
    mask = dist > 0.1
    xx_valid = xx[mask]
    yy_valid = yy[mask]

    # Analytical deflection
    ax_analytic, ay_analytic = lens.deflection(xx_valid, yy_valid)

    # Potential wrapper
    def potential_func(x, y):
        return lens.potential(x, y, mask_singular=True)

    # Finite difference gradient with step size h = 1e-6
    h = 1.0e-6
    grad_x, grad_y = gradient_finite_difference(potential_func, xx_valid, yy_valid, step_size=h)

    # Gradient must match deflection within finite-difference approximation tolerance
    # Central difference error is O(h^2)
    assert np.allclose(grad_x, ax_analytic, rtol=1e-5, atol=1e-5)
    assert np.allclose(grad_y, ay_analytic, rtol=1e-5, atol=1e-5)


def test_potential_reference_scale_invariance():
    """Test that altering theta_ref changes potential by a constant but leaves gradient invariant."""
    theta_e = 1.0
    x, y = 1.8, 0.7

    psi1 = point_mass_potential(x, y, theta_e=theta_e, theta_ref=1.0)
    psi2 = point_mass_potential(x, y, theta_e=theta_e, theta_ref=2.5)

    # Difference psi1 - psi2 = theta_E^2 * ln(theta_ref_2 / theta_ref_1)
    expected_diff = (theta_e**2) * np.log(2.5 / 1.0)
    assert np.isclose(psi1 - psi2, expected_diff, rtol=1e-14)

    # Gradient is independent of reference scale
    h = 1e-6
    gx1, gy1 = gradient_finite_difference(lambda x, y: point_mass_potential(x, y, theta_e, theta_ref=1.0), x, y, h)
    gx2, gy2 = gradient_finite_difference(lambda x, y: point_mass_potential(x, y, theta_e, theta_ref=2.5), x, y, h)
    assert np.isclose(gx1, gx2, rtol=1e-6)
    assert np.isclose(gy1, gy2, rtol=1e-6)
