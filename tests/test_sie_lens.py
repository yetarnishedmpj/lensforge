"""Tests for Singular Isothermal Ellipsoid (SIE) gravitational lens model."""

import numpy as np
import pytest

from lensforge.physics.lens_potential import SingularityError, gradient_finite_difference
from lensforge.physics.models.sie import SingularIsothermalEllipsoid
from lensforge.physics.models.sis import SingularIsothermalSphere


def test_sie_parameter_validation():
    """Verify input validation for Einstein radius and axis ratio."""
    # Invalid Einstein radius
    with pytest.raises(ValueError, match="Einstein radius theta_e must be strictly positive"):
        SingularIsothermalEllipsoid(theta_e=0.0, axis_ratio=0.8)
    with pytest.raises(ValueError, match="Einstein radius theta_e must be strictly positive"):
        SingularIsothermalEllipsoid(theta_e=-1.2, axis_ratio=0.8)

    # Invalid axis ratio q (must be 0 < q <= 1)
    with pytest.raises(ValueError, match="Axis ratio q must be in"):
        SingularIsothermalEllipsoid(theta_e=1.0, axis_ratio=0.0)
    with pytest.raises(ValueError, match="Axis ratio q must be in"):
        SingularIsothermalEllipsoid(theta_e=1.0, axis_ratio=-0.5)
    with pytest.raises(ValueError, match="Axis ratio q must be in"):
        SingularIsothermalEllipsoid(theta_e=1.0, axis_ratio=1.05)


def test_sie_spherical_limit_matches_sis():
    """Verify that when q -> 1.0, SIE deflection identically matches SIS deflection."""
    theta_e = 1.2
    center = (0.2, -0.15)
    sis = SingularIsothermalSphere(theta_e=theta_e, center=center)
    sie_sph = SingularIsothermalEllipsoid(theta_e=theta_e, axis_ratio=1.0, center=center)

    # Test coordinate grid
    xs = np.linspace(-3.0, 3.0, 25)
    ys = np.linspace(-3.0, 3.0, 25)
    gx, gy = np.meshgrid(xs, ys)
    # Mask center
    mask = np.hypot(gx - center[0], gy - center[1]) > 0.05
    tx = gx[mask]
    ty = gy[mask]

    ax_sis, ay_sis = sis.deflection(tx, ty)
    ax_sie, ay_sie = sie_sph.deflection(tx, ty)

    assert np.allclose(ax_sie, ax_sis, rtol=1e-12, atol=1e-12)
    assert np.allclose(ay_sie, ay_sis, rtol=1e-12, atol=1e-12)


def test_sie_potential_gradient_matches_deflection():
    """Verify that numerical gradient of psi matches analytical deflection for rotated SIE."""
    theta_e = 1.15
    center = (0.1, -0.25)
    q_values = [0.4, 0.7, 0.85]
    phi_deg_values = [0.0, 35.0, 78.0, 140.0]

    tx = np.array([0.7, -0.5, 1.2, -1.1])
    ty = np.array([0.9, 0.6, -0.8, -1.3])

    for q in q_values:
        for phi_deg in phi_deg_values:
            sie = SingularIsothermalEllipsoid.from_degrees(
                theta_e=theta_e,
                axis_ratio=q,
                pos_angle_deg=phi_deg,
                center=center,
            )
            ax, ay = sie.deflection(tx, ty)
            grad_x, grad_y = gradient_finite_difference(sie.potential, tx, ty, step_size=1e-6)

            assert np.allclose(grad_x, ax, rtol=1e-5, atol=1e-5)
            assert np.allclose(grad_y, ay, rtol=1e-5, atol=1e-5)


def test_sie_deflection_anisotropy_analytical():
    """Verify analytical deflection values along principal major and minor axes."""
    theta_e = 1.0
    q = 0.6
    s = np.sqrt(1.0 - q**2)
    sie = SingularIsothermalEllipsoid(theta_e=theta_e, axis_ratio=q, pos_angle=0.0, center=(0.0, 0.0))

    # Along major axis (x > 0, y = 0):
    # r = q * x, so s * x / r = s / q
    # alpha_x = (theta_E * sqrt(q) / s) * arctan(s / q)
    # alpha_y = 0
    x_test = 1.5
    ax, ay = sie.deflection(x_test, 0.0)
    expected_ax = (theta_e * np.sqrt(q) / s) * np.arctan(s / q)
    assert np.isclose(ax, expected_ax, rtol=1e-13)
    assert np.isclose(ay, 0.0, atol=1e-13)

    # Along minor axis (x = 0, y > 0):
    # r = y, so s * y / r = s
    # alpha_x = 0
    # alpha_y = (theta_E * sqrt(q) / s) * artanh(s)
    y_test = 1.5
    ax_y, ay_y = sie.deflection(0.0, y_test)
    expected_ay = (theta_e * np.sqrt(q) / s) * np.arctanh(s)
    assert np.isclose(ax_y, 0.0, atol=1e-13)
    assert np.isclose(ay_y, expected_ay, rtol=1e-13)

    # For q < 1, the minor axis deflection is larger than major axis deflection (higher surface density)
    assert expected_ay > expected_ax


def test_sie_convergence_analytic_vs_finite_difference():
    """Verify that analytical convergence kappa matches numerical Jacobian trace."""
    theta_e = 1.0
    q = 0.75
    sie = SingularIsothermalEllipsoid.from_degrees(theta_e=theta_e, axis_ratio=q, pos_angle_deg=25.0)

    tx = np.array([0.8, -0.6, 1.4])
    ty = np.array([0.5, 0.9, -1.0])

    kappa_analytic = sie.convergence_analytic(tx, ty)
    kappa_numeric = sie.convergence(tx, ty, eps=1e-5)

    assert np.allclose(kappa_numeric, kappa_analytic, rtol=1e-4, atol=1e-4)


def test_sie_singularity_handling():
    """Verify that evaluation at center raises SingularityError or returns NaN if masked."""
    center = (-0.4, 0.8)
    sie = SingularIsothermalEllipsoid(theta_e=1.0, axis_ratio=0.7, center=center)

    with pytest.raises(SingularityError, match="SIE deflection field is singular at lens center"):
        sie.deflection(center[0], center[1], mask_singular=False)

    ax, ay = sie.deflection(center[0], center[1], mask_singular=True)
    assert np.isnan(ax) and np.isnan(ay)

