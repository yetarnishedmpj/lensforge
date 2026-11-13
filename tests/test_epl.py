"""Unit tests for the Elliptical Power-Law (EPL / PEMD) lens model."""

import numpy as np
import pytest

from lensforge.constants import deg2rad
from lensforge.physics.models.epl import EllipticalPowerLaw
from lensforge.physics.models.sis import SingularIsothermalSphere


def test_epl_initialization_and_bounds():
    """Verify EPL parameter validation."""
    epl = EllipticalPowerLaw(theta_e=1.2, gamma=2.1, axis_ratio=0.85)
    assert epl.theta_e == 1.2
    assert epl.gamma == 2.1
    assert epl.axis_ratio == 0.85

    with pytest.raises(ValueError, match="Einstein radius"):
        EllipticalPowerLaw(theta_e=-1.0)

    with pytest.raises(ValueError, match="slope gamma"):
        EllipticalPowerLaw(theta_e=1.0, gamma=0.8)

    with pytest.raises(ValueError, match="slope gamma"):
        EllipticalPowerLaw(theta_e=1.0, gamma=3.2)

    with pytest.raises(ValueError, match="Axis ratio"):
        EllipticalPowerLaw(theta_e=1.0, axis_ratio=-0.2)


def test_epl_spherical_isothermal_matches_sis():
    """Verify EPL with gamma=2.0 and q=1.0 identically matches SIS."""
    theta_e = 1.35
    epl = EllipticalPowerLaw(theta_e=theta_e, gamma=2.0, axis_ratio=1.0, core_radius=1e-9)
    sis = SingularIsothermalSphere(theta_e=theta_e)

    x = np.array([0.8, -1.2, 2.5, -0.4])
    y = np.array([1.1, 0.9, -1.8, -2.1])

    ax_epl, ay_epl = epl.deflection(x, y)
    ax_sis, ay_sis = sis.deflection(x, y)

    np.testing.assert_allclose(ax_epl, ax_sis, rtol=1e-5)
    np.testing.assert_allclose(ay_epl, ay_sis, rtol=1e-5)


def test_epl_spherical_analytical_power_law():
    """Verify spherical EPL (q=1.0) matches alpha = theta_E * (r / theta_E)^(2 - gamma)."""
    theta_e = 1.5
    gamma = 2.3  # Steeper than isothermal
    epl = EllipticalPowerLaw(theta_e=theta_e, gamma=gamma, axis_ratio=1.0, core_radius=1e-9)

    r = 2.4
    angle = np.pi / 4.0
    tx, ty = r * np.cos(angle), r * np.sin(angle)

    ax, ay = epl.deflection(tx, ty)
    alpha_mag = np.hypot(ax, ay)

    expected_mag = theta_e * ((r / theta_e) ** (2.0 - gamma))
    np.testing.assert_allclose(alpha_mag, expected_mag, rtol=1e-5)


def test_epl_gradient_matches_deflection():
    """Verify grad(psi) == alpha for arbitrary gamma and axis ratio q."""
    theta_e = 1.25
    gamma = 2.15
    q = 0.78
    pa_deg = 37.0
    center = (0.3, -0.2)
    epl = EllipticalPowerLaw.from_degrees(
        theta_e=theta_e, gamma=gamma, axis_ratio=q, pos_angle_deg=pa_deg, center=center
    )

    test_points = [(1.4, 0.8), (-1.1, 1.5), (0.2, -1.9)]
    eps = 1e-5

    for tx, ty in test_points:
        ax_true, ay_true = epl.deflection(tx, ty)

        p_x_plus = epl.potential(tx + eps, ty)
        p_x_minus = epl.potential(tx - eps, ty)
        dpsi_dx = (p_x_plus - p_x_minus) / (2.0 * eps)

        p_y_plus = epl.potential(tx, ty + eps)
        p_y_minus = epl.potential(tx, ty - eps)
        dpsi_dy = (p_y_plus - p_y_minus) / (2.0 * eps)

        np.testing.assert_allclose(ax_true, dpsi_dx, rtol=1e-4)
        np.testing.assert_allclose(ay_true, dpsi_dy, rtol=1e-4)
