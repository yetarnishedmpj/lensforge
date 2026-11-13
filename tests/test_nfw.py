"""Unit tests for the Navarro-Frenk-White (NFW) dark matter halo lens model."""

import numpy as np
import pytest

from lensforge.physics.models.nfw import NFWHalo


def test_nfw_initialization():
    """Verify NFW initialization and parameter validation."""
    halo = NFWHalo(theta_s=15.0, kappa_s=0.25, center=(0.5, -0.3))
    assert halo.theta_s == 15.0
    assert halo.kappa_s == 0.25
    assert halo.center == (0.5, -0.3)

    with pytest.raises(ValueError, match="scale radius"):
        NFWHalo(theta_s=-5.0, kappa_s=0.2)

    with pytest.raises(ValueError, match="characteristic convergence"):
        NFWHalo(theta_s=10.0, kappa_s=-0.1)


def test_nfw_origin_behavior():
    """Verify deflection smoothly approaches zero at the halo center."""
    halo = NFWHalo(theta_s=10.0, kappa_s=0.3)
    ax, ay = halo.deflection(0.0, 0.0)
    assert ax == 0.0
    assert ay == 0.0


def test_nfw_gradient_matches_deflection():
    """Verify grad(psi) == alpha across inner (x < 1) and outer (x > 1) regions."""
    theta_s = 5.0
    kappa_s = 0.35
    halo = NFWHalo(theta_s=theta_s, kappa_s=kappa_s)

    # Test points at x = 0.5 (inner) and x = 2.0 (outer)
    test_points = [(1.5, 2.0), (7.0, -6.0), (3.5, 3.5)]
    eps = 1e-5

    for tx, ty in test_points:
        ax_true, ay_true = halo.deflection(tx, ty)

        # Finite difference gradient of potential
        p_x_plus = halo.potential(tx + eps, ty)
        p_x_minus = halo.potential(tx - eps, ty)
        dpsi_dx = (p_x_plus - p_x_minus) / (2.0 * eps)

        p_y_plus = halo.potential(tx, ty + eps)
        p_y_minus = halo.potential(tx, ty - eps)
        dpsi_dy = (p_y_plus - p_y_minus) / (2.0 * eps)

        np.testing.assert_allclose(ax_true, dpsi_dx, rtol=1e-4)
        np.testing.assert_allclose(ay_true, dpsi_dy, rtol=1e-4)


def test_nfw_laplacian_matches_convergence():
    """Verify 0.5 * Laplacian(psi) matches the analytical NFW convergence."""
    halo = NFWHalo(theta_s=8.0, kappa_s=0.4)
    tx = np.array([2.0, 5.0, 12.0])
    ty = np.array([1.5, -4.0, 9.0])

    kappa_analytical = halo.convergence_analytical(tx, ty)
    # Convergence inherited via base class finite-difference divergence of alpha
    kappa_num = halo.convergence(tx, ty, eps=1e-5)

    np.testing.assert_allclose(kappa_num, kappa_analytical, rtol=1e-3)


def test_nfw_radial_symmetry():
    """Verify deflection magnitude is invariant under rotation for spherical halo."""
    halo = NFWHalo(theta_s=10.0, kappa_s=0.2)
    r = 4.5
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    tx = r * np.cos(angles)
    ty = r * np.sin(angles)

    ax, ay = halo.deflection(tx, ty)
    alpha_mag = np.hypot(ax, ay)

    # All deflection magnitudes should be identical
    np.testing.assert_allclose(alpha_mag, alpha_mag[0], rtol=1e-5)
