"""Tests for ExternalShear and CompositeLens models."""

import numpy as np
import pytest

from lensforge.physics.lens_potential import gradient_finite_difference
from lensforge.physics.models.composite import CompositeLens
from lensforge.physics.models.shear import ExternalShear
from lensforge.physics.models.sie import SingularIsothermalEllipsoid
from lensforge.physics.models.sis import SingularIsothermalSphere


def test_external_shear_polar_conversions():
    """Verify polar and Cartesian parameterization conversions for external shear."""
    gamma = 0.08
    phi_deg = 30.0
    phi_rad = np.radians(phi_deg)

    shear = ExternalShear.from_polar_degrees(gamma=gamma, phi_gamma_deg=phi_deg)

    expected_g1 = gamma * np.cos(2.0 * phi_rad)
    expected_g2 = gamma * np.sin(2.0 * phi_rad)

    assert np.isclose(shear.gamma_1, expected_g1, rtol=1e-14)
    assert np.isclose(shear.gamma_2, expected_g2, rtol=1e-14)
    assert np.isclose(shear.gamma, gamma, rtol=1e-14)
    assert np.isclose(shear.phi_gamma_deg, phi_deg, rtol=1e-12)


def test_external_shear_potential_gradient():
    """Verify that numerical gradient of shear potential equals deflection vector."""
    shear = ExternalShear(gamma_1=0.05, gamma_2=-0.04, center=(0.1, -0.2))

    tx = np.array([0.5, -0.8, 1.2])
    ty = np.array([-0.3, 0.7, 1.0])

    ax, ay = shear.deflection(tx, ty)
    grad_x, grad_y = gradient_finite_difference(shear.potential, tx, ty, step_size=1e-6)

    assert np.allclose(grad_x, ax, rtol=1e-6, atol=1e-6)
    assert np.allclose(grad_y, ay, rtol=1e-6, atol=1e-6)


def test_external_shear_zero_convergence_and_constant_magnification():
    """Verify that external shear has zero convergence (Ricci-flat) and constant magnification."""
    gamma = 0.1
    shear = ExternalShear.from_polar(gamma=gamma, phi_gamma=0.2)

    tx = np.array([0.4, -0.5, 1.0])
    ty = np.array([0.2, 0.9, -0.7])

    # Zero convergence
    kappa = shear.convergence(tx, ty, eps=1e-5)
    assert np.allclose(kappa, 0.0, atol=1e-8)

    # Constant magnification mu = 1 / (1 - gamma^2)
    expected_mu = 1.0 / (1.0 - gamma**2)
    mu = shear.magnification(tx, ty, eps=1e-5)
    assert np.allclose(mu, expected_mu, rtol=1e-5)


def test_composite_lens_validation_and_composition():
    """Verify CompositeLens initialization, component summation, and potential gradient."""
    # Invalid initialization
    with pytest.raises(ValueError, match="requires at least one lens model component"):
        CompositeLens([])

    with pytest.raises(TypeError, match="must be an instance of BaseLensModel"):
        CompositeLens(["not_a_model"])  # type: ignore

    # Build composite: SIE galaxy halo + external environmental shear
    sie = SingularIsothermalEllipsoid.from_degrees(
        theta_e=1.2,
        axis_ratio=0.75,
        pos_angle_deg=40.0,
        center=(0.0, 0.0),
    )
    shear = ExternalShear.from_polar_degrees(gamma=0.07, phi_gamma_deg=15.0)

    composite = CompositeLens([sie, shear])
    assert len(composite) == 2
    assert composite[0] is sie
    assert composite[1] is shear

    tx = np.array([0.6, -0.9, 1.3])
    ty = np.array([0.4, 0.8, -1.1])

    # Total deflection is the sum of deflections
    ax_comp, ay_comp = composite.deflection(tx, ty)
    ax_sie, ay_sie = sie.deflection(tx, ty)
    ax_sh, ay_sh = shear.deflection(tx, ty)

    assert np.allclose(ax_comp, ax_sie + ax_sh, rtol=1e-14)
    assert np.allclose(ay_comp, ay_sie + ay_sh, rtol=1e-14)

    # Total potential is the sum of potentials
    psi_comp = composite.potential(tx, ty)
    psi_sie = sie.potential(tx, ty)
    psi_sh = shear.potential(tx, ty)

    assert np.allclose(psi_comp, psi_sie + psi_sh, rtol=1e-14)

    # Numerical gradient of total potential matches composite deflection
    grad_x, grad_y = gradient_finite_difference(composite.potential, tx, ty, step_size=1e-6)
    assert np.allclose(grad_x, ax_comp, rtol=1e-5, atol=1e-5)
    assert np.allclose(grad_y, ay_comp, rtol=1e-5, atol=1e-5)

