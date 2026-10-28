"""Tests for Singular Isothermal Sphere (SIS) gravitational lens model."""

import numpy as np
import pytest

from lensforge.constants import C_LIGHT, MEGAPARSEC, rad2arcsec
from lensforge.physics.distances import (
    einstein_radius_sis,
    einstein_radius_sis_arcsec,
    validate_sis_distances,
)
from lensforge.physics.lens_potential import SingularityError, gradient_finite_difference
from lensforge.physics.models.sis import SingularIsothermalSphere


def test_einstein_radius_sis_analytic():
    """Verify SIS Einstein radius calculation against physical formula."""
    # Typical massive elliptical galaxy: sigma_v = 250 km/s = 2.5e5 m/s
    sigma_v = 250_000.0  # m/s
    d_s = 1800.0 * MEGAPARSEC
    d_ls = 1100.0 * MEGAPARSEC

    theta_e_rad = einstein_radius_sis(sigma_v, d_s, d_ls)
    expected_rad = 4.0 * np.pi * ((sigma_v / C_LIGHT) ** 2) * (d_ls / d_s)
    assert np.isclose(theta_e_rad, expected_rad, rtol=1e-15)

    # In arcsec: for 250 km/s galaxy, theta_E ~ 1.0 - 1.5 arcsec
    theta_e_arcsec = einstein_radius_sis_arcsec(sigma_v, d_s, d_ls)
    assert 0.8 < theta_e_arcsec < 2.0
    assert np.isclose(theta_e_arcsec, rad2arcsec(theta_e_rad), rtol=1e-15)


def test_sis_invalid_physical_parameters():
    """Verify validation of velocity dispersion and distances."""
    valid_sigma = 220_000.0
    valid_ds = 1500.0 * MEGAPARSEC
    valid_dls = 900.0 * MEGAPARSEC

    # Non-positive velocity dispersion
    with pytest.raises(ValueError, match="Velocity dispersion sigma_v must be strictly positive"):
        einstein_radius_sis(-10.0, valid_ds, valid_dls)
    with pytest.raises(ValueError, match="Velocity dispersion sigma_v must be strictly positive"):
        einstein_radius_sis(0.0, valid_ds, valid_dls)

    # Non-positive distances
    with pytest.raises(ValueError, match="Observer-source distance d_s must be strictly positive"):
        einstein_radius_sis(valid_sigma, -100.0, valid_dls)
    with pytest.raises(ValueError, match="Lens-source distance d_ls must be strictly positive"):
        einstein_radius_sis(valid_sigma, valid_ds, 0.0)

    # Cosmological ordering violation: d_ls >= d_s
    with pytest.raises(ValueError, match="cannot exceed or equal observer-to-source"):
        einstein_radius_sis(valid_sigma, valid_ds, valid_ds + 10.0)


def test_sis_constant_deflection_magnitude():
    """Verify that SIS deflection angle has constant magnitude theta_E at all radii."""
    theta_e = 1.35
    lens = SingularIsothermalSphere(theta_e=theta_e, center=(0.2, -0.1))

    # Test various points at different distances and directions
    radii = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    angles = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, -np.pi / 3]

    for r in radii:
        for phi in angles:
            tx = lens.center[0] + r * np.cos(phi)
            ty = lens.center[1] + r * np.sin(phi)
            ax, ay = lens.deflection(tx, ty)
            mag = float(np.hypot(ax, ay))
            assert np.isclose(mag, theta_e, rtol=1e-13)

            # Direction must be radially outward from lens center
            dir_x = (tx - lens.center[0]) / r
            dir_y = (ty - lens.center[1]) / r
            assert np.isclose(ax / theta_e, dir_x, atol=1e-13)
            assert np.isclose(ay / theta_e, dir_y, atol=1e-13)


def test_sis_potential_gradient_matches_deflection():
    """Verify that numerical gradient of psi matches deflection angle alpha."""
    theta_e = 1.25
    center = (0.3, -0.4)
    lens = SingularIsothermalSphere(theta_e=theta_e, center=center)

    tx = np.array([0.8, -0.2, 1.5, -1.0])
    ty = np.array([0.5, 1.2, -0.7, -1.5])

    ax, ay = lens.deflection(tx, ty)
    grad_x, grad_y = gradient_finite_difference(lens.potential, tx, ty, step_size=1e-6)

    assert np.allclose(grad_x, ax, rtol=1e-5, atol=1e-5)
    assert np.allclose(grad_y, ay, rtol=1e-5, atol=1e-5)


def test_sis_analytical_image_positions_and_invariants():
    """Verify scalar image positions, ray tracing satisfaction, and separation invariant."""
    theta_e = 1.0
    lens = SingularIsothermalSphere(theta_e=theta_e, center=(0.0, 0.0))

    # Case 1: Source inside Einstein radius (0 < beta < theta_e) -> two images
    beta_vals = [0.1, 0.3, 0.5, 0.8, 0.99]
    for b in beta_vals:
        t_plus, t_minus = lens.scalar_image_positions(b)
        assert t_plus > theta_e  # outer image
        assert -theta_e < t_minus < 0.0  # inner image on opposite side

        # Invariant 1: Distance between images is exactly 2 * theta_E
        assert np.isclose(t_plus - t_minus, 2.0 * theta_e, rtol=1e-14)

        # Invariant 2: Both satisfy the scalar lens equation beta = theta - theta_E * sgn(theta)
        assert np.isclose(t_plus - theta_e, b, rtol=1e-14)
        assert np.isclose(t_minus - (-theta_e), b, rtol=1e-14)

    # Case 2: Source outside Einstein radius (beta > theta_e) -> single image
    b_out = 1.5
    t_plus_out, t_minus_out = lens.scalar_image_positions(b_out)
    assert np.isclose(t_plus_out, b_out + theta_e)
    assert np.isnan(t_minus_out)


def test_sis_2d_image_geometry_and_ring():
    """Verify 2D image geometries: Einstein ring for beta=0, 2 images for u < theta_E, 1 for u > theta_E."""
    theta_e = 1.2
    lens = SingularIsothermalSphere(theta_e=theta_e, center=(0.1, 0.2))

    # Perfect alignment (beta = center)
    res_ring = lens.image_positions_2d(0.1, 0.2)
    assert res_ring["is_ring"] is True
    assert np.isclose(res_ring["einstein_ring_radius"], theta_e)
    assert len(res_ring["images"]) == 0

    # Inside caustic: u = 0.5 < theta_e
    res_double = lens.image_positions_2d(0.1 + 0.3, 0.2 + 0.4)
    assert res_double["is_ring"] is False
    assert len(res_double["images"]) == 2

    # Outside caustic: u = 2.0 > theta_e
    res_single = lens.image_positions_2d(0.1 + 1.2, 0.2 + 1.6)
    assert res_single["is_ring"] is False
    assert len(res_single["images"]) == 1


def test_sis_image_magnifications_and_invariants():
    """Verify signed and absolute magnifications, parity flip, and invariant mu_+ + mu_- = 2."""
    theta_e = 1.0
    lens = SingularIsothermalSphere(theta_e=theta_e, center=(0.0, 0.0))

    u_vals = [0.1, 0.25, 0.5, 0.75, 0.95]
    for u in u_vals:
        res = lens.image_magnifications_2d(u, 0.0)
        mu_plus = res["mu_plus"]
        mu_minus = res["mu_minus"]

        # Parity checks
        assert mu_plus > 1.0  # outer image positive parity
        assert mu_minus < 0.0  # inner image negative parity

        # Magnification sum invariant for SIS: mu_+ + mu_- = 2.0 exactly!
        assert np.isclose(mu_plus + mu_minus, 2.0, rtol=1e-14)

        # Total magnification: |mu_+| + |mu_-| = 2 * theta_E / u
        expected_total = 2.0 * theta_e / u
        assert np.isclose(res["mu_total"], expected_total, rtol=1e-14)

    # u = 0: Caustic singularity
    with pytest.raises(ZeroDivisionError, match="Source lies at caustic center"):
        lens.image_magnifications_2d(0.0, 0.0, allow_infinite=False)

    res_inf = lens.image_magnifications_2d(0.0, 0.0, allow_infinite=True)
    assert np.isinf(res_inf["mu_total"])


def test_sis_singularity_handling():
    """Verify that evaluating deflection exactly at the lens center behaves according to policy."""
    lens = SingularIsothermalSphere(theta_e=1.0, center=(0.5, -0.5))

    # Without masking: raise SingularityError
    with pytest.raises(SingularityError, match="deflection field is singular at lens center"):
        lens.deflection(0.5, -0.5, mask_singular=False)

    # With masking: return NaN
    ax, ay = lens.deflection(0.5, -0.5, mask_singular=True)
    assert np.isnan(ax) and np.isnan(ay)
