"""Tests for analytical image positions of point-mass gravitational lenses."""

import numpy as np
import pytest

from lensforge.physics.models.point_mass import PointMassLens


def test_scalar_image_positions_satisfy_lens_equation():
    """Verify that theta_+ and theta_- satisfy the scalar lens equation theta - theta_E^2 / theta = beta."""
    theta_e = 1.25
    lens = PointMassLens(theta_e=theta_e)

    # Test an array of non-zero source offsets (positive and negative)
    betas = np.array([0.01, 0.1, 0.5, 1.0, 2.5, 10.0, -0.2, -1.5, -5.0])

    theta_plus, theta_minus = lens.scalar_image_positions(betas, theta_e)

    # Check lens equation for outer image: theta_+ - theta_E^2 / theta_+ == beta
    lhs_plus = theta_plus - (theta_e**2) / theta_plus
    assert np.allclose(lhs_plus, betas, rtol=1e-13, atol=1e-14)

    # Check lens equation for inner image: theta_- - theta_E^2 / theta_- == beta
    lhs_minus = theta_minus - (theta_e**2) / theta_minus
    assert np.allclose(lhs_minus, betas, rtol=1e-13, atol=1e-14)


def test_image_position_algebraic_invariants():
    """Verify standard algebraic identities for point-mass lensing images:

    1. Product of positions: theta_+ * theta_- = -theta_E^2 identically.
    2. Difference of positions: theta_+ - theta_- = sqrt(beta^2 + 4 * theta_E^2).
    3. Sum of positions: theta_+ + theta_- = beta.
    """
    theta_e = 1.4
    betas = np.linspace(-5.0, 5.0, 50)
    # Exclude exact 0 for division stability
    betas = betas[~np.isclose(betas, 0.0)]

    theta_plus, theta_minus = PointMassLens.scalar_image_positions(betas, theta_e)

    # Invariant 1: Product equals -theta_E^2
    products = theta_plus * theta_minus
    assert np.allclose(products, -(theta_e**2), rtol=1e-13)

    # Invariant 2: Sum equals beta
    sums = theta_plus + theta_minus
    assert np.allclose(sums, betas, rtol=1e-13)

    # Invariant 3: Difference equals sqrt(beta^2 + 4 * theta_E^2)
    diffs = theta_plus - theta_minus
    expected_diffs = np.sqrt(betas**2 + 4.0 * (theta_e**2))
    assert np.allclose(diffs, expected_diffs, rtol=1e-13)


def test_image_position_bounds_and_ordering():
    """Verify spatial bounds of images relative to the Einstein ring.

    For beta > 0:
        theta_+ > theta_E  (outer image lies outside the Einstein ring)
        0 > theta_- > -theta_E  (|theta_-| < theta_E, inner image lies inside)
    """
    theta_e = 1.0
    positive_betas = np.array([0.001, 0.2, 0.8, 2.0, 10.0])

    theta_plus, theta_minus = PointMassLens.scalar_image_positions(positive_betas, theta_e)

    # Outer image strictly outside Einstein ring
    assert np.all(theta_plus > theta_e)

    # Inner image inside Einstein ring (with negative sign)
    assert np.all(np.abs(theta_minus) < theta_e)
    assert np.all(theta_minus < 0.0)


def test_2d_image_positions_ray_trace_back_to_source():
    """Verify that 2D image positions ray-trace back precisely to the source position."""
    center = (0.2, -0.1)
    theta_e = 1.15
    lens = PointMassLens(theta_e=theta_e, center=center)

    # Arbitrary source coordinates
    test_sources = [
        (0.6, 0.4),
        (-0.5, 0.8),
        (0.2, -1.2),
        (-1.0, -0.7),
    ]

    for beta_x, beta_y in test_sources:
        result = lens.image_positions_2d(beta_x, beta_y)
        assert not result["is_ring"]

        img_plus, img_minus = result["images"]

        # Ray-trace img_plus
        bx_plus, by_plus = lens.ray_trace(img_plus[0], img_plus[1])
        assert np.isclose(bx_plus, beta_x, rtol=1e-12, atol=1e-12)
        assert np.isclose(by_plus, beta_y, rtol=1e-12, atol=1e-12)

        # Ray-trace img_minus
        bx_minus, by_minus = lens.ray_trace(img_minus[0], img_minus[1])
        assert np.isclose(bx_minus, beta_x, rtol=1e-12, atol=1e-12)
        assert np.isclose(by_minus, beta_y, rtol=1e-12, atol=1e-12)


def test_perfect_alignment_special_case():
    """Verify that beta = center (perfect alignment) produces an Einstein ring result."""
    center = (0.5, -0.5)
    theta_e = 1.3
    lens = PointMassLens(theta_e=theta_e, center=center)

    result = lens.image_positions_2d(center[0], center[1])
    assert result["is_ring"] is True
    assert np.isclose(result["einstein_ring_radius"], theta_e)
    assert len(result["images"]) == 0
    assert "Einstein ring" in result["description"]
