"""Tests for point-mass gravitational lens magnification, parity, and asymptotics."""

import numpy as np
import pytest

from lensforge.physics.magnification import (
    point_mass_absolute_magnification,
    point_mass_magnification,
    point_mass_total_magnification,
)


def test_signed_magnification_properties():
    """Verify signed magnification values and topological parity.

    For all u > 0:
        mu_+ > 1.0 (Outer image: positive parity, amplified flux)
        mu_- < 0.0 (Inner image: negative parity, mirrored image)
        mu_+ + mu_- == 1.0 identically (Point-mass lensing invariant)
    """
    # Sample u from 1e-4 to 10.0
    u_vals = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0])

    mu_plus, mu_minus = point_mass_magnification(u_vals)

    # Parity checks
    assert np.all(mu_plus > 1.0)
    assert np.all(mu_minus < 0.0)

    # Invariant check: mu_+ + mu_- == 1.0
    invariant = mu_plus + mu_minus
    assert np.allclose(invariant, 1.0, rtol=1e-13, atol=1e-14)


def test_absolute_and_total_magnification():
    """Verify absolute and total flux magnification."""
    u = 1.0
    # At u = 1.0:
    # mu_+ = 1/2 + 3 / (2 * sqrt(5)) ~ 0.5 + 0.67082 = 1.170820
    # mu_- = 1/2 - 3 / (2 * sqrt(5)) ~ 0.5 - 0.67082 = -0.170820
    # mu_tot = 3 / sqrt(5) ~ 1.341641
    mu_plus, mu_minus = point_mass_magnification(u)
    abs_plus, abs_minus = point_mass_absolute_magnification(u)
    mu_tot = point_mass_total_magnification(u)

    expected_plus = 0.5 + 3.0 / (2.0 * np.sqrt(5.0))
    expected_minus = 0.5 - 3.0 / (2.0 * np.sqrt(5.0))
    expected_tot = 3.0 / np.sqrt(5.0)

    assert np.isclose(mu_plus, expected_plus, rtol=1e-14)
    assert np.isclose(mu_minus, expected_minus, rtol=1e-14)
    assert np.isclose(abs_plus, abs(expected_plus), rtol=1e-14)
    assert np.isclose(abs_minus, abs(expected_minus), rtol=1e-14)
    assert np.isclose(mu_tot, expected_tot, rtol=1e-14)
    assert np.isclose(mu_tot, abs_plus + abs_minus, rtol=1e-14)


def test_caustic_singularity_at_zero():
    """Verify explicit zero handling and divergence at u = 0."""
    # When allow_infinite=False (default) -> ZeroDivisionError
    with pytest.raises(ZeroDivisionError, match="Point-mass magnification diverges to infinity"):
        point_mass_magnification(0.0)

    with pytest.raises(ZeroDivisionError, match="Point-mass magnification diverges to infinity"):
        point_mass_total_magnification(0.0)

    # When allow_infinite=True -> returns (+inf, -inf) and +inf
    mu_p, mu_m = point_mass_magnification(0.0, allow_infinite=True)
    assert np.isposinf(mu_p)
    assert np.isneginf(mu_m)

    tot = point_mass_total_magnification(0.0, allow_infinite=True)
    assert np.isposinf(tot)


def test_invalid_negative_offset():
    """Verify that negative u raises ValueError."""
    with pytest.raises(ValueError, match="must be non-negative"):
        point_mass_magnification(-0.5)


def test_asymptotic_behavior_large_u():
    """Test asymptotic behavior when source is far from lens (u >> 1).

    As u -> infinity:
        mu_+ -> 1.0 (unlensed primary image)
        mu_- -> 0.0 (secondary image vanishes)
        mu_tot -> 1.0 (unperturbed flux)
    """
    large_u = 1.0e4
    mu_plus, mu_minus = point_mass_magnification(large_u)
    mu_tot = point_mass_total_magnification(large_u)

    assert np.isclose(mu_plus, 1.0, atol=1e-6)
    assert np.isclose(mu_minus, 0.0, atol=1e-6)
    assert np.isclose(mu_tot, 1.0, atol=1e-6)


def test_asymptotic_behavior_small_u():
    """Test asymptotic behavior near caustic (0 < u << 1).

    For small u:
        mu_+ ~ 1 / (2u)
        mu_- ~ -1 / (2u)
        mu_tot ~ 1 / u
    """
    small_u = 1.0e-4
    mu_plus, mu_minus = point_mass_magnification(small_u)
    mu_tot = point_mass_total_magnification(small_u)

    # Note: mu_+ = 1/(2u) + 1/2 + O(u), so leading-order matches with rtol=1e-3,
    # or including the 0.5 term matches to O(u^2).
    approx_plus = 0.5 / small_u + 0.5
    approx_tot = 1.0 / small_u

    assert np.isclose(mu_plus, approx_plus, rtol=1e-5)
    assert np.isclose(mu_minus, -approx_plus + 1.0, rtol=1e-5)
    assert np.isclose(mu_tot, approx_tot, rtol=1e-3)


def test_numerical_stability_broad_range():
    """Verify numerical stability and no NaNs across 10 orders of magnitude."""
    u_range = np.logspace(-5, 5, 100)
    mu_plus, mu_minus = point_mass_magnification(u_range)
    mu_tot = point_mass_total_magnification(u_range)

    assert np.all(np.isfinite(mu_plus))
    assert np.all(np.isfinite(mu_minus))
    assert np.all(np.isfinite(mu_tot))

    # mu_minus is strictly negative for all u > 0 due to rationalized formulation
    assert np.all(mu_minus < 0.0)

    # For u <= 1000, 1/u^4 > machine epsilon (2.2e-16), so mu_plus is strictly > 1.0
    assert np.all(mu_plus[u_range <= 1000.0] > 1.0)

    # For all u, mu_plus >= 1.0 and mu_tot >= 1.0 in float64
    assert np.all(mu_plus >= 1.0)
    assert np.all(mu_tot >= 1.0)

    # Fundamental invariant mu_+ + mu_- == 1 holds to high precision
    assert np.allclose(mu_plus + mu_minus, 1.0, rtol=1e-12)
