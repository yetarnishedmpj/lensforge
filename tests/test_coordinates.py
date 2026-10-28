"""Tests for angular coordinate transformations, grids, and conversions."""

import numpy as np
import pytest

from lensforge.constants import (
    ARCSEC_PER_RAD,
    RAD_PER_ARCSEC,
    arcsec2deg,
    arcsec2rad,
    deg2arcsec,
    deg2rad,
    rad2arcsec,
    rad2deg,
)
from lensforge.physics.coordinates import (
    CoordinateGrid,
    angular_radius,
    make_coordinate_grid,
    validate_coordinates,
)


def test_angular_conversion_roundtrips():
    """Test round-trip conversions between radians, arcseconds, and degrees."""
    # Test array of random angular values spanning microarcseconds to degrees
    test_angles_arcsec = np.array([0.001, 0.05, 1.0, 3.14159, 45.0, 3600.0])

    # arcsec -> rad -> arcsec
    rad_vals = arcsec2rad(test_angles_arcsec)
    roundtrip_arcsec = rad2arcsec(rad_vals)
    assert np.allclose(roundtrip_arcsec, test_angles_arcsec, rtol=1e-14, atol=1e-15)

    # deg -> rad -> deg
    test_degrees = np.array([0.0, 0.1, 45.0, 90.0, 180.0, 360.0])
    rad_deg = deg2rad(test_degrees)
    roundtrip_deg = rad2deg(rad_deg)
    assert np.allclose(roundtrip_deg, test_degrees, rtol=1e-14, atol=1e-15)

    # arcsec -> deg -> arcsec
    deg_vals = arcsec2deg(test_angles_arcsec)
    roundtrip_deg_arcsec = deg2arcsec(deg_vals)
    assert np.allclose(roundtrip_deg_arcsec, test_angles_arcsec, rtol=1e-14, atol=1e-15)


def test_conversion_constants():
    """Verify conversion constants against theoretical definitions."""
    # 1 radian = 180 * 3600 / pi arcseconds
    assert np.isclose(ARCSEC_PER_RAD, 648000.0 / np.pi, rtol=1e-15)
    assert np.isclose(RAD_PER_ARCSEC * ARCSEC_PER_RAD, 1.0, rtol=1e-15)


def test_make_coordinate_grid():
    """Test coordinate grid creation, shape, symmetry, and extent."""
    num_pix = 100
    fov = 2.0
    grid = make_coordinate_grid(num_pix=num_pix, field_of_view=fov, unit="arcsec")

    assert grid.shape == (num_pix, num_pix)
    assert grid.unit == "arcsec"
    assert np.isclose(grid.pixel_scale, fov / num_pix)

    # Grid should be centered at 0.0
    assert np.isclose(grid.x.mean(), 0.0, atol=1e-12)
    assert np.isclose(grid.y.mean(), 0.0, atol=1e-12)

    # Pixel centers should range from -FOV/2 + px/2 to FOV/2 - px/2
    half_px = grid.pixel_scale / 2.0
    assert np.isclose(grid.x[0, 0], -1.0 + half_px)
    assert np.isclose(grid.x[0, -1], 1.0 - half_px)

    # Test extent for imshow
    extent = grid.extent
    assert np.isclose(extent[0], -1.0)
    assert np.isclose(extent[1], 1.0)
    assert np.isclose(extent[2], -1.0)
    assert np.isclose(extent[3], 1.0)


def test_coordinate_grid_unit_conversion():
    """Test converting a CoordinateGrid between arcseconds and radians."""
    grid_arcsec = make_coordinate_grid(num_pix=50, field_of_view=1.0, unit="arcsec")
    grid_rad = grid_arcsec.to_unit("rad")

    assert grid_rad.unit == "rad"
    assert np.allclose(grid_rad.x, arcsec2rad(grid_arcsec.x))
    assert np.allclose(grid_rad.y, arcsec2rad(grid_arcsec.y))
    assert np.isclose(grid_rad.pixel_scale, arcsec2rad(grid_arcsec.pixel_scale))

    # Roundtrip back to arcsec
    grid_roundtrip = grid_rad.to_unit("arcsec")
    assert np.allclose(grid_roundtrip.x, grid_arcsec.x)


def test_make_coordinate_grid_invalid():
    """Test validation errors for invalid grid creation parameters."""
    with pytest.raises(ValueError, match="num_pix must be a positive integer"):
        make_coordinate_grid(num_pix=0, field_of_view=2.0)

    with pytest.raises(ValueError, match="field_of_view must be strictly positive"):
        make_coordinate_grid(num_pix=100, field_of_view=-1.0)


def test_validate_coordinates():
    """Test coordinate pair validation and broadcasting."""
    # Scalar broadcast
    x_bc, y_bc = validate_coordinates(1.0, 2.0)
    assert x_bc.shape == ()
    assert y_bc.shape == ()

    # 1D with 1D
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    xb, yb = validate_coordinates(x, y)
    assert xb.shape == (3,)

    # Incompatible shape raises ValueError
    with pytest.raises(ValueError, match="could not be broadcast together"):
        validate_coordinates(np.zeros((2, 3)), np.zeros((4, 5)))


def test_angular_radius():
    """Test radial coordinate calculations."""
    r = angular_radius(3.0, 4.0, center=(0.0, 0.0))
    assert np.isclose(r, 5.0)

    r_offset = angular_radius(4.0, 5.0, center=(1.0, 1.0))
    assert np.isclose(r_offset, 5.0)
