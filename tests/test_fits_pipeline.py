"""Tests for FITS data ingestion and preprocessing pipeline."""

from pathlib import Path
import numpy as np
import pytest

from lensforge.data import (
    create_annular_mask,
    estimate_sky_background,
    load_fits_observation,
    save_fits_observation,
)
from lensforge.physics.coordinates import make_coordinate_grid

try:
    import astropy
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


def test_sky_background_estimation():
    """Verify iterative sigma-clipping accurately recovers sky background and RMS."""
    rng = np.random.default_rng(42)
    sky_true = 15.2
    rms_true = 2.4
    img = rng.normal(loc=sky_true, scale=rms_true, size=(100, 100))

    # Add a few bright source outliers (stars/galaxies)
    img[20:25, 20:25] += 100.0

    med, rms = estimate_sky_background(img, sigma_clip=3.0)
    assert np.isclose(med, sky_true, atol=0.2)
    assert np.isclose(rms, rms_true, atol=0.2)


def test_annular_mask():
    """Verify annular mask selects correct radial pixel aperture."""
    grid = make_coordinate_grid(num_pix=40, field_of_view=4.0, unit="arcsec")
    mask = create_annular_mask(grid, r_inner=0.8, r_outer=1.6, center=(0.0, 0.0))

    r = np.hypot(grid.x, grid.y)
    assert np.all(r[mask] >= 0.8)
    assert np.all(r[mask] <= 1.6)
    assert np.any(mask)
    assert not np.all(mask)


def test_fits_io_roundtrip(tmp_path: Path):
    """Verify saving and loading FITS observations with metadata."""
    if not ASTROPY_AVAILABLE:
        pytest.skip("Astropy not installed")

    out_file = tmp_path / "test_lens_obs.fits"
    img = np.ones((50, 50), dtype=np.float64) * 5.0
    var = np.ones((50, 50), dtype=np.float64) * 0.25
    psf = np.zeros((15, 15), dtype=np.float64)
    psf[7, 7] = 1.0

    save_fits_observation(
        filepath=out_file,
        image=img,
        variance=var,
        psf_kernel=psf,
        pixel_scale=0.04,
        header_info={"TARGET": "SDSS_J0946", "TELESCOP": "HST"},
    )

    assert out_file.exists()

    obs = load_fits_observation(out_file, subtract_background=False)
    assert obs.image.shape == (50, 50)
    assert obs.pixel_scale == 0.04
    assert np.allclose(obs.variance, 0.25)
    assert obs.psf_kernel is not None
    assert obs.psf_kernel.shape == (15, 15)
    assert obs.header is not None
    assert obs.header["TARGET"] == "SDSS_J0946"
