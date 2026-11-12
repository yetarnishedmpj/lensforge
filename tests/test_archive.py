"""Tests for astronomical archival data ingestion and FITS cutout extraction."""

import numpy as np
import pytest

from lensforge.data.archive import (
    KNOWN_LENS_CATALOG,
    extract_fits_cutout,
    resolve_coordinates,
)

try:
    from astropy.io import fits
    from astropy.wcs import WCS
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


def test_known_lens_catalog_coordinates():
    """Verify that all preset gravitational lenses have valid celestial coordinates."""
    assert len(KNOWN_LENS_CATALOG) >= 3
    for key, info in KNOWN_LENS_CATALOG.items():
        assert 0.0 <= info["ra"] <= 360.0
        assert -90.0 <= info["dec"] <= 90.0
        assert "z_lens" in info and info["z_lens"] > 0
        assert "z_source" in info and info["z_source"] > info["z_lens"]


def test_resolve_coordinates():
    """Verify coordinate resolution from target strings and numerical coordinates."""
    # 1. Preset target resolution
    ra, dec, label = resolve_coordinates("SDSSJ0946+1006")
    assert np.isclose(ra, 146.73617)
    assert np.isclose(dec, 10.11461)
    assert "SDSS" in label

    # 2. Numerical coordinates
    ra_num, dec_num, label_num = resolve_coordinates(150.0, 20.0)
    assert np.isclose(ra_num, 150.0)
    assert np.isclose(dec_num, 20.0)

    # 3. Invalid inputs
    with pytest.raises(ValueError):
        resolve_coordinates(150.0)  # Missing dec


@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="Astropy required for cutout extraction test.")
def test_extract_fits_cutout(tmp_path):
    """Verify that extract_fits_cutout extracts correct pixel dimensions and preserves WCS."""
    # 1. Create a synthetic full-frame FITS image with valid WCS
    w = WCS(naxis=2)
    w.wcs.crpix = [100.0, 100.0]
    w.wcs.cdelt = np.array([-0.05 / 3600.0, 0.05 / 3600.0])  # 0.05 arcsec/pixel
    w.wcs.crval = [146.73617, 10.11461]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    data = np.zeros((200, 200), dtype=np.float32)
    # Put a synthetic source right at the center
    data[95:105, 95:105] = 100.0

    header = w.to_header()
    hdu = fits.PrimaryHDU(data=data, header=header)
    input_fits = tmp_path / "synthetic_full_frame.fits"
    hdu.writeto(input_fits)

    # 2. Extract a 4.0 arcsecond cutout centered on the source
    output_cutout = tmp_path / "extracted_cutout.fits"
    result_path = extract_fits_cutout(
        fits_path=input_fits,
        ra=146.73617,
        dec=10.11461,
        cutout_size_arcsec=4.0,
        output_path=output_cutout,
        overwrite=True,
    )

    assert result_path.exists()

    # 3. Verify extracted cutout dimensions (4.0 arcsec / 0.05 arcsec/pixel = 80 pixels)
    with fits.open(result_path) as hdul:
        cutout_data = hdul[0].data
        cutout_hdr = hdul[0].header
        assert cutout_data.shape == (80, 80)
        assert np.max(cutout_data) == 100.0
        assert "PIXSCALE" in cutout_hdr
        assert np.isclose(cutout_hdr["PIXSCALE"], 0.05, rtol=1e-3)

