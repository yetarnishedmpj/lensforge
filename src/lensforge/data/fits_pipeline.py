"""Astronomical FITS file ingestion, preprocessing, and World Coordinate System (WCS) pipeline.

Provides tools to read, calibrate, mask, and save standard astronomical observations
from space telescopes (HST, JWST) and ground surveys (SDSS, DES, Euclid).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

try:
    from astropy.io import fits
    from astropy.wcs import WCS
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

from lensforge.physics.coordinates import CoordinateGrid, make_coordinate_grid


@dataclass
class AstroObservation:
    """Standardized astronomical telescope observation.

    Attributes:
        image: 2D array of calibrated surface brightness (flux rate, e.g. e-/s).
        variance: 2D array of pixel noise variance.
        noise_sigma: 2D array of 1-sigma uncertainty standard deviation.
        pixel_scale: Pixel resolution in arcseconds/pixel.
        grid: CoordinateGrid spanning the observation field.
        psf_kernel: Optional 2D Point Spread Function kernel.
        mask: Optional 2D boolean array (True = good/unmasked, False = bad/masked pixel).
        header: Optional FITS header metadata dictionary.
    """

    image: np.ndarray
    variance: np.ndarray
    noise_sigma: np.ndarray
    pixel_scale: float
    grid: CoordinateGrid
    psf_kernel: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    header: Optional[Dict[str, Any]] = None


def estimate_sky_background(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    sigma_clip: float = 3.0,
    max_iters: int = 5,
) -> Tuple[float, float]:
    """Estimate sky background level and RMS noise using iterative sigma-clipping.

    Args:
        image: 2D image array.
        mask: Optional boolean mask (True = include in sky calculation).
        sigma_clip: Sigma threshold for rejection.
        max_iters: Maximum clipping iterations.

    Returns:
        Tuple of (sky_median, sky_rms).
    """
    valid = np.isfinite(image)
    if mask is not None:
        valid = valid & mask

    pixels = image[valid]
    if len(pixels) == 0:
        return 0.0, 1.0

    for _ in range(max_iters):
        med = float(np.median(pixels))
        mad = float(np.median(np.abs(pixels - med)))
        std = 1.4826 * mad  # normalized MAD ~ Gaussian sigma
        if std == 0.0:
            break
        keep = np.abs(pixels - med) < (sigma_clip * std)
        if np.all(keep) or np.sum(keep) < 10:
            break
        pixels = pixels[keep]

    sky_med = float(np.median(pixels))
    sky_rms = max(1e-12, 1.4826 * float(np.median(np.abs(pixels - sky_med))))
    return sky_med, sky_rms


def create_annular_mask(
    grid: CoordinateGrid,
    r_inner: float,
    r_outer: float,
    center: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Create a 2D boolean mask selecting an annular aperture around the deflector.

    Useful for isolating Einstein rings and arcs while masking the central bright
    deflector galaxy core and uninformative outer background.

    Args:
        grid: CoordinateGrid instance.
        r_inner: Inner radius in grid units (arcsec).
        r_outer: Outer radius in grid units (arcsec).
        center: Center (x, y) coordinates of the aperture.

    Returns:
        2D boolean array (True where r_inner <= radius <= r_outer).
    """
    dx = grid.x - center[0]
    dy = grid.y - center[1]
    r = np.hypot(dx, dy)
    return (r >= r_inner) & (r <= r_outer)


def save_fits_observation(
    filepath: Union[str, Path],
    image: np.ndarray,
    variance: Optional[np.ndarray] = None,
    psf_kernel: Optional[np.ndarray] = None,
    pixel_scale: float = 0.05,
    header_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save an astronomical observation to a multi-extension FITS file.

    Primary HDU: Science image
    Extension 1: Variance map (if provided)
    Extension 2: PSF kernel (if provided)

    Args:
        filepath: Destination path (.fits).
        image: 2D science image.
        variance: Optional 2D noise variance map.
        psf_kernel: Optional 2D PSF kernel.
        pixel_scale: Pixel scale in arcsec/pixel.
        header_info: Optional header keyword dictionary.

    Returns:
        Path to the saved FITS file.
    """
    if not ASTROPY_AVAILABLE:
        raise ImportError("Astropy is required to save FITS files. Install via 'pip install astropy'.")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    hdus = []

    # 1. Primary HDU: Science Image
    primary_header = fits.Header()
    primary_header["PIXSCALE"] = (float(pixel_scale), "Pixel scale in arcsec/pixel")
    primary_header["BUNIT"] = ("ELECTRONS/S", "Physical brightness unit")
    primary_header["CREATOR"] = ("LensForge", "Computational Astrophysics Imaging Framework")

    if header_info:
        for k, v in header_info.items():
            primary_header[str(k)[:8]] = v

    primary_hdu = fits.PrimaryHDU(data=np.asarray(image, dtype=np.float32), header=primary_header)
    hdus.append(primary_hdu)

    # 2. Extension: Variance Map
    if variance is not None:
        var_header = fits.Header()
        var_header["EXTNAME"] = ("VARIANCE", "Noise variance map")
        var_hdu = fits.ImageHDU(data=np.asarray(variance, dtype=np.float32), header=var_header)
        hdus.append(var_hdu)

    # 3. Extension: PSF Kernel
    if psf_kernel is not None:
        psf_header = fits.Header()
        psf_header["EXTNAME"] = ("PSF", "Telescope Point Spread Function")
        psf_hdu = fits.ImageHDU(data=np.asarray(psf_kernel, dtype=np.float32), header=psf_header)
        hdus.append(psf_hdu)

    hdul = fits.HDUList(hdus)
    hdul.writeto(path, overwrite=True)
    return path


def load_fits_observation(
    filepath: Union[str, Path],
    ext_data: Union[int, str] = 0,
    ext_variance: Optional[Union[int, str]] = None,
    ext_psf: Optional[Union[int, str]] = None,
    pixel_scale_override: Optional[float] = None,
    subtract_background: bool = True,
) -> AstroObservation:
    """Load an astronomical observation from a FITS file.

    Args:
        filepath: Path to the FITS file.
        ext_data: HDU index or name for science data (default 0).
        ext_variance: Optional HDU index or name for variance map.
        ext_psf: Optional HDU index or name for PSF kernel.
        pixel_scale_override: Override pixel scale if missing from header.
        subtract_background: If True, estimate and subtract median sky background.

    Returns:
        AstroObservation instance.
    """
    if not ASTROPY_AVAILABLE:
        raise ImportError("Astropy is required to load FITS files. Install via 'pip install astropy'.")

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"FITS file not found: {path}")

    with fits.open(path) as hdul:
        data = np.asarray(hdul[ext_data].data, dtype=np.float64)
        header = dict(hdul[ext_data].header)

        # Variance map
        if ext_variance is not None and ext_variance in hdul:
            variance = np.asarray(hdul[ext_variance].data, dtype=np.float64)
        elif "VARIANCE" in hdul:
            variance = np.asarray(hdul["VARIANCE"].data, dtype=np.float64)
        else:
            variance = None

        # PSF kernel
        if ext_psf is not None and ext_psf in hdul:
            psf_kernel = np.asarray(hdul[ext_psf].data, dtype=np.float64)
        elif "PSF" in hdul:
            psf_kernel = np.asarray(hdul["PSF"].data, dtype=np.float64)
        else:
            psf_kernel = None

    # Detect pixel scale from header keywords (PIXSCALE, CD1_1/CD2_2, or CDELT1)
    pixel_scale = pixel_scale_override
    if pixel_scale is None:
        if "PIXSCALE" in header:
            pixel_scale = float(header["PIXSCALE"])
        elif "CD1_1" in header:
            pixel_scale = float(abs(header["CD1_1"]) * 3600.0)
        elif "CDELT1" in header:
            pixel_scale = float(abs(header["CDELT1"]) * 3600.0)
        else:
            pixel_scale = 0.05  # Default HST-like scale

    # Optional background subtraction
    if subtract_background:
        sky_bg, sky_rms = estimate_sky_background(data)
        data = data - sky_bg
    else:
        _, sky_rms = estimate_sky_background(data)

    # If variance is absent, estimate from shot noise + background RMS
    if variance is None:
        variance = np.maximum(data, 0.0) + sky_rms**2
    noise_sigma = np.sqrt(np.maximum(variance, 1e-12))

    # Construct coordinate grid
    num_y, num_x = data.shape
    fov_x = num_x * pixel_scale
    grid = make_coordinate_grid(
        num_pix=num_x,
        field_of_view=fov_x,
        unit="arcsec",
    )

    return AstroObservation(
        image=data,
        variance=variance,
        noise_sigma=noise_sigma,
        pixel_scale=pixel_scale,
        grid=grid,
        psf_kernel=psf_kernel,
        header=header,
    )
