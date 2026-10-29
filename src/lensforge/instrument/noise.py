"""Astronomical detector noise modeling and noise map generation.

This module models observational uncertainties:
1. Gaussian detector readout noise (electronic amplifier noise).
2. Poissonian photon shot noise from galaxy signal and background sky.
3. Generation of pixel-level variance / uncertainty maps for chi-squared weighting.
"""

from typing import Optional, Tuple
import numpy as np


def add_observation_noise(
    image: np.ndarray,
    read_noise: float = 2.0,
    exposure_time: float = 1200.0,
    sky_background: float = 10.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add realistic Poisson photon shot noise and Gaussian read noise to a noiseless image.

    The physical detector count equation is:
        Counts_e = Image * Exposure_Time + Sky_e
        Noisy_Counts_e = Poisson(Counts_e) + Normal(0, Read_Noise^2)
        Observed_Flux = (Noisy_Counts_e - Sky_e) / Exposure_Time

    Args:
        image: 2D array of noiseless surface brightness (electrons / second or flux).
        read_noise: RMS detector read noise in electrons (e.g. 2-5 e- for CCD/CMOS).
        exposure_time: Effective exposure time in seconds (> 0).
        sky_background: Background sky brightness rate (electrons / second / pixel).
        seed: Optional random seed for deterministic reproduction.

    Returns:
        Tuple of (observed_image, noise_map_sigma) where noise_map_sigma contains
        the pixel-by-pixel 1-sigma uncertainty standard deviation.

    Raises:
        ValueError: If read_noise < 0 or exposure_time <= 0.
    """
    if read_noise < 0.0:
        raise ValueError(f"read_noise must be non-negative, got {read_noise}.")
    if exposure_time <= 0.0:
        raise ValueError(f"exposure_time must be strictly positive, got {exposure_time}.")

    rng = np.random.default_rng(seed)
    img = np.asarray(image, dtype=np.float64)

    # Expected photon electrons from source + sky background
    expected_electrons = np.maximum(0.0, img + sky_background) * exposure_time

    # 1. Poisson shot noise
    poisson_electrons = rng.poisson(expected_electrons)

    # 2. Gaussian readout noise
    gaussian_electrons = rng.normal(loc=0.0, scale=read_noise, size=img.shape)

    # Total observed electrons minus subtracted background
    total_electrons = poisson_electrons + gaussian_electrons - (sky_background * exposure_time)

    # Convert back to flux rate
    observed_image = total_electrons / exposure_time

    # Variance per pixel in flux rate units:
    # Var(Flux) = (Var(Poisson) + Var(ReadNoise)) / (Exposure_Time^2)
    #           = (expected_electrons + read_noise^2) / (exposure_time^2)
    var_flux = (expected_electrons + read_noise**2) / (exposure_time**2)
    noise_sigma = np.sqrt(var_flux)

    return observed_image, noise_sigma
