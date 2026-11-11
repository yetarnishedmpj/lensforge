"""Tests for telescope instrument simulation (PSF convolution and observational noise)."""

import numpy as np
import pytest

from lensforge.instrument.noise import add_observation_noise
from lensforge.instrument.psf import GaussianPSF, PixelatedPSF


def test_gaussian_psf_normalization_and_validation():
    """Verify that Gaussian PSF kernel sums to 1.0 and enforces valid parameters."""
    psf = GaussianPSF(fwhm=0.15, pixel_scale=0.03, num_pix=21)
    k = psf.kernel()

    assert k.shape == (21, 21)
    assert np.isclose(np.sum(k), 1.0, rtol=1e-14)
    assert np.all(k >= 0.0)

    # Invalid parameters
    with pytest.raises(ValueError, match="FWHM must be strictly positive"):
        GaussianPSF(fwhm=0.0, pixel_scale=0.03)
    with pytest.raises(ValueError, match="pixel_scale must be strictly positive"):
        GaussianPSF(fwhm=0.15, pixel_scale=-0.01)
    with pytest.raises(ValueError, match="num_pix must be a positive odd integer"):
        GaussianPSF(fwhm=0.15, pixel_scale=0.03, num_pix=20)  # even


def test_psf_flux_conservation_on_convolution():
    """Verify that PSF convolution conserves total flux of an extended image."""
    img = np.zeros((60, 60))
    img[25:35, 25:35] = 10.0  # square block of flux

    psf = GaussianPSF(fwhm=0.12, pixel_scale=0.03, num_pix=15)
    blurred = psf.convolve(img)

    assert blurred.shape == img.shape
    # Flux should be conserved within numerical margin
    assert np.isclose(np.sum(blurred), np.sum(img), rtol=1e-4)


def test_observation_noise_statistics():
    """Verify photon Poisson and readout Gaussian noise statistical properties."""
    img = np.full((100, 100), 5.0)  # 5 e-/s per pixel
    read_noise = 3.0
    exposure_time = 1000.0
    sky_bg = 2.0

    noisy_img, noise_sigma = add_observation_noise(
        image=img,
        read_noise=read_noise,
        exposure_time=exposure_time,
        sky_background=sky_bg,
        seed=42,
    )

    assert noisy_img.shape == img.shape
    assert noise_sigma.shape == img.shape

    # Expected mean should match original image (unbiased)
    mean_obs = np.mean(noisy_img)
    assert np.isclose(mean_obs, 5.0, atol=0.1)

    # Normalized residuals (noisy - truth) / sigma should have std ~ 1.0
    pulls = (noisy_img - img) / noise_sigma
    assert np.isclose(np.mean(pulls), 0.0, atol=0.1)
    assert np.isclose(np.std(pulls), 1.0, atol=0.08)

