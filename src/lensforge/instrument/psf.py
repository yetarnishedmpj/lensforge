"""Telescope Point Spread Function (PSF) modeling and convolution.

This module models optical diffraction and atmospheric/detector blurring through:
- GaussianPSF: Symmetric or elliptical Gaussian blurring kernel.
- PixelatedPSF: Discrete 2D kernel convolution using fast 2D Fast Fourier Transforms (FFT).
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np
from scipy import signal


class BasePSF(ABC):
    """Abstract base class for telescope point spread functions."""

    @abstractmethod
    def kernel(self) -> np.ndarray:
        """Return the normalized 2D convolution kernel (summing to 1.0)."""

    def convolve(self, image: np.ndarray) -> np.ndarray:
        """Convolve a 2D image with this telescope PSF via FFT.

        Args:
            image: 2D image array.

        Returns:
            Blurred 2D image with identical dimensions.
        """
        k = self.kernel()
        # signal.fftconvolve preserves spatial dimensions with mode="same"
        return signal.fftconvolve(image, k, mode="same")


class GaussianPSF(BasePSF):
    """Gaussian Point Spread Function.

    Attributes:
        fwhm: Full Width at Half Maximum in angular units (e.g. arcsec).
        pixel_scale: Pixel resolution in the same angular units.
        num_pix: Kernel width in pixels (must be odd, default 21).
    """

    def __init__(
        self,
        fwhm: float,
        pixel_scale: float,
        num_pix: int = 21,
    ) -> None:
        """Initialize Gaussian PSF.

        Args:
            fwhm: FWHM in angular units (> 0).
            pixel_scale: Pixel scale in angular units (> 0).
            num_pix: Size of square kernel in pixels (must be odd positive integer).

        Raises:
            ValueError: If fwhm <= 0, pixel_scale <= 0, or num_pix is not positive odd.
        """
        if fwhm <= 0.0:
            raise ValueError(f"FWHM must be strictly positive, got {fwhm}.")
        if pixel_scale <= 0.0:
            raise ValueError(f"pixel_scale must be strictly positive, got {pixel_scale}.")
        if num_pix <= 0 or num_pix % 2 == 0:
            raise ValueError(f"num_pix must be a positive odd integer, got {num_pix}.")

        self._fwhm = float(fwhm)
        self._pixel_scale = float(pixel_scale)
        self._num_pix = int(num_pix)

        # sigma in pixels: FWHM = 2 * sqrt(2 * ln(2)) * sigma
        sigma_ang = self._fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        self._sigma_pix = sigma_ang / self._pixel_scale

        # Generate normalized kernel
        half = self._num_pix // 2
        y, x = np.mgrid[-half : half + 1, -half : half + 1]
        r2 = x**2 + y**2
        k = np.exp(-0.5 * r2 / (self._sigma_pix**2))
        self._kernel = k / np.sum(k)

    @property
    def fwhm(self) -> float:
        """FWHM in angular units."""
        return self._fwhm

    @property
    def sigma_pixels(self) -> float:
        """Standard deviation in pixel units."""
        return self._sigma_pix

    def kernel(self) -> np.ndarray:
        """Return the precomputed 2D normalized Gaussian kernel."""
        return self._kernel.copy()


class PixelatedPSF(BasePSF):
    """Arbitrary discrete 2D telescope PSF kernel (e.g., from TinyTim or empirical WebbPSF)."""

    def __init__(self, kernel: np.ndarray) -> None:
        """Initialize pixelated PSF from array.

        Args:
            kernel: 2D array representing telescope PSF.
        """
        k = np.asarray(kernel, dtype=np.float64)
        if k.ndim != 2:
            raise ValueError(f"PSF kernel must be 2D, got shape {k.shape}.")
        if np.sum(k) <= 0.0:
            raise ValueError("PSF kernel must have positive total sum.")
        self._kernel = k / np.sum(k)

    def kernel(self) -> np.ndarray:
        """Return normalized PSF kernel."""
        return self._kernel.copy()
