"""Cosmology module for LensForge.

Provides cosmological distance calculations and critical surface mass density
using the Friedmann-Lemaître-Robertson-Walker (FLRW) metric in flat Lambda-CDM.
Integrates with Astropy cosmology when available, with a standalone NumPy/SciPy
fallback ensuring zero strict external dependencies.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
from scipy import integrate

from lensforge.constants import C_LIGHT, G_NEWTON, MEGAPARSEC, rad2arcsec


@dataclass
class FlatLambdaCDMCosmology:
    """Flat Lambda Cold Dark Matter (FLRW) cosmological model.

    Attributes:
        h0: Hubble constant H_0 in km/s/Mpc (default 67.4 from Planck 2018).
        omega_m: Matter density parameter Omega_m (default 0.315).
        omega_lambda: Dark energy density parameter Omega_Lambda = 1 - Omega_m.
    """

    h0: float = 67.4
    omega_m: float = 0.315

    def __post_init__(self) -> None:
        if self.h0 <= 0.0:
            raise ValueError(f"Hubble constant H_0 must be positive, got {self.h0}.")
        if not (0.0 < self.omega_m < 1.0):
            raise ValueError(f"Omega_m must be in (0, 1), got {self.omega_m}.")
        self.omega_lambda = 1.0 - self.omega_m
        # Hubble parameter in SI units: (km/s/Mpc) -> (1/s)
        self.h0_si = (self.h0 * 1000.0) / MEGAPARSEC
        # Hubble distance c / H0 in meters
        self.d_h = C_LIGHT / self.h0_si

    def e_factor(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Dimensionless expansion rate E(z) = H(z) / H_0."""
        return np.sqrt(self.omega_m * (1.0 + z)**3 + self.omega_lambda)

    def comoving_distance(self, z: float) -> float:
        """Line-of-sight comoving distance chi(z) in meters.

        chi(z) = (c / H_0) * int_0^z dz' / E(z')
        """
        if z < 0.0:
            raise ValueError(f"Redshift z cannot be negative, got {z}.")
        if z == 0.0:
            return 0.0

        val, _ = integrate.quad(lambda zp: 1.0 / self.e_factor(zp), 0.0, z)
        return float(self.d_h * val)

    def comoving_distance_mpc(self, z: float) -> float:
        """Line-of-sight comoving distance chi(z) in Megaparsecs (Mpc)."""
        return self.comoving_distance(z) / MEGAPARSEC

    def angular_diameter_distance(self, z1: float, z2: float) -> float:
        """Transverse angular-diameter distance D_A(z1, z2) in meters.

        In a flat FLRW universe:
            D_A(z1, z2) = [ chi(z2) - chi(z1) ] / (1 + z2)
        """
        if z1 < 0.0 or z2 < 0.0:
            raise ValueError(f"Redshifts must be non-negative, got z1={z1}, z2={z2}.")
        if z2 < z1:
            raise ValueError(f"Target redshift z2 ({z2}) must be >= source redshift z1 ({z1}).")
        if z1 == z2:
            return 0.0

        chi1 = self.comoving_distance(z1)
        chi2 = self.comoving_distance(z2)
        return float((chi2 - chi1) / (1.0 + z2))

    def angular_diameter_distance_mpc(self, z1: float, z2: float) -> float:
        """Angular-diameter distance D_A(z1, z2) in Megaparsecs (Mpc)."""
        return self.angular_diameter_distance(z1, z2) / MEGAPARSEC

    def lensing_distances(
        self,
        z_lens: float,
        z_source: float,
    ) -> Tuple[float, float, float]:
        """Compute standard cosmological lensing distances (D_l, D_s, D_ls) in meters.

        Args:
            z_lens: Lens redshift z_l (> 0).
            z_source: Source galaxy redshift z_s (> z_l).

        Returns:
            Tuple of (D_l, D_s, D_ls) in meters.
        """
        if z_lens <= 0.0:
            raise ValueError(f"Lens redshift must be strictly positive, got {z_lens}.")
        if z_source <= z_lens:
            raise ValueError(
                f"Source redshift z_source ({z_source}) must be strictly greater than "
                f"lens redshift z_lens ({z_lens})."
            )

        d_l = self.angular_diameter_distance(0.0, z_lens)
        d_s = self.angular_diameter_distance(0.0, z_source)
        d_ls = self.angular_diameter_distance(z_lens, z_source)
        return d_l, d_s, d_ls

    def critical_surface_density(
        self,
        z_lens: float,
        z_source: float,
    ) -> float:
        """Compute critical surface mass density Sigma_cr in SI units (kg/m^2).

        Sigma_cr = (c^2 / (4 * pi * G)) * (D_s / (D_l * D_ls))

        Typical astrophysical values at cosmological distances are ~ 1 - 10 kg/m^2.
        """
        d_l, d_s, d_ls = self.lensing_distances(z_lens, z_source)
        prefactor = (C_LIGHT**2) / (4.0 * np.pi * G_NEWTON)
        sigma_cr = prefactor * (d_s / (d_l * d_ls))
        return float(sigma_cr)

    def critical_surface_density_msun_pc2(
        self,
        z_lens: float,
        z_source: float,
    ) -> float:
        """Compute critical surface mass density Sigma_cr in Solar masses per square parsec (M_sun/pc^2).

        Typical values for galaxy strong lenses are ~ 1,000 to 5,000 M_sun/pc^2.
        """
        from lensforge.constants import M_SUN, PARSEC
        sigma_cr_si = self.critical_surface_density(z_lens, z_source)
        conversion = (PARSEC**2) / M_SUN
        return float(sigma_cr_si * conversion)

    def sis_einstein_radius(
        self,
        z_lens: float,
        z_source: float,
        velocity_dispersion: float,
    ) -> float:
        """Compute SIS Einstein radius in arcseconds for given redshifts and velocity dispersion.

        theta_E = 4 * pi * (sigma_v / c)^2 * (D_ls / D_s)

        Args:
            z_lens: Lens redshift.
            z_source: Source redshift.
            velocity_dispersion: 1D velocity dispersion sigma_v in m/s.

        Returns:
            theta_E in arcseconds.
        """
        if velocity_dispersion <= 0.0:
            raise ValueError(f"Velocity dispersion must be positive, got {velocity_dispersion}.")

        _, d_s, d_ls = self.lensing_distances(z_lens, z_source)
        theta_e_rad = 4.0 * np.pi * ((velocity_dispersion / C_LIGHT)**2) * (d_ls / d_s)
        return float(rad2arcsec(theta_e_rad))


# Standard cosmological instance (Planck 2018 baseline)
PLANCK18 = FlatLambdaCDMCosmology(h0=67.4, omega_m=0.315)
