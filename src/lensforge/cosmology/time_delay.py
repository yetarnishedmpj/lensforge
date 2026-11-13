"""Time-delay cosmography and Fermat potential engine for LensForge.

Provides tools to compute the gravitational Fermat potential, physical arrival time delays
between multiple lensed images (e.g. from variable quasars or supernovae), and infer
the Hubble constant H_0 to help resolve the cosmic Hubble Tension.

Mathematical Framework:
-----------------------
According to Fermat's Principle in General Relativity, images form at stationary points
of the arrival time surface (Blandford & Narayan 1986):

    tau(theta, beta) = 0.5 * |theta - beta|^2 - psi(theta)

where:
    tau is the dimensionless Fermat potential (in radians^2).
    theta is image position on the celestial sky.
    beta is the unlensed source position.
    psi(theta) is the effective 2D gravitational lens potential.

The physical light travel time delay between two images A and B is:

    Delta t_AB = (D_Delta_t / c) * [ tau(theta_A, beta) - tau(theta_B, beta) ]

where the cosmological Time-Delay Distance is:

    D_Delta_t = (1 + z_l) * (D_l * D_s / D_ls)  [meters or Mpc]

Because D_Delta_t is inversely proportional to H_0, measuring Delta t_AB directly
determines H_0 with zero reliance on the astronomical distance ladder.
"""

from typing import Tuple, Union
import numpy as np

from lensforge.constants import C_LIGHT, arcsec2rad, rad2arcsec
from lensforge.cosmology.distances import FlatLambdaCDMCosmology, PLANCK18
from lensforge.physics.models.base import BaseLensModel

SECONDS_PER_DAY = 86400.0


def fermat_potential(
    theta_x: Union[float, np.ndarray],
    theta_y: Union[float, np.ndarray],
    beta_x: Union[float, np.ndarray],
    beta_y: Union[float, np.ndarray],
    lens_model: BaseLensModel,
    angular_unit: str = "arcsec",
) -> np.ndarray:
    """Compute the dimensionless Fermat potential tau(theta, beta) in radians^2.

    tau(theta, beta) = 0.5 * |theta - beta|^2 - psi(theta)

    Args:
        theta_x: Image plane horizontal coordinate(s).
        theta_y: Image plane vertical coordinate(s).
        beta_x: Source plane horizontal coordinate(s).
        beta_y: Source plane vertical coordinate(s).
        lens_model: BaseLensModel instance providing potential(theta_x, theta_y).
        angular_unit: Input unit ('arcsec' or 'rad').

    Returns:
        Fermat potential in radians^2.
    """
    tx = np.asarray(theta_x, dtype=np.float64)
    ty = np.asarray(theta_y, dtype=np.float64)
    bx = np.asarray(beta_x, dtype=np.float64)
    by = np.asarray(beta_y, dtype=np.float64)

    # Convert coordinates to radians for physical SI Fermat potential
    if angular_unit == "arcsec":
        tx_rad = arcsec2rad(tx)
        ty_rad = arcsec2rad(ty)
        bx_rad = arcsec2rad(bx)
        by_rad = arcsec2rad(by)
        # Evaluate potential in arcsec units and convert to rad^2
        # psi(rad) = psi(arcsec) * (arcsec2rad)^2
        psi_raw = lens_model.potential(tx, ty)
        psi_rad2 = psi_raw * (arcsec2rad(1.0) ** 2)
    elif angular_unit == "rad":
        tx_rad, ty_rad = tx, ty
        bx_rad, by_rad = bx, by
        psi_rad2 = lens_model.potential(tx, ty)
    else:
        raise ValueError(f"Unsupported angular_unit: {angular_unit}. Must be 'arcsec' or 'rad'.")

    geom_term = 0.5 * ((tx_rad - bx_rad) ** 2 + (ty_rad - by_rad) ** 2)
    return geom_term - psi_rad2


def time_delay_seconds(
    theta_x1: float,
    theta_y1: float,
    theta_x2: float,
    theta_y2: float,
    beta_x: float,
    beta_y: float,
    lens_model: BaseLensModel,
    z_lens: float,
    z_source: float,
    cosmology: FlatLambdaCDMCosmology = PLANCK18,
    angular_unit: str = "arcsec",
) -> float:
    """Compute physical arrival time delay Delta t = t1 - t2 in seconds.

    A positive value indicates image 1 arrives LATER than image 2.

    Args:
        theta_x1, theta_y1: Celestial coordinates of image 1.
        theta_x2, theta_y2: Celestial coordinates of image 2.
        beta_x, beta_y: Source position coordinates.
        lens_model: Gravitational lens model.
        z_lens: Deflector redshift.
        z_source: Source redshift.
        cosmology: Cosmological model (default: PLANCK18).
        angular_unit: 'arcsec' or 'rad'.

    Returns:
        Time delay Delta t = t1 - t2 in seconds.
    """
    tau1 = float(fermat_potential(theta_x1, theta_y1, beta_x, beta_y, lens_model, angular_unit))
    tau2 = float(fermat_potential(theta_x2, theta_y2, beta_x, beta_y, lens_model, angular_unit))
    delta_tau = tau1 - tau2

    d_dt_m = cosmology.time_delay_distance(z_lens, z_source)
    dt_seconds = (d_dt_m / C_LIGHT) * delta_tau
    return float(dt_seconds)


def time_delay_days(
    theta_x1: float,
    theta_y1: float,
    theta_x2: float,
    theta_y2: float,
    beta_x: float,
    beta_y: float,
    lens_model: BaseLensModel,
    z_lens: float,
    z_source: float,
    cosmology: FlatLambdaCDMCosmology = PLANCK18,
    angular_unit: str = "arcsec",
) -> float:
    """Compute physical arrival time delay Delta t = t1 - t2 in astronomical days."""
    dt_sec = time_delay_seconds(
        theta_x1, theta_y1, theta_x2, theta_y2,
        beta_x, beta_y, lens_model,
        z_lens, z_source, cosmology, angular_unit
    )
    return dt_sec / SECONDS_PER_DAY


def infer_h0_from_time_delay(
    delta_t_days: float,
    theta_x1: float,
    theta_y1: float,
    theta_x2: float,
    theta_y2: float,
    beta_x: float,
    beta_y: float,
    lens_model: BaseLensModel,
    z_lens: float,
    z_source: float,
    cosmology: FlatLambdaCDMCosmology = PLANCK18,
    angular_unit: str = "arcsec",
) -> float:
    """Infer the Hubble constant H_0 (km/s/Mpc) from an observed image time delay.

    Given measured Delta t (in days) and lens potential parameters, solves:
        D_Delta_t = c * Delta t / Delta tau
    and inverts for H_0.

    Args:
        delta_t_days: Measured time delay t1 - t2 in days.
        theta_x1, theta_y1: Position of image 1.
        theta_x2, theta_y2: Position of image 2.
        beta_x, beta_y: Source position.
        lens_model: Fitted lens model.
        z_lens: Lens redshift.
        z_source: Source redshift.
        cosmology: Cosmological model (specifies Omega_m).
        angular_unit: Coordinate units ('arcsec' or 'rad').

    Returns:
        Inferred Hubble constant H_0 in km/s/Mpc.
    """
    if delta_t_days == 0.0:
        raise ValueError("Time delay cannot be identically zero.")

    tau1 = float(fermat_potential(theta_x1, theta_y1, beta_x, beta_y, lens_model, angular_unit))
    tau2 = float(fermat_potential(theta_x2, theta_y2, beta_x, beta_y, lens_model, angular_unit))
    delta_tau = tau1 - tau2

    if abs(delta_tau) < 1e-18:
        raise ValueError("Fermat potential difference is negligible; images appear coincident.")

    delta_t_sec = delta_t_days * SECONDS_PER_DAY
    # D_Delta_t in meters = c * Delta t / Delta tau
    d_dt_m = (C_LIGHT * delta_t_sec) / delta_tau
    from lensforge.constants import MEGAPARSEC
    d_dt_mpc = d_dt_m / MEGAPARSEC

    return cosmology.infer_h0_from_time_delay_distance(z_lens, z_source, abs(d_dt_mpc))
