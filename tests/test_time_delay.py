"""Unit tests for time-delay cosmography and Fermat potential calculations."""

import numpy as np
import pytest

from lensforge.constants import C_LIGHT, arcsec2rad
from lensforge.cosmology import (
    PLANCK18,
    FlatLambdaCDMCosmology,
    fermat_potential,
    infer_h0_from_time_delay,
    time_delay_days,
    time_delay_seconds,
)
from lensforge.physics.models.point_mass import PointMassLens
from lensforge.physics.models.sis import SingularIsothermalSphere


def test_time_delay_distance_scaling():
    """Verify time-delay distance scales strictly as 1 / H_0."""
    z_l, z_s = 0.3, 1.2
    cosmo_ref = FlatLambdaCDMCosmology(h0=70.0, omega_m=0.3)
    cosmo_alt = FlatLambdaCDMCosmology(h0=73.0, omega_m=0.3)

    d_dt_ref = cosmo_ref.time_delay_distance_mpc(z_l, z_s)
    d_dt_alt = cosmo_alt.time_delay_distance_mpc(z_l, z_s)

    # Inverse scaling: d_dt_ref * 70 == d_dt_alt * 73
    np.testing.assert_allclose(d_dt_ref * 70.0, d_dt_alt * 73.0, rtol=1e-5)


def test_sis_fermat_potential_analytical():
    """Verify Fermat potential difference for an SIS matches analytical derivation Delta tau = -2 * theta_E * beta."""
    theta_e = 1.4  # arcsec
    beta = 0.35    # arcsec
    sis = SingularIsothermalSphere(theta_e=theta_e)

    # Collinear image positions for SIS:
    theta_1 = beta + theta_e   # Positive image (1.75'')
    theta_2 = beta - theta_e   # Negative image (-1.05'')

    tau_1 = fermat_potential(theta_1, 0.0, beta, 0.0, sis, angular_unit="arcsec")
    tau_2 = fermat_potential(theta_2, 0.0, beta, 0.0, sis, angular_unit="arcsec")
    delta_tau = tau_1 - tau_2

    # Analytical: Delta tau (in rad^2) = -2 * theta_E * beta (in rad^2)
    expected_delta_tau = -2.0 * arcsec2rad(theta_e) * arcsec2rad(beta)
    np.testing.assert_allclose(delta_tau, expected_delta_tau, rtol=1e-4)


def test_time_delay_and_h0_recovery():
    """Verify forward computation of time delay and subsequent exact recovery of H_0."""
    z_l = 0.295
    z_s = 0.657
    theta_e = 1.35
    beta = 0.25
    cosmo_true = FlatLambdaCDMCosmology(h0=72.0, omega_m=0.30)
    sis = SingularIsothermalSphere(theta_e=theta_e)

    theta_1 = beta + theta_e
    theta_2 = beta - theta_e

    # Forward: compute physical time delay in days
    dt_days = time_delay_days(
        theta_x1=theta_1, theta_y1=0.0,
        theta_x2=theta_2, theta_y2=0.0,
        beta_x=beta, beta_y=0.0,
        lens_model=sis,
        z_lens=z_l,
        z_source=z_s,
        cosmology=cosmo_true,
        angular_unit="arcsec",
    )

    # Realistic galaxy-scale time delay is typically a few weeks to months (positive/negative depending on arrival)
    assert abs(dt_days) > 1.0

    # Inverse: infer H_0 from the simulated time delay
    h0_recovered = infer_h0_from_time_delay(
        delta_t_days=dt_days,
        theta_x1=theta_1, theta_y1=0.0,
        theta_x2=theta_2, theta_y2=0.0,
        beta_x=beta, beta_y=0.0,
        lens_model=sis,
        z_lens=z_l,
        z_source=z_s,
        cosmology=cosmo_true,
        angular_unit="arcsec",
    )

    np.testing.assert_allclose(h0_recovered, 72.0, rtol=1e-4)


def test_fermat_potential_invalid_unit():
    """Verify invalid angular unit raises ValueError."""
    sis = SingularIsothermalSphere(theta_e=1.0)
    with pytest.raises(ValueError, match="Unsupported angular_unit"):
        fermat_potential(1.0, 0.0, 0.5, 0.0, sis, angular_unit="degrees")
