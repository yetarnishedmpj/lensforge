"""Tests for LensForge cosmological distance engine."""

import numpy as np
import pytest

from lensforge.cosmology import PLANCK18, FlatLambdaCDMCosmology

try:
    from astropy.cosmology import Planck18 as AstropyPlanck18
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


def test_cosmology_parameter_validation():
    """Verify input validation for cosmological parameters."""
    with pytest.raises(ValueError, match="Hubble constant H_0 must be positive"):
        FlatLambdaCDMCosmology(h0=0.0)
    with pytest.raises(ValueError, match="Omega_m must be in"):
        FlatLambdaCDMCosmology(omega_m=1.2)


def test_cosmological_distances_and_ordering():
    """Verify distances at z=0, z>0 and cosmological ordering."""
    cosmo = PLANCK18

    assert cosmo.comoving_distance(0.0) == 0.0
    assert cosmo.angular_diameter_distance(0.0, 0.0) == 0.0

    # Cosmological ordering violations
    with pytest.raises(ValueError, match="Source redshift z_source"):
        cosmo.lensing_distances(z_lens=0.5, z_source=0.3)
    with pytest.raises(ValueError, match="Lens redshift must be strictly positive"):
        cosmo.lensing_distances(z_lens=0.0, z_source=0.5)

    # For typical lens z_l = 0.3, z_s = 1.0:
    dl, ds, dls = cosmo.lensing_distances(0.3, 1.0)
    assert 0 < dl < ds
    assert 0 < dls < ds
    # Non-additivity in FLRW: D_ls != D_s - D_l
    assert not np.isclose(dls, ds - dl)


def test_cosmology_agreement_with_astropy():
    """Verify LensForge distances match Astropy Planck18 within 1%."""
    if not ASTROPY_AVAILABLE:
        pytest.skip("Astropy not installed")

    cosmo = PLANCK18
    z_vals = [0.1, 0.3, 0.5, 0.8, 1.2, 2.0]

    for z in z_vals:
        da_lensforge = cosmo.angular_diameter_distance_mpc(0.0, z)
        da_astropy = AstropyPlanck18.angular_diameter_distance(z).value
        # 1% tolerance accounts for massive neutrinos and radiation in Astropy Planck18
        assert np.isclose(da_lensforge, da_astropy, rtol=0.01)

    # Transverse distance D_ls between z1 and z2
    dls_lensforge = cosmo.angular_diameter_distance_mpc(0.3, 1.2)
    dls_astropy = AstropyPlanck18.angular_diameter_distance(0.3, 1.2).value
    assert np.isclose(dls_lensforge, dls_astropy, rtol=0.01)


def test_critical_surface_density():
    """Verify critical surface density has correct order of magnitude in SI and M_sun/pc^2."""
    cosmo = PLANCK18
    sigma_cr_si = cosmo.critical_surface_density(z_lens=0.3, z_source=1.0)
    # Typical cosmological sigma_cr in SI is around 1 to 10 kg/m^2
    assert 1.0 < sigma_cr_si < 20.0

    sigma_cr_astro = cosmo.critical_surface_density_msun_pc2(z_lens=0.3, z_source=1.0)
    # In M_sun/pc^2, typical values are ~ 1,000 to 5,000 M_sun/pc^2 (~ 0.5 g/cm^2)
    assert 500.0 < sigma_cr_astro < 10000.0
