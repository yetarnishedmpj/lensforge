#!/usr/bin/env python3
"""LensForge Rigorous Bayesian MCMC Parameter Inference: SDSS J0946+1006.

Performs affine-invariant ensemble Markov Chain Monte Carlo (MCMC) sampling (Goodman & Weare 2010 / emcee)
over the macro-lens parameters on real HST ACS/WFC F814W imaging of SDSS J0946+1006.

Quantifies the full multi-dimensional posterior probability distribution P(eta | d),
yielding:
    1. Median values with rigorous 16th/84th percentile (68% / 1-sigma) credible intervals.
    2. Multi-parameter covariance and degeneracies (e.g., ellipticity q vs external shear gamma).
    3. Publication-quality Corner Plot showing 1D marginalized posteriors and 2D joint contours.
"""

from pathlib import Path
import sys
import time

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import center_of_mass

# Ensure src/ is on python path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from lensforge.constants import C_LIGHT, M_SUN, MEGAPARSEC, PARSEC, arcsec2rad
from lensforge.cosmology import PLANCK18
from lensforge.data import load_fits_observation
from lensforge.instrument.psf import GaussianPSF
from lensforge.inversion.reconstructor import SourceReconstructor
from lensforge.optimization import fit_lens_parameters
from lensforge.physics.coordinates import make_coordinate_grid
from lensforge.physics.models.composite import CompositeLens
from lensforge.physics.models.shear import ExternalShear
from lensforge.physics.models.sie import SingularIsothermalEllipsoid


def run_rigorous_mcmc_inference(
    fits_cutout_path: Path,
    output_dir: Path,
    n_walkers: int = 16,
    n_steps: int = 35,
    burn_in: int = 15,
) -> Path:
    """Run full Bayesian MCMC posterior sampling on real space telescope observation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    corner_path = output_dir / "real_sdss_j0946_mcmc_corner.png"

    print("=" * 80)
    print(" LENSFORGE RIGOROUS BAYESIAN MCMC INFERENCE: SDSS J0946+1006")
    print("=" * 80)

    # 1. Ingest Observation & Centroiding
    obs = load_fits_observation(fits_cutout_path, subtract_background=True)
    ny_raw, nx_raw = obs.image.shape
    mid_y, mid_x = ny_raw // 2, nx_raw // 2
    search_half = 20
    central_box = obs.image[mid_y - search_half : mid_y + search_half, mid_x - search_half : mid_x + search_half]
    max_idx = np.unravel_index(np.argmax(central_box), central_box.shape)
    peak_y = (mid_y - search_half) + max_idx[0]
    peak_x = (mid_x - search_half) + max_idx[1]

    win = 6
    sub_box = obs.image[peak_y - win : peak_y + win, peak_x - win : peak_x + win]
    sub_com = center_of_mass(np.maximum(sub_box, 0.0))
    sub_y = (peak_y - win) + sub_com[0]
    sub_x = (peak_x - win) + sub_com[1]

    cx_arcsec = float(obs.grid.x[0, int(round(sub_x))])
    cy_arcsec = float(obs.grid.y[int(round(sub_y)), 0])
    print(f"[1/4] Calibrated Observation Centroid: cx = {cx_arcsec:+.3f}'', cy = {cy_arcsec:+.3f}''")

    # 2. Cosmological Distances
    z_lens, z_source = 0.222, 0.609
    d_l_m = PLANCK18.angular_diameter_distance_mpc(0.0, z_lens) * MEGAPARSEC
    d_s_m = PLANCK18.angular_diameter_distance_mpc(0.0, z_source) * MEGAPARSEC
    d_ls_m = PLANCK18.angular_diameter_distance_mpc(z_lens, z_source) * MEGAPARSEC
    sigma_cr_si = PLANCK18.critical_surface_density(z_lens, z_source)

    # 3. Active Grid Setup & Arc Aperture
    fov_active = 4.2
    num_active = 46  # 0.091 arcsec/pixel for fast MCMC likelihood evaluation
    opt_grid = make_coordinate_grid(
        num_pix=num_active,
        field_of_view=fov_active,
        unit="arcsec",
        center=(cx_arcsec, cy_arcsec),
    )

    y_orig = obs.grid.y[:, 0]
    x_orig = obs.grid.x[0, :]
    interp_fn = RegularGridInterpolator((y_orig, x_orig), obs.image, bounds_error=False, fill_value=0.0)
    opt_data = interp_fn((opt_grid.y, opt_grid.x))

    interp_noise = RegularGridInterpolator(
        (y_orig, x_orig), obs.noise_sigma, bounds_error=False, fill_value=float(np.median(obs.noise_sigma))
    )
    opt_noise = interp_noise((opt_grid.y, opt_grid.x))

    dx = opt_grid.x - cx_arcsec
    dy = opt_grid.y - cy_arcsec
    r_dist = np.hypot(dx, dy)
    arc_mask = (r_dist >= 0.75) & (r_dist <= 2.15)
    masked_data = np.where(arc_mask, opt_data, 0.0)
    masked_noise = np.where(arc_mask, opt_noise, 1e5)

    hst_psf = GaussianPSF(fwhm=0.090, pixel_scale=opt_grid.pixel_scale, num_pix=9)
    src_grid = make_coordinate_grid(num_pix=18, field_of_view=1.4, unit="arcsec")

    # 4. Preliminary MAP estimate (Nelder-Mead)
    print(f"[2/4] Finding Maximum A Posteriori (MAP) Initialization...")
    initial_params = {
        "theta_e": 1.25,
        "axis_ratio": 0.85,
        "pos_angle_deg": 43.0,
        "gamma": 0.045,
        "phi_gamma_deg": 30.0,
    }
    bounds = {
        "theta_e": (1.05, 1.45),
        "axis_ratio": (0.65, 0.98),
        "pos_angle_deg": (25.0, 65.0),
        "gamma": (0.010, 0.090),
        "phi_gamma_deg": (10.0, 60.0),
    }

    map_res = fit_lens_parameters(
        image_grid=opt_grid,
        source_grid=src_grid,
        observed_image=masked_data,
        noise_sigma=masked_noise,
        initial_params=initial_params,
        param_bounds=bounds,
        psf=hst_psf,
        include_shear=True,
        regularization_lambda=2.0,
        max_iter=25,
    )
    map_p = map_res.best_params
    print(f"      MAP Estimate: theta_E={map_p['theta_e']:.3f}'', q={map_p['axis_ratio']:.3f}, "
          f"PA={map_p['pos_angle_deg']:.1f} deg, gamma={map_p['gamma']:.3f}")

    # 5. Define Bayesian Log-Prior and Log-Likelihood
    param_keys = ["theta_e", "axis_ratio", "pos_angle_deg", "gamma", "phi_gamma_deg"]
    p_map_vec = np.array([map_p[k] for k in param_keys])

    def log_prior(theta: np.ndarray) -> float:
        t_e, q, pa, gam, phi_g = theta
        if not (1.00 <= t_e <= 1.50):
            return -np.inf
        if not (0.60 <= q <= 0.99):
            return -np.inf
        if not (15.0 <= pa <= 75.0):
            return -np.inf
        if not (0.005 <= gam <= 0.10):
            return -np.inf
        if not (0.0 <= phi_g <= 90.0):
            return -np.inf
        return 0.0

    def log_likelihood(theta: np.ndarray) -> float:
        t_e, q, pa, gam, phi_g = theta
        try:
            sie = SingularIsothermalEllipsoid.from_degrees(
                theta_e=t_e, axis_ratio=q, pos_angle_deg=pa, center=(cx_arcsec, cy_arcsec)
            )
            shear = ExternalShear.from_polar_degrees(gamma=gam, phi_gamma_deg=phi_g)
            lens_m = CompositeLens([sie, shear])
            reconstructor = SourceReconstructor(
                image_grid=opt_grid,
                source_grid=src_grid,
                lens_model=lens_m,
                psf=hst_psf,
                regularization_method="curvature",
            )
            res = reconstructor.reconstruct(
                data_image=masked_data,
                noise_sigma=masked_noise,
                regularization_lambda=2.0,
            )
            return -0.5 * res.chi2
        except Exception:
            return -np.inf

    def log_posterior(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    # 6. Initialize MCMC Walkers
    ndim = len(param_keys)
    print(f"\n[3/4] Initializing {n_walkers} Affine-Invariant MCMC Walkers across {ndim} Dimensions...")
    perturb = np.array([0.02, 0.02, 1.5, 0.005, 1.5])
    p0 = p_map_vec + perturb * np.random.randn(n_walkers, ndim)

    # Enforce prior bounds on initial positions
    for i in range(n_walkers):
        while not np.isfinite(log_prior(p0[i])):
            p0[i] = p_map_vec + 0.5 * perturb * np.random.randn(ndim)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
    t_mcmc_start = time.time()
    print(f"      Sampling Posterior Distribution ({n_steps} steps per walker)...")
    sampler.run_mcmc(p0, n_steps, progress=False)
    t_mcmc_elapsed = time.time() - t_mcmc_start
    print(f"      MCMC Completed in {t_mcmc_elapsed:.1f}s ({n_walkers * n_steps} total likelihood evaluations)")

    # 7. Extract Samples & Compute Posterior Credible Intervals
    flat_samples = sampler.get_chain(discard=burn_in, flat=True)
    print(f"\n[4/4] Inferred Bayesian Posterior Credible Intervals (16th, 50th, 84th percentiles):")
    
    labels = [
        r"$\theta_E\ [^{\prime\prime}]$",
        r"$q$",
        r"$\phi\ [^\circ]$",
        r"$\gamma$",
        r"$\phi_\gamma\ [^\circ]$",
    ]

    results_table = []
    for i, k in enumerate(param_keys):
        mcmc_vals = np.percentile(flat_samples[:, i], [16, 50, 84])
        median_val = mcmc_vals[1]
        err_minus = median_val - mcmc_vals[0]
        err_plus = mcmc_vals[2] - median_val
        results_table.append((k, median_val, err_plus, err_minus))
        print(f"      {k:<14}: {median_val:.3f} (+{err_plus:.3f} / -{err_minus:.3f})")

    # Derived physical mass and velocity dispersion posterior distributions
    theta_e_samples = flat_samples[:, 0]
    r_einstein_m_samples = d_l_m * arcsec2rad(theta_e_samples)
    mass_samples_msun = (np.pi * (r_einstein_m_samples ** 2) * sigma_cr_si) / M_SUN
    sigma_v_samples = C_LIGHT * np.sqrt((arcsec2rad(theta_e_samples) / (4.0 * np.pi)) * (d_s_m / d_ls_m)) / 1000.0

    mass_perc = np.percentile(mass_samples_msun, [16, 50, 84])
    sigv_perc = np.percentile(sigma_v_samples, [16, 50, 84])
    print(f"\n      Derived Enclosed Mass : {mass_perc[1]:.2e} (+{mass_perc[2]-mass_perc[1]:.2e} / -{mass_perc[1]-mass_perc[0]:.2e}) M_sun")
    print(f"      Derived sigma_v       : {sigv_perc[1]:.1f} (+{sigv_perc[2]-sigv_perc[1]:.1f} / -{sigv_perc[1]-sigv_perc[0]:.1f}) km/s")

    # 8. Generate Publication Corner Plot
    fig = corner.corner(
        flat_samples,
        labels=labels,
        truths=[map_p[k] for k in param_keys],
        truth_color="crimson",
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 11, "fontweight": "bold"},
        title_fmt=".3f",
        color="navy",
        hist_kwargs={"density": True, "color": "royalblue"},
        levels=(0.68, 0.95),
        smooth=1.0,
    )
    fig.suptitle(
        "Bayesian Posterior Corner Plot: SDSS J0946+1006 (Real HST ACS Data)\n"
        r"$\mathbf{P}(\boldsymbol{\eta} | \mathbf{d})$ Affine-Invariant MCMC Posterior Distribution",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(corner_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("=" * 80)
    print(f"[SUCCESS] Rigorous Bayesian MCMC Inference Complete!")
    print(f"          Corner Plot Saved: {corner_path.resolve()}")
    print("=" * 80)
    return corner_path


if __name__ == "__main__":
    cutout_candidates = list(project_root.glob("real_data/*cutout*.fits"))
    if not cutout_candidates:
        print("[ERROR] No cutout FITS found in real_data/.", file=sys.stderr)
        sys.exit(1)

    target_cutout = cutout_candidates[0]
    out_dir = project_root / "outputs"
    run_rigorous_mcmc_inference(target_cutout, out_dir)
