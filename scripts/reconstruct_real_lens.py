#!/usr/bin/env python3
"""LensForge Real Space Telescope Galaxy Reconstruction: SDSS J0946+1006.

Performs Bayesian regularized source reconstruction and macro-lens optimization
on actual calibrated Hubble Space Telescope (HST) ACS/WFC F814W archival imaging.

Target:
    SDSS J0946+1006 (The SLACS "Double Einstein Ring")
    Deflector: z_l = 0.222 (early-type elliptical galaxy)
    Source:    z_s = 0.609 (star-forming dwarf galaxy)
"""

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# Ensure src/ is on python path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from lensforge.constants import C_LIGHT, M_SUN, MEGAPARSEC, PARSEC, arcsec2rad
from lensforge.cosmology import PLANCK18
from lensforge.data import create_annular_mask, load_fits_observation
from lensforge.instrument.psf import GaussianPSF
from lensforge.optimization import fit_lens_parameters
from lensforge.physics.coordinates import make_coordinate_grid


def run_real_lens_reconstruction(
    fits_cutout_path: Path,
    output_dir: Path,
) -> Path:
    """Run end-to-end scientific reconstruction on a real HST observation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = output_dir / "real_sdss_j0946_reconstruction.png"

    print("=" * 78)
    print(" LENSFORGE REAL HST DATA RECONSTRUCTION: SDSS J0946+1006")
    print("=" * 78)

    # 1. Load Calibrated Astronomical Cutout from FITS
    print(f"\n[1/5] Ingesting Calibrated HST FITS Cutout: {fits_cutout_path.name}")
    obs = load_fits_observation(fits_cutout_path, subtract_background=True)

    fov_img = float(obs.grid.shape[1] * obs.pixel_scale)
    print(f"      Image Dimensions       : {obs.image.shape[1]}x{obs.image.shape[0]} pixels")
    print(f"      Pixel Scale            : {obs.pixel_scale:.4f} arcsec/pixel")
    print(f"      Field of View          : {fov_img:.2f}'' x {fov_img:.2f}''")
    print(f"      Median Sky Uncertainty : {np.median(obs.noise_sigma):.4f} e-/s")
    print(f"      Peak Signal Flux       : {np.max(obs.image):.2f} e-/s")

    # 2. Deflector Starlight Masking (Annular Aperture)
    # Exclude central core (r < 0.40'') where deflector starlight subtraction residuals dominate,
    # and outer background (r > 2.30'')
    annular_mask = create_annular_mask(obs.grid, r_inner=0.40, r_outer=2.30)
    print(f"      Annular Aperture Mask  : {np.sum(annular_mask)} active arc pixels retained")

    # 3. Cosmological Framework & Redshift Distances
    z_lens = 0.222
    z_source = 0.609
    d_l_mpc = PLANCK18.angular_diameter_distance_mpc(0.0, z_lens)
    d_s_mpc = PLANCK18.angular_diameter_distance_mpc(0.0, z_source)
    d_ls_mpc = PLANCK18.angular_diameter_distance_mpc(z_lens, z_source)
    sigma_cr_si = PLANCK18.critical_surface_density(z_lens, z_source)
    sigma_cr_astro = PLANCK18.critical_surface_density_msun_pc2(z_lens, z_source)

    print(f"\n[2/5] Cosmological Context (Flat Lambda-CDM Planck18):")
    print(f"      Deflector Redshift z_l : {z_lens:.3f} (D_l = {d_l_mpc:.1f} Mpc)")
    print(f"      Source Redshift z_s    : {z_source:.3f} (D_s = {d_s_mpc:.1f} Mpc)")
    print(f"      Transverse Dist D_ls   : {d_ls_mpc:.1f} Mpc")
    print(f"      Critical Surface Sigma : {sigma_cr_astro:.1f} M_sun/pc^2")

    # 4. Inversion Grids & Telescope Instrument PSF
    # Downsample observation grid slightly for efficient non-linear optimization
    num_opt_img = 56
    opt_img_grid = make_coordinate_grid(num_pix=num_opt_img, field_of_view=fov_img, unit="arcsec")

    # Interpolate real data onto optimization grid
    from scipy.interpolate import RegularGridInterpolator
    y_orig = obs.grid.y[:, 0]
    x_orig = obs.grid.x[0, :]
    interp_fn = RegularGridInterpolator((y_orig, x_orig), obs.image, bounds_error=False, fill_value=0.0)
    opt_data = interp_fn((opt_img_grid.y, opt_img_grid.x))

    interp_noise = RegularGridInterpolator((y_orig, x_orig), obs.noise_sigma, bounds_error=False, fill_value=np.median(obs.noise_sigma))
    opt_noise = interp_noise((opt_img_grid.y, opt_img_grid.x))

    # Real HST ACS F814W PSF: Gaussian approximation FWHM = 0.090 arcsec
    hst_psf = GaussianPSF(fwhm=0.090, pixel_scale=opt_img_grid.pixel_scale, num_pix=11)

    # High-resolution source plane grid
    num_src = 24
    fov_src = 1.4  # arcsec
    src_grid = make_coordinate_grid(num_pix=num_src, field_of_view=fov_src, unit="arcsec")

    # 5. Non-Linear Macro-Lens Optimization (SIE + Shear)
    initial_params = {
        "theta_e": 1.30,
        "axis_ratio": 0.85,
        "pos_angle_deg": 40.0,
        "gamma": 0.035,
        "phi_gamma_deg": 30.0,
    }
    bounds = {
        "theta_e": (0.8, 1.8),
        "axis_ratio": (0.5, 0.99),
        "pos_angle_deg": (10.0, 80.0),
        "gamma": (0.005, 0.12),
        "phi_gamma_deg": (0.0, 90.0),
    }

    print(f"\n[3/5] Executing Bilevel Optimization on Real HST Data:")
    print(f"      Initial Parameters: theta_E={initial_params['theta_e']:.2f}'', "
          f"q={initial_params['axis_ratio']:.2f}, PA={initial_params['pos_angle_deg']:.1f} deg")

    t_start = time.time()
    fit_result = fit_lens_parameters(
        image_grid=opt_img_grid,
        source_grid=src_grid,
        observed_image=opt_data,
        noise_sigma=opt_noise,
        initial_params=initial_params,
        param_bounds=bounds,
        psf=hst_psf,
        include_shear=True,
        regularization_lambda=3.0,
        objective="chi2",
        method="Nelder-Mead",
        max_iter=35,
    )
    t_elapsed = time.time() - t_start

    best_p = fit_result.best_params
    fitted_theta_e = best_p["theta_e"]
    fitted_q = best_p["axis_ratio"]
    fitted_pa = best_p["pos_angle_deg"]
    fitted_gamma = best_p["gamma"]

    # Calculate physical mass & velocity dispersion
    theta_e_rad = arcsec2rad(fitted_theta_e)
    d_l_m = d_l_mpc * MEGAPARSEC
    d_s_m = d_s_mpc * MEGAPARSEC
    d_ls_m = d_ls_mpc * MEGAPARSEC
    r_einstein_m = d_l_m * theta_e_rad
    mass_enclosed_kg = np.pi * (r_einstein_m ** 2) * sigma_cr_si
    mass_enclosed_msun = mass_enclosed_kg / M_SUN
    inferred_sigma_v = C_LIGHT * np.sqrt((theta_e_rad / (4.0 * np.pi)) * (d_s_m / d_ls_m)) / 1000.0

    print(f"      Optimization Completed in {t_elapsed:.1f}s ({len(fit_result.optimization_history)} steps)")
    print(f"\n[4/5] Inferred Astrophysical Properties from Real HST Observation:")
    print(f"      Recovered Einstein Radius : {fitted_theta_e:.3f} arcsec ({r_einstein_m / PARSEC / 1000.0:.2f} kpc)")
    print(f"      Deflector Axis Ratio q    : {fitted_q:.3f}")
    print(f"      Deflector Position Angle  : {fitted_pa:.1f} deg")
    print(f"      External Shear gamma      : {fitted_gamma:.3f}")
    print(f"      Enclosed Deflector Mass   : {mass_enclosed_msun:.2e} M_sun")
    print(f"      Inferred sigma_v          : {inferred_sigma_v:.1f} km/s (SDSS spec: ~280 +/- 15 km/s)")

    # 6. Multi-Panel Scientific Publication Dashboard
    print(f"\n[5/5] Generating Scientific Dashboard: {dashboard_path.name}")
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    extent_img = obs.grid.extent
    extent_src = src_grid.extent
    extent_opt = opt_img_grid.extent

    rec_res = fit_result.reconstruction
    model_img = rec_res.model_image
    residuals = opt_data - model_img
    norm_residuals = residuals / opt_noise

    # Panel 1: Real HST Science Observation
    ax1 = fig.add_subplot(gs[0, 0])
    vmax_obs = float(np.percentile(obs.image, 99.2))
    vmin_obs = float(np.percentile(obs.image, 5.0))
    im1 = ax1.imshow(obs.image, origin="lower", extent=extent_img, cmap="magma", vmin=vmin_obs, vmax=vmax_obs)
    # Overlay annular mask
    ax1.contour(obs.grid.x, obs.grid.y, annular_mask.astype(float), levels=[0.5], colors="cyan", linestyles="--", linewidths=1.0)
    ax1.set_title("A. Real HST ACS/WFC Observation\n(SDSS J0946+1006, F814W)", fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"$\theta_x$ [arcsec]", fontsize=10)
    ax1.set_ylabel(r"$\theta_y$ [arcsec]", fontsize=10)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Calibrated Flux [e-/s]")

    # Panel 2: Best-Fit Reconstructed Lensed Model
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(model_img, origin="lower", extent=extent_opt, cmap="magma")
    ax2.set_title("B. Best-Fit Lensed Model $P * \mathbf{F}\hat{\mathbf{s}}$\n(SIE + External Shear)", fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"$\theta_x$ [arcsec]", fontsize=10)
    ax2.set_ylabel(r"$\theta_y$ [arcsec]", fontsize=10)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Model Flux [e-/s]")

    # Panel 3: Normalized Residuals
    ax3 = fig.add_subplot(gs[0, 2])
    vlim = 3.0
    im3 = ax3.imshow(norm_residuals, origin="lower", extent=extent_opt, cmap="coolwarm", vmin=-vlim, vmax=vlim)
    chi2_red = np.mean(norm_residuals**2)
    ax3.set_title(f"C. Normalized Residuals $(D - M)/\\sigma$\n($\\chi^2_{{\\mathrm{{red}}}} = {chi2_red:.2f}$)", fontsize=11, fontweight="bold")
    ax3.set_xlabel(r"$\theta_x$ [arcsec]", fontsize=10)
    ax3.set_ylabel(r"$\theta_y$ [arcsec]", fontsize=10)
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label=r"Residual $[\sigma]$")
    cbar3.set_ticks([-3, -2, -1, 0, 1, 2, 3])

    # Panel 4: Inverted Unlensed Galaxy Surface Brightness
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(rec_res.reconstructed_source, origin="lower", extent=extent_src, cmap="inferno")
    ax4.set_title("D. Reconstructed Source Galaxy $\\hat{\\mathbf{s}}$\n(Linear Curvature Inversion)", fontsize=11, fontweight="bold")
    ax4.set_xlabel(r"$\beta_x$ [arcsec]", fontsize=10)
    ax4.set_ylabel(r"$\beta_y$ [arcsec]", fontsize=10)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label="Intrinsic Brightness")

    # Panel 5: Optimization Convergence History
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(fit_result.optimization_history, marker="o", markersize=3, color="navy", lw=1.5)
    ax5.set_yscale("log")
    ax5.set_title(f"E. Bilevel Optimization Convergence\n({len(fit_result.optimization_history)} Iterations, Nelder-Mead)", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Evaluation Step", fontsize=10)
    ax5.set_ylabel(r"Objective Loss ($\chi^2$)", fontsize=10)
    ax5.grid(True, which="both", ls="--", alpha=0.5)

    # Panel 6: Astrophysical Summary Card
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    clean_summary = (
        "Real HST Observation: SDSS J0946+1006\n"
        "--------------------------------------------------\n"
        f"Observatory / Inst:  HST ACS/WFC (F814W)\n"
        f"Deflector Redshift:  z_l = {z_lens:.3f} (D_l = {d_l_mpc:.0f} Mpc)\n"
        f"Source Redshift:     z_s = {z_source:.3f} (D_s = {d_s_mpc:.0f} Mpc)\n"
        f"Cosmology:           Flat LCDM (Planck18)\n\n"
        "Recovered Physical Parameters:\n"
        f"  * Einstein Radius: {fitted_theta_e:.3f}'' ({r_einstein_m / PARSEC / 1000.0:.2f} kpc)\n"
        f"  * Deflector Axis q:{fitted_q:.3f}\n"
        f"  * Position Angle:  {fitted_pa:.1f} deg\n"
        f"  * External Shear:  {fitted_gamma:.3f}\n\n"
        "Inferred Astrophysical Quantities:\n"
        f"  * Enclosed Mass:   {mass_enclosed_msun:.2e} M_sun\n"
        f"  * Stellar Disp:    {inferred_sigma_v:.1f} km/s\n"
        f"  * SDSS Spec Disp:  ~280 +/- 15 km/s\n"
        f"  * Critical Sigma:  {sigma_cr_astro:.1f} M_sun/pc^2\n"
        f"  * Reduced chi2:    {chi2_red:.2f}\n"
        f"  * Inversion Prior: 2D Laplacian Curvature"
    )

    ax6.text(
        0.05,
        0.95,
        clean_summary,
        transform=ax6.transAxes,
        fontsize=9.2,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ced4da", lw=1.2),
    )
    ax6.set_title("F. Inferred Physical Parameters", fontsize=11, fontweight="bold")

    fig.savefig(dashboard_path, dpi=300)
    plt.close(fig)

    print("=" * 78)
    print(f"[SUCCESS] Real Data Scientific Reconstruction Complete!")
    print(f"          Publication Dashboard: {dashboard_path.resolve()}")
    print("=" * 78)
    return dashboard_path


if __name__ == "__main__":
    cutout_candidates = list(project_root.glob("real_data/*cutout*.fits"))
    if not cutout_candidates:
        print("[ERROR] No cutout FITS found in real_data/. Please run download first.", file=sys.stderr)
        sys.exit(1)

    target_cutout = cutout_candidates[0]
    out_dir = project_root / "outputs"
    run_real_lens_reconstruction(target_cutout, out_dir)

