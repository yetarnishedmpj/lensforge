#!/usr/bin/env python3
"""LensForge SLACS Benchmark: Gravitational Lens Reconstruction of SDSS J0946+1006.

Case study replicating the published properties of the Sloan Lens ACS Survey (SLACS)
Einstein ring SDSS J0946+1006 (Bolton et al. 2008, Gavazzi et al. 2008, Dye et al. 2014):
    - Deflector: Early-type galaxy at z_l = 0.222
    - Source: Star-forming galaxy at z_s = 0.609
    - Inner Einstein Radius: theta_E ~ 1.38 arcsec
    - Central Velocity Dispersion: sigma_v ~ 280 km/s

This script demonstrates:
    1. Cosmological distance calculation and critical surface density via FlatLambdaCDMCosmology (PLANCK18).
    2. High-fidelity synthetic HST ACS/WFC F814W observation generation with realistic PSF & noise.
    3. FITS data packaging with WCS and multi-extension HDUs (SCI, ERR, MASK).
    4. Non-linear bilevel optimization recovering macro-lens parameters (SIE + Shear).
    5. Linear regularized reconstruction of the unlensed pixelated source galaxy.
    6. Physical mass and velocity dispersion derivation from gravitational lensing.
    7. Publication-grade 6-panel summary dashboard exported to outputs/slacs_j0946_benchmark.png.
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

from lensforge.constants import C_LIGHT, G_NEWTON, M_SUN, MEGAPARSEC, PARSEC, arcsec2rad
from lensforge.cosmology import PLANCK18
from lensforge.data import AstroObservation, create_annular_mask, save_fits_observation
from lensforge.imaging.source_models import GaussianSource
from lensforge.instrument.noise import add_observation_noise
from lensforge.instrument.psf import GaussianPSF
from lensforge.inversion.reconstructor import SourceReconstructor
from lensforge.optimization import fit_lens_parameters
from lensforge.physics.coordinates import make_coordinate_grid
from lensforge.physics.models.composite import CompositeLens
from lensforge.physics.models.shear import ExternalShear
from lensforge.physics.models.sie import SingularIsothermalEllipsoid


def compute_critical_and_caustic_curves(lens_model, grid_fov: float = 4.2, num_grid: int = 150):
    """Compute critical curves (image plane) and caustics (source plane)."""
    x = np.linspace(-grid_fov / 2, grid_fov / 2, num_grid)
    y = np.linspace(-grid_fov / 2, grid_fov / 2, num_grid)
    xx, yy = np.meshgrid(x, y)
    a_xx, a_xy, a_yx, a_yy = lens_model.jacobian(xx, yy, eps=1e-4)
    det_a = a_xx * a_yy - a_xy * a_yx
    return xx, yy, det_a


def run_slacs_benchmark(output_dir: Path) -> None:
    """Execute the end-to-end SLACS SDSS J0946+1006 benchmark reconstruction."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fits_path = output_dir / "slacs_j0946_observation.fits"
    dashboard_path = output_dir / "slacs_j0946_benchmark.png"

    print("=" * 78)
    print(" LENSFORGE RESEARCH BENCHMARK: SDSS J0946+1006 (SLACS EINSTEIN RING)")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # 1. Cosmological Calculations (Flat Lambda-CDM Planck18)
    # -------------------------------------------------------------------------
    z_lens = 0.222
    z_source = 0.609

    d_l_mpc = PLANCK18.angular_diameter_distance_mpc(0.0, z_lens)
    d_s_mpc = PLANCK18.angular_diameter_distance_mpc(0.0, z_source)
    d_ls_mpc = PLANCK18.angular_diameter_distance_mpc(z_lens, z_source)

    sigma_cr_si = PLANCK18.critical_surface_density(z_lens, z_source)
    sigma_cr_astro = PLANCK18.critical_surface_density_msun_pc2(z_lens, z_source)

    print(f"\n[1/6] Cosmological Framework (Planck 2018):")
    print(f"      Deflector Redshift z_l     : {z_lens:.3f} (D_l = {d_l_mpc:.1f} Mpc)")
    print(f"      Source Redshift z_s        : {z_source:.3f} (D_s = {d_s_mpc:.1f} Mpc)")
    print(f"      Angular Diameter Dist D_ls : {d_ls_mpc:.1f} Mpc")
    print(f"      Critical Surface Density   : {sigma_cr_astro:.1f} M_sun/pc^2 ({sigma_cr_si:.3f} kg/m^2)")

    # -------------------------------------------------------------------------
    # 2. Ground-Truth Lens & Source Specification
    # -------------------------------------------------------------------------
    # Ground truth parameters matching Bolton et al. 2008 & Dye et al. 2014
    true_theta_e = 1.38      # arcsec
    true_axis_ratio = 0.88   # q
    true_pa_deg = 42.0       # deg (East of North)
    true_gamma = 0.045       # external shear
    true_phi_gamma_deg = 30.0

    sie_true = SingularIsothermalEllipsoid.from_degrees(
        theta_e=true_theta_e,
        axis_ratio=true_axis_ratio,
        pos_angle_deg=true_pa_deg,
        center=(0.0, 0.0),
    )
    shear_true = ExternalShear.from_polar_degrees(
        gamma=true_gamma,
        phi_gamma_deg=true_phi_gamma_deg,
        center=(0.0, 0.0),
    )
    lens_truth = CompositeLens([sie_true, shear_true])

    # Theoretical SIS velocity dispersion: theta_E = 4 pi (sigma_v/c)^2 (D_ls / D_s)
    # sigma_v = c * sqrt(theta_E_rad / (4 pi) * (D_s / D_ls))
    theta_e_rad = arcsec2rad(true_theta_e)
    d_s_m = d_s_mpc * MEGAPARSEC
    d_ls_m = d_ls_mpc * MEGAPARSEC
    d_l_m = d_l_mpc * MEGAPARSEC
    theory_sigma_v = C_LIGHT * np.sqrt((theta_e_rad / (4.0 * np.pi)) * (d_s_m / d_ls_m)) / 1000.0  # km/s

    # Enclosed mass within Einstein radius: M(< R_E) = pi * (D_l * theta_E)^2 * Sigma_cr
    r_einstein_m = d_l_m * theta_e_rad
    mass_enclosed_kg = np.pi * (r_einstein_m ** 2) * sigma_cr_si
    mass_enclosed_msun = mass_enclosed_kg / M_SUN

    print(f"\n[2/6] Deflector Astrophysics (Ground Truth):")
    print(f"      Einstein Radius theta_E    : {true_theta_e:.2f} arcsec ({r_einstein_m / PARSEC / 1000.0:.2f} kpc)")
    print(f"      Axis Ratio q               : {true_axis_ratio:.2f}")
    print(f"      Position Angle             : {true_pa_deg:.1f} deg")
    print(f"      External Shear gamma       : {true_gamma:.3f} at {true_phi_gamma_deg:.1f} deg")
    print(f"      Predicted Central sigma_v  : {theory_sigma_v:.1f} km/s (Observed SDSS: ~280 +/- 15 km/s)")
    print(f"      Enclosed Mass M(< theta_E) : {mass_enclosed_msun:.2e} M_sun")

    # High-z clumpy source galaxy (central bulge + star-forming knot)
    source_core = GaussianSource.from_degrees(
        center=(0.10, 0.08),
        sigma=0.09,
        amp=5.0,
        axis_ratio=0.75,
        pos_angle_deg=55.0,
    )
    source_knot = GaussianSource.from_degrees(
        center=(0.22, 0.16),
        sigma=0.045,
        amp=3.2,
        axis_ratio=0.85,
        pos_angle_deg=20.0,
    )

    def evaluate_true_source(bx: np.ndarray, by: np.ndarray) -> np.ndarray:
        return source_core.evaluate(bx, by) + source_knot.evaluate(bx, by)

    # -------------------------------------------------------------------------
    # 3. Synthetic HST ACS/WFC Observation Generation & FITS Export
    # -------------------------------------------------------------------------
    num_img = 56
    fov_img = 4.2  # arcsec, pixel scale = 0.075 arcsec/pixel
    img_grid = make_coordinate_grid(num_pix=num_img, field_of_view=fov_img, unit="arcsec")

    # Ray trace to source plane
    beta_x, beta_y = lens_truth.ray_trace(img_grid.x, img_grid.y, mask_singular=True)
    noiseless_lensed = np.nan_to_num(evaluate_true_source(beta_x, beta_y), nan=0.0)

    # HST ACS PSF: FWHM = 0.09 arcsec
    psf_fwhm = 0.09
    hst_psf = GaussianPSF(fwhm=psf_fwhm, pixel_scale=img_grid.pixel_scale, num_pix=11)
    convolved_lensed = hst_psf.convolve(noiseless_lensed)

    # Realistic noise: background noise + Poisson shot noise (HST ACS-like)
    noisy_observation, std_map = add_observation_noise(
        convolved_lensed,
        read_noise=3.0,
        exposure_time=2000.0,
        sky_background=0.5,
        seed=101,
    )

    # Deflector annular mask: remove core (deflector starlight subtraction residual) and outer margins
    annular_mask = create_annular_mask(img_grid, r_inner=0.35, r_outer=2.05)

    # Package as AstroObservation and write to FITS
    fits_header = {
        "OBJECT": "SDSS J0946+1006",
        "TELESCOP": "HST",
        "INSTRUME": "ACS/WFC",
        "FILTER": "F814W",
        "EXPTIME": 2000.0,
        "REDSH_L": z_lens,
        "REDSH_S": z_source,
        "THETA_E": true_theta_e,
    }
    save_fits_observation(
        filepath=fits_path,
        image=noisy_observation,
        variance=std_map**2,
        psf_kernel=hst_psf.kernel(),
        pixel_scale=img_grid.pixel_scale,
        header_info=fits_header,
    )

    print(f"\n[3/6] Synthetic HST ACS Observation Generated:")
    print(f"      Grid Dimensions            : {num_img}x{num_img} ({fov_img:.1f}'' x {fov_img:.1f}'')")
    print(f"      Pixel Scale                : {img_grid.pixel_scale:.3f} arcsec/pixel")
    print(f"      PSF Model                  : HST Gaussian (FWHM = {psf_fwhm:.3f}'')")
    print(f"      Peak Signal-to-Noise       : {np.max(noisy_observation) / np.median(std_map):.1f}")
    print(f"      FITS Observation Exported  : {fits_path.name}")

    # -------------------------------------------------------------------------
    # 4. Bilevel Semilinear Macro-Lens Parameter Fitting
    # -------------------------------------------------------------------------
    # Compact source grid for fast, well-conditioned linear inversion
    num_src = 20
    fov_src = 1.4  # arcsec
    src_grid = make_coordinate_grid(num_pix=num_src, field_of_view=fov_src, unit="arcsec")

    # Intentionally perturb initial parameters to test parameter recovery
    initial_guess = {
        "theta_e": 1.25,         # -9.4% perturbation
        "axis_ratio": 0.78,      # -11.4% perturbation
        "pos_angle_deg": 35.0,   # -7 deg offset
        "gamma": 0.030,          # -33% offset
        "phi_gamma_deg": 25.0,   # -5 deg offset
    }

    bounds = {
        "theta_e": (0.8, 1.8),
        "axis_ratio": (0.5, 0.99),
        "pos_angle_deg": (10.0, 80.0),
        "gamma": (0.005, 0.15),
        "phi_gamma_deg": (0.0, 90.0),
    }

    print(f"\n[4/6] Executing Non-Linear Macro-Lens Optimization (Nelder-Mead):")
    print(f"      Initial Guess : theta_E={initial_guess['theta_e']:.2f}'', q={initial_guess['axis_ratio']:.2f}, "
          f"PA={initial_guess['pos_angle_deg']:.1f} deg, gamma={initial_guess['gamma']:.3f}")

    start_time = time.time()
    fit_result = fit_lens_parameters(
        image_grid=img_grid,
        source_grid=src_grid,
        observed_image=noisy_observation,
        noise_sigma=std_map,
        initial_params=initial_guess,
        param_bounds=bounds,
        psf=hst_psf,
        include_shear=True,
        regularization_lambda=2.0,
        objective="chi2",
        method="Nelder-Mead",
        max_iter=30,
    )
    elapsed = time.time() - start_time

    best_p = fit_result.best_params
    fitted_theta_e = best_p["theta_e"]
    fitted_q = best_p["axis_ratio"]
    fitted_pa = best_p["pos_angle_deg"]
    fitted_gamma = best_p["gamma"]

    # Calculate derived mass & velocity dispersion from fitted Einstein radius
    fitted_theta_e_rad = arcsec2rad(fitted_theta_e)
    fitted_r_einstein_m = d_l_m * fitted_theta_e_rad
    fitted_mass_kg = np.pi * (fitted_r_einstein_m ** 2) * sigma_cr_si
    fitted_mass_msun = fitted_mass_kg / M_SUN
    fitted_sigma_v = C_LIGHT * np.sqrt((fitted_theta_e_rad / (4.0 * np.pi)) * (d_s_m / d_ls_m)) / 1000.0

    print(f"      Optimization Completed in  : {elapsed:.2f} seconds ({len(fit_result.optimization_history)} evaluations)")
    print(f"      Recovered theta_E          : {fitted_theta_e:.3f}'' (Error: {abs(fitted_theta_e - true_theta_e)/true_theta_e*100:.2f}%)")
    print(f"      Recovered Axis Ratio q     : {fitted_q:.3f} (Error: {abs(fitted_q - true_axis_ratio)/true_axis_ratio*100:.2f}%)")
    print(f"      Recovered PA               : {fitted_pa:.1f} deg (Error: {abs(fitted_pa - true_pa_deg):.1f} deg)")
    print(f"      Recovered Shear gamma      : {fitted_gamma:.3f} (Error: {abs(fitted_gamma - true_gamma)/true_gamma*100:.2f}%)")
    print(f"      Recovered Lens Mass        : {fitted_mass_msun:.2e} M_sun (True: {mass_enclosed_msun:.2e} M_sun)")
    print(f"      Recovered sigma_v          : {fitted_sigma_v:.1f} km/s (True: {theory_sigma_v:.1f} km/s)")

    # -------------------------------------------------------------------------
    # 5. Scientific Validation & Diagnostics
    # -------------------------------------------------------------------------
    rec_res = fit_result.reconstruction
    model_img = rec_res.model_image
    residuals = noisy_observation - model_img
    norm_residuals = residuals / std_map

    # Masked chi-squared evaluation
    masked_norm_res = norm_residuals[annular_mask]
    chi2_red = np.mean(masked_norm_res**2)

    print(f"\n[5/6] Goodness-of-Fit & Reconstruction Quality:")
    print(f"      Reduced Chi-squared (chi2_red): {chi2_red:.3f} (Target: ~1.00)")
    print(f"      Log-Evidence                : {rec_res.log_evidence:.2f}")

    # Evaluate caustics & critical curves for best-fit model
    xx_crit, yy_crit, det_a = compute_critical_and_caustic_curves(
        fit_result.best_lens_model,
        grid_fov=fov_img,
        num_grid=160,
    )

    # -------------------------------------------------------------------------
    # 6. Multi-Panel Publication-Grade Benchmark Dashboard
    # -------------------------------------------------------------------------
    print(f"\n[6/6] Generating Publication Dashboard...")
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    extent_img = [-fov_img / 2, fov_img / 2, -fov_img / 2, fov_img / 2]
    extent_src = [-fov_src / 2, fov_src / 2, -fov_src / 2, fov_src / 2]

    # Panel 1: Observed Mock HST Image
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(noisy_observation, origin="lower", extent=extent_img, cmap="magma")
    # Overlay annular mask boundary
    ax1.contour(img_grid.x, img_grid.y, annular_mask.astype(float), levels=[0.5], colors="cyan", linestyles="--", linewidths=1.0)
    ax1.set_title("A. Mock HST ACS Observation\n(SDSS J0946+1006, F814W)", fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"$\theta_x$ [arcsec]", fontsize=10)
    ax1.set_ylabel(r"$\theta_y$ [arcsec]", fontsize=10)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Flux [counts/s]")

    # Panel 2: Best-Fit Reconstructed Lensed Model
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(model_img, origin="lower", extent=extent_img, cmap="magma")
    # Overlay critical curves
    cs_crit = ax2.contour(xx_crit, yy_crit, det_a, levels=[0.0], colors=["lime"], linewidths=1.2)
    ax2.set_title("B. Best-Fit Lensed Model\n($P * \mathbf{F}\hat{\mathbf{s}}$, SIE + Shear)", fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"$\theta_x$ [arcsec]", fontsize=10)
    ax2.set_ylabel(r"$\theta_y$ [arcsec]", fontsize=10)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Model Flux [counts/s]")

    # Panel 3: Normalized Residuals (D - M) / sigma
    ax3 = fig.add_subplot(gs[0, 2])
    vlim = 3.0
    im3 = ax3.imshow(norm_residuals, origin="lower", extent=extent_img, cmap="coolwarm", vmin=-vlim, vmax=vlim)
    ax3.set_title(f"C. Normalized Residuals $(D - M)/\\sigma$\n($\\chi^2_{{\\mathrm{{red}}}} = {chi2_red:.2f}$)", fontsize=11, fontweight="bold")
    ax3.set_xlabel(r"$\theta_x$ [arcsec]", fontsize=10)
    ax3.set_ylabel(r"$\theta_y$ [arcsec]", fontsize=10)
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label=r"Residual $[\sigma]$")
    cbar3.set_ticks([-3, -2, -1, 0, 1, 2, 3])

    # Panel 4: Inverted Source Galaxy with Caustic Curves
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(rec_res.reconstructed_source, origin="lower", extent=extent_src, cmap="inferno")
    # Trace caustics by ray tracing critical curve points back to source plane
    crit_paths = cs_crit.get_paths() if hasattr(cs_crit, "get_paths") else []
    for path in crit_paths:
        v = path.vertices
        if len(v) > 2:
            bx_c, by_c = fit_result.best_lens_model.ray_trace(v[:, 0], v[:, 1])
            # filter points outside source grid extent for clean plot
            valid = (np.abs(bx_c) <= fov_src / 2 * 1.2) & (np.abs(by_c) <= fov_src / 2 * 1.2)
            if np.any(valid):
                ax4.plot(bx_c[valid], by_c[valid], color="cyan", lw=1.2, ls="-")
    ax4.set_xlim(-fov_src / 2, fov_src / 2)
    ax4.set_ylim(-fov_src / 2, fov_src / 2)
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

    # Panel 6: Astrophysical Recovery Summary Card
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    summary_text = (
        "\\textbf{SLACS SDSS J0946+1006 Benchmark}\n"
        "\\noindent\\rule{5.0cm}{0.4pt}\n"
        f"Deflector Redshift: $z_l = {z_lens:.3f}$\n"
        f"Source Redshift: $z_s = {z_source:.3f}$\n"
        f"Cosmology: $\\text{{Flat }}\\Lambda\\text{{CDM}}$ (Planck18)\n\n"
        "\\textbf{Parameter Recovery:}\n"
        f"$\\bullet$ $\\theta_E$: {fitted_theta_e:.3f}'' (True: {true_theta_e:.2f}'')\n"
        f"$\\bullet$ Axis ratio $q$: {fitted_q:.3f} (True: {true_axis_ratio:.2f})\n"
        f"$\\bullet$ Pos Angle $\\phi$: {fitted_pa:.1f}$^\\circ$ (True: {true_pa_deg:.1f}$^\\circ$)\n"
        f"$\\bullet$ Shear $\\gamma$: {fitted_gamma:.3f} (True: {true_gamma:.3f})\n\n"
        "\\textbf{Derived Physical Properties:}\n"
        f"$\\bullet$ $\\sigma_v$: {fitted_sigma_v:.1f} km/s (SDSS: $\\approx 280$ km/s)\n"
        f"$\\bullet$ $M(< \\theta_E)$: ${fitted_mass_msun:.2e}\\,M_\\odot$\n"
        f"$\\bullet$ $\\Sigma_{{\\text{{cr}}}}$: ${sigma_cr_astro:.1f}\\,M_\\odot/\\text{{pc}}^2$\n"
        f"$\\bullet$ Reduced $\\chi^2$: {chi2_red:.2f}"
    )

    # Use plain text formatting for robust rendering across all Matplotlib backends
    clean_summary = (
        "SLACS SDSS J0946+1006 Benchmark\n"
        "--------------------------------------------------\n"
        f"Deflector Redshift:  z_l = {z_lens:.3f}\n"
        f"Source Redshift:     z_s = {z_source:.3f}\n"
        f"Cosmology:           Flat LCDM (Planck18)\n\n"
        "Parameter Recovery:\n"
        f"  * Einstein Radius: {fitted_theta_e:.3f}'' (True: {true_theta_e:.2f}'')\n"
        f"  * Axis Ratio q:    {fitted_q:.3f} (True: {true_axis_ratio:.2f})\n"
        f"  * Position Angle:  {fitted_pa:.1f} deg (True: {true_pa_deg:.1f} deg)\n"
        f"  * External Shear:  {fitted_gamma:.3f} (True: {true_gamma:.3f})\n\n"
        "Derived Physical Properties:\n"
        f"  * Velocity Disp:   {fitted_sigma_v:.1f} km/s (SDSS: ~280 km/s)\n"
        f"  * Enclosed Mass:   {fitted_mass_msun:.2e} M_sun\n"
        f"  * Critical Sigma:  {sigma_cr_astro:.1f} M_sun/pc^2\n"
        f"  * Reduced chi2:    {chi2_red:.2f} (DOF ~ {len(masked_norm_res)})\n"
        f"  * Source Inversion: Curvature Regularization (lambda={2.5})"
    )

    ax6.text(
        0.05,
        0.95,
        clean_summary,
        transform=ax6.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ced4da", lw=1.2),
    )
    ax6.set_title("F. Astrophysical Parameter Summary", fontsize=11, fontweight="bold")

    fig.savefig(dashboard_path, dpi=300)
    plt.close(fig)

    print(f"\n[SUCCESS] Benchmark Complete!")
    print(f"          Publication Figure Saved : {dashboard_path.resolve()}")
    print(f"          FITS Science Data Saved  : {fits_path.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    out = project_root / "outputs"
    run_slacs_benchmark(out)
