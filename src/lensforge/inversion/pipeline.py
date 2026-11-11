"""High-level galaxy lens reconstruction pipeline.

Provides `run_galaxy_reconstruction` to execute the full end-to-end inverse problem:
1. Simulates realistic observation (distorted arcs + telescope PSF + Poisson/read noise).
2. Formulates linear inverse problem.
3. Automatically finds Bayesian optimal regularization lambda.
4. Recovers unlensed galaxy morphology.
5. Renders 6-panel scientific dashboard.
"""

from pathlib import Path
from typing import Optional
import numpy as np

from lensforge.imaging.source_models import GaussianSource
from lensforge.instrument.noise import add_observation_noise
from lensforge.instrument.psf import GaussianPSF
from lensforge.inversion.reconstructor import SourceReconstructor
from lensforge.physics.coordinates import make_coordinate_grid
from lensforge.physics.models.composite import CompositeLens
from lensforge.physics.models.shear import ExternalShear
from lensforge.physics.models.sie import SingularIsothermalEllipsoid
from lensforge.visualization.lens_plots import plot_reconstruction_dashboard


def run_galaxy_reconstruction(output_dir: Path) -> Path:
    """Execute end-to-end galaxy lens reconstruction pipeline.

    Args:
        output_dir: Path where output figures and reports will be saved.

    Returns:
        Path to the generated reconstruction dashboard figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = output_dir / "reconstruction_dashboard.png"

    print("=" * 82)
    print(" LensForge — Bayesian Regularized Galaxy Source Reconstruction Pipeline")
    print("=" * 82)

    # 1. Image and Source Plane Grids
    fov_image = 3.6  # arcsec
    n_image = 72     # 72x72 image pixels (0.05 arcsec/pixel)
    image_grid = make_coordinate_grid(num_pix=n_image, field_of_view=fov_image, unit="arcsec")

    fov_source = 1.6  # arcsec
    n_source = 36     # 36x36 source reconstruction pixels
    source_grid = make_coordinate_grid(num_pix=n_source, field_of_view=fov_source, unit="arcsec")

    print("\n[1] Observational & Discretization Setup:")
    print(f"    Image Plane:  {fov_image:.2f}'' x {fov_image:.2f}'' ({n_image}x{n_image} = {n_image**2} pixels)")
    print(f"    Pixel Scale:  {image_grid.pixel_scale:.4f} arcsec/pixel")
    print(f"    Source Plane: {fov_source:.2f}'' x {fov_source:.2f}'' ({n_source}x{n_source} = {n_source**2} pixels)")
    print(f"    Source Scale: {source_grid.pixel_scale:.4f} arcsec/pixel (High-Res Inversion Mesh)")

    # 2. Deflector Model: SIE + External Shear
    lens_center = (0.0, 0.0)
    sie = SingularIsothermalEllipsoid.from_degrees(
        theta_e=1.15,
        axis_ratio=0.75,
        pos_angle_deg=35.0,
        center=lens_center,
    )
    shear = ExternalShear.from_polar_degrees(
        gamma=0.05,
        phi_gamma_deg=20.0,
        center=lens_center,
    )
    lens_model = CompositeLens([sie, shear])

    print("\n[2] Gravitational Deflector Parameters:")
    print(f"    SIE Einstein Radius:    {sie.theta_e:.2f} arcsec")
    print(f"    SIE Axis Ratio (q):     {sie.axis_ratio:.2f}")
    print(f"    SIE Position Angle:     {sie.pos_angle_deg:.1f} deg")
    print(f"    External Tidal Shear:   gamma = {shear.gamma:.3f} at {shear.phi_gamma_deg:.1f} deg")

    # 3. Ground Truth Background Galaxy (Core + 2 Clumps)
    core = GaussianSource.from_degrees(center=(0.14, 0.10), sigma=0.10, amp=12.0, axis_ratio=0.7, pos_angle_deg=40.0)
    clump1 = GaussianSource.from_degrees(center=(0.20, 0.18), sigma=0.04, amp=8.0, axis_ratio=0.9, pos_angle_deg=0.0)
    clump2 = GaussianSource.from_degrees(center=(0.07, 0.04), sigma=0.05, amp=6.0, axis_ratio=0.8, pos_angle_deg=60.0)

    true_source = core.evaluate(source_grid.x, source_grid.y) + \
                  clump1.evaluate(source_grid.x, source_grid.y) + \
                  clump2.evaluate(source_grid.x, source_grid.y)

    # 4. Generate Lensed Surface Brightness
    bx, by = lens_model.ray_trace(image_grid.x, image_grid.y, mask_singular=True)
    ideal_lensed = core.evaluate(bx, by) + clump1.evaluate(bx, by) + clump2.evaluate(bx, by)
    ideal_lensed = np.nan_to_num(ideal_lensed, nan=0.0)

    # 5. Telescope PSF and Noise
    psf = GaussianPSF(fwhm=0.12, pixel_scale=image_grid.pixel_scale, num_pix=15)
    blurred_lensed = psf.convolve(ideal_lensed)

    observed_data, noise_sigma = add_observation_noise(
        image=blurred_lensed,
        read_noise=2.5,
        exposure_time=1500.0,
        sky_background=1.5,
        seed=101,
    )

    print("\n[3] Simulated Telescope Observation:")
    print(f"    Telescope PSF:          FWHM = {psf.fwhm:.3f} arcsec ({psf.sigma_pixels:.2f} pixels)")
    print(f"    Detector Readout Noise: 2.5 e- RMS")
    print(f"    Effective Exposure:     1500 seconds")
    print(f"    Observed Data Peak SNR: {np.max(observed_data) / np.median(noise_sigma):.1f}")

    # 6. Linear Inversion Engine
    print("\n[4] Building Response Operator & Inversion Engine...")
    reconstructor = SourceReconstructor(
        image_grid=image_grid,
        source_grid=source_grid,
        lens_model=lens_model,
        psf=psf,
        regularization_method="curvature",
    )

    # 7. Optimize Regularization Lambda & Solve Linear Inversion
    print("    Searching for Bayesian optimal regularization strength lambda...")
    result = reconstructor.optimize_regularization(
        data_image=observed_data,
        noise_sigma=noise_sigma,
        log_lambda_bounds=(-1.0, 4.0),
    )

    print(f"    Optimal Regularization lambda*: {result.regularization_lambda:.2e}")
    print(f"    Total Chi-Squared:              {result.chi2:.1f}")
    print(f"    Reduced Chi-Squared (chi2/dof): {result.reduced_chi2:.3f}  [Target: ~1.0]")
    print(f"    Bayesian Log-Evidence:          {result.log_evidence:.1f}")

    # 8. Render 6-Panel Scientific Dashboard
    print(f"\n[5] Rendering Scientific Reconstruction Dashboard to {dashboard_path}...")
    plot_reconstruction_dashboard(
        result=result,
        observed_image=observed_data,
        ground_truth_source=true_source,
        output_path=str(dashboard_path),
        title="LensForge Bayesian Gravitational Lens Source Inversion",
    )

    print("\n" + "=" * 82)
    print(" RECONSTRUCTION SUCCESSFUL!")
    print(f" Recovered unlensed galaxy morphology and saved dashboard to: {dashboard_path.resolve()}")
    print("=" * 82)

    return dashboard_path

