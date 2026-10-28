#!/usr/bin/env python3
"""LensForge Demonstration Script: Realistic Galaxy Gravitational Lensing (SIE + Shear).

This script demonstrates the Phase 2 forward modeling pipeline:
1. Configures a realistic Singular Isothermal Ellipsoid (SIE) lens model (q=0.72, PA=45 deg).
2. Adds environmental tidal external shear (gamma=0.06, PA=20 deg) using CompositeLens.
3. Instantiates an analytic background galaxy source (Gaussian profile with intrinsic ellipticity).
4. Generates an angular coordinate grid spanning 4.5 x 4.5 arcsec.
5. Performs backwards ray-tracing using the composite deflection field:
       beta = theta - alpha_SIE(theta) - alpha_shear(theta)
6. Samples the intrinsic surface brightness under Liouville's theorem.
7. Computes the projected mass density (convergence kappa).
8. Produces a 3-panel scientific visualization showing source plane, lensed image with arcs,
   and convergence kappa.

NOTE:
This represents an idealized continuous surface-brightness simulation (Phase 2).
Telescope optics (PSF convolution) and detector noise (Poisson shot noise & read noise)
will be introduced in Phase 3.
"""

from pathlib import Path
import sys
import numpy as np

# Ensure src/ is on python path if running standalone
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from lensforge.physics.coordinates import make_coordinate_grid
from lensforge.physics.models.composite import CompositeLens
from lensforge.physics.models.shear import ExternalShear
from lensforge.physics.models.sie import SingularIsothermalEllipsoid
from lensforge.imaging.source_models import GaussianSource, simulate_lensed_image
from lensforge.visualization.lens_plots import plot_galaxy_lensing_system


def run_galaxy_demonstration(output_dir: Path) -> Path:
    """Execute the end-to-end SIE + Shear galaxy lensing demonstration.

    Args:
        output_dir: Directory where generated figures should be saved.

    Returns:
        Path to the saved visualization figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "demonstration_galaxy_sie.png"

    print("=" * 80)
    print(" LensForge — Phase 2 Galaxy Gravitational Lens Demonstration (SIE + Shear)")
    print("=" * 80)

    # 1. Lens Model Parameters
    theta_e_arcsec = 1.35
    axis_ratio = 0.72
    pos_angle_deg = 45.0
    lens_center = (0.0, 0.0)

    gamma = 0.06
    phi_gamma_deg = 20.0

    print("\n[1] Configuring Lens Model:")
    print(f"    SIE Einstein Radius: {theta_e_arcsec:.2f} arcsec")
    print(f"    SIE Axis Ratio (q): {axis_ratio:.2f}")
    print(f"    SIE Position Angle: {pos_angle_deg:.1f} deg")
    print(f"    External Shear Gamma: {gamma:.3f} at {phi_gamma_deg:.1f} deg")

    sie = SingularIsothermalEllipsoid.from_degrees(
        theta_e=theta_e_arcsec,
        axis_ratio=axis_ratio,
        pos_angle_deg=pos_angle_deg,
        center=lens_center,
    )
    shear = ExternalShear.from_polar_degrees(
        gamma=gamma,
        phi_gamma_deg=phi_gamma_deg,
        center=lens_center,
    )
    composite_lens = CompositeLens([sie, shear])

    # 2. Source Galaxy Parameters
    src_center = (0.18, 0.12)
    src_sigma = 0.10
    src_q = 0.80
    src_pa = 30.0

    print("\n[2] Configuring Source Galaxy:")
    print(f"    Centroid: ({src_center[0]:.2f}, {src_center[1]:.2f}) arcsec")
    print(f"    Characteristic Size (sigma): {src_sigma:.2f} arcsec")
    print(f"    Intrinsic Axis Ratio: {src_q:.2f}, PA: {src_pa:.1f} deg")

    source = GaussianSource.from_degrees(
        center=src_center,
        sigma=src_sigma,
        amp=1.0,
        axis_ratio=src_q,
        pos_angle_deg=src_pa,
    )

    # 3. Coordinate Grid
    num_pix = 450
    fov_arcsec = 4.5
    print("\n[3] Generating Image-Plane Grid:")
    print(f"    Field of View: {fov_arcsec:.1f} arcsec, Resolution: {num_pix}x{num_pix} pixels")
    print(f"    Pixel Scale: {fov_arcsec / num_pix:.4f} arcsec/pixel")

    grid = make_coordinate_grid(
        num_pix=num_pix,
        field_of_view=fov_arcsec,
        unit="arcsec",
    )

    # 4. Forward Simulation via Backwards Ray-Tracing
    print("\n[4] Executing Backwards Ray-Tracing...")
    unlensed_img = source.evaluate(grid.x, grid.y)
    beta_x, beta_y = composite_lens.ray_trace(grid.x, grid.y, mask_singular=True)
    lensed_img = source.evaluate(beta_x, beta_y)
    # Mask central singularity
    lensed_img = np.nan_to_num(lensed_img, nan=0.0)

    # Compute convergence kappa
    kappa_map = sie.convergence_analytic(grid.x, grid.y, mask_singular=True)

    # 5. Render Publication-Quality Visualization
    print(f"\n[5] Rendering Diagnostic Figure to {figure_path}...")
    plot_galaxy_lensing_system(
        grid=grid,
        unlensed_image=unlensed_img,
        lensed_image=lensed_img,
        convergence_map=kappa_map,
        lens_center=lens_center,
        source_center=src_center,
        output_path=str(figure_path),
        title="Singular Isothermal Ellipsoid (SIE) + External Shear",
    )

    print("\n" + "=" * 80)
    print(" Demonstration Complete! Lensed arcs successfully simulated and saved.")
    print(f" Figure written to: {figure_path.resolve()}")
    print("=" * 80)
    return figure_path


if __name__ == "__main__":
    out_dir = project_root / "outputs"
    run_galaxy_demonstration(out_dir)
