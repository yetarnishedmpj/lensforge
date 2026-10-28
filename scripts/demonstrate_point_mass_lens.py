#!/usr/bin/env python3
"""LensForge Demonstration Script: Point-Mass Gravitational Lensing.

This script demonstrates the Phase 1 forward modeling pipeline:
1. Initializes physical lens parameters (Mass, cosmological angular-diameter distances).
2. Calculates the Einstein radius in radians and arcseconds.
3. Instantiates an analytic Gaussian source galaxy profile.
4. Generates a regular image-plane angular coordinate grid.
5. Applies backwards ray-tracing using the point-mass lens equation.
6. Samples the intrinsic source surface brightness via Liouville's theorem.
7. Computes analytical image positions and magnification factors for the source centroid.
8. Produces a publication-quality visualization of the source and image planes.

IMPORTANT NOTE:
This represents an idealized theoretical lensing simulation. In this Phase 1 foundation,
neither telescope Point Spread Function (PSF) convolution nor observational noise (N)
is applied. These components will be introduced in subsequent phases of LensForge.
"""

from pathlib import Path
import sys
import numpy as np

# Ensure src/ is on python path if running standalone
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from lensforge.constants import (
    M_SUN,
    MEGAPARSEC,
    arcsec2rad,
    rad2arcsec,
    deg2rad,
)
from lensforge.physics.coordinates import make_coordinate_grid
from lensforge.physics.distances import einstein_radius, einstein_radius_arcsec
from lensforge.physics.models.point_mass import PointMassLens
from lensforge.imaging.source_models import GaussianSource, simulate_lensed_image
from lensforge.visualization.lens_plots import plot_lensing_system


def run_demonstration(output_dir: Path) -> Path:
    """Execute the end-to-end point-mass lensing demonstration.

    Args:
        output_dir: Directory where generated figures should be saved.

    Returns:
        Path to the saved visualization figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "demonstration_point_mass.png"

    print("=" * 78)
    print(" LensForge — Phase 1 Point-Mass Gravitational Lens Demonstration")
    print("=" * 78)

    # 1. Physical astrophysical parameters
    lens_mass_kg = 1.0e11 * M_SUN          # 10^11 Solar masses (typical galaxy bulge)
    d_l_meters = 1000.0 * MEGAPARSEC       # D_l = 1000 Mpc (z_l ~ 0.3)
    d_s_meters = 1800.0 * MEGAPARSEC       # D_s = 1800 Mpc (z_s ~ 1.0)
    d_ls_meters = 1100.0 * MEGAPARSEC      # D_ls = 1100 Mpc (angular diameter distance)

    print("\n[1] Physical Setup:")
    print(f"    - Lens Mass M:                   {lens_mass_kg:.3e} kg  (1.0e11 M_sun)")
    print(f"    - Observer-Lens Distance D_l:    {d_l_meters:.3e} m  (1000.0 Mpc)")
    print(f"    - Observer-Source Distance D_s:  {d_s_meters:.3e} m  (1800.0 Mpc)")
    print(f"    - Lens-Source Distance D_ls:     {d_ls_meters:.3e} m  (1100.0 Mpc)")

    # 2. Einstein Radius Calculation
    theta_e_rad = einstein_radius(lens_mass_kg, d_l_meters, d_s_meters, d_ls_meters)
    theta_e_arcsec = rad2arcsec(theta_e_rad)
    print("\n[2] Einstein Angle Calculation:")
    print(f"    - theta_E (radians):             {theta_e_rad:.6e} rad")
    print(f"    - theta_E (arcseconds):          {theta_e_arcsec:.4f} arcsec")

    # 3. Instantiate PointMassLens model (working in arcseconds for sky visualization)
    lens_center_arcsec = (0.0, 0.0)
    lens = PointMassLens(theta_e=theta_e_arcsec, center=lens_center_arcsec)

    # 4. Background Source Model (Offset Gaussian galaxy)
    source_center_arcsec = (0.28, 0.18)
    source_sigma_arcsec = 0.12
    source_axis_ratio = 0.80
    source_pa_rad = deg2rad(35.0)

    source = GaussianSource(
        center=source_center_arcsec,
        sigma=source_sigma_arcsec,
        amp=1.0,
        axis_ratio=source_axis_ratio,
        pos_angle=source_pa_rad,
    )
    print("\n[3] Background Galaxy Source Model:")
    print(f"    - Profile:                       2D Elliptical Gaussian")
    print(f"    - Centroid (beta_x, beta_y):     ({source_center_arcsec[0]:.2f}, {source_center_arcsec[1]:.2f}) arcsec")
    print(f"    - Size scale (sigma):            {source_sigma_arcsec:.2f} arcsec (FWHM: {source.fwhm:.2f} arcsec)")
    print(f"    - Ellipticity axis ratio q:      {source_axis_ratio:.2f}")

    # 5. Image plane Coordinate Grid
    num_pix = 400
    fov_arcsec = 4.0
    grid = make_coordinate_grid(
        num_pix=num_pix,
        field_of_view=fov_arcsec,
        unit="arcsec",
        center=(0.0, 0.0),
    )
    print("\n[4] Image Plane Discretization:")
    print(f"    - Field of View:                 {fov_arcsec:.1f} x {fov_arcsec:.1f} arcsec")
    print(f"    - Grid Resolution:               {num_pix} x {num_pix} pixels")
    print(f"    - Pixel Scale:                   {grid.pixel_scale:.4f} arcsec/pixel")

    # 6. Ray-tracing and Surface Brightness Sampling (Conservation of surface brightness)
    print("\n[5] Executing Backwards Ray-Tracing...")
    unlensed_image, lensed_image = simulate_lensed_image(lens, source, grid)

    # 7. Analytical image positions and magnifications for source center
    analysis = lens.image_positions_2d(*source_center_arcsec)
    magnifications = lens.image_magnifications_2d(*source_center_arcsec)

    print("\n[6] Analytical Image Positions & Magnifications for Source Centroid:")
    print(f"    - Normalized offset u:           {magnifications['u']:.4f}")
    if not analysis["is_ring"]:
        pos_plus, pos_minus = analysis["images"]
        print(f"    - Outer Image (theta_+):         ({pos_plus[0]:.3f}, {pos_plus[1]:.3f}) arcsec")
        print(f"      * Signed Magnification mu_+:   {magnifications['mu_plus']:+.4f} (parity: +1)")
        print(f"    - Inner Image (theta_-):         ({pos_minus[0]:.3f}, {pos_minus[1]:.3f}) arcsec")
        print(f"      * Signed Magnification mu_-:   {magnifications['mu_minus']:+.4f} (parity: -1, reversed)")
        print(f"    - Total Point-Source Flux Amp:   {magnifications['mu_total']:.4f}")
        print(f"    - Verification (mu_+ + mu_-):    {magnifications['mu_plus'] + magnifications['mu_minus']:.6f} (Identity == 1.0)")
    else:
        print("    - Perfect alignment detected: Einstein ring of radius theta_E.")

    # 8. Visualization
    print("\n[7] Generating Publication-Quality Plot...")
    img_coords = analysis["images"] if not analysis["is_ring"] else None
    plot_lensing_system(
        grid=grid,
        unlensed_image=unlensed_image,
        lensed_image=lensed_image,
        theta_e=theta_e_arcsec,
        lens_center=lens_center_arcsec,
        source_center=source_center_arcsec,
        image_positions=img_coords,
        output_path=str(figure_path),
        title_prefix="Point-Mass Gravitational Lens",
    )
    print(f"    -> Plot saved successfully to: {figure_path}")

    print("\n" + "=" * 78)
    print(" Demonstration completed successfully.")
    print(" NOTE: Observational noise (N) and telescope PSF (P) will be incorporated")
    print(" in subsequent phases of the LensForge forward modeling framework.")
    print("=" * 78)

    return figure_path


if __name__ == "__main__":
    out_dir = project_root / "outputs"
    run_demonstration(out_dir)
