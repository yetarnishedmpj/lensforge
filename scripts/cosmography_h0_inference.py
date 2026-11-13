#!/usr/bin/env python3
"""LensForge Time-Delay Cosmography & Hubble Constant (H0) Inference.

Demonstrates measuring the cosmological Hubble constant H0 via strong gravitational lens
time delays (Refsdal 1964, Suyu et al. 2013, Wong et al. 2020 / H0LiCOW & TDCOSMO).

Cosmic Significance:
-------------------
The "Hubble Tension" between early-universe CMB measurements (Planck: H0 ~ 67.4 km/s/Mpc)
and local distance ladder Cepheids (SH0ES: H0 ~ 73.0 km/s/Mpc) is a 5-sigma discrepancy.
Time-delay cosmography provides an independent geometric probe that directly measures
the time-delay distance D_Delta_t proportional to 1 / H0.

Workflow:
---------
1. Define a quadruply-imaged lensed quasar system under flat Lambda-CDM (True H0 = 70.0 km/s/Mpc).
2. Compute the 2D gravitational Fermat potential surface tau(theta, beta).
3. Find multiple image positions at stationary points (minima and saddle points).
4. Compute the theoretical and observed light travel time delays Delta t_ij in days.
5. Invert for the time-delay distance D_Delta_t and sample the posterior P(H0 | Delta t).
6. Export publication-quality 4-panel figure to outputs/cosmography_h0_inference.png.
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

from lensforge.constants import C_LIGHT, MEGAPARSEC, arcsec2rad
from lensforge.cosmology import (
    FlatLambdaCDMCosmology,
    fermat_potential,
    infer_h0_from_time_delay,
    time_delay_days,
)
from lensforge.physics.models.composite import CompositeLens
from lensforge.physics.models.shear import ExternalShear
from lensforge.physics.models.sie import SingularIsothermalEllipsoid


def run_time_delay_cosmography_benchmark(output_dir: Path) -> Path:
    """Execute complete time-delay cosmography pipeline and H0 posterior inference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = output_dir / "cosmography_h0_inference.png"

    print("=" * 80)
    print(" LENSFORGE TIME-DELAY COSMOGRAPHY: HUBBLE CONSTANT (H0) INFERENCE")
    print("=" * 80)

    # 1. Cosmological Truth & Redshift Framework
    z_lens = 0.295      # Characteristic early-type galaxy deflector
    z_source = 0.657    # Background variable quasar (e.g. RXJ1131-like)
    true_h0 = 70.0      # km/s/Mpc
    omega_m = 0.300
    true_cosmo = FlatLambdaCDMCosmology(h0=true_h0, omega_m=omega_m)

    d_l_m, d_s_m, d_ls_m = true_cosmo.lensing_distances(z_lens, z_source)
    true_d_dt_mpc = true_cosmo.time_delay_distance_mpc(z_lens, z_source)

    print(f"[1/5] Cosmological Setup (True H0 = {true_h0:.1f} km/s/Mpc):")
    print(f"      Deflector Redshift z_l     : {z_lens:.3f}")
    print(f"      Quasar Source Redshift z_s : {z_source:.3f}")
    print(f"      Time-Delay Dist D_Delta_t  : {true_d_dt_mpc:.1f} Mpc")

    # 2. Macro-Lens Deflector & Source Geometry
    theta_e = 1.35  # arcsec
    axis_ratio = 0.85
    pa_deg = 45.0
    gamma = 0.04
    phi_gamma = 30.0

    sie = SingularIsothermalEllipsoid.from_degrees(theta_e=theta_e, axis_ratio=axis_ratio, pos_angle_deg=pa_deg)
    shear = ExternalShear.from_polar_degrees(gamma=gamma, phi_gamma_deg=phi_gamma)
    lens_model = CompositeLens([sie, shear])

    # Source slightly offset from caustic cusp to form distinct multiple images
    beta_x, beta_y = 0.15, 0.10

    # 3. Fermat Potential Surface & Stationary Points (Images)
    print(f"\n[2/5] Evaluating Fermat Potential Surface & Image Positions...")
    grid_fov = 3.6
    num_grid = 120
    x_1d = np.linspace(-grid_fov / 2, grid_fov / 2, num_grid)
    y_1d = np.linspace(-grid_fov / 2, grid_fov / 2, num_grid)
    xx, yy = np.meshgrid(x_1d, y_1d)

    fermat_surface = fermat_potential(xx, yy, beta_x, beta_y, lens_model, angular_unit="arcsec")

    # Approximate 4 multiple image positions for this configuration
    images = [
        ("A", 1.25, 0.75),   # Leading minimum
        ("B", -1.15, 0.82),  # Saddle point
        ("C", -0.85, -1.05), # Trailing minimum
        ("D", 0.95, -0.92),  # Saddle point
    ]

    # Refine image positions to exact roots of lens equation via local optimization
    from scipy.optimize import minimize
    refined_images = []
    for name, gx, gy in images:
        res = minimize(
            lambda pos: float(np.hypot(*lens_model.ray_trace(pos[0], pos[1]))),
            x0=[gx, gy],
            method="Nelder-Mead",
            tol=1e-5,
        )
        rx, ry = res.x
        refined_images.append((name, float(rx), float(ry)))

    # Compute arrival times relative to the leading image A
    img_a = refined_images[0]
    time_delays_true = {}
    print(f"\n[3/5] Theoretical Light Travel Time Delays (Relative to Image A):")
    for name, ix, iy in refined_images:
        dt = time_delay_days(
            theta_x1=ix, theta_y1=iy,
            theta_x2=img_a[1], theta_y2=img_a[2],
            beta_x=beta_x, beta_y=beta_y,
            lens_model=lens_model,
            z_lens=z_lens,
            z_source=z_source,
            cosmology=true_cosmo,
        )
        time_delays_true[name] = dt
        print(f"      Image {name} at ({ix:+.3f}'', {iy:+.3f}''): Delta t = {dt:+.2f} days")

    # 4. Simulated Observation with Measurement Uncertainty
    # Typical monitoring campaign (COSMOGRAIL) achieves ~ 1.5 - 2.5 day precision
    np.random.seed(42)
    dt_obs = time_delays_true["B"] + np.random.normal(0.0, 1.8)
    sigma_dt = 2.0  # days uncertainty

    print(f"\n[4/5] Simulated Observational Measurement:")
    print(f"      Observed Delta t_AB        : {dt_obs:.2f} +/- {sigma_dt:.2f} days")

    # Invert for H0 directly
    h0_inferred = infer_h0_from_time_delay(
        delta_t_days=dt_obs,
        theta_x1=refined_images[1][1], theta_y1=refined_images[1][2],
        theta_x2=refined_images[0][1], theta_y2=refined_images[0][2],
        beta_x=beta_x, beta_y=beta_y,
        lens_model=lens_model,
        z_lens=z_lens,
        z_source=z_source,
        cosmology=true_cosmo,
    )
    print(f"      Directly Inferred H0       : {h0_inferred:.2f} km/s/Mpc (True: {true_h0:.1f})")

    # 5. Bayesian Posterior Sampling for H0
    h0_grid = np.linspace(55.0, 85.0, 300)
    # Forward predicted delay as a function of H0: dt(H0) = dt_true * (true_H0 / H0)
    dt_pred = time_delays_true["B"] * (true_h0 / h0_grid)
    chi2_h0 = ((dt_obs - dt_pred) / sigma_dt) ** 2
    posterior_h0 = np.exp(-0.5 * (chi2_h0 - np.min(chi2_h0)))
    posterior_h0 /= np.trapezoid(posterior_h0, h0_grid)

    # Compute 16th, 50th, 84th percentiles
    cum_post = np.cumsum(posterior_h0) / np.sum(posterior_h0)
    h0_16 = float(np.interp(0.16, cum_post, h0_grid))
    h0_50 = float(np.interp(0.50, cum_post, h0_grid))
    h0_84 = float(np.interp(0.84, cum_post, h0_grid))

    print(f"\n[5/5] Inferred Hubble Constant Posterior Distribution:")
    print(f"      H0 = {h0_50:.2f} (+{h0_84 - h0_50:.2f} / -{h0_50 - h0_16:.2f}) km/s/Mpc")

    # 6. Generate 4-Panel Publication-Quality Figure
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)

    # Panel 1: Fermat Potential Surface & Stationary Points
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(xx, yy, fermat_surface, levels=30, cmap="viridis")
    ax1.contour(xx, yy, fermat_surface, levels=20, colors="white", alpha=0.3, linewidths=0.6)
    # Plot image positions
    colors = ["cyan", "magenta", "orange", "yellow"]
    for idx, (name, ix, iy) in enumerate(refined_images):
        ax1.scatter(ix, iy, color=colors[idx], s=90, edgecolors="black", zorder=5, label=f"Image {name}")
        ax1.text(ix + 0.08, iy + 0.08, name, color="white", fontsize=11, fontweight="bold")
    ax1.scatter(beta_x, beta_y, color="red", marker="*", s=140, edgecolors="black", zorder=6, label="Quasar Source")
    ax1.set_title("A. Gravitational Fermat Potential Surface $\\tau(\\theta, \\beta)$\n(Images Form at Stationary Points $\\nabla \\tau = 0$)", fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"$\theta_x$ [arcsec]", fontsize=10)
    ax1.set_ylabel(r"$\theta_y$ [arcsec]", fontsize=10)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label=r"$\tau$ [rad$^2$]")
    ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: Simulated Quasar Light Curves Showing Time Delays
    ax2 = fig.add_subplot(gs[0, 1])
    t_days = np.linspace(0, 180, 200)
    # Intrinsic stochastic variability: damped random walk + flare
    intrinsic_curve = 1.0 + 0.3 * np.sin(2 * np.pi * t_days / 60.0) + 0.5 * np.exp(-((t_days - 70.0) / 12.0) ** 2)
    
    # Plot light curves shifted by delays
    ax2.plot(t_days, intrinsic_curve, color="cyan", lw=2.0, label="Image A (Reference)")
    ax2.plot(t_days + time_delays_true["B"], intrinsic_curve * 0.75, color="magenta", lw=2.0, ls="--", label=f"Image B (Delay = {time_delays_true['B']:+.1f}d)")
    ax2.plot(t_days + time_delays_true["C"], intrinsic_curve * 0.60, color="orange", lw=2.0, ls="-.", label=f"Image C (Delay = {time_delays_true['C']:+.1f}d)")
    ax2.set_xlim(0, 180)
    ax2.set_title("B. Simulated Variable Quasar Light Curves\n(Physical Delays from Distinct Geodesic Paths)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Observation Time [days]", fontsize=10)
    ax2.set_ylabel("Normalized Quasar Flux", fontsize=10)
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    # Panel 3: Time-Delay Distance D_Delta_t vs H0 Relation
    ax3 = fig.add_subplot(gs[1, 0])
    h0_sweep = np.linspace(50.0, 90.0, 100)
    d_dt_sweep = [true_cosmo.time_delay_distance_mpc(z_lens, z_source) * (true_h0 / h) for h in h0_sweep]
    ax3.plot(h0_sweep, d_dt_sweep, color="navy", lw=2.5, label=r"$D_{\Delta t} \propto 1 / H_0$")
    ax3.axhline(true_d_dt_mpc, color="green", ls="--", label=f"True $D_{{\\Delta t}} = {true_d_dt_mpc:.1f}$ Mpc")
    ax3.scatter([true_h0], [true_d_dt_mpc], color="red", s=80, zorder=5, label=f"Truth ($H_0 = {true_h0}$)")
    ax3.set_title("C. Cosmological Time-Delay Distance Scaling\n(Pure Geometric Probe Independent of Standard Candles)", fontsize=11, fontweight="bold")
    ax3.set_xlabel(r"Hubble Constant $H_0$ [km/s/Mpc]", fontsize=10)
    ax3.set_ylabel(r"Time-Delay Distance $D_{\Delta t}$ [Mpc]", fontsize=10)
    ax3.grid(True, ls="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8)

    # Panel 4: Inferred H0 Posterior Distribution & Hubble Tension Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(h0_grid, posterior_h0, color="purple", lw=2.5, label="LensForge Posterior $P(H_0 | \Delta t)$")
    ax4.fill_between(
        h0_grid,
        posterior_h0,
        where=(h0_grid >= h0_16) & (h0_grid <= h0_84),
        color="purple",
        alpha=0.25,
        label=f"$1\\sigma$ Interval: ${h0_50:.1f}^{{+{h0_84-h0_50:.1f}}}_{{-{h0_50-h0_16:.1f}}}$",
    )
    # Benchmark lines
    ax4.axvline(67.4, color="blue", ls=":", lw=2.0, label="Planck CMB ($67.4 \pm 0.5$)")
    ax4.axvline(73.0, color="crimson", ls=":", lw=2.0, label="SH0ES Cepheids ($73.0 \pm 1.0$)")
    ax4.axvline(true_h0, color="black", ls="-", lw=1.5, label=f"True Truth ($H_0 = {true_h0:.1f}$)")
    ax4.set_title("D. Posterior Probability Distribution $P(H_0 | \\mathbf{d})$\n(Direct Constraint on the Cosmic Hubble Tension)", fontsize=11, fontweight="bold")
    ax4.set_xlabel(r"Hubble Constant $H_0$ [km/s/Mpc]", fontsize=10)
    ax4.set_ylabel("Posterior Probability Density", fontsize=10)
    ax4.set_xlim(58, 82)
    ax4.grid(True, ls="--", alpha=0.5)
    ax4.legend(loc="upper right", fontsize=8)

    fig.savefig(dashboard_path, dpi=300)
    plt.close(fig)

    print("=" * 80)
    print(f"[SUCCESS] Time-Delay Cosmography Complete!")
    print(f"          Cosmography Dashboard: {dashboard_path.resolve()}")
    print("=" * 80)
    return dashboard_path


if __name__ == "__main__":
    out_dir = project_root / "outputs"
    run_time_delay_cosmography_benchmark(out_dir)
