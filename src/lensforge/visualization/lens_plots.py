"""Scientific visualization utilities for gravitational lensing.

Functions in this module generate publication-quality figures illustrating
source and image planes, Einstein rings, analytical image positions, and lens potentials.
Physics calculations are strictly decoupled from plotting routines.
"""

from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from lensforge.physics.coordinates import CoordinateGrid


def plot_lensing_system(
    grid: CoordinateGrid,
    unlensed_image: np.ndarray,
    lensed_image: np.ndarray,
    theta_e: float,
    lens_center: Tuple[float, float] = (0.0, 0.0),
    source_center: Optional[Tuple[float, float]] = None,
    image_positions: Optional[List[Tuple[float, float]]] = None,
    output_path: Optional[str] = None,
    title_prefix: str = "Point-Mass Lens",
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot source plane, lensed image plane, and geometric critical structures.

    Args:
        grid: CoordinateGrid instance providing spatial extent and units.
        unlensed_image: 2D array of intrinsic source surface brightness.
        lensed_image: 2D array of ray-traced lensed surface brightness.
        theta_e: Einstein radius in units matching the grid.
        lens_center: (x, y) coordinates of the lens center.
        source_center: (x, y) coordinates of the source centroid (optional).
        image_positions: List of (x, y) analytical image positions (optional).
        output_path: If specified, save figure to this file path.
        title_prefix: Title prefix string for the figure.

    Returns:
        Tuple of (fig, axes).
    """
    unit_str = grid.unit
    extent = grid.extent

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    # Panel 1: Intrinsic Source Plane
    im0 = axes[0].imshow(
        unlensed_image,
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    axes[0].set_title(f"Source Plane: Intrinsic Surface Brightness $S(\\beta)$", fontsize=11, fontweight="bold")
    axes[0].set_xlabel(f"$\\beta_x$ [{unit_str}]", fontsize=11)
    axes[0].set_ylabel(f"$\\beta_y$ [{unit_str}]", fontsize=11)
    axes[0].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    axes[0].axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

    if source_center is not None:
        axes[0].plot(
            source_center[0],
            source_center[1],
            "c+",
            markersize=10,
            markeredgewidth=2,
            label=f"Source Center ({source_center[0]:.2f}, {source_center[1]:.2f})",
        )
        axes[0].legend(loc="upper right", framealpha=0.8, fontsize=9)

    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Relative Surface Brightness", fontsize=10)

    # Panel 2: Observed Image Plane
    im1 = axes[1].imshow(
        lensed_image,
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    axes[1].set_title(f"Image Plane: Ideal Gravitational Lens $I(\\theta)$", fontsize=11, fontweight="bold")
    axes[1].set_xlabel(f"$\\theta_x$ [{unit_str}]", fontsize=11)
    axes[1].set_ylabel(f"$\\theta_y$ [{unit_str}]", fontsize=11)
    axes[1].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    axes[1].axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

    # Overlay Lens Center
    axes[1].plot(
        lens_center[0],
        lens_center[1],
        "yx",
        markersize=9,
        markeredgewidth=2,
        label=f"Lens Center ({lens_center[0]:.2f}, {lens_center[1]:.2f})",
    )

    # Overlay Einstein Ring
    ring = patches.Circle(
        lens_center,
        radius=theta_e,
        color="cyan",
        fill=False,
        linestyle="--",
        linewidth=1.5,
        label=f"Einstein Ring ($\\theta_E = {theta_e:.2f}\\,${unit_str})",
    )
    axes[1].add_patch(ring)

    # Overlay Analytical Point Image Positions
    if image_positions:
        labels = ["Outer Image ($\\theta_+$)", "Inner Image ($\\theta_-$)"]
        colors = ["lime", "orange"]
        for idx, (img_x, img_y) in enumerate(image_positions):
            lbl = labels[idx] if idx < len(labels) else f"Image {idx+1}"
            clr = colors[idx] if idx < len(colors) else "white"
            axes[1].plot(
                img_x,
                img_y,
                "o",
                color=clr,
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=1.2,
                label=f"{lbl}: ({img_x:.2f}, {img_y:.2f})",
            )

    axes[1].legend(loc="upper right", framealpha=0.8, fontsize=8.5)
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Relative Surface Brightness", fontsize=10)

    fig.suptitle(
        f"{title_prefix} — Idealized Ray-Traced Lensing Model (Phase 1)",
        fontsize=13,
        fontweight="bold",
    )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_deflection_and_potential(
    grid: CoordinateGrid,
    potential: np.ndarray,
    alpha_x: np.ndarray,
    alpha_y: np.ndarray,
    theta_e: float,
    subsample: int = 16,
    output_path: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot lens potential contours and deflection angle vector field.

    Args:
        grid: CoordinateGrid instance.
        potential: 2D array of lensing potential.
        alpha_x: 2D array of horizontal deflection angles.
        alpha_y: 2D array of vertical deflection angles.
        theta_e: Einstein radius.
        subsample: Spatial subsampling interval for vector quiver arrows.
        output_path: File path to save figure (optional).

    Returns:
        Tuple of (fig, axes).
    """
    unit_str = grid.unit
    extent = grid.extent

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    # Potential contour heatmap
    cs = ax.contour(
        grid.x,
        grid.y,
        potential,
        levels=20,
        cmap="viridis",
        linewidths=1.2,
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    # Subsampled quiver vectors for deflection field
    step = subsample
    sub_x = grid.x[::step, ::step]
    sub_y = grid.y[::step, ::step]
    sub_ax = alpha_x[::step, ::step]
    sub_ay = alpha_y[::step, ::step]

    # Mask any singular NaNs
    valid = np.isfinite(sub_ax) & np.isfinite(sub_ay)
    ax.quiver(
        sub_x[valid],
        sub_y[valid],
        sub_ax[valid],
        sub_ay[valid],
        color="crimson",
        angles="xy",
        scale_units="xy",
        scale=2.0,
        width=0.003,
        label="Deflection $\\alpha(\\theta)$",
    )

    # Einstein ring
    ring = patches.Circle(
        (0.0, 0.0),
        radius=theta_e,
        color="cyan",
        fill=False,
        linestyle="--",
        linewidth=1.8,
        label=f"Einstein Ring ($\\theta_E = {theta_e:.2f}\\,${unit_str})",
    )
    ax.add_patch(ring)

    ax.set_title("Lensing Potential $\\psi(\\theta)$ & Deflection Field $\\alpha(\\theta)$", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"$\\theta_x$ [{unit_str}]", fontsize=11)
    ax.set_ylabel(f"$\\theta_y$ [{unit_str}]", fontsize=11)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.legend(loc="upper right", framealpha=0.8)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, np.array([ax])


def plot_galaxy_lensing_system(
    grid: CoordinateGrid,
    unlensed_image: np.ndarray,
    lensed_image: np.ndarray,
    convergence_map: Optional[np.ndarray] = None,
    lens_center: Tuple[float, float] = (0.0, 0.0),
    source_center: Optional[Tuple[float, float]] = None,
    output_path: Optional[str] = None,
    title: str = "Singular Isothermal Ellipsoid (SIE) Galaxy Lens",
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot a 3-panel diagnostic visualization for a realistic galaxy lens system.

    Panels:
    1. Intrinsic Source Plane (unlensed galaxy surface brightness).
    2. Image Plane (lensed arcs, multiple images, and lens position).
    3. Mass Model Convergence kappa (projected surface mass distribution).

    Args:
        grid: CoordinateGrid instance.
        unlensed_image: 2D array of intrinsic source surface brightness.
        lensed_image: 2D array of ray-traced lensed image.
        convergence_map: Optional 2D array of convergence kappa.
        lens_center: (x, y) coordinates of the lens centroid.
        source_center: (x, y) coordinates of the background source centroid.
        output_path: Optional file path to save the generated figure.
        title: Main title for the figure.

    Returns:
        Tuple of (fig, axes).
    """
    unit_str = grid.unit
    extent = grid.extent

    num_panels = 3 if convergence_map is not None else 2
    fig, axes = plt.subplots(1, num_panels, figsize=(5.5 * num_panels, 5.0), constrained_layout=True)
    if num_panels == 2:
        axes = np.array(axes)

    # Panel 1: Intrinsic Source Plane
    im0 = axes[0].imshow(unlensed_image, origin="lower", extent=extent, cmap="magma")
    axes[0].set_title("Source Plane $S(\\beta)$", fontsize=11, fontweight="bold")
    axes[0].set_xlabel(f"$\\beta_x$ [{unit_str}]", fontsize=10)
    axes[0].set_ylabel(f"$\\beta_y$ [{unit_str}]", fontsize=10)
    axes[0].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    axes[0].axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    if source_center is not None:
        axes[0].plot(
            source_center[0], source_center[1],
            "c+", markersize=9, markeredgewidth=2,
            label=f"Source ({source_center[0]:.2f}, {source_center[1]:.2f})",
        )
        axes[0].legend(loc="upper right", fontsize=8.5, framealpha=0.8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).set_label("Surface Brightness", fontsize=9)

    # Panel 2: Observed Image Plane
    im1 = axes[1].imshow(lensed_image, origin="lower", extent=extent, cmap="magma")
    axes[1].set_title("Lensed Image $I(\\theta)$", fontsize=11, fontweight="bold")
    axes[1].set_xlabel(f"$\\theta_x$ [{unit_str}]", fontsize=10)
    axes[1].set_ylabel(f"$\\theta_y$ [{unit_str}]", fontsize=10)
    axes[1].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    axes[1].axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    axes[1].plot(
        lens_center[0], lens_center[1],
        "yx", markersize=9, markeredgewidth=2,
        label=f"Lens ({lens_center[0]:.2f}, {lens_center[1]:.2f})",
    )
    axes[1].legend(loc="upper right", fontsize=8.5, framealpha=0.8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).set_label("Observed Flux", fontsize=9)

    # Panel 3: Convergence kappa (if provided)
    if convergence_map is not None:
        # Clip kappa to avoid singularity spike washing out the color scale
        vmax = float(np.nanpercentile(convergence_map, 98))
        im2 = axes[2].imshow(
            convergence_map,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmax=max(vmax, 2.0),
        )
        axes[2].set_title("Mass Density $\\kappa(\\theta)$", fontsize=11, fontweight="bold")
        axes[2].set_xlabel(f"$\\theta_x$ [{unit_str}]", fontsize=10)
        axes[2].set_ylabel(f"$\\theta_y$ [{unit_str}]", fontsize=10)
        axes[2].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        axes[2].axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04).set_label("Convergence $\\kappa$", fontsize=9)

    fig.suptitle(f"{title} (LensForge Phase 2)", fontsize=13, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes
