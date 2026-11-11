"""Lens mapping matrix construction via backwards ray-tracing and bilinear interpolation.

Constructs the sparse linear operator L mapping unlensed source pixels s to lensed image pixels:
    i_lensed = L * s
"""

from typing import Optional
import numpy as np
from scipy import sparse

from lensforge.physics.coordinates import CoordinateGrid
from lensforge.physics.models.base import BaseLensModel


def build_lens_mapping_matrix(
    image_grid: CoordinateGrid,
    source_grid: CoordinateGrid,
    lens_model: BaseLensModel,
) -> sparse.csr_matrix:
    """Construct the sparse lens mapping matrix L.

    For each image pixel, ray-traces coordinates back to the source plane and assigns
    bilinear interpolation weights to the 4 surrounding source-plane pixels.

    Args:
        image_grid: CoordinateGrid defining image-plane discretization.
        source_grid: CoordinateGrid defining source-plane reconstruction mesh.
        lens_model: Gravitational lens model implementing ray_trace().

    Returns:
        sparse.csr_matrix of shape (N_image_pixels, N_source_pixels).
    """
    n_img_y, n_img_x = image_grid.shape
    n_img_tot = n_img_y * n_img_x

    n_src_y, n_src_x = source_grid.shape
    n_src_tot = n_src_y * n_src_x

    # 1. Backwards ray-trace image pixel coordinates to source plane
    beta_x, beta_y = lens_model.ray_trace(image_grid.x, image_grid.y, mask_singular=True)
    flat_bx = beta_x.ravel()
    flat_by = beta_y.ravel()

    # 2. Source grid geometry
    dx_src = source_grid.pixel_scale
    dy_src = source_grid.pixel_scale

    # Source coordinate axes (1D)
    x_min_center = source_grid.x[0, 0]
    y_min_center = source_grid.y[0, 0]

    # Continuous pixel indices on source grid
    # For a coordinate x, index float = (x - x_min_center) / dx
    u = (flat_bx - x_min_center) / dx_src
    v = (flat_by - y_min_center) / dy_src

    # Lower integer indices
    j0 = np.floor(u).astype(np.int64)
    i0 = np.floor(v).astype(np.int64)

    j1 = j0 + 1
    i1 = i0 + 1

    # Fractional weights
    wx1 = u - j0
    wx0 = 1.0 - wx1
    wy1 = v - i0
    wy0 = 1.0 - wy1

    # Check boundaries: only include rays landing within the source grid
    valid = (
        np.isfinite(flat_bx)
        & np.isfinite(flat_by)
        & (j0 >= 0)
        & (j1 < n_src_x)
        & (i0 >= 0)
        & (i1 < n_src_y)
    )

    valid_img_indices = np.where(valid)[0]
    if len(valid_img_indices) == 0:
        return sparse.csr_matrix((n_img_tot, n_src_tot), dtype=np.float64)

    # Filter arrays to valid pixels
    v_img = valid_img_indices
    v_j0 = j0[v_img]
    v_j1 = j1[v_img]
    v_i0 = i0[v_img]
    v_i1 = i1[v_img]
    v_wx0 = wx0[v_img]
    v_wx1 = wx1[v_img]
    v_wy0 = wy0[v_img]
    v_wy1 = wy1[v_img]

    # Four corners in source grid: (i0, j0), (i0, j1), (i1, j0), (i1, j1)
    k_00 = v_i0 * n_src_x + v_j0
    k_01 = v_i0 * n_src_x + v_j1
    k_10 = v_i1 * n_src_x + v_j0
    k_11 = v_i1 * n_src_x + v_j1

    w_00 = v_wx0 * v_wy0
    w_01 = v_wx1 * v_wy0
    w_10 = v_wx0 * v_wy1
    w_11 = v_wx1 * v_wy1

    # Build CSR coordinate arrays
    rows = np.concatenate([v_img, v_img, v_img, v_img])
    cols = np.concatenate([k_00, k_01, k_10, k_11])
    data = np.concatenate([w_00, w_01, w_10, w_11])

    mapping_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_img_tot, n_src_tot), dtype=np.float64)
    return mapping_matrix

