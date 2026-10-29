"""Regularization matrices for linear galaxy source reconstruction.

Provides spatial prior operators:
- Zeroth-order: Identity norm (penalizes total flux amplitude).
- First-order: Gradient / Dirichlet energy (penalizes sharp slopes / steps).
- Second-order: Curvature / Laplacian (penalizes curvature / oscillations, produces smooth galaxies).
"""

from typing import Literal
import numpy as np
from scipy import sparse


def build_regularization_matrix(
    num_x: int,
    num_y: int,
    method: Literal["zeroth", "gradient", "curvature"] = "curvature",
) -> np.ndarray:
    """Build the N_s x N_s quadratic regularization matrix R.

    Args:
        num_x: Number of source grid columns.
        num_y: Number of source grid rows.
        method: 'zeroth' (identity), 'gradient' (1st derivative), or 'curvature' (2nd derivative Laplacian).

    Returns:
        Dense symmetric positive semi-definite (or positive definite) matrix of shape (num_pix, num_pix).

    Raises:
        ValueError: If invalid method or dimensions <= 0.
    """
    if num_x <= 0 or num_y <= 0:
        raise ValueError(f"Dimensions must be positive integers, got num_x={num_x}, num_y={num_y}.")

    n_tot = num_x * num_y

    if method == "zeroth":
        return np.eye(n_tot, dtype=np.float64)

    elif method == "gradient":
        # 1st order finite difference in x and y directions
        dx_rows, dx_cols, dx_data = [], [], []
        dy_rows, dy_cols, dy_data = [], [], []

        # x-gradients: s(i, j+1) - s(i, j)
        idx_diff = 0
        for i in range(num_y):
            for j in range(num_x - 1):
                idx0 = i * num_x + j
                idx1 = i * num_x + (j + 1)
                dx_rows.extend([idx_diff, idx_diff])
                dx_cols.extend([idx0, idx1])
                dx_data.extend([-1.0, 1.0])
                idx_diff += 1
        d_x = sparse.csr_matrix((dx_data, (dx_rows, dx_cols)), shape=(idx_diff, n_tot))

        # y-gradients: s(i+1, j) - s(i, j)
        idx_diff_y = 0
        for i in range(num_y - 1):
            for j in range(num_x):
                idx0 = i * num_x + j
                idx1 = (i + 1) * num_x + j
                dy_rows.extend([idx_diff_y, idx_diff_y])
                dy_cols.extend([idx0, idx1])
                dy_data.extend([-1.0, 1.0])
                idx_diff_y += 1
        d_y = sparse.csr_matrix((dy_data, (dy_rows, dy_cols)), shape=(idx_diff_y, n_tot))

        r_matrix = (d_x.T @ d_x) + (d_y.T @ d_y)
        # Add tiny epsilon to diagonal to ensure strict positive-definiteness
        r_matrix += sparse.eye(n_tot) * 1e-6
        return r_matrix.toarray()

    elif method == "curvature":
        # 2D Discrete Laplacian stencil:
        # [ 0,  1,  0]
        # [ 1, -4,  1]
        # [ 0,  1,  0]
        lap_rows, lap_cols, lap_data = [], [], []
        row_idx = 0
        for i in range(num_y):
            for j in range(num_x):
                center_idx = i * num_x + j
                weight_center = 0.0

                # Top neighbor (i - 1, j)
                if i > 0:
                    lap_rows.append(row_idx)
                    lap_cols.append((i - 1) * num_x + j)
                    lap_data.append(1.0)
                    weight_center -= 1.0
                # Bottom neighbor (i + 1, j)
                if i < num_y - 1:
                    lap_rows.append(row_idx)
                    lap_cols.append((i + 1) * num_x + j)
                    lap_data.append(1.0)
                    weight_center -= 1.0
                # Left neighbor (i, j - 1)
                if j > 0:
                    lap_rows.append(row_idx)
                    lap_cols.append(i * num_x + (j - 1))
                    lap_data.append(1.0)
                    weight_center -= 1.0
                # Right neighbor (i, j + 1)
                if j < num_x - 1:
                    lap_rows.append(row_idx)
                    lap_cols.append(i * num_x + (j + 1))
                    lap_data.append(1.0)
                    weight_center -= 1.0

                # Center weight
                lap_rows.append(row_idx)
                lap_cols.append(center_idx)
                lap_data.append(weight_center)
                row_idx += 1

        lap = sparse.csr_matrix((lap_data, (lap_rows, lap_cols)), shape=(n_tot, n_tot))
        r_matrix = lap.T @ lap
        # Add slight ridge regularization to diagonal to ensure full rank
        r_matrix += sparse.eye(n_tot) * 1e-6
        return r_matrix.toarray()

    else:
        raise ValueError(
            f"Unknown regularization method '{method}'. Must be 'zeroth', 'gradient', or 'curvature'."
        )
