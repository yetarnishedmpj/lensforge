"""Regularized Semilinear Inversion (Warren & Dye / Treu & Koopmans method).

Solves the inverse problem in strong gravitational lensing:
    d = F * s + n
    where F = B * L (PSF blurring B * lens deflection mapping L).

Given observed data d and noise covariance C_D, the maximum a posteriori (MAP)
reconstructed unlensed source galaxy s_hat satisfies:
    ( F^T * C_D^-1 * F + lambda * R ) * s_hat = F^T * C_D^-1 * d
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union
import numpy as np
from scipy import linalg, optimize, sparse

from lensforge.instrument.psf import BasePSF
from lensforge.inversion.mapping import build_lens_mapping_matrix
from lensforge.inversion.regularization import build_regularization_matrix
from lensforge.physics.coordinates import CoordinateGrid
from lensforge.physics.models.base import BaseLensModel


@dataclass
class ReconstructionResult:
    """Encapsulates the complete scientific results of a source reconstruction.

    Attributes:
        reconstructed_source: 2D array of recovered unlensed galaxy surface brightness.
        model_image: 2D array of best-fit lensed/blurred model prediction in the image plane.
        residuals: 2D array of normalized residuals (data - model) / noise_sigma.
        chi2: Total chi-squared statistic.
        reduced_chi2: Reduced chi-squared (chi^2 / degrees_of_freedom).
        log_evidence: Bayesian log-evidence ln P(d | lambda, eta).
        regularization_lambda: Regularization strength lambda used.
        source_grid: CoordinateGrid of the reconstructed source plane.
        image_grid: CoordinateGrid of the image plane.
    """

    reconstructed_source: np.ndarray
    model_image: np.ndarray
    residuals: np.ndarray
    chi2: float
    reduced_chi2: float
    log_evidence: float
    regularization_lambda: float
    source_grid: CoordinateGrid
    image_grid: CoordinateGrid


class SourceReconstructor:
    """Linear inversion engine for regularized galaxy source reconstruction.

    Implements the Warren & Dye (2003) and Suyu et al. (2006) formulation.
    """

    def __init__(
        self,
        image_grid: CoordinateGrid,
        source_grid: CoordinateGrid,
        lens_model: BaseLensModel,
        psf: Optional[BasePSF] = None,
        regularization_method: Literal["curvature", "gradient", "zeroth"] = "curvature",
    ) -> None:
        """Initialize the reconstructor and precompute linear operators.

        Args:
            image_grid: CoordinateGrid defining image-plane pixels.
            source_grid: CoordinateGrid defining source-plane reconstruction pixels.
            lens_model: Gravitational lens model.
            psf: Optional telescope Point Spread Function model.
            regularization_method: Spatial prior ('curvature', 'gradient', 'zeroth').
        """
        self.image_grid = image_grid
        self.source_grid = source_grid
        self.lens_model = lens_model
        self.psf = psf
        self.regularization_method = regularization_method

        # 1. Build lens mapping matrix L (sparse N_image x N_source)
        self.mapping_matrix = build_lens_mapping_matrix(
            image_grid=self.image_grid,
            source_grid=self.source_grid,
            lens_model=self.lens_model,
        )

        # 2. Build response operator F = B * L
        self.f_matrix = self._build_response_matrix()

        # 3. Build regularization prior matrix R (N_source x N_source)
        n_sy, n_sx = self.source_grid.shape
        self.r_matrix = build_regularization_matrix(
            num_x=n_sx,
            num_y=n_sy,
            method=self.regularization_method,
        )

    def _build_response_matrix(self) -> np.ndarray:
        """Construct the dense response operator F = B * L."""
        l_dense = self.mapping_matrix.toarray()
        if self.psf is None:
            return l_dense

        # Convolve each source basis column with telescope PSF
        n_img_y, n_img_x = self.image_grid.shape
        n_src_tot = self.source_grid.shape[0] * self.source_grid.shape[1]
        f_dense = np.zeros_like(l_dense)

        for col_idx in range(n_src_tot):
            col_l = l_dense[:, col_idx]
            if np.any(col_l > 0):
                img_2d = col_l.reshape((n_img_y, n_img_x))
                conv_2d = self.psf.convolve(img_2d)
                f_dense[:, col_idx] = conv_2d.ravel()

        return f_dense

    def reconstruct(
        self,
        data_image: np.ndarray,
        noise_sigma: Union[float, np.ndarray],
        regularization_lambda: float = 10.0,
    ) -> ReconstructionResult:
        """Execute linear regularized inversion for given data and regularization lambda.

        Solves: (F^T * C_D^-1 * F + lambda * R) * s = F^T * C_D^-1 * d

        Args:
            data_image: 2D observed image array.
            noise_sigma: 1-sigma uncertainty (scalar or 2D array matching data_image).
            regularization_lambda: Regularization strength lambda (> 0).

        Returns:
            ReconstructionResult instance containing recovered source and diagnostics.
        """
        if regularization_lambda <= 0.0:
            raise ValueError(f"Regularization lambda must be positive, got {regularization_lambda}.")

        d_flat = np.asarray(data_image, dtype=np.float64).ravel()
        if np.isscalar(noise_sigma):
            inv_sigma = np.full_like(d_flat, 1.0 / float(noise_sigma))
        else:
            inv_sigma = 1.0 / np.asarray(noise_sigma, dtype=np.float64).ravel()

        # Weight response matrix by 1 / sigma
        # F_w = C_D^(-1/2) * F,  d_w = C_D^(-1/2) * d
        f_w = self.f_matrix * inv_sigma[:, np.newaxis]
        d_w = d_flat * inv_sigma

        # Curvature matrix M = F^T C_D^-1 F + lambda * R
        ft_cd_f = f_w.T @ f_w
        m_matrix = ft_cd_f + regularization_lambda * self.r_matrix

        # Right-hand vector v = F^T C_D^-1 d
        v_vector = f_w.T @ d_w

        # Solve symmetric positive-definite linear system
        try:
            c, lower = linalg.cho_factor(m_matrix, lower=True)
            s_hat = linalg.cho_solve((c, lower), v_vector)
            # log det(M) = 2 * sum(ln(diag(c)))
            log_det_m = 2.0 * np.sum(np.log(np.diag(c)))
        except linalg.LinAlgError:
            # Fallback to standard solver
            s_hat = linalg.solve(m_matrix, v_vector, assume_a="sym")
            sign, log_det_m = np.linalg.slogdet(m_matrix)

        # Reshape reconstructed source to 2D
        n_sy, n_sx = self.source_grid.shape
        rec_source_2d = s_hat.reshape((n_sy, n_sx))

        # Forward model prediction: d_model = F * s_hat
        model_flat = self.f_matrix @ s_hat
        n_iy, n_ix = self.image_grid.shape
        model_image_2d = model_flat.reshape((n_iy, n_ix))

        # Residuals and Chi-squared
        diff = d_flat - model_flat
        residuals_flat = diff * inv_sigma
        residuals_2d = residuals_flat.reshape((n_iy, n_ix))
        chi2 = float(np.sum(residuals_flat**2))

        # Effective number of degrees of freedom
        # N_eff = trace( (F^T C_D^-1 F) * M^-1 )
        n_data = len(d_flat)
        try:
            m_inv = linalg.inv(m_matrix)
            n_eff = float(np.trace(ft_cd_f @ m_inv))
        except Exception:
            n_eff = float(len(s_hat))

        dof = max(1.0, float(n_data - n_eff))
        reduced_chi2 = chi2 / dof

        # Bayesian Log-Evidence:
        # ln E = -0.5 * chi2 - 0.5 * lambda * (s^T R s) + 0.5 * ln det(lambda R) - 0.5 * ln det(M)
        reg_term = float(regularization_lambda * (s_hat.T @ (self.r_matrix @ s_hat)))
        sign_r, log_det_r = np.linalg.slogdet(regularization_lambda * self.r_matrix)
        log_evidence = -0.5 * chi2 - 0.5 * reg_term + 0.5 * log_det_r - 0.5 * log_det_m

        return ReconstructionResult(
            reconstructed_source=rec_source_2d,
            model_image=model_image_2d,
            residuals=residuals_2d,
            chi2=chi2,
            reduced_chi2=reduced_chi2,
            log_evidence=float(log_evidence),
            regularization_lambda=regularization_lambda,
            source_grid=self.source_grid,
            image_grid=self.image_grid,
        )

    def optimize_regularization(
        self,
        data_image: np.ndarray,
        noise_sigma: Union[float, np.ndarray],
        log_lambda_bounds: Tuple[float, float] = (-2.0, 5.0),
    ) -> ReconstructionResult:
        """Find the optimal regularization parameter lambda maximizing Bayesian evidence.

        Args:
            data_image: 2D observed image.
            noise_sigma: 1-sigma uncertainty.
            log_lambda_bounds: Search range in log10(lambda), e.g. (-2.0, 5.0) for 10^-2 to 10^5.

        Returns:
            ReconstructionResult evaluated at optimal lambda.
        """
        def objective(log_l: float) -> float:
            lam = float(10.0**log_l)
            res = self.reconstruct(data_image, noise_sigma, regularization_lambda=lam)
            # Minimize negative log-evidence
            return -res.log_evidence

        res_opt = optimize.minimize_scalar(
            objective,
            bounds=log_lambda_bounds,
            method="bounded",
            options={"xatol": 0.05, "maxiter": 30},
        )
        best_lambda = float(10.0**res_opt.x)
        return self.reconstruct(data_image, noise_sigma, regularization_lambda=best_lambda)

