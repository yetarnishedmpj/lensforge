"""Macro-lens parameter optimization and bilevel semilinear inversion.

Solves the joint inverse problem in gravitational lensing:
    Optimizes non-linear mass parameters eta = (theta_E, q, phi, gamma, center_x, center_y)
    in an outer loop while analytically reconstructing the optimal pixelated source s_hat(eta)
    in an inner loop via Cholesky linear inversion.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import optimize

from lensforge.instrument.psf import BasePSF
from lensforge.inversion.reconstructor import ReconstructionResult, SourceReconstructor
from lensforge.physics.coordinates import CoordinateGrid
from lensforge.physics.models.composite import CompositeLens
from lensforge.physics.models.shear import ExternalShear
from lensforge.physics.models.sie import SingularIsothermalEllipsoid


@dataclass
class MacroLensFitResult:
    """Encapsulates the complete results of a macro-lens parameter fit.

    Attributes:
        best_params: Dictionary of optimized macro-lens parameters.
        param_uncertainties: Estimated 1-sigma uncertainties for fitted parameters.
        best_lens_model: Optimized lens model instance.
        reconstruction: ReconstructionResult corresponding to the best-fit lens model.
        optimization_history: List of loss evaluations across iterations.
        success: Whether the non-linear optimization converged.
        message: Convergence status message.
    """

    best_params: Dict[str, float]
    param_uncertainties: Dict[str, float]
    best_lens_model: Union[SingularIsothermalEllipsoid, CompositeLens]
    reconstruction: ReconstructionResult
    optimization_history: List[float] = field(default_factory=list)
    success: bool = True
    message: str = "Optimization converged."


def fit_lens_parameters(
    image_grid: CoordinateGrid,
    source_grid: CoordinateGrid,
    observed_image: np.ndarray,
    noise_sigma: Union[float, np.ndarray],
    initial_params: Dict[str, float],
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    psf: Optional[BasePSF] = None,
    include_shear: bool = True,
    regularization_lambda: float = 1.0,
    objective: str = "chi2",  # 'chi2' or 'evidence'
    method: str = "Nelder-Mead",
    max_iter: int = 40,
    compute_uncertainties: bool = False,
    callback: Optional[Callable[[Dict[str, float], float], None]] = None,
) -> MacroLensFitResult:
    """Fit macro-lens parameters simultaneously with regularized source reconstruction.

    Args:
        image_grid: Image plane CoordinateGrid.
        source_grid: Source plane CoordinateGrid.
        observed_image: 2D observed telescope image.
        noise_sigma: 1-sigma pixel uncertainty map or scalar.
        initial_params: Initial parameter guesses (e.g. {'theta_e': 1.2, 'axis_ratio': 0.8, 'pos_angle_deg': 45.0}).
        param_bounds: Dictionary mapping parameter names to (min, max) bounds.
        psf: Telescope PSF model.
        include_shear: If True, include external shear (gamma, phi_gamma_deg) in fitting.
        regularization_lambda: Regularization strength for source reconstruction.
        objective: 'chi2' (minimize chi-squared) or 'evidence' (minimize negative log-evidence).
        method: SciPy optimization algorithm ('L-BFGS-B', 'Nelder-Mead', 'Powell').
        max_iter: Maximum outer optimization iterations.
        callback: Optional callback invoked after each step: fn(current_params, current_loss).

    Returns:
        MacroLensFitResult instance.
    """
    # 1. Determine active parameter keys and defaults
    keys = ["theta_e", "axis_ratio", "pos_angle_deg", "center_x", "center_y"]
    if include_shear:
        keys.extend(["gamma", "phi_gamma_deg"])

    # Default parameter values
    defaults = {
        "theta_e": 1.0,
        "axis_ratio": 0.8,
        "pos_angle_deg": 0.0,
        "center_x": 0.0,
        "center_y": 0.0,
        "gamma": 0.0,
        "phi_gamma_deg": 0.0,
    }

    # Default parameter bounds
    default_bounds = {
        "theta_e": (0.2, 4.0),
        "axis_ratio": (0.3, 0.999),
        "pos_angle_deg": (0.0, 180.0),
        "center_x": (-0.8, 0.8),
        "center_y": (-0.8, 0.8),
        "gamma": (0.0, 0.25),
        "phi_gamma_deg": (0.0, 180.0),
    }

    active_keys = [k for k in keys if k in initial_params or k in (param_bounds or {})]
    if not active_keys:
        active_keys = ["theta_e", "axis_ratio", "pos_angle_deg"]
        if include_shear:
            active_keys.extend(["gamma", "phi_gamma_deg"])

    # Vectorize initial values and bounds
    x0 = []
    bounds = []
    for k in active_keys:
        val = initial_params.get(k, defaults[k])
        x0.append(val)
        if param_bounds and k in param_bounds:
            bounds.append(param_bounds[k])
        else:
            bounds.append(default_bounds.get(k, (None, None)))

    x0_arr = np.array(x0, dtype=np.float64)

    def _params_to_lens_model(p_dict: Dict[str, float]) -> Union[SingularIsothermalEllipsoid, CompositeLens]:
        """Helper to instantiate lens model from parameter dictionary."""
        sie = SingularIsothermalEllipsoid.from_degrees(
            theta_e=p_dict.get("theta_e", defaults["theta_e"]),
            axis_ratio=p_dict.get("axis_ratio", defaults["axis_ratio"]),
            pos_angle_deg=p_dict.get("pos_angle_deg", defaults["pos_angle_deg"]),
            center=(
                p_dict.get("center_x", defaults["center_x"]),
                p_dict.get("center_y", defaults["center_y"]),
            ),
        )
        if include_shear and (p_dict.get("gamma", 0.0) > 1e-5 or "gamma" in active_keys):
            shear = ExternalShear.from_polar_degrees(
                gamma=p_dict.get("gamma", defaults["gamma"]),
                phi_gamma_deg=p_dict.get("phi_gamma_deg", defaults["phi_gamma_deg"]),
                center=(
                    p_dict.get("center_x", defaults["center_x"]),
                    p_dict.get("center_y", defaults["center_y"]),
                ),
            )
            return CompositeLens([sie, shear])
        return sie

    history: List[float] = []

    def loss_function(x_vec: np.ndarray) -> float:
        """Inner-loop evaluation returning loss."""
        current_dict = initial_params.copy()
        for idx, k in enumerate(active_keys):
            current_dict[k] = float(x_vec[idx])

        # Enforce bounds via quadratic penalty
        for idx, k in enumerate(active_keys):
            b_min, b_max = bounds[idx]
            if b_min is not None and x_vec[idx] < b_min:
                return 1e9 + 1e5 * float((b_min - x_vec[idx])**2)
            if b_max is not None and x_vec[idx] > b_max:
                return 1e9 + 1e5 * float((x_vec[idx] - b_max)**2)

        # Enforce physical constraints
        if current_dict.get("axis_ratio", 1.0) <= 0.1 or current_dict.get("axis_ratio", 1.0) > 1.0:
            return 1e9
        if current_dict.get("theta_e", 1.0) <= 0.05:
            return 1e9

        try:
            lens_m = _params_to_lens_model(current_dict)
            rec = SourceReconstructor(
                image_grid=image_grid,
                source_grid=source_grid,
                lens_model=lens_m,
                psf=psf,
                regularization_method="curvature",
            )
            res = rec.reconstruct(
                data_image=observed_image,
                noise_sigma=noise_sigma,
                regularization_lambda=regularization_lambda,
            )

            val = -res.log_evidence if objective == "evidence" else res.chi2
            history.append(float(val))

            if callback is not None:
                callback(current_dict, float(val))

            return float(val)
        except Exception:
            return 1e9

    # Run optimization
    opt_res = optimize.minimize(
        loss_function,
        x0=x0_arr,
        bounds=bounds if method in ["L-BFGS-B", "SLSQP", "trust-constr", "Nelder-Mead"] else None,
        method=method,
        options={"maxiter": max_iter},
    )

    best_x = opt_res.x
    best_params = initial_params.copy()
    for idx, k in enumerate(active_keys):
        best_params[k] = float(best_x[idx])

    # Estimate parameter uncertainties (diagonal second differences) if requested
    uncertainties: Dict[str, float] = {}
    if compute_uncertainties:
        try:
            eps = 1e-2
            f0 = loss_function(best_x)
            for i, k in enumerate(active_keys):
                x_pi = best_x.copy()
                x_pi[i] += eps
                f_pi = loss_function(x_pi)
                x_mi = best_x.copy()
                x_mi[i] -= eps
                f_mi = loss_function(x_mi)
                d2f = (f_pi - 2.0 * f0 + f_mi) / (eps**2)
                var = 2.0 / d2f if (d2f > 0 and objective == "chi2") else 1.0 / max(1e-6, abs(d2f))
                uncertainties[k] = float(np.sqrt(max(1e-8, var)))
        except Exception:
            for k in active_keys:
                uncertainties[k] = float(np.nan)
    else:
        for k in active_keys:
            uncertainties[k] = float(np.nan)

    # Final reconstruction at best-fit parameters
    best_lens = _params_to_lens_model(best_params)
    final_reconstructor = SourceReconstructor(
        image_grid=image_grid,
        source_grid=source_grid,
        lens_model=best_lens,
        psf=psf,
        regularization_method="curvature",
    )
    final_res = final_reconstructor.reconstruct(
        data_image=observed_image,
        noise_sigma=noise_sigma,
        regularization_lambda=regularization_lambda,
    )

    return MacroLensFitResult(
        best_params=best_params,
        param_uncertainties=uncertainties,
        best_lens_model=best_lens,
        reconstruction=final_res,
        optimization_history=history,
        success=bool(opt_res.success),
        message=str(opt_res.message),
    )
