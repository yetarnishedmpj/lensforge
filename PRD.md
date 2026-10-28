# Product Requirements Document (PRD) — LensForge

**Project**: LensForge: Gravitational Lens Reconstruction Research System  
**Version**: 0.2.0 (Phase 1 & Phase 2 Baseline)  
**Status**: Approved & Validated  
**Author**: Computational Astrophysics & Imaging Research Engineering Team  

---

## 1. Executive Summary & Research Vision

Gravitational lensing offers an extraordinary astrophysical "cosmic telescope", magnifying distant galaxies that would otherwise remain unresolvable. However, gravitational lensing severely distorts the source morphology into giant arcs, multiple images, or Einstein rings. 

**LensForge** is a research-oriented computational astrophysics imaging framework engineered to invert this distortion and reconstruct the intrinsic morphology, stellar structure, and kinematics of distant galaxies.

The complete physical forward model is:

$$\mathcal{D} = \mathcal{P} * \mathcal{L}_{\boldsymbol{\eta}}(\mathcal{S}) + \mathcal{N}$$

where:
- $\mathcal{S}$ is the intrinsic 2D surface-brightness distribution of the background source galaxy.
- $\mathcal{L}_{\boldsymbol{\eta}}$ is the gravitational lens mapping parameterized by mass parameters $\boldsymbol{\eta}$.
- $\mathcal{P}$ is the telescope point-spread function (instrument optics and detector blurring).
- $\mathcal{N}$ represents observational noise (photon shot noise and detector read noise).
- $\mathcal{D}$ is the observed 2D CCD/FITS imaging data.

---

## 2. Project Scope Boundaries

### In-Scope (Phase 1 & Phase 2 Baseline):
- Core repository scaffolding using standard `src/` layout and PEP 518/621 packaging.
- Rigorous SI physical constants and bidirectional angular conversion utilities.
- Relativistic Einstein radius calculation using cosmological angular-diameter distances for point masses and SIS halos.
- Point-mass deflection field $\boldsymbol{\alpha}(\boldsymbol{\theta})$ and effective potential $\psi(\boldsymbol{\theta})$.
- Mathematical verification that $\nabla \psi = \boldsymbol{\alpha}$ via central finite differences.
- Analytical point-source image position solver and algebraic invariant verification.
- Signed, absolute, and total flux magnification computation with cancellation-free numerical formulations.
- Singular Isothermal Sphere (SIS) with velocity dispersion $\sigma_v$, analytical image solutions, and invariants.
- Singular Isothermal Ellipsoid (SIE) using Kormann, Schneider & Bartelmann (1994) / Keeton (2001) analytical deflection, Euler's homogeneous potential theorem, and arbitrary position angle $\phi$.
- External gravitational shear $(\gamma_1, \gamma_2)$ and polar parameterizations.
- Composite multi-component lens model (`CompositeLens`) for superposing galaxy halos and external shear.
- Continuous 2D Gaussian galaxy source modeling and backwards ray-traced surface brightness sampling under Liouville's theorem.
- Publication-quality visualization routines and diagnostic forward modeling scripts.
- 100% deterministic test coverage across all mathematical foundations (44 unit and integration tests).
- Complete Google Colab transfer and execution workflow.

### Out-of-Scope (Deferred to Subsequent Phases):
- Instrument point spread function (PSF) convolution (Phase 3).
- Observational noise injection (Gaussian read noise and Poisson shot noise) (Phase 3).
- Sub-pixel supersampling and detector pixel integration (Phase 3).
- Regularized linear source inversion (Warren & Dye adaptive pixelated meshes) (Phase 4).
- Nonlinear MCMC Bayesian parameter estimation and evidence computation (Phase 5).
- Ingestion and preprocessing of real astronomical FITS files (HST & JWST) (Phase 6).

---

## 3. Functional Requirements (FR)

### FR-1: Units and Physical Constants
- **FR-1.1**: The system shall provide exact CODATA values for the speed of light $c$ ($299,792,458 \text{ m/s}$) and gravitational constant $G$ ($6.67430 \times 10^{-11} \text{ m}^3 \text{ kg}^{-1} \text{ s}^{-2}$).
- **FR-1.2**: Physical calculations shall be conducted strictly in SI base units (`m`, `kg`, `s`).
- **FR-1.3**: The system shall provide bidirectional conversion functions between radians, arcseconds, and degrees supporting both scalars and NumPy arrays.

### FR-2: Cosmological Distances & Einstein Radius
- **FR-2.1**: The system shall compute the Einstein radius $\theta_E = \sqrt{\frac{4GM}{c^2} \frac{D_{ls}}{D_l D_s}}$ in radians and arcseconds.
- **FR-2.2**: The system shall validate all physical inputs: $M > 0$, $D_l > 0$, $D_s > 0$, $D_{ls} > 0$.
- **FR-2.3**: Under cosmological checks, the system shall enforce $D_s > D_l$ and $D_{ls} < D_s$, raising explicit `ValueError` on violations.

### FR-3: Point-Mass Deflection Field & Lens Equation
- **FR-3.1**: The system shall compute 2D reduced deflection vectors:
  $$\boldsymbol{\alpha}(\boldsymbol{\theta}) = \theta_E^2 \frac{\boldsymbol{\theta} - \boldsymbol{\theta}_l}{|\boldsymbol{\theta} - \boldsymbol{\theta}_l|^2}$$
- **FR-3.2**: When evaluated exactly at the lens center, the system shall raise `SingularityError` by default, or return `np.nan` if `mask_singular=True`.
- **FR-3.3**: The system shall execute thin-screen ray tracing via the lens equation $\boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}(\boldsymbol{\theta})$ for single coordinates and 2D arrays.

### FR-4: Lensing Potential & Gradient Verification
- **FR-4.1**: The system shall evaluate the point-mass effective potential $\psi(\boldsymbol{\theta}) = \theta_E^2 \ln(|\boldsymbol{\theta} - \boldsymbol{\theta}_l| / \theta_{\text{ref}})$.
- **FR-4.2**: The system shall provide a central finite-difference gradient calculator verifying that $\nabla \psi(\boldsymbol{\theta})$ matches $\boldsymbol{\alpha}(\boldsymbol{\theta})$ within $10^{-5}$ relative tolerance away from the singularity.

### FR-5: Analytical Image Positions
- **FR-5.1**: For a collinear scalar offset $\beta$, the system shall compute:
  $$\theta_+ = \frac{\beta + \sqrt{\beta^2 + 4\theta_E^2}}{2}, \quad \theta_- = \frac{\beta - \sqrt{\beta^2 + 4\theta_E^2}}{2}$$
- **FR-5.2**: The system shall confirm that $\theta_{\pm} - \frac{\theta_E^2}{\theta_{\pm}} = \beta$ within $10^{-12}$ relative precision.
- **FR-5.3**: For $\beta = 0$, the system shall detect perfect alignment and return an Einstein ring structure of radius $\theta_E$.

### FR-6: Image Magnifications & Parity
- **FR-6.1**: For normalized source offset $u = |\boldsymbol{\beta}| / \theta_E$, the system shall compute signed magnifications $\mu_+$ and $\mu_-$.
- **FR-6.2**: The system shall enforce $\mu_+ > 1.0$ (positive parity) and $\mu_- < 0.0$ (negative parity indicating image inversion).
- **FR-6.3**: The system shall verify the identity $\mu_+ + \mu_- = 1.0$ across all valid $u > 0$.
- **FR-6.4**: The system shall employ rationalized formulations eliminating catastrophic cancellation for $u \gg 1$.
- **FR-6.5**: At $u = 0$, the system shall raise `ZeroDivisionError` by default or return infinite magnification if `allow_infinite=True`.

### FR-7: Source Profiles & Surface Brightness Simulation
- **FR-7.1**: The system shall provide an extensible `BaseSourceModel` and a 2D `GaussianSource` model supporting centroid, size ($\sigma$), amplitude, axis ratio ($q$), and position angle.
- **FR-7.2**: The system shall simulate lensed surface brightness on a `CoordinateGrid` via backwards ray-tracing under Liouville's theorem ($I_{\text{obs}}(\boldsymbol{\theta}) = S(\boldsymbol{\beta}(\boldsymbol{\theta}))$).

### FR-8: Visualization & Reporting
- **FR-8.1**: The system shall render multi-panel figures illustrating the source plane, image plane, Einstein ring, lens center, and analytical image positions.
- **FR-8.2**: All spatial axes and critical curves shall display explicit angular units (`arcsec`).

---

## 4. Non-Functional Requirements (NFR)

### NFR-1: Portability & Environments
- Compatible with Python 3.11+.
- Fully executable in headless Linux/macOS environments and Google Colab virtual machines.
- Standard dependencies limited to NumPy, SciPy, Matplotlib, and PyYAML.

### NFR-2: Numerical Accuracy & Stability
- Deterministic numerical execution.
- No memory leaks or construction of dense $N^2 \times N^2$ pixel covariance matrices in Phase 1.
- Complete absence of unhandled `NaN` or `Inf` propagation during regular simulation flows.

### NFR-3: Code Quality & Software Architecture
- Clean package structure adhering to modern `src/` layout.
- Decoupled physics, imaging, and visualization subsystems.
- 100% type annotations and Google/NumPy-style docstrings.

---

## 5. Acceptance Criteria

| ID | Criteria | Validation Method | Status |
| :--- | :--- | :--- | :--- |
| **AC-1** | All angular conversions round-trip with $< 10^{-14}$ error | Automated pytest suite (`test_coordinates.py`) | PASS ✅ |
| **AC-2** | Einstein radius matches analytical GR formula | Automated pytest suite (`test_point_mass_lens.py`) | PASS ✅ |
| **AC-3** | Deflection agrees with $\nabla \psi$ within $10^{-5}$ tolerance | Automated pytest suite (`test_point_mass_lens.py`) | PASS ✅ |
| **AC-4** | Analytical image positions satisfy scalar lens equation | Automated pytest suite (`test_image_positions.py`) | PASS ✅ |
| **AC-5** | Magnification invariant $\mu_+ + \mu_- = 1.0$ holds across all $u$ | Automated pytest suite (`test_magnification.py`) | PASS ✅ |
| **AC-6** | Catastrophic cancellation eliminated up to $u = 10^5$ | Automated pytest suite (`test_magnification.py`) | PASS ✅ |
| **AC-7** | Singularity at lens center raises explicit `SingularityError` | Automated pytest suite (`test_point_mass_lens.py`) | PASS ✅ |
| **AC-8** | Demonstration script generates visual publication-ready output | End-to-end execution of `demonstrate_point_mass_lens.py` | PASS ✅ |

---

## 6. Phased Evolution Roadmap

- **Phase 2**: Singular Isothermal Sphere (SIS), Singular Isothermal Ellipsoid (SIE), and external tidal shear.
- **Phase 3**: 2D telescope PSF convolution (FFT), detector pixelation, and Gaussian/Poisson noise injection.
- **Phase 4**: Semi-linear regularized source inversion (Warren & Dye 2003) on Delaunay and adaptive grids.
- **Phase 5**: Joint MCMC / nested sampling Bayesian parameter estimation.
- **Phase 6**: Benchmarking on real space telescope strong lensing datasets (HST and JWST).
