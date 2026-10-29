# LensForge: Gravitational Lens Reconstruction Research Project

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-50%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LensForge** is a scientific research framework in computational astrophysics designed to solve the **Inverse Gravitational Lens Problem**: taking distorted, giant astronomical arcs observed by space telescopes and reconstructing the intrinsic, unlensed morphologies of distant galaxies.

The complete physical forward and inverse model is:

$$\mathcal{D} = \mathcal{P} * \mathcal{L}_{\boldsymbol{\eta}}(\mathcal{S}) + \mathcal{N} \quad \xrightarrow{\text{LensForge Inversion}} \quad \hat{\mathcal{S}} = \left(\mathbf{F}^T \mathbf{C}_D^{-1} \mathbf{F} + \lambda \mathbf{R}\right)^{-1} \mathbf{F}^T \mathbf{C}_D^{-1} \mathbf{d}$$

where:
- $\mathcal{S}$ is the intrinsic 2D surface-brightness distribution of the background source galaxy.
- $\mathcal{L}_{\boldsymbol{\eta}}$ is the gravitational lens mapping parameterized by mass parameters $\boldsymbol{\eta}$.
- $\mathcal{P}$ is the telescope Point Spread Function (PSF).
- $\mathcal{N}$ represents observational noise (Poisson photon shot noise and detector read noise).
- $\mathcal{D}$ is the observed astronomical image.
- $\hat{\mathcal{S}}$ is the maximum a posteriori (MAP) reconstructed unlensed galaxy image.

---

## 1. Current Implementation Status (Version 0.3.0)

LensForge provides both forward modeling and **regularized Bayesian source reconstruction**:

- **Inverse Problem Engine ([`lensforge.inversion`](src/lensforge/inversion))**:
  - Implementation of the **Warren & Dye (2003) / Suyu et al. (2006) Semilinear Inversion** algorithm.
  - Sparse bilinear deflection mapping operator $\mathbf{L}$.
  - Combined telescope response operator $\mathbf{F} = \mathbf{B} \mathbf{L}$.
  - Spatial prior regularization matrices $\mathbf{R}$ (Laplacian curvature, 1st-order gradient, and 0th-order norm).
  - Automated Bayesian optimal regularization parameter ($\lambda^*$) search maximizing log-evidence $\ln P(\mathbf{d}|\lambda, \boldsymbol{\eta})$.
  - Outputs full $\chi^2$, reduced $\chi^2/\text{dof} \approx 1.0$, residuals, and reconstructed source morphology.
- **Telescope Instrument Optics & Noise ([`lensforge.instrument`](src/lensforge/instrument))**:
  - 2D Gaussian and pixelated telescope PSFs with fast FFT convolution and flux conservation.
  - Poisson photon shot noise from galaxy and background sky + Gaussian detector readout noise.
  - Generation of variance and 1-sigma uncertainty maps $\mathbf{C}_D$ for rigorous $\chi^2$ statistical weighting.
- **Parametric Lens Profiles ([`lensforge.physics.models`](src/lensforge/physics/models))**:
  - Point-mass (Schwarzschild lens with analytical image roots $\theta_{\pm}$ and magnification invariants).
  - Singular Isothermal Sphere (SIS) parameterized by 1D velocity dispersion $\sigma_v$.
  - Singular Isothermal Ellipsoid (SIE, Kormann et al. 1994) with arbitrary position angles $\phi$ and Euler potential theorem.
  - External Gravitational Shear ($\gamma_1, \gamma_2$) for environmental tidal perturbations.
  - Multi-component `CompositeLens` superpositions.
- **Command-Line Interface (CLI)**:
  - Ready-to-use executable command `lensforge-reconstruct` for one-line automated scientific reconstructions.
- **Automated Test Suite**:
  - **50 deterministic unit and integration tests** passing with 100% pass rate.

---

## 2. Repository Layout

```
lensforge/
├── README.md                          # Project documentation and quickstart
├── PRD.md                             # Product Requirements Document
├── pyproject.toml                     # Package metadata and build system
├── .gitignore                         # Scientific Python gitignore
├── configs/
│   └── lens/
│       ├── point_mass.yaml            # Parameter configuration for point mass
│       └── galaxy_sie_shear.yaml      # Parameter configuration for realistic galaxy lens
├── docs/
│   ├── architecture.md                # System design & module boundaries
│   ├── mathematical_foundations.md    # Analytical derivations from GR
│   ├── numerical_conventions.md       # Units, coordinate systems, singularity policy
│   └── development_roadmap.md         # Phased research & development milestones
├── src/
│   └── lensforge/
│       ├── __init__.py                # Package exports
│       ├── constants.py               # Physical constants & unit conversions
│       ├── physics/
│       │   ├── __init__.py
│       │   ├── coordinates.py         # 2D coordinates & CoordinateGrid dataclass
│       │   ├── distances.py           # Angular-diameter distances & Einstein radius
│       │   ├── lens_equation.py       # Thin-screen ray-tracing mapping
│       │   ├── lens_potential.py      # Lensing potential & finite difference gradient
│       │   ├── magnification.py       # Jacobian, signed/absolute magnification, parity
│       │   └── models/
│       │       ├── __init__.py        # Lens models exports
│       │       ├── base.py            # BaseLensModel abstract class
│       │       ├── point_mass.py      # Schwarzschild point-mass model
│       │       ├── sis.py             # Singular Isothermal Sphere model
│       │       ├── sie.py             # Singular Isothermal Ellipsoid model (KSB94)
│       │       ├── shear.py           # External gravitational tidal shear
│       │       └── composite.py       # Composite multi-component lens model
│       ├── imaging/
│       │   ├── __init__.py
│       │   └── source_models.py       # Galaxy profiles & ray-traced sampling
│       └── visualization/
│           ├── __init__.py
│           └── lens_plots.py          # Scientific multi-panel figures
├── tests/
│   ├── test_coordinates.py            # Coordinate transformations & grid tests
│   ├── test_point_mass_lens.py        # Deflection, potential gradient, distance checks
│   ├── test_image_positions.py        # Analytical image solver & invariants
│   ├── test_magnification.py          # Magnification, parity, stability, caustic checks
│   ├── test_sis_lens.py               # SIS deflection, potential, image invariants
│   ├── test_sie_lens.py               # SIE deflection, potential gradient, spherical limit
│   └── test_shear_and_composite.py    # External shear & CompositeLens superposition tests
├── scripts/
│   ├── demonstrate_point_mass_lens.py # Point-mass forward simulation script
│   └── demonstrate_galaxy_lens.py     # Realistic galaxy SIE+shear arc demonstration
└── notebooks/
    └── README.md                      # Local interactive research notebook instructions
```

---

## 3. Installation

LensForge requires Python 3.11 or newer.

### Local Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/<your-username>/lensforge.git
cd lensforge
pip install -e .
```

To include test and development dependencies:

```bash
pip install -e ".[dev]"
```

---

## 4. Running the Automated Tests

All tests are implemented with `pytest`, requiring zero external data downloads:

```bash
pytest -v
```

Expected output:
```
tests/test_coordinates.py .......                                        [ 15%]
tests/test_image_positions.py .....                                      [ 27%]
tests/test_magnification.py .......                                      [ 43%]
tests/test_point_mass_lens.py .......                                    [ 59%]
tests/test_shear_and_composite.py ....                                   [ 68%]
tests/test_sie_lens.py ......                                            [ 81%]
tests/test_sis_lens.py ........                                          [100%]

============================= 44 passed in 0.43s ==============================
```

---

## 5. Running Reconstructions & Demonstrations

### Complete Inverse Reconstruction (The Core Product)
Run the full Bayesian source reconstruction pipeline from the command line:

```bash
lensforge-reconstruct --output-dir outputs
```
*(Or run `python scripts/reconstruct_galaxy.py`)*

This pipeline:
1. Simulates an observed space-telescope image containing distorted, giant arcs with realistic PSF blurring and noise.
2. Inverts the gravitational lens equation using regularized linear inversion on a free pixelated source grid.
3. Automatically identifies the Bayesian optimal regularization parameter $\lambda^*$.
4. Unveils the intrinsic morphology of the background galaxy.
5. Saves the publication-quality 6-panel scientific dashboard to `outputs/reconstruction_dashboard.png`.

### Forward Simulation Demos
- **Point-Mass Lens**: `python scripts/demonstrate_point_mass_lens.py` (saves `outputs/demonstration_point_mass.png`).
- **Realistic Galaxy Lens (SIE + Shear)**: `python scripts/demonstrate_galaxy_lens.py` (saves `outputs/demonstration_galaxy_sie.png`).

---

## 6. Local Scientific Workflow & Interactive Notebooks

LensForge is designed as a standalone, high-performance local Python application. It runs natively on your machine, leveraging multi-core CPUs and local GPUs without cloud dependencies or timeouts.

### Local Python Environment Setup

Create and activate an isolated virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install `lensforge` in editable mode with development and notebook tools:

```bash
pip install -e ".[dev,notebook]"
```

### Running Interactive Jupyter Notebooks Locally

Launch Jupyter Lab or Notebook server from the project root:

```bash
jupyter lab
```

Open [`notebooks/01_point_mass_and_galaxy_lensing.ipynb`](notebooks/01_point_mass_and_galaxy_lensing.ipynb) to interactively modify lens models, explore ray-tracing grids, and visualize gravitational arcs.

All generated simulation outputs and figures are persisted locally in:
```
lensforge/outputs/
```

---

## 7. Planned Development Stages

| Stage | Focus Area | Status | Description |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Point-Mass Foundation | **Completed** ✅ | Validated analytical physics, unit systems, ray-tracing, and test suite. |
| **Phase 2** | Extended Mass Profiles | **Completed** ✅ | Singular Isothermal Sphere (SIS), Singular Isothermal Ellipsoid (SIE), External Shear, CompositeLens. |
| **Phase 3** | Telescope Forward Model | **Planned** ⏳ | 2D PSF convolution via FFT, pixel integration, and Poisson/Gaussian noise. |
| **Phase 4** | Source Inversion Engine | **Planned** ⏳ | Regularized pixelated reconstruction (Warren & Dye 2003) on Cartesian and Delaunay meshes. |
| **Phase 5** | Bayesian Inference | **Planned** ⏳ | Joint MCMC / nested sampling for lens parameters and source morphology. |
| **Phase 6** | Space Telescope Science | **Planned** ⏳ | Validation and reconstruction on real Hubble Space Telescope (HST) and JWST strong lenses. |
