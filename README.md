# LensForge: Gravitational Lens Reconstruction Research Project

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-44%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LensForge** is a scientific research framework in computational astrophysics designed to model gravitational lensing and reconstruct the intrinsic morphologies of distant galaxies from astronomical observations.

The ultimate physical forward model realized by LensForge is:

$$\mathcal{D} = \mathcal{P} * \mathcal{L}_{\boldsymbol{\eta}}(\mathcal{S}) + \mathcal{N}$$

where:
- $\mathcal{S}$ is the intrinsic 2D surface-brightness distribution of the background source galaxy.
- $\mathcal{L}_{\boldsymbol{\eta}}$ is the gravitational lensing transformation parameterized by mass parameters $\boldsymbol{\eta}$.
- $\mathcal{P}$ is the telescope Point Spread Function (PSF).
- $\mathcal{N}$ represents observational noise (Poisson shot noise and CCD read noise).
- $\mathcal{D}$ is the observed astronomical image.

---

## 1. Current Implementation Status (Phases 1 & 2 Completed)

LensForge currently implements the complete, validated mathematical foundation for both **point-mass gravitational lensing** and **realistic galaxy lens profiles (SIS, SIE, External Shear, CompositeLens)**:

- **SI Constants & Angular Conversions**: Standardized CODATA constants ($c, G, M_{\odot}$) and verified unit transformations between radians, arcseconds, and degrees.
- **Relativistic Einstein Radius**: Rigorous cosmological angular-diameter distance modeling ($D_l, D_s, D_{ls}$) for point masses and Singular Isothermal Spheres ($\theta_E = 4\pi (\sigma_v/c)^2 D_{ls}/D_s$).
- **Deflection & Potential**: Thin-screen ray-tracing $\boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}(\boldsymbol{\theta})$, effective 2D lens potential $\psi(\boldsymbol{\theta})$, and numerical verification that $\nabla \psi = \boldsymbol{\alpha}$ via central finite differences.
- **Analytical Image Solutions & Invariants**:
  - Point-mass: roots ($\theta_+, \theta_-$), algebraic invariants ($\theta_+ \theta_- = -\theta_E^2$, $\theta_+ + \theta_- = \beta$), and Einstein rings for $\beta = 0$.
  - SIS: constant image separation $\theta_+ - \theta_- = 2\theta_E$, magnification invariant $\mu_+ + \mu_- = 2.0$, and single-image cutoff outside caustic.
- **Magnification & Parity Engine**: Signed image magnifications with rationalized cancellation-free expressions eliminating numerical underflow for large impact parameters ($u \gg 1$), verifying the invariant $\mu_+ + \mu_- = 1.0$.
- **Singular Isothermal Ellipsoid (SIE)**: Closed-form complex deflection formulas following Kormann, Schneider & Bartelmann (1994) / Keeton (2001), arbitrary position angles $\phi$, potential evaluated via Euler's homogeneous function theorem ($\psi = \boldsymbol{\theta} \cdot \boldsymbol{\alpha}$), and smooth limit to SIS as $q \to 1$.
- **External Gravitational Shear**: Cartesian $(\gamma_1, \gamma_2)$ and polar $(\gamma, \phi_\gamma)$ tidal formulations with zero convergence ($\kappa = 0$) and constant magnification $\mu = 1 / (1 - \gamma^2)$.
- **Composite Lens Superposition**: Unified `CompositeLens` class to assemble multi-component models (e.g. SIE halo + external environmental shear).
- **Source Simulation & Visualization**: 2D Gaussian galaxy profiles sampled via backwards ray tracing under Liouville's theorem, with publication-quality Matplotlib figures.
- **Automated Test Suite**: 44 deterministic unit and integration tests passing in $< 0.5$ seconds with 100% pass rate.

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
    └── README.md                      # Google Colab instructions & Drive persistence
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

## 5. Running the Demonstration Scripts

### Point-Mass Gravitational Lens
Execute the point-mass forward simulation:

```bash
python scripts/demonstrate_point_mass_lens.py
```
This generates `outputs/demonstration_point_mass.png` illustrating the source plane, image plane, Einstein ring, and analytical image positions ($\theta_+, \theta_-$).

### Realistic Galaxy Lens (SIE + External Shear)
Execute the realistic galaxy forward simulation:

```bash
python scripts/demonstrate_galaxy_lens.py
```
This simulates a massive elliptical galaxy ($q=0.72, \phi=45^{\circ}, \gamma=0.06$) deflecting a background galaxy into giant gravitational arcs, saving a 3-panel diagnostic visualization to `outputs/demonstration_galaxy_sie.png`.

---

## 6. Google Colab Execution Guide

LensForge is structured for seamless portability to Google Colab.

### Step 1: Transfer Repository to Colab
Open a new notebook in Google Colab and run:

```python
!git clone https://github.com/<your-username>/lensforge.git
%cd lensforge
```

Or upload `lensforge.zip` and extract:
```python
!unzip -q lensforge.zip
%cd lensforge
```

### Step 2: Install in Editable Mode
```python
!pip install -e .
```

### Step 3: Run the Galaxy Demonstration
```python
from scripts.demonstrate_galaxy_lens import run_galaxy_demonstration
from pathlib import Path
from IPython.display import Image, display

output_path = run_galaxy_demonstration(Path("outputs"))
display(Image(filename=str(output_path)))
```

### Step 4: Persistent Storage Warning

> [!WARNING]
> Google Colab runtime instances are temporary virtual machines. All local files will be deleted upon disconnection.
>
> Mount Google Drive to preserve experiment artifacts:
> ```python
> from google.colab import drive
> drive.mount('/content/drive')
> ```

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
