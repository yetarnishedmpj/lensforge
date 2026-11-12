# LensForge: Gravitational Lens Modeling and Galaxy Source Reconstruction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-61%20passed-brightgreen.svg)]()
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue.svg)](.github/workflows/ci.yml)
[![JOSS Paper](https://img.shields.io/badge/JOSS-paper%20draft-orange.svg)](paper/paper.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LensForge** is a research-grade computational astrophysics framework designed to solve the **Inverse Gravitational Lens Problem**: taking distorted, giant astronomical arcs observed by space telescopes (such as HST and JWST) and reconstructing the intrinsic, unlensed morphologies of distant infant galaxies while simultaneously inferring the dark matter mass distribution of the foreground deflector.

The mathematical forward and semilinear inverse model is:

$$\mathcal{D} = \mathcal{P} * \mathcal{L}_{\boldsymbol{\eta}}(\mathcal{S}) + \mathcal{N} \quad \xrightarrow{\text{LensForge Inversion}} \quad \hat{\mathbf{s}}(\boldsymbol{\eta}) = \left(\mathbf{F}^T \mathbf{C}_D^{-1} \mathbf{F} + \lambda \mathbf{R}\right)^{-1} \mathbf{F}^T \mathbf{C}_D^{-1} \mathbf{d}$$

where:
- $\mathcal{S}$ is the intrinsic 2D surface-brightness distribution of the background source galaxy.
- $\mathcal{L}_{\boldsymbol{\eta}}$ is gravitational lensing parameterized by non-linear mass parameters $\boldsymbol{\eta} = (\theta_E, q, \phi, \gamma)$.
- $\mathcal{P}$ is the telescope Point Spread Function (PSF).
- $\mathcal{N}$ represents observational noise (Poisson photon shot noise and detector readout noise).
- $\mathcal{D}$ is the observed astronomical telescope image.
- $\hat{\mathbf{s}}$ is the maximum a posteriori (MAP) reconstructed unlensed galaxy image on an adaptive source mesh.

---

## 1. Research-Grade Capabilities (Version 0.4.0)

LensForge provides an end-to-end computational pipeline for strong gravitational lensing research:

1. **Macro-Lens Non-Linear Optimization & Bilevel Inversion ([`lensforge.optimization`](src/lensforge/optimization))**:
   - Joint estimation of deflector parameters $\boldsymbol{\eta} = (\theta_E, q, \phi, \gamma, \phi_\gamma)$ via outer-loop simplex search (Nelder-Mead with quadratic penalty bounds).
   - Inner-loop analytical linear reconstruction of the pixelated source $\hat{\mathbf{s}}(\boldsymbol{\eta})$ via Cholesky decomposition of $(\mathbf{F}^T \mathbf{C}_D^{-1} \mathbf{F} + \lambda \mathbf{R})$.
   - Hessian-based parameter uncertainty estimation and Bayesian evidence evaluation.

2. **Cosmological Distance & Mass Engine ([`lensforge.cosmology`](src/lensforge/cosmology))**:
   - Full FLRW comoving and angular-diameter distance integration ($D_l, D_s, D_{ls}$) under flat $\Lambda\text{CDM}$ (Planck 2018 parameters).
   - Critical surface mass density $\Sigma_{\text{cr}}$ in both physical SI ($\text{kg/m}^2$) and astronomical units ($M_\odot/\text{pc}^2$).
   - Physical conversion from angular Einstein radii to enclosed deflector mass $M(\le R_E)$ and theoretical central velocity dispersion $\sigma_v$.
   - Validated against `astropy.cosmology` within $1\%$.

3. **FITS Astronomical Data Pipeline ([`lensforge.data`](src/lensforge/data))**:
   - Multi-extension FITS file ingestion and export with WCS coordinate headers.
   - Robust iterative $\sigma$-clipped sky background estimation and RMS noise determination.
   - Annular deflector masking to isolate gravitationally lensed arcs and exclude deflector starlight subtraction residuals.

4. **Linear Regularized Inversion Engine ([`lensforge.inversion`](src/lensforge/inversion))**:
   - Implementation of the **Warren & Dye (2003) / Suyu et al. (2006)** semilinear inversion formulation.
   - Sparse bilinear deflection mapping operator $\mathbf{L}$.
   - Telescope response operator $\mathbf{F} = \mathbf{B} \mathbf{L}$ incorporating optical PSF blurring.
   - Spatial prior regularization matrices $\mathbf{R}$ (discrete 2D Laplacian curvature, 1st-order gradient, and 0th-order norm).
   - Automated Bayesian optimal regularization parameter ($\lambda^*$) search maximizing marginal log-evidence $\ln P(\mathbf{d}|\lambda, \boldsymbol{\eta})$.

5. **Telescope Optics & Detector Noise ([`lensforge.instrument`](src/lensforge/instrument))**:
   - 2D Gaussian and pixelated telescope PSFs with fast FFT convolution and flux conservation.
   - Realistic Poisson photon shot noise from galaxy and background sky + Gaussian detector readout noise.
   - Pixel-by-pixel variance and $1\sigma$ uncertainty maps $\mathbf{C}_D$ for rigorous $\chi^2$ statistical weighting.

6. **Astrophysical Lens Models ([`lensforge.physics.models`](src/lensforge/physics/models))**:
   - Point-mass (Schwarzschild lens with analytical image roots $\theta_{\pm}$ and magnification invariants).
   - Singular Isothermal Sphere (SIS) parameterized by 1D velocity dispersion $\sigma_v$.
   - Singular Isothermal Ellipsoid (SIE, Kormann et al. 1994) with arbitrary position angles $\phi$.
   - External Gravitational Shear ($\gamma_1, \gamma_2$) for environmental tidal perturbations.
   - Multi-component `CompositeLens` superpositions.

7. **SLACS Astrophysical Benchmark Case Study ([`scripts/benchmark_slacs_reconstruction.py`](scripts/benchmark_slacs_reconstruction.py))**:
   - End-to-end reconstruction reproducing the published properties of the iconic SLACS lens **SDSS J0946+1006** ($z_l=0.222, z_s=0.609$).
   - Recovers the Einstein radius to $1.07\%$ accuracy ($\theta_E = 1.395''$ vs $1.38''$), enclosed mass $M(\le \theta_E) = 3.07 \times 10^{11} M_\odot$, and velocity dispersion $\sigma_v = 285.1\text{ km/s}$ (matching SDSS spectroscopy of $\sim 280\text{ km/s}$).
   - Exports calibrated FITS data and a publication-quality 6-panel scientific dashboard to `outputs/slacs_j0946_benchmark.png`.

8. **Research Software Engineering Artifacts**:
   - Complete draft paper for the *Journal of Open Source Software* (JOSS) in [`paper/paper.md`](paper/paper.md) and [`paper/paper.bib`](paper/paper.bib).
   - Multi-platform continuous integration (CI) via GitHub Actions across Linux, macOS, and Windows for Python 3.11 & 3.12.
   - 100% deterministic test suite (**58 tests passing**).

---

## 2. Repository Layout

```
lensforge/
├── README.md                          # Project documentation and quickstart
├── PRD.md                             # Product Requirements Document (Phases 1-6)
├── pyproject.toml                     # Package metadata, dependencies, and build config
├── .gitignore                         # Scientific Python gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI matrix (Linux, macOS, Windows)
├── paper/
│   ├── paper.md                       # JOSS submission paper draft
│   └── paper.bib                      # BibTeX citations for astrophysics literature
├── configs/
│   └── lens/
│       ├── point_mass.yaml            # Parameter configuration for point mass
│       └── galaxy_sie_shear.yaml      # Parameter configuration for realistic galaxy lens
├── docs/
│   ├── architecture.md                # System design & module boundaries
│   ├── mathematical_foundations.md    # Analytical derivations from GR and optics
│   ├── numerical_conventions.md       # Units, coordinate systems, singularity policy
│   └── development_roadmap.md         # Phased research & development milestones
├── src/
│   └── lensforge/
│       ├── __init__.py                # Version 0.4.0 exports
│       ├── constants.py               # SI constants & angular conversion utilities
│       ├── cli.py                     # CLI entrypoints (lensforge-reconstruct)
│       ├── cosmology/
│       │   ├── __init__.py
│       │   └── distances.py           # FLRW cosmological distance & Sigma_cr engine
│       ├── data/
│       │   ├── __init__.py
│       │   └── fits_pipeline.py       # FITS I/O, sigma-clipping sky, annular masking
│       ├── optimization/
│       │   ├── __init__.py
│       │   └── lens_optimizer.py      # Macro-lens non-linear fitting & semilinear MAP
│       ├── physics/
│       │   ├── coordinates.py         # 2D coordinates & CoordinateGrid dataclass
│       │   ├── distances.py           # Angular-diameter distances & Einstein radius
│       │   ├── lens_equation.py       # Thin-screen ray-tracing mapping
│       │   ├── lens_potential.py      # Lensing potential & finite difference gradient
│       │   ├── magnification.py       # Jacobian, signed/absolute magnification, parity
│       │   └── models/
│       │       ├── base.py            # BaseLensModel abstract class
│       │       ├── point_mass.py      # Schwarzschild point-mass model
│       │       ├── sis.py             # Singular Isothermal Sphere model
│       │       ├── sie.py             # Singular Isothermal Ellipsoid model (KSB94)
│       │       ├── shear.py           # External gravitational tidal shear
│       │       └── composite.py       # Composite multi-component lens model
│       ├── instrument/
│       │   ├── psf.py                 # Gaussian & Pixelated PSF convolution
│       │   └── noise.py               # Poisson photon shot noise & Gaussian read noise
│       ├── inversion/
│       │   ├── mapping.py             # Sparse bilinear deflection mapping matrix L
│       │   ├── regularization.py      # Laplacian curvature & gradient prior matrices R
│       │   ├── reconstructor.py       # Semilinear regularized inversion engine
│       │   └── pipeline.py            # Automated galaxy reconstruction workflow
│       ├── imaging/
│       │   └── source_models.py       # Galaxy profiles & ray-traced sampling
│       └── visualization/
│           └── lens_plots.py          # Publication-quality diagnostic dashboards
├── tests/
│   ├── test_coordinates.py            # Coordinate transformations & grid tests
│   ├── test_cosmology.py              # FLRW distance, Astropy comparison, Sigma_cr tests
│   ├── test_fits_pipeline.py          # FITS I/O, sky background, annular mask tests
│   ├── test_image_positions.py        # Analytical image solver & invariants
│   ├── test_instrument.py             # PSF convolution & detector noise tests
│   ├── test_inversion.py              # Mapping, regularization & Cholesky MAP tests
│   ├── test_lens_optimizer.py         # Semilinear macro-lens convergence tests
│   ├── test_magnification.py          # Magnification, parity, stability, caustic checks
│   ├── test_point_mass_lens.py        # Deflection, potential gradient, distance checks
│   ├── test_shear_and_composite.py    # External shear & CompositeLens superposition tests
│   ├── test_sie_lens.py               # SIE deflection, potential gradient, spherical limit
│   └── test_sis_lens.py               # SIS deflection, potential, image invariants
├── scripts/
│   ├── benchmark_slacs_reconstruction.py  # SLACS SDSS J0946+1006 benchmark case study
│   ├── reconstruct_galaxy.py              # Bayesian galaxy source reconstruction
│   ├── demonstrate_galaxy_lens.py         # Realistic galaxy SIE+shear arc demonstration
│   └── demonstrate_point_mass_lens.py     # Point-mass forward simulation script
└── notebooks/
    └── README.md                      # Interactive research notebook instructions
```

---

## 3. Installation

LensForge requires Python 3.11 or newer.

```bash
git clone https://github.com/maharana2002/lensforge.git
cd lensforge
pip install -e .
```

To include test and development dependencies:

```bash
pip install -e ".[dev]"
```

---

## 4. Running the Automated Test Suite

All 58 tests are deterministic, run with `pytest`, and require zero external network calls:

```bash
pytest -v
```

Expected output:
```
tests/test_coordinates.py .......                                        [ 12%]
tests/test_cosmology.py ....                                             [ 18%]
tests/test_fits_pipeline.py ...                                          [ 24%]
tests/test_image_positions.py .....                                      [ 32%]
tests/test_instrument.py ...                                             [ 37%]
tests/test_inversion.py ...                                              [ 43%]
tests/test_lens_optimizer.py .                                           [ 44%]
tests/test_magnification.py .......                                      [ 56%]
tests/test_point_mass_lens.py .......                                    [ 68%]
tests/test_shear_and_composite.py ....                                   [ 75%]
tests/test_sie_lens.py ......                                            [ 86%]
tests/test_sis_lens.py ........                                          [100%]

============================= 58 passed in 19.26s =============================
```

---

## 5. Running Case Studies & Real Space Telescope Pipeline

### 1. Real Hubble Space Telescope (HST) Archival Ingestion & Reconstruction
LensForge includes a built-in NASA MAST query and streaming ingestion pipeline to download real science data and perform complete Bayesian lens modeling:

```bash
# Step 1: Query NASA MAST and download real HST ACS/WFC F814W cutout
python scripts/download_real_lens_data.py --lens SDSSJ0946+1006 --cutout-size 8.0

# Step 2: Run end-to-end non-linear optimization & linear regularized source inversion
python scripts/reconstruct_real_lens.py
```

Outputs generated:
- `real_data/SDSSJ0946+1006_F814W_cutout_8.0arcsec.fits`: Calibrated $8.0'' \times 8.0''$ science cutout with astrometric WCS coordinates.
- `outputs/real_sdss_j0946_reconstruction.png`: High-resolution 6-panel scientific publication dashboard:
  1. Real HST ACS/WFC observation ($160 \times 160$ px, $0.05''/\text{px}$) with annular aperture mask.
  2. Best-fit forward-lensed model ($P * \mathbf{F}\hat{\mathbf{s}}$).
  3. Normalized fit residuals $(D - M)/\sigma$.
  4. Inverted unlensed source galaxy surface brightness $\hat{\mathbf{s}}$.
  5. Optimization convergence history across simplex iterations.
  6. Inferred physical quantities (Einstein radius $\theta_E$, axis ratio $q$, position angle $\phi$, enclosed dark matter mass $M$, and stellar velocity dispersion $\sigma_v$).

### 2. SLACS SDSS J0946+1006 Benchmark Case Study
Run the complete scientific benchmark replicating the published properties of the Sloan Lens ACS Survey Einstein ring:

```bash
python scripts/benchmark_slacs_reconstruction.py
```

Outputs generated:
- `outputs/slacs_j0946_observation.fits`: Multi-extension FITS observation with WCS headers, calibrated science data, noise variance map, and PSF kernel.
- `outputs/slacs_j0946_benchmark.png`: High-resolution 6-panel publication dashboard.

### 3. Standalone Bayesian Galaxy Reconstruction
Run the regularized linear inversion pipeline from CLI or script:

```bash
lensforge-reconstruct --output-dir outputs
# or:
python scripts/reconstruct_galaxy.py
```

Saves the 6-panel reconstruction diagnostic dashboard to `outputs/reconstruction_dashboard.png`.

---

## 6. Phased Research Roadmap Status

| Stage | Focus Area | Status | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Mathematical Foundations | **Completed** ✅ | SI unit system, angular coordinates, point-mass physics, analytical image roots. |
| **Phase 2** | Extended Mass Profiles | **Completed** ✅ | SIS, Singular Isothermal Ellipsoid (SIE; KSB94), External Shear, CompositeLens. |
| **Phase 3** | Telescope Forward Model | **Completed** ✅ | 2D Gaussian/pixelated PSF convolution, Poisson shot noise + Gaussian read noise. |
| **Phase 4** | Semilinear Source Inversion | **Completed** ✅ | Bilinear response matrix $\mathbf{F}$, Laplacian curvature prior $\mathbf{R}$, Cholesky MAP solver. |
| **Phase 5** | Macro-Lens Optimization | **Completed** ✅ | Nelder-Mead simplex search with penalty bounds, Bayesian evidence, SLACS benchmark. |
| **Phase 6** | Astronomical Pipeline & FITS | **Completed** ✅ | Astropy FITS I/O, WCS metadata, sigma-clipped sky background, annular masking. |
| **Phase 7** | Real Space Telescope Pipeline | **Completed** ✅ | NASA MAST API ingestion, WCS cutouts, and real HST ACS lens reconstruction. |

---

## 7. License and Citation

LensForge is released under the [MIT License](LICENSE).

If you use LensForge in your research or academic applications, please cite:

```bibtex
@article{lensforge2026,
  author = {LensForge Research Team},
  title = {LensForge: A Bayesian Framework for Gravitational Lens Modeling and Galaxy Source Reconstruction},
  journal = {Journal of Open Source Software (Draft)},
  year = {2026}
}
```
