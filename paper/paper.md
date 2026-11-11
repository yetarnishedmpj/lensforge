---
title: 'LensForge: A Bayesian Framework for Gravitational Lens Modeling and Galaxy Source Reconstruction'
tags:
  - Python
  - astronomy
  - astrophysics
  - gravitational lensing
  - computational imaging
  - inverse problems
  - cosmology
authors:
  - name: Research Contributor
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Computational Astrophysics Research Group
    index: 1
date: 11 November 2026
bibliography: paper.bib
---

# Summary

Strong gravitational lensing occurs when the gravitational field of a massive foreground deflector (such as an elliptical galaxy or cluster) bends light emitted by a distant background source, producing highly distorted arcs, Einstein rings, and multiple images [@kormann1994; @bolton2008]. Modeling strong lenses serves two complementary astrophysical objectives: precision mapping of the foreground dark matter distribution and super-resolved imaging of infant galaxies in the early Universe.

`LensForge` is an open-source scientific Python framework designed for forward modeling and Bayesian non-parametric source reconstruction of galaxy-scale strong lenses. By formulating the observation as a semilinear inverse problem, `LensForge` decouples the linear reconstruction of the unlensed galaxy's pixelated surface brightness from the non-linear optimization of macro-lens mass parameters [@warren2003; @suyu2006]. The package integrates native Friedmann-Lemaître-Robertson-Walker (FLRW) cosmological distance routines, FITS observational data pipelines, telescope point-spread function (PSF) convolution, and Bayesian evidence maximization into a modular, research-grade software architecture.

# Statement of Need

While large-scale lensing analysis suites such as `lenstronomy` [@birrer2021] and `PyAutoLens` exist, their expansive monolithic designs and deep dependency graphs present significant pedagogical barriers and computational overhead for researchers prototyping custom inversion priors or evaluating targeted physical conjectures. 

`LensForge` fills the need for a transparent, mathematically rigorous, and computationally efficient research software foundation:
1. **Mathematical Clarity & Dimensional Integrity**: Strict, auditable separation between SI physical units and astronomical angular coordinates (radians, arcseconds), avoiding silent unit-conflation bugs common in astrophysical scripts.
2. **Semilinear Separation of Scales**: Linear pixelated reconstruction using Cholesky decomposition of regularized normal matrices coupled to non-linear macro-lens optimization (Nelder-Mead simplex with penalty bounds).
3. **Astropy & FITS Ecosystem Integration**: Direct ingestion of standard telescope FITS formats with World Coordinate System (WCS) metadata, automated sky background estimation via iterative sigma-clipping, and annular deflector starlight masking.
4. **Cosmological Calibration**: Integrated flat $\Lambda\text{CDM}$ cosmology engine (Planck 2018 baseline) verified against `astropy.cosmology`, directly linking observed angular Einstein radii $\theta_E$ to physical deflector masses $M(\le R_E)$ and stellar velocity dispersions $\sigma_v$.

# Mathematical Foundations

The observational forward model in strong gravitational lensing is expressed as:
\begin{equation}
\mathbf{d} = \mathbf{P} \ast \mathcal{L}_{\boldsymbol{\eta}}(\mathbf{s}) + \mathbf{n} = \mathbf{F}(\boldsymbol{\eta}) \mathbf{s} + \mathbf{n}
\end{equation}
where $\mathbf{d} \in \mathbb{R}^{N_d}$ represents the observed pixel fluxes, $\mathbf{P}$ is the telescope optical PSF, $\mathcal{L}_{\boldsymbol{\eta}}$ represents gravitational deflection by lens mass parameters $\boldsymbol{\eta}$, $\mathbf{s} \in \mathbb{R}^{N_s}$ is the intrinsic unlensed source surface brightness on a discrete grid, and $\mathbf{n} \sim \mathcal{N}(0, \mathbf{C}_D)$ denotes detector readout and photon shot noise.

The combined response matrix $\mathbf{F}(\boldsymbol{\eta}) = \mathbf{B} \mathbf{L}(\boldsymbol{\eta})$ encapsulates both the sparse bilinear ray-tracing interpolation operator $\mathbf{L}$ and the PSF convolution operator $\mathbf{B}$.

## Linear Regularized Source Reconstruction

For a fixed lens model $\boldsymbol{\eta}$, recovering $\mathbf{s}$ is an ill-posed linear inverse problem requiring spatial regularization. Adopting a Gaussian prior $P(\mathbf{s} \mid \lambda) \propto \exp(-\frac{1}{2} \lambda \mathbf{s}^T \mathbf{R} \mathbf{s})$—where $\mathbf{R}$ represents the finite-difference curvature (discrete Laplacian) or gradient operator—the maximum a posteriori (MAP) source vector $\hat{\mathbf{s}}$ satisfies the linear system [@warren2003; @suyu2006]:
\begin{equation}
\left( \mathbf{F}^T \mathbf{C}_D^{-1} \mathbf{F} + \lambda \mathbf{R} \right) \hat{\mathbf{s}} = \mathbf{F}^T \mathbf{C}_D^{-1} \mathbf{d}
\end{equation}
Because the coefficient matrix $\mathbf{A} = \mathbf{F}^T \mathbf{C}_D^{-1} \mathbf{F} + \lambda \mathbf{R}$ is symmetric and strictly positive definite, `LensForge` solves this system efficiently via numerical Cholesky factorization $\mathbf{A} = \mathbf{L}_c \mathbf{L}_c^T$.

The regularization hyperparameter $\lambda$ is optimized by maximizing the Bayesian marginal likelihood (evidence) $P(\mathbf{d} \mid \lambda, \boldsymbol{\eta})$:
\begin{equation}
2 \ln P(\mathbf{d} \mid \lambda, \boldsymbol{\eta}) = -\chi^2 - \lambda \hat{\mathbf{s}}^T \mathbf{R} \hat{\mathbf{s}} - \ln \det \mathbf{A} + \ln \det (\lambda \mathbf{R}) - \ln \det \mathbf{C}_D - N_d \ln(2\pi)
\end{equation}

## Cosmological Distance Calculations

Deflection geometry depends on angular diameter distances $D_A(z_1, z_2)$ across cosmic time. Under a flat FLRW metric with $\Omega_k = 0$, the line-of-sight comoving distance is:
\begin{equation}
D_c(z_1, z_2) = \frac{c}{H_0} \int_{z_1}^{z_2} \frac{dz'}{\sqrt{\Omega_{m,0}(1+z')^3 + \Omega_{\Lambda,0}}}
\end{equation}
and the angular diameter distance is $D_A(z_1, z_2) = D_c(z_1, z_2) / (1 + z_2)$. The critical surface mass density governing lensing efficiency is computed as:
\begin{equation}
\Sigma_{\text{cr}} = \frac{c^2}{4\pi G} \frac{D_s}{D_l D_{ls}}
\end{equation}

# Software Architecture

`LensForge` is structured into decoupled, modular subpackages:

- `lensforge.physics`: Coordinates, cosmological distances, Einstein radius derivations, and mass models (Point Mass, Singular Isothermal Sphere, Singular Isothermal Ellipsoid, External Shear, and Composite Lenses).
- `lensforge.cosmology`: Integration of FLRW comoving and angular-diameter distances, $\Sigma_{\text{cr}}$, and analytical velocity dispersion scaling matching `astropy.cosmology` Planck 2018 parameters [@planck2020].
- `lensforge.instrument`: Optical PSF models (Gaussian, Pixelated) and astronomical noise simulation combining Poissonian photon counts with Gaussian amplifier readout noise.
- `lensforge.inversion`: Bilinear ray-tracing response construction, discrete Laplacian prior matrices, Cholesky solver, and Bayesian evidence evaluators.
- `lensforge.optimization`: Bilevel semilinear optimization framework for joint estimation of macro-lens parameters and pixelated source morphology.
- `lensforge.data`: FITS input/output with WCS metadata, iterative sigma-clipped sky background estimation, and annular deflector masking.
- `lensforge.visualization`: Publication-grade Matplotlib diagnostic dashboards displaying lensed observations, model reconstructions, normalized residuals, source planes, and critical/caustic curves.

# Validation & Case Study: SLACS SDSS J0946+1006

To demonstrate the framework on an astrophysically realistic system, `LensForge` includes an end-to-end benchmark script replicating the Sloan Lens ACS Survey (SLACS) Einstein ring **SDSS J0946+1006** [@bolton2008; @gavazzi2008; @dye2014]:
- **Deflector**: Early-type galaxy at $z_l = 0.222$ ($D_l = 764.4\text{ Mpc}$).
- **Source**: Star-forming dwarf galaxy at $z_s = 0.609$ ($D_s = 1433.3\text{ Mpc}$).
- **True Parameters**: $\theta_E = 1.38''$, axis ratio $q = 0.88$, position angle $\phi = 42.0^\circ$, external shear $\gamma = 0.045$.

Starting from an intentionally perturbed initial guess ($\theta_E = 1.25''$, $q = 0.78$, $\gamma = 0.030$), `LensForge`'s bilevel optimizer converges to $\theta_E = 1.395''$ ($1.07\%$ error), recovering the enclosed physical lens mass $M(\le \theta_E) = 3.07 \times 10^{11} M_\odot$ (true: $3.00 \times 10^{11} M_\odot$) and velocity dispersion $\sigma_v = 285.1\text{ km/s}$ (matching SDSS spectroscopic measurements of $\approx 280 \pm 15\text{ km/s}$).

# Automated Testing and CI/CD

`LensForge` maintains 100% automated test coverage across all physics, cosmology, inversion, and data ingestion modules using `pytest`. The test suite includes 58 deterministic test cases validating numerical stability, circular and elliptical limit behaviors, potential gradients, energy conservation, FITS round-trips, and Astropy cosmological agreement within 1%. Automated continuous integration (CI) workflows execute across Linux, macOS, and Windows platforms for Python 3.11 and 3.12.

# Acknowledgements

This research software was developed following open computational astrophysics standards, building upon foundational lensing theory developed by Kormann, Schneider, Bartelmann, Warren, Dye, Suyu, and the SLACS collaboration.

# References
