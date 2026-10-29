# LensForge Interactive Research Notebooks

This directory contains interactive Jupyter research notebooks for gravitational lens modeling, forward simulation, and galaxy source reconstruction.

---

## 1. Local Interactive Research Workflow

LensForge is engineered as a first-class local scientific Python package. You can run all notebooks locally on your machine with full access to your hardware (multi-core CPU and NVIDIA CUDA GPU).

### Step 1: Install Package & Notebook Support

Ensure your virtual environment is active, then install `lensforge` in editable mode with development and notebook dependencies:

```bash
pip install -e ".[dev,notebook]"
```

Verify the installation:
```bash
python -c "import lensforge; print('LensForge version:', lensforge.__version__)"
```

---

### Step 2: Launch Jupyter Locally

You can launch Jupyter Lab or Jupyter Notebook directly from your terminal:

```bash
jupyter lab
```
*(Or `jupyter notebook`)*

Alternatively, open any `.ipynb` file directly inside your IDE (e.g., VS Code or Cursor) and select your active Python environment as the Jupyter kernel.

---

### Step 3: Local Artifact Persistence

All notebook simulations, reconstructed source matrices, and publication-quality diagnostic plots are saved directly to your local project directory:

```
lensforge/outputs/
```

Outputs are automatically organized locally on your fast SSD, with zero runtime limits, timeout disconnections, or cloud storage restrictions.

---

## 2. Available & Planned Research Notebooks

| Notebook | Focus | Status |
| :--- | :--- | :--- |
| `01_point_mass_and_galaxy_lensing.ipynb` | Point-mass deflection, Einstein rings, SIS velocity dispersion, SIE (Kormann et al. 1994), and external shear. | **Available** ✅ |
| `02_telescope_forward_model.ipynb` | Point Spread Function (PSF) 2D FFT convolution, pixel integration, and Poisson/read noise. | Phase 3 (Planned) |
| `03_semilinear_source_inversion.ipynb` | Regularized galaxy source reconstruction using Warren & Dye / Treu linear matrix inversion. | Phase 4 (Planned) |
| `04_lens_parameter_inference.ipynb` | Bayesian MCMC posterior sampling and nonlinear optimization of macro-lens parameters. | Phase 5 (Planned) |
| `05_space_telescope_reconstruction.ipynb` | Reconstruction pipelines for real HST and JWST strong lensing observations. | Phase 6 (Planned) |
