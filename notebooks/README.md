# LensForge Interactive Notebooks & Google Colab Workflow

This directory is designated for interactive research notebooks exploring gravitational lens modeling, forward simulation, and galaxy source reconstruction.

---

## 1. Google Colab Execution Workflow

LensForge is engineered to be 100% compatible with Google Colab (CPU and GPU runtimes) with zero proprietary dependencies.

### Step 1: Transferring Repository to Colab

You can transfer the repository to Google Colab using either **GitHub Clone** or **ZIP Archive Upload**:

#### Option A: Clone from GitHub
In a new Colab notebook cell:
```python
!git clone https://github.com/<your-username>/lensforge.git
%cd lensforge
```

#### Option B: Upload ZIP Archive
1. Compress your local repository into `lensforge.zip`.
2. In Colab, upload `lensforge.zip` using the file browser or via code:
```python
from google.colab import files
uploaded = files.upload()  # select lensforge.zip
!unzip -q lensforge.zip
%cd lensforge
```

---

### Step 2: Install Package in Editable Mode

Install `lensforge` in editable mode directly into the active Colab virtual environment:

```python
!pip install -e .
```

Verify the installation:
```python
import lensforge
print(f"LensForge version: {lensforge.__version__}")
```

---

### Step 3: Persistent Storage Warning

> [!WARNING]
> Google Colab runtime instances are **ephemeral** and will terminate after idle timeout or disconnection. All files stored on the local virtual machine will be permanently erased.
>
> To persist experimental results, figures, MCMC chains, and reconstructed source matrices, always mount Google Drive:
> ```python
> from google.colab import drive
> drive.mount('/content/drive')
>
> # Save outputs to persistent storage
> output_dir = "/content/drive/MyDrive/LensForge_Experiments/run_01"
> ```

---

## 2. Planned Research Notebooks

The following exploratory notebooks are planned across the development roadmap:

| Notebook | Focus | Phase |
| :--- | :--- | :--- |
| `01_point_mass_lensing.ipynb` | Point-mass deflection, Einstein rings, analytical image positions, and parity. | Phase 1 |
| `02_singular_isothermal_sphere.ipynb` | Singular Isothermal Sphere (SIS) and Singular Isothermal Ellipsoid (SIE) mass profiles. | Phase 2 |
| `03_telescope_forward_model.ipynb` | Point Spread Function (PSF) convolution, pixelation, and observational noise modeling. | Phase 3 |
| `04_semilinear_source_inversion.ipynb` | Inverting lens mapping using Warren & Dye / Treu regularized matrix methods. | Phase 4 |
| `05_lens_parameter_inference.ipynb` | MCMC posterior sampling and nonlinear optimization of macro-lens parameters. | Phase 5 |
