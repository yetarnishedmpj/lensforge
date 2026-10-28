# Numerical Conventions and Guidelines

## 1. Coordinate Systems & Sky Orientations

LensForge adopts Cartesian angular coordinates defined in the tangent plane of the celestial sphere:

- **Image Plane**: $\boldsymbol{\theta} = (\theta_x, \theta_y)$
- **Source Plane**: $\boldsymbol{\beta} = (\beta_x, \beta_y)$

### Directional Conventions:
- $\theta_x$: Positive towards the **East** (Right Ascension offset $\Delta \alpha \cos\delta$).
- $\theta_y$: Positive towards the **North** (Declination offset $\Delta \delta$).
- Origin $(0, 0)$: Placed at the fiducial optical axis or user-defined lens centroid.

---

## 2. Unit Management Policy

Mixing angular and physical units without explicit conversion is the leading source of bugs in computational astrophysics. LensForge enforces a strict three-tier unit hierarchy:

| Tier | Context | Primary Unit | Rationale |
| :--- | :--- | :--- | :--- |
| **Physical (SI)** | Relativistic constants & distances | `m`, `kg`, `s` | Internal calculations ($c, G, M, D_l, D_s, D_{ls}$) use pure SI units. |
| **Theoretical** | Analytical lens equations & potentials | `rad` (radians) | Dimensionless or SI-consistent derivation ($r_s / D_A$). |
| **Observational** | Coordinates, sizes, sky grids, plots | `arcsec` (arcseconds) | Standard astronomical observing scale ($\theta_E \sim 1''$). |

### Conversion Invariants:
- $1 \text{ rad} = \frac{648,000}{\pi} \approx 206,264.806247 \text{ arcsec}$.
- $1 \text{ arcsec} = \frac{\pi}{648,000} \approx 4.8481368 \times 10^{-6} \text{ rad}$.
- Conversion utilities in `lensforge.constants` (`rad2arcsec`, `arcsec2rad`, `deg2rad`, `rad2deg`) must be called explicitly whenever crossing domain boundaries.

---

## 3. Discretization & Grid Geometry

A 2D observation field is discretized as a `CoordinateGrid`:

- **Array Ordering**: `(num_y, num_x)` where the first dimension corresponds to vertical rows ($\theta_y$) and the second dimension corresponds to horizontal columns ($\theta_x$).
- **Pixel Centering**: Pixel coordinates represent the **geometric center** of each pixel, not the bottom-left corner.
  For a grid spanning field of view $\text{FOV}$ with $N$ pixels and pixel scale $\Delta = \text{FOV} / N$:
  $$\theta_{x, i} = -\frac{\text{FOV}}{2} + \left(i + \frac{1}{2}\right) \Delta, \quad i \in \{0, \dots, N-1\}$$
- **Matplotlib Display Extent**:
  $$\text{extent} = [x_{\min} - \Delta/2, \; x_{\max} + \Delta/2, \; y_{\min} - \Delta/2, \; y_{\max} + \Delta/2]$$
  Always render with `origin="lower"` to ensure positive $\theta_y$ points North.

---

## 4. Singularity Handling & Error Policies

Gravitational lensing by ideal point masses contains true physical singularities:

1. **Central Coordinate Singularity ($r = 0$)**:
   - $\alpha(\theta) \to \infty$ and $\psi(\theta) \to -\infty$.
   - **Default Policy**: Calling `deflection` or `potential` at $\boldsymbol{\theta} = \boldsymbol{\theta}_l$ raises `SingularityError`.
   - **Masking Policy**: When `mask_singular=True`, singular coordinates are populated with `np.nan` (for deflection) and `-np.inf` (for potential), enabling seamless array masking in grid operations.

2. **Caustic Divergence ($u = 0$)**:
   - For a point source located exactly on the point caustic ($u = 0$), point-source magnification formally diverges: $\mu \to \infty$.
   - **Default Policy**: `point_mass_magnification(0.0)` raises `ZeroDivisionError`.
   - **Extended Policy**: Passing `allow_infinite=True` returns `(np.inf, -np.inf)`.
   - **Extended Sources**: Finite source profiles (e.g. `GaussianSource`) integrate over finite area and never produce infinite flux, naturally regularizing the caustic divergence during ray-tracing.

---

## 5. Numerical Stability & Floating-Point Precision

Calculations are performed in standard IEEE 754 64-bit double precision (`float64`, machine epsilon $\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$).

### Elimination of Catastrophic Cancellation:
In point-mass magnification calculations, as $u \to \infty$, the naive formula:
$$\mu_- = \frac{1}{2} - \frac{u^2 + 2}{2 u \sqrt{u^2 + 4}}$$
subtracts two numbers converging to $0.5$, losing precision for $u > 10^3$ and erroneously flipping sign at $u = 10^5$.
LensForge employs the rationalized formulation:
$$\mu_- = \frac{-2}{u\sqrt{u^2+4}(u\sqrt{u^2+4} + u^2 + 2)}$$
which evaluates strictly negative values with full relative precision across all dynamic ranges ($10^{-5} \le u \le 10^5$).
