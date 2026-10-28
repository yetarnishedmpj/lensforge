# Mathematical Foundations of Gravitational Lensing

## 1. Relativistic Deflection of Light

According to Einstein's General Relativity, a photon traversing the linearized Schwarzschild metric of a static mass $M$ with physical impact parameter $\xi$ experiences a small deflection angle $\hat{\alpha}$:

$$\hat{\alpha} = \frac{4 G M}{c^2 \xi}$$

where:
- $G = 6.67430 \times 10^{-11} \text{ m}^3 \text{ kg}^{-1} \text{ s}^{-2}$ is Newton's gravitational constant.
- $c = 299,792,458 \text{ m/s}$ is the speed of light in vacuum.
- $\xi$ is the physical perpendicular distance of closest approach to the mass.

---

## 2. Thin Lens Geometry and Cosmological Distances

In astrophysical lensing, the physical extent of the lens along the line of sight is minuscule compared to cosmological distances between the observer, lens, and source ($\sim \text{kpc}$ vs $\sim \text{Gpc}$). Consequently, the thin-lens approximation projects the gravitational potential onto a single transverse lens plane.

### Angular-Diameter Distances in FLRW Cosmology

Let:
- $D_l$: Angular-diameter distance from the observer to the lens plane.
- $D_s$: Angular-diameter distance from the observer to the source plane.
- $D_{ls}$: Angular-diameter distance from the lens to the source.

In an expanding universe described by the Friedmann-Lemaître-Robertson-Walker (FLRW) metric, **angular-diameter distances are non-additive**:

$$D_{ls} \ne D_s - D_l$$

The angular-diameter distance is defined by the ratio of physical transverse size $d\xi$ to angular subtended angle $d\theta$:

$$d\xi = D_A \, d\theta$$

In a spatially flat universe with comoving distance $\chi(z) = c \int_0^z \frac{dz'}{H(z')}$, the angular-diameter distance from redshift $z_1$ to $z_2$ is:

$$D_A(z_1, z_2) = \frac{1}{1 + z_2} \left[ \chi(z_2) - \chi(z_1) \right]$$

Hence:

$$D_l = \frac{\chi(z_l)}{1 + z_l}, \quad D_s = \frac{\chi(z_s)}{1 + z_s}, \quad D_{ls} = \frac{\chi(z_s) - \chi(z_l)}{1 + z_s}$$

Therefore, $D_{ls} = D_s - \frac{1 + z_l}{1 + z_s} D_l \ne D_s - D_l$. LensForge strictly treats $D_l, D_s, D_{ls}$ as distinct geometric quantities.

### The Lens Equation

From simple planar geometry on the celestial sphere, the physical position of the source $\boldsymbol{\eta}$ and image $\boldsymbol{\xi}$ satisfy:

$$\boldsymbol{\eta} = \frac{D_s}{D_l} \boldsymbol{\xi} - D_{ls} \hat{\boldsymbol{\alpha}}(\boldsymbol{\xi})$$

Dividing by $D_s$ and expressing coordinates as angular positions $\boldsymbol{\theta} = \boldsymbol{\xi} / D_l$ and $\boldsymbol{\beta} = \boldsymbol{\eta} / D_s$:

$$\boldsymbol{\beta} = \boldsymbol{\theta} - \frac{D_{ls}}{D_s} \hat{\boldsymbol{\alpha}}(D_l \boldsymbol{\theta})$$

Defining the **reduced deflection angle** $\boldsymbol{\alpha}(\boldsymbol{\theta})$:

$$\boldsymbol{\alpha}(\boldsymbol{\theta}) \equiv \frac{D_{ls}}{D_s} \hat{\boldsymbol{\alpha}}(D_l \boldsymbol{\theta})$$

we arrive at the classical **lens equation**:

$$\boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}(\boldsymbol{\theta})$$

---

## 3. The Einstein Radius

For a spherically symmetric point mass $M$ centered on the optical axis, $\xi = D_l |\boldsymbol{\theta}|$, and the physical deflection is $\hat{\alpha} = \frac{4GM}{c^2 D_l |\boldsymbol{\theta}|}$.
The reduced deflection angle is:

$$\alpha(\theta) = \frac{D_{ls}}{D_s} \frac{4GM}{c^2 D_l \theta} = \frac{4GM}{c^2} \frac{D_{ls}}{D_l D_s} \frac{1}{\theta}$$

Under perfect collinear alignment between observer, lens, and source, the source position is $\beta = 0$. The lens equation becomes:

$$0 = \theta - \frac{4GM}{c^2} \frac{D_{ls}}{D_l D_s} \frac{1}{\theta} \implies \theta^2 = \frac{4GM}{c^2} \frac{D_{ls}}{D_l D_s}$$

The characteristic angular scale is the **Einstein radius**:

$$\theta_E \equiv \sqrt{\frac{4GM}{c^2} \frac{D_{ls}}{D_l D_s}}$$

Expressed in terms of the Schwarzschild radius $r_s = \frac{2GM}{c^2}$:

$$\theta_E = \sqrt{2 r_s \frac{D_{ls}}{D_l D_s}}$$

---

## 4. Deflection Field and Effective Potential

In 2D angular vector notation, the reduced deflection angle for a point mass centered at $\boldsymbol{\theta}_l$ is:

$$\boldsymbol{\alpha}(\boldsymbol{\theta}) = \theta_E^2 \frac{\boldsymbol{\theta} - \boldsymbol{\theta}_l}{|\boldsymbol{\theta} - \boldsymbol{\theta}_l|^2}$$

The 2D effective gravitational lens potential $\psi(\boldsymbol{\theta})$ is defined such that its gradient equals the deflection field:

$$\boldsymbol{\alpha}(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \psi(\boldsymbol{\theta})$$

For a point mass:

$$\psi(\boldsymbol{\theta}) = \theta_E^2 \ln \left( \frac{|\boldsymbol{\theta} - \boldsymbol{\theta}_l|}{\theta_{\text{ref}}} \right)$$

where $\theta_{\text{ref}} > 0$ is an arbitrary angular reference scale.

### Proof of Gradient Agreement:

Let $r = |\boldsymbol{\theta} - \boldsymbol{\theta}_l| = \sqrt{(\theta_x - \theta_{lx})^2 + (\theta_y - \theta_{ly})^2}$.
Using the chain rule:

$$\frac{\partial \psi}{\partial \theta_x} = \theta_E^2 \frac{\partial}{\partial \theta_x} \left( \ln r - \ln \theta_{\text{ref}} \right) = \theta_E^2 \frac{1}{r} \frac{\partial r}{\partial \theta_x}$$

Since $\frac{\partial r}{\partial \theta_x} = \frac{\theta_x - \theta_{lx}}{r}$:

$$\frac{\partial \psi}{\partial \theta_x} = \theta_E^2 \frac{\theta_x - \theta_{lx}}{r^2} = \alpha_x(\boldsymbol{\theta})$$

Similarly for $\theta_y$:

$$\frac{\partial \psi}{\partial \theta_y} = \theta_E^2 \frac{\theta_y - \theta_{ly}}{r^2} = \alpha_y(\boldsymbol{\theta})$$

Note that changing $\theta_{\text{ref}}$ adds an unobservable global additive constant $-\theta_E^2 \ln \theta_{\text{ref}}$ to $\psi(\boldsymbol{\theta})$ and leaves the gradient $\boldsymbol{\alpha}(\boldsymbol{\theta})$ strictly invariant.

---

## 5. Analytical Image Positions

Consider a point source displaced from the lens center by a signed collinear angular distance $\beta$. The 1D scalar lens equation is:

$$\beta = \theta - \frac{\theta_E^2}{\theta}$$

Multiplying through by $\theta \ne 0$:

$$\theta^2 - \beta \theta - \theta_E^2 = 0$$

This quadratic equation has two analytical real solutions:

$$\theta_+ = \frac{\beta + \sqrt{\beta^2 + 4 \theta_E^2}}{2}$$

$$\theta_- = \frac{\beta - \sqrt{\beta^2 + 4 \theta_E^2}}{2}$$

### Algebraic Invariants:

1. **Product Invariant**:
   $$\theta_+ \cdot \theta_- = \frac{(\beta + \sqrt{\beta^2 + 4\theta_E^2})(\beta - \sqrt{\beta^2 + 4\theta_E^2})}{4} = \frac{\beta^2 - (\beta^2 + 4\theta_E^2)}{4} = -\theta_E^2$$
   The product of image positions is always $-\theta_E^2$, meaning the two images always appear on opposite sides of the lens center.

2. **Sum Invariant**:
   $$\theta_+ + \theta_- = \beta$$

3. **Separation Invariant**:
   $$\Delta \theta = \theta_+ - \theta_- = \sqrt{\beta^2 + 4\theta_E^2} \ge 2\theta_E$$
   The angular separation between images is always strictly greater than $2\theta_E$.

4. **Radial Bounds (for $\beta > 0$)**:
   $$\theta_+ > \theta_E \quad \text{and} \quad 0 > \theta_- > -\theta_E \implies |\theta_-| < \theta_E$$
   The outer image always lies outside the Einstein ring; the inner image always lies inside.

### Degenerate Case: Perfect Alignment ($\beta = 0$)

When $\beta = 0$:
$$\theta_+ = +\theta_E, \quad \theta_- = -\theta_E$$
In two dimensions, rotational symmetry implies that every azimuthal angle $\phi \in [0, 2\pi)$ satisfies the lens equation:
$$\boldsymbol{\theta} = \theta_E (\cos\phi, \sin\phi)$$
The source is imaged into a continuous circular **Einstein ring** rather than two discrete points.

---

## 6. Magnification, Parity, and Jacobian Matrix

The magnification of an infinitesimal solid angle is given by the inverse Jacobian determinant of the lens mapping:

$$A_{ij} = \frac{\partial \beta_i}{\partial \theta_j} = \delta_{ij} - \frac{\partial^2 \psi}{\partial \theta_i \partial \theta_j}$$

$$\mu = \frac{1}{\det A}$$

In polar coordinates $(r, \phi)$ aligned with the lens center:

$$\det A = \left( 1 - \frac{d\alpha}{dr} \right) \left( 1 - \frac{\alpha}{r} \right)$$

For a point mass where $\alpha(r) = \frac{\theta_E^2}{r}$:

$$\frac{d\alpha}{dr} = -\frac{\theta_E^2}{r^2}, \quad \frac{\alpha}{r} = \frac{\theta_E^2}{r^2}$$

$$\det A = \left( 1 + \frac{\theta_E^2}{r^2} \right) \left( 1 - \frac{\theta_E^2}{r^2} \right) = 1 - \frac{\theta_E^4}{r^4}$$

Evaluating at the two image positions $\theta_{\pm}$ in terms of the normalized impact parameter $u \equiv |\beta| / \theta_E$:

$$\mu_+ = \frac{1}{2} + \frac{u^2 + 2}{2 u \sqrt{u^2 + 4}}$$

$$\mu_- = \frac{1}{2} - \frac{u^2 + 2}{2 u \sqrt{u^2 + 4}}$$

### Physical Interpretation of Parity:

- $\mu_+ > 1.0$: Outer image. Det $A > 0$ (positive parity). Spatial orientation matches the source.
- $\mu_- < 0.0$: Inner image. Det $A < 0$ (negative parity). The image is mirrored across the tangential critical curve (the Einstein ring).
- **Observed Flux is Positive**: Observed flux is proportional to $|\mu|$:
  $$F_{\text{obs}} = |\mu| F_{\text{source}}$$
  The negative sign is a mathematical and topological indicator of parity reflection, never negative radiation energy.

### Total Unresolved Magnification:

$$\mu_{\text{tot}} = |\mu_+| + |\mu_-| = \mu_+ - \mu_- = \frac{u^2 + 2}{u \sqrt{u^2 + 4}}$$

- As $u \to 0$ (caustic approach): $\mu_{\text{tot}} \sim \frac{1}{u} \to \infty$.
- As $u \to \infty$ (distant limit): $\mu_{\text{tot}} \to 1.0$.
- For all $u > 0$: $\mu_{\text{tot}} > 1.0$ (gravitational lensing always amplifies total flux for a point mass).
- Magnification Sum Invariant: $\mu_+ + \mu_- = 1.0$ identically for all $u > 0$.

### Numerically Stable Rationalization:

For large $u \gg 1$, computing $\mu_- = 0.5 - \frac{u^2+2}{2u\sqrt{u^2+4}}$ naively suffers from catastrophic floating-point cancellation.
Rationalizing the numerator:

$$u\sqrt{u^2+4} - (u^2+2) = \frac{u^2(u^2+4) - (u^2+2)^2}{u\sqrt{u^2+4} + u^2+2} = \frac{-4}{u\sqrt{u^2+4} + u^2+2}$$

Dividing by $2u\sqrt{u^2+4}$:

$$\mu_- = \frac{-2}{u\sqrt{u^2+4} \left( u\sqrt{u^2+4} + u^2 + 2 \right)}$$

$$\mu_+ = 1 - \mu_-$$

This rationalized formulation is algebraically exact for all $u > 0$, preserves full machine precision without catastrophic cancellation, and guarantees $\mu_- < 0$ and $\mu_+ \ge 1$ across all floating-point scales.

---

## 7. Singular Isothermal Sphere (SIS)

The Singular Isothermal Sphere represents an idealized, spherically symmetric galaxy with constant stellar velocity dispersion $\sigma_v$, mimicking the observed flat rotation curves of galaxies.

### Density Profile and Einstein Radius
The 3D mass density profile is:
$$\rho(r) = \frac{\sigma_v^2}{2\pi G r^2}$$

Projecting along the line of sight yields the surface mass density:
$$\Sigma(R) = \int_{-\infty}^{\infty} \rho\left(\sqrt{R^2 + z^2}\right) dz = \frac{\sigma_v^2}{2 G R}$$

Dividing by the critical surface mass density $\Sigma_{\text{cr}} = \frac{c^2}{4\pi G} \frac{D_s}{D_l D_{ls}}$, the convergence is:
$$\kappa(\theta) = \frac{\theta_E}{2 |\boldsymbol{\theta}|}$$

where the SIS Einstein radius is:
$$\theta_E = 4\pi \left(\frac{\sigma_v}{c}\right)^2 \frac{D_{ls}}{D_s}$$

Unlike a point mass, the SIS Einstein radius depends only on the distance ratio $D_{ls}/D_s$, not on the observer-lens distance $D_l$ individually.

### Deflection Angle and Potential
The 2D reduced deflection angle has constant magnitude $\theta_E$ everywhere:
$$\boldsymbol{\alpha}(\boldsymbol{\theta}) = \theta_E \frac{\boldsymbol{\theta}}{|\boldsymbol{\theta}|}$$

The effective lensing potential is:
$$\psi(\boldsymbol{\theta}) = \theta_E |\boldsymbol{\theta}|$$
satisfying $\nabla \psi(\boldsymbol{\theta}) = \boldsymbol{\alpha}(\boldsymbol{\theta})$ for all $\boldsymbol{\theta} \ne \mathbf{0}$.

### Analytical Image Positions and Invariants
Along the collinear axis for source position $\beta$:
- If $|\beta| < \theta_E$: Two images form:
  $$\theta_+ = \beta + \theta_E, \quad \theta_- = \beta - \theta_E$$
  - Image separation invariant: $\theta_+ - \theta_- = 2\theta_E$ (independent of source position $\beta$).
- If $|\beta| > \theta_E$: Only one image forms at $\theta_+ = \beta + \theta_E$.
- If $\beta = 0$: Perfect alignment forms an Einstein ring of radius $\theta_E$.

### Magnifications and Parity
$$\mu_+ = 1 + \frac{\theta_E}{|\beta|} > 1.0 \quad (\text{outer image, positive parity})$$
$$\mu_- = 1 - \frac{\theta_E}{|\beta|} < 0.0 \quad (\text{inner image, negative parity, for } |\beta| < \theta_E)$$
- Magnification sum invariant: $\mu_+ + \mu_- = 2.0$ identically for all $0 < |\beta| < \theta_E$.
- Total magnification: $\mu_{\text{tot}} = |\mu_+| + |\mu_-| = \frac{2\theta_E}{|\beta|}$.

---

## 8. Singular Isothermal Ellipsoid (SIE)

The Singular Isothermal Ellipsoid (Kormann, Schneider & Bartelmann 1994; Keeton 2001) extends the isothermal profile to elliptical galaxy mass distributions with axis ratio $q = b/a \in (0, 1]$ and major axis position angle $\phi$.

### Coordinate Rotation
Given sky coordinates $(\theta_x, \theta_y)$ and lens centroid $(x_l, y_l)$:
$$\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} \cos\phi & \sin\phi \\ -\sin\phi & \cos\phi \end{pmatrix} \begin{pmatrix} \theta_x - x_l \\ \theta_y - y_l \end{pmatrix}$$

### Analytical Deflection Components (KSB94)
Defining elliptical radius $r = \sqrt{q^2 x_1^2 + x_2^2}$ and eccentricity parameter $s = \sqrt{1 - q^2}$:
$$\alpha_1(x_1, x_2) = \frac{\theta_E \sqrt{q}}{s} \arctan\left( \frac{s x_1}{r} \right)$$
$$\alpha_2(x_1, x_2) = \frac{\theta_E \sqrt{q}}{s} \operatorname{artanh}\left( \frac{s x_2}{r} \right)$$

In the spherical limit $q \to 1$ ($s \to 0$):
$$\lim_{s \to 0} \frac{\arctan(s x_1 / r)}{s} = \frac{x_1}{\sqrt{x_1^2 + x_2^2}}, \quad \lim_{s \to 0} \frac{\operatorname{artanh}(s x_2 / r)}{s} = \frac{x_2}{\sqrt{x_1^2 + x_2^2}}$$
identically recovering the Singular Isothermal Sphere.

Rotating deflection back into the celestial frame:
$$\alpha_x = \cos\phi \, \alpha_1 - \sin\phi \, \alpha_2$$
$$\alpha_y = \sin\phi \, \alpha_1 + \cos\phi \, \alpha_2$$

### Lensing Potential and Euler's Homogeneous Function Theorem
Because the projected isothermal mass distribution is scale-free ($\rho \propto r^{-2}$), the lensing potential $\psi$ is a homogeneous function of degree 1:
$$\psi(\lambda x_1, \lambda x_2) = \lambda \psi(x_1, x_2)$$

By Euler's theorem for homogeneous functions:
$$\psi(x_1, x_2) = x_1 \frac{\partial \psi}{\partial x_1} + x_2 \frac{\partial \psi}{\partial x_2} = x_1 \alpha_1 + x_2 \alpha_2 = (\theta_x - x_l) \alpha_x + (\theta_y - y_l) \alpha_y$$

This ensures that $\nabla \psi = \boldsymbol{\alpha}$ holds algebraically and exactly across all smooth regions.

### Analytical Convergence
$$\kappa(x_1, x_2) = \frac{\theta_E \sqrt{q}}{2 \sqrt{q^2 x_1^2 + x_2^2}}$$

---

## 9. External Gravitational Shear

Tidal gravitational perturbations from neighboring galaxies or cluster environments are modeled as external shear:

### Potential and Deflection
$$\psi_{\text{shear}}(\boldsymbol{\theta}) = \frac{1}{2} \gamma_1 (\Delta x^2 - \Delta y^2) + \gamma_2 \Delta x \Delta y$$

$$\alpha_x = \gamma_1 \Delta x + \gamma_2 \Delta y, \quad \alpha_y = \gamma_2 \Delta x - \gamma_1 \Delta y$$

where $\gamma_1 = \gamma \cos(2\phi_\gamma)$ and $\gamma_2 = \gamma \sin(2\phi_\gamma)$.

### Properties
- **Zero Convergence**: $\kappa = \frac{1}{2}\left(\frac{\partial \alpha_x}{\partial x} + \frac{\partial \alpha_y}{\partial y}\right) = \frac{1}{2}(\gamma_1 - \gamma_1) = 0.0$ (Ricci-flat vacuum tidal field).
- **Constant Magnification**: $\det A = 1 - \gamma^2 \implies \mu = \frac{1}{1 - \gamma^2}$.

---

## 10. Composite Lens Superposition

In realistic galaxy lensing, total deflection and potential are linear superpositions of constituent components:
$$\boldsymbol{\alpha}_{\text{total}}(\boldsymbol{\theta}) = \sum_i \boldsymbol{\alpha}_i(\boldsymbol{\theta})$$
$$\psi_{\text{total}}(\boldsymbol{\theta}) = \sum_i \psi_i(\boldsymbol{\theta})$$
$$\boldsymbol{\beta}(\boldsymbol{\theta}) = \boldsymbol{\theta} - \boldsymbol{\alpha}_{\text{total}}(\boldsymbol{\theta})$$
