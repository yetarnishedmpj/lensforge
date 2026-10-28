"""Composite gravitational lens model.

This module provides `CompositeLens`, which superposes multiple lens mass components
(e.g., an SIE galaxy halo + external environmental shear, or a central SMBH + galaxy halo).
"""

from typing import Iterator, List, Sequence, Tuple, Union
import numpy as np

from lensforge.physics.coordinates import validate_coordinates
from lensforge.physics.models.base import BaseLensModel


class CompositeLens(BaseLensModel):
    """Composite gravitational lens model superposing multiple lens components.

    Attributes:
        components: List of `BaseLensModel` instances.
    """

    def __init__(self, components: Sequence[BaseLensModel]) -> None:
        """Initialize composite lens model.

        Args:
            components: Non-empty sequence of BaseLensModel instances.

        Raises:
            ValueError: If components sequence is empty or contains non-BaseLensModel objects.
        """
        if not components:
            raise ValueError("CompositeLens requires at least one lens model component.")

        for i, comp in enumerate(components):
            if not isinstance(comp, BaseLensModel):
                raise TypeError(
                    f"Component {i} must be an instance of BaseLensModel, got {type(comp)}."
                )

        self._components: List[BaseLensModel] = list(components)

    @property
    def components(self) -> List[BaseLensModel]:
        """List of constituent lens model components."""
        return list(self._components)

    def __len__(self) -> int:
        """Number of lens components."""
        return len(self._components)

    def __getitem__(self, index: int) -> BaseLensModel:
        """Access a constituent component by index."""
        return self._components[index]

    def __iter__(self) -> Iterator[BaseLensModel]:
        """Iterate over constituent components."""
        return iter(self._components)

    def deflection(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        mask_singular: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the total 2D deflection angle as the sum of all components.

        alpha_total(theta) = sum_i alpha_i(theta)

        Args:
            theta_x: Horizontal image-plane coordinate(s).
            theta_y: Vertical image-plane coordinate(s).
            mask_singular: Singularity masking policy passed to each component.

        Returns:
            Tuple of (alpha_x_total, alpha_y_total).
        """
        tx, ty = validate_coordinates(theta_x, theta_y)
        alpha_x_total = np.zeros_like(tx, dtype=np.float64)
        alpha_y_total = np.zeros_like(ty, dtype=np.float64)

        for comp in self._components:
            ax, ay = comp.deflection(tx, ty, mask_singular=mask_singular)
            alpha_x_total += ax
            alpha_y_total += ay

        return alpha_x_total, alpha_y_total

    def potential(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        mask_singular: bool = False,
    ) -> np.ndarray:
        """Compute the total effective lens potential as the sum of all components.

        psi_total(theta) = sum_i psi_i(theta)

        Args:
            theta_x: Horizontal coordinate(s).
            theta_y: Vertical coordinate(s).
            mask_singular: Singularity masking policy passed to each component.

        Returns:
            Total lensing potential array.
        """
        tx, ty = validate_coordinates(theta_x, theta_y)
        psi_total = np.zeros_like(tx, dtype=np.float64)

        for comp in self._components:
            psi = comp.potential(tx, ty, mask_singular=mask_singular)
            psi_total += psi

        return psi_total

