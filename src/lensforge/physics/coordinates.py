"""Angular coordinate conventions, validation, and grid generation for LensForge.

In gravitational lensing, the observer measures angular positions on the celestial sphere.
We adopt standard Cartesian angular coordinates defined in the tangent plane:
    theta = (theta_x, theta_y) : Angular position on the image plane.
    beta  = (beta_x, beta_y)   : Angular position on the source plane.

Conventions & Units:
- Coordinate axes: theta_x is oriented towards East / positive right-ascension offset,
  and theta_y is oriented towards North / positive declination offset.
- Physical / Theoretical Lens calculations: Radians are the fundamental SI-consistent unit.
- Observational / Astronomy conventions: Arcseconds ('arcsec') are the standard presentation unit.
- Functions in this module strictly require or return explicitly named units to avoid silent errors.
"""

from dataclasses import dataclass
from typing import Literal, Tuple, Union
import numpy as np

from lensforge.constants import arcsec2rad, rad2arcsec

AngularUnit = Literal["rad", "arcsec"]


@dataclass(frozen=True)
class CoordinateGrid:
    """Represents a 2D regular angular coordinate meshgrid.

    Attributes:
        x: 2D array of horizontal angular coordinates (shape: (num_y, num_x)).
        y: 2D array of vertical angular coordinates (shape: (num_y, num_x)).
        pixel_scale: Angular width of one pixel along each dimension.
        unit: Angular unit ('rad' or 'arcsec').
    """

    x: np.ndarray
    y: np.ndarray
    pixel_scale: float
    unit: AngularUnit

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the grid shape (num_y, num_x)."""
        return self.x.shape

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Return the bounding box extent [x_min, x_max, y_min, y_max] suitable for imshow."""
        half_px = self.pixel_scale / 2.0
        return (
            float(self.x[0, 0] - half_px),
            float(self.x[0, -1] + half_px),
            float(self.y[0, 0] - half_px),
            float(self.y[-1, 0] + half_px),
        )

    def to_unit(self, target_unit: AngularUnit) -> "CoordinateGrid":
        """Return a new CoordinateGrid converted to the requested angular unit."""
        if target_unit == self.unit:
            return self
        if self.unit == "arcsec" and target_unit == "rad":
            return CoordinateGrid(
                x=arcsec2rad(self.x),
                y=arcsec2rad(self.y),
                pixel_scale=float(arcsec2rad(self.pixel_scale)),
                unit="rad",
            )
        elif self.unit == "rad" and target_unit == "arcsec":
            return CoordinateGrid(
                x=rad2arcsec(self.x),
                y=rad2arcsec(self.y),
                pixel_scale=float(rad2arcsec(self.pixel_scale)),
                unit="arcsec",
            )
        else:
            raise ValueError(f"Unsupported unit conversion: {self.unit} -> {target_unit}")


def make_coordinate_grid(
    num_pix: int,
    field_of_view: float,
    unit: AngularUnit = "arcsec",
    center: Tuple[float, float] = (0.0, 0.0),
) -> CoordinateGrid:
    """Construct a square 2D angular coordinate grid centered at a given position.

    Args:
        num_pix: Number of pixels along each side (must be positive integer >= 1).
        field_of_view: Total angular width/height of the grid in units of `unit` (> 0).
        unit: Angular unit for coordinates, either 'arcsec' or 'rad'.
        center: (x_center, y_center) coordinates in units of `unit`.

    Returns:
        CoordinateGrid instance with 2D coordinate arrays x and y.

    Raises:
        ValueError: If num_pix <= 0 or field_of_view <= 0.
    """
    if num_pix <= 0:
        raise ValueError(f"num_pix must be a positive integer, got {num_pix}")
    if field_of_view <= 0:
        raise ValueError(f"field_of_view must be strictly positive, got {field_of_view}")

    pixel_scale = field_of_view / float(num_pix)
    # Centers of pixels: offset from -FOV/2 to +FOV/2
    half_fov = field_of_view / 2.0
    x_1d = np.linspace(-half_fov + pixel_scale / 2.0, half_fov - pixel_scale / 2.0, num_pix) + center[0]
    y_1d = np.linspace(-half_fov + pixel_scale / 2.0, half_fov - pixel_scale / 2.0, num_pix) + center[1]

    x_2d, y_2d = np.meshgrid(x_1d, y_1d)
    return CoordinateGrid(x=x_2d, y=y_2d, pixel_scale=pixel_scale, unit=unit)


def validate_coordinates(
    theta_x: Union[float, np.ndarray],
    theta_y: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and broadcast angular coordinate pairs.

    Args:
        theta_x: Scalar or array of horizontal coordinates.
        theta_y: Scalar or array of vertical coordinates.

    Returns:
        Tuple of (theta_x, theta_y) as numpy arrays with identical broadcast shape.

    Raises:
        ValueError: If theta_x and theta_y cannot be broadcast to a common shape.
    """
    x_arr = np.asarray(theta_x, dtype=np.float64)
    y_arr = np.asarray(theta_y, dtype=np.float64)

    try:
        x_bc, y_bc = np.broadcast_arrays(x_arr, y_arr)
    except ValueError as exc:
        raise ValueError(
            f"Coordinate arrays theta_x (shape {x_arr.shape}) and theta_y (shape {y_arr.shape}) "
            "could not be broadcast together."
        ) from exc

    return x_bc, y_bc


def angular_radius(
    theta_x: Union[float, np.ndarray],
    theta_y: Union[float, np.ndarray],
    center: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Compute the angular radial distance |theta - center|.

    Args:
        theta_x: Horizontal coordinate(s).
        theta_y: Vertical coordinate(s).
        center: (x_0, y_0) center coordinates.

    Returns:
        Radial distance array sqrt((theta_x - x_0)^2 + (theta_y - y_0)^2).
    """
    x_bc, y_bc = validate_coordinates(theta_x, theta_y)
    dx = x_bc - center[0]
    dy = y_bc - center[1]
    return np.hypot(dx, dy)
