
import numpy as np

from scipy.interpolate import splprep, splev


def validate_points(points: np.ndarray) -> np.ndarray:
    """Validate 2D coordinate input."""

    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must have shape (N, 2)")

    if len(points) < 4:
        raise ValueError("At least 4 points are required")

    if not np.isfinite(points).all():
        raise ValueError("Points contain invalid coordinates")

    # Remove consecutive duplicate points
    distances = np.linalg.norm(
        np.diff(points, axis=0),
        axis=1,
    )

    keep = np.concatenate(([True], distances > 1e-9))
    points = points[keep]

    if len(points) < 4:
        raise ValueError("Not enough unique points")

    return points


def fit_spline(
    points: np.ndarray,
    smoothing: float = 0.0,
    degree: int = 3,
    samples: int = 200,
) -> np.ndarray:
    """
    Fit a parametric B-spline through 2D points.

    Returns sampled (x, y) coordinates.
    """

    points = validate_points(points)

    if degree not in (1, 2, 3):
        raise ValueError("Degree must be 1, 2, or 3")

    if len(points) <= degree:
        raise ValueError("Not enough points for spline degree")

    if samples < 2:
        raise ValueError("samples must be at least 2")

    tck, _ = splprep(
        [points[:, 0], points[:, 1]],
        s=smoothing,
        k=degree,
    )

    u_new = np.linspace(0, 1, samples)

    x_new, y_new = splev(u_new, tck)

    return np.column_stack((x_new, y_new))


def calculate_normals(
    points: np.ndarray,
) -> np.ndarray:
    """Calculate unit normal vectors along a sampled curve."""

    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must have shape (N, 2)")

    if len(points) < 2:
        raise ValueError("At least 2 points are required")

    tangent = np.gradient(points, axis=0)

    lengths = np.linalg.norm(
        tangent,
        axis=1,
        keepdims=True,
    )

    if np.any(lengths < 1e-12):
        raise ValueError("Curve contains degenerate tangent")

    tangent = tangent / lengths

    # Rotate tangent by 90 degrees
    normals = np.column_stack((
        -tangent[:, 1],
        tangent[:, 0],
    ))

    return normals


def offset_curve(
    points: np.ndarray,
    distance: float,
) -> np.ndarray:
    """
    Offset a sampled curve along its normal vectors.

    Positive and negative distances produce
    opposite offset directions.
    """

    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must have shape (N, 2)")

    normals = calculate_normals(points)

    return points + distance * normals
