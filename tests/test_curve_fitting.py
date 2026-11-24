
import numpy as np

from geometry.curve_fitting import (
    fit_spline,
    calculate_normals,
    offset_curve,
)


def test_spline_generation():
    points = np.array([
        [0, 0],
        [10, 5],
        [20, 12],
        [30, 15],
        [40, 12],
        [50, 5],
    ])

    curve = fit_spline(
        points,
        smoothing=0,
        samples=200,
    )

    assert curve.shape == (200, 2)
    assert np.isfinite(curve).all()


def test_normal_vectors():
    points = np.array([
        [0, 0],
        [10, 0],
        [20, 0],
        [30, 0],
    ])

    normals = calculate_normals(points)

    assert normals.shape == points.shape
    assert np.allclose(
        np.linalg.norm(normals, axis=1),
        1.0,
    )


def test_curve_offset():
    points = np.array([
        [0, 0],
        [10, 0],
        [20, 0],
        [30, 0],
    ])

    offset = offset_curve(points, 5)

    assert offset.shape == points.shape
    assert np.allclose(offset[:, 1], 5)


def test_invalid_points():
    points = np.array([
        [0, 0],
        [10, 0],
    ])

    try:
        fit_spline(points)
        assert False, "Expected ValueError"
    except ValueError:
        pass
