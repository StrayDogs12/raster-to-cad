
import cv2
import numpy as np
import pytest

from vision.preprocessing import (
    load_image,
    validate_image,
    resize_image,
    convert_to_hsv,
    save_image,
)


def create_test_image():
    return np.zeros((500, 1000, 3), dtype=np.uint8)


def test_validate_image():
    image = create_test_image()

    assert validate_image(image)


def test_resize_preserves_aspect_ratio():
    image = create_test_image()

    resized = resize_image(
        image,
        max_dimension=500
    )

    assert resized.shape == (250, 500, 3)


def test_hsv_conversion():
    image = create_test_image()

    hsv = convert_to_hsv(image)

    assert hsv.shape == image.shape
    assert hsv.dtype == np.uint8


def test_save_and_load_image(tmp_path):
    image = create_test_image()

    path = tmp_path / "test.png"

    save_image(image, path)

    loaded = load_image(path)

    assert loaded.shape == image.shape


def test_invalid_image():
    image = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(ValueError):
        validate_image(image)


def test_missing_image():
    with pytest.raises(FileNotFoundError):
        load_image("not_existing_image.png")
