
import cv2
import numpy as np
import pytest

from vision.color_segmentation import (
    load_color_config,
    segment_colors,
    save_masks,
    mask_statistics,
)


CONFIG = "config/colors.yaml"


def test_load_color_config():
    colors = load_color_config(CONFIG)

    assert "cyan" in colors
    assert "yellow" in colors
    assert "red" in colors
    assert "magenta" in colors
    assert "green" in colors
    assert "white_gray" in colors


def test_color_segmentation():
    colors = load_color_config(CONFIG)

    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # BGR synthetic test patches
    image[10:30, 10:30] = [255, 255, 0]   # Cyan
    image[10:30, 40:60] = [0, 255, 255]  # Yellow
    image[40:60, 10:30] = [0, 0, 255]    # Red
    image[40:60, 40:60] = [255, 0, 255]  # Magenta
    image[70:90, 10:30] = [0, 255, 0]    # Green
    image[70:90, 40:60] = [255, 255, 255] # White

    masks = segment_colors(image, colors)

    assert len(masks) == 6

    for mask in masks.values():
        assert mask.dtype == np.uint8
        assert mask.shape == (100, 100)


def test_mask_detection():
    colors = load_color_config(CONFIG)

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[10:30, 10:30] = [0, 255, 255]

    masks = segment_colors(image, colors)

    assert np.count_nonzero(masks["yellow"]) > 0


def test_save_masks(tmp_path):
    masks = {
        "cyan": np.zeros((50, 50), dtype=np.uint8),
        "yellow": np.ones((50, 50), dtype=np.uint8) * 255,
    }

    saved = save_masks(masks, tmp_path)

    assert len(saved) == 2

    for path in saved.values():
        assert path.exists()

        loaded = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )

        assert loaded is not None


def test_mask_statistics():
    masks = {
        "cyan": np.ones((100, 100), dtype=np.uint8) * 255
    }

    stats = mask_statistics(masks)

    assert stats["cyan"]["pixels"] == 10000
    assert stats["cyan"]["percentage"] == 100.0


def test_invalid_config():
    with pytest.raises(FileNotFoundError):
        load_color_config("config/not_found.yaml")


def test_empty_masks():
    with pytest.raises(ValueError):
        mask_statistics({})
