import cv2
import numpy as np
import pytest

from vision.mask_cleaning import CleaningConfig, clean_mask, save_mask, load_grayscale


def test_fixed_threshold_returns_binary_mask():
    gray = np.zeros((40, 60), dtype=np.uint8)
    gray[10:30, 20:22] = 255
    result = clean_mask(gray, CleaningConfig(threshold_mode="fixed", threshold=42, close_kernel=1))
    assert set(np.unique(result)).issubset({0, 255})
    assert np.count_nonzero(result) == 40


def test_small_closing_bridges_one_pixel_gap():
    gray = np.zeros((30, 30), dtype=np.uint8)
    gray[14:17, 5:13] = 255
    gray[14:17, 14:22] = 255
    result = clean_mask(gray, CleaningConfig(close_kernel=3, close_iterations=1))
    assert result[15, 13] == 255


def test_component_filter_is_opt_in():
    gray = np.zeros((30, 30), dtype=np.uint8)
    gray[3, 3] = 255
    gray[10:20, 10] = 255
    keep_all = clean_mask(gray, CleaningConfig(close_kernel=1, min_component_area=1))
    filtered = clean_mask(gray, CleaningConfig(close_kernel=1, min_component_area=2))
    assert keep_all[3, 3] == 255
    assert filtered[3, 3] == 0
    assert np.count_nonzero(filtered) == 10


def test_otsu_mode_handles_binary_like_input():
    gray = np.zeros((20, 20), dtype=np.uint8)
    gray[5:15, 5:15] = 220
    result = clean_mask(gray, CleaningConfig(threshold_mode="otsu", close_kernel=1))
    assert np.count_nonzero(result) == 100


def test_invalid_even_closing_kernel_raises():
    with pytest.raises(ValueError, match="odd"):
        clean_mask(np.zeros((5, 5), dtype=np.uint8), CleaningConfig(close_kernel=2))


def test_save_and_load_roundtrip(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "out" / "mask.png"
    image = np.zeros((12, 16), dtype=np.uint8)
    image[2:10, 4:12] = 255
    assert cv2.imwrite(str(source), image)
    loaded = load_grayscale(source)
    mask = clean_mask(loaded, CleaningConfig(close_kernel=1))
    saved = save_mask(mask, output)
    reread = load_grayscale(saved)
    assert saved.exists()
    assert np.array_equal(mask, reread)
