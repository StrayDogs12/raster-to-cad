import cv2
import numpy as np
import pytest

from vision.mask_validation import ValidationConfig, validate_mask, validate_folder, save_report


def write_gray(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), arr)


def test_valid_binary_mask_passes(tmp_path):
    source = np.zeros((40, 50), np.uint8)
    source[10:30, 10:40] = 255
    mask = np.zeros_like(source)
    mask[10:30, 10:40] = 255
    sp, mp = tmp_path / "source.png", tmp_path / "mask_clean.png"
    write_gray(sp, source); write_gray(mp, mask)
    result = validate_mask(sp, mp)
    assert result["status"] == "PASS"
    assert result["is_binary"] is True
    assert result["dimensions"]["matches_source"] is True
    assert result["source_support_ratio"] == 1.0


def test_dimension_mismatch_fails(tmp_path):
    sp, mp = tmp_path / "source.png", tmp_path / "mask_clean.png"
    write_gray(sp, np.zeros((20, 20), np.uint8))
    m = np.zeros((21, 20), np.uint8); m[5:10, 5:10] = 255
    write_gray(mp, m)
    result = validate_mask(sp, mp)
    assert result["status"] == "FAIL"
    assert any("dimension mismatch" in e for e in result["errors"])


def test_empty_mask_fails(tmp_path):
    sp, mp = tmp_path / "source.png", tmp_path / "mask_clean.png"
    write_gray(sp, np.zeros((20, 20), np.uint8))
    write_gray(mp, np.zeros((20, 20), np.uint8))
    assert validate_mask(sp, mp)["status"] == "FAIL"


def test_nonbinary_values_are_flagged_for_review(tmp_path):
    source = np.zeros((20, 20), np.uint8); source[5:15, 5:15] = 255
    mask = np.zeros_like(source); mask[5:15, 5:15] = 180
    sp, mp = tmp_path / "source.png", tmp_path / "mask_clean.png"
    write_gray(sp, source); write_gray(mp, mask)
    result = validate_mask(sp, mp)
    assert result["is_binary"] is False
    assert result["status"] == "REVIEW"


def test_folder_validation_and_report(tmp_path):
    src, masks = tmp_path / "src", tmp_path / "masks"
    src.mkdir(); masks.mkdir()
    image = np.zeros((30, 30), np.uint8); image[5:25, 5:25] = 255
    write_gray(src / "layer.jpg", image)
    write_gray(masks / "layer_clean.png", image)
    report = validate_folder(src, masks)
    out = save_report(report, tmp_path / "report.json")
    assert report["validated"] == 1
    assert out.exists()


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        ValidationConfig(min_source_support_ratio=1.5).validate()
