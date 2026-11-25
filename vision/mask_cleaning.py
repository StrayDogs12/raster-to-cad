"""Clean grayscale line-art layers into stable binary masks.

Expected input: dark/black background with light/white drawing lines. The
implementation deliberately avoids aggressive component-size filtering so
small text, arrowheads, symbols, and dashed grid lines are not discarded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

ThresholdMode = Literal["fixed", "otsu", "adaptive"]


@dataclass(frozen=True)
class CleaningConfig:
    """Parameters for mask extraction and conservative cleanup.

    threshold_mode:
        ``fixed`` is predictable for black-background CAD exports; ``otsu``
        estimates a global threshold; ``adaptive`` is useful for uneven scans.
    threshold:
        Grayscale threshold for fixed mode (0..255). Pixels >= threshold are
        considered drawing/foreground.
    close_kernel:
        Odd kernel size. 1 disables closing. Keep this small (2-3) to avoid
        merging nearby architectural details.
    close_iterations:
        Number of morphological closing passes.
    min_component_area:
        Components smaller than this are removed. Default 1 preserves all
        detected pixels; increase only after inspecting a diagnostic preview.
    adaptive_block_size / adaptive_c:
        Parameters for adaptive Gaussian thresholding.
    """

    threshold_mode: ThresholdMode = "fixed"
    threshold: int = 42
    close_kernel: int = 3
    close_iterations: int = 1
    min_component_area: int = 1
    adaptive_block_size: int = 31
    adaptive_c: int = 7

    def validate(self) -> None:
        if self.threshold_mode not in ("fixed", "otsu", "adaptive"):
            raise ValueError(f"Unsupported threshold_mode: {self.threshold_mode}")
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be between 0 and 255")
        if self.close_kernel < 1:
            raise ValueError("close_kernel must be >= 1")
        if self.close_kernel > 1 and self.close_kernel % 2 == 0:
            raise ValueError("close_kernel must be odd (or 1 to disable)")
        if self.close_iterations < 0:
            raise ValueError("close_iterations cannot be negative")
        if self.min_component_area < 1:
            raise ValueError("min_component_area must be >= 1")
        if self.adaptive_block_size < 3 or self.adaptive_block_size % 2 == 0:
            raise ValueError("adaptive_block_size must be an odd integer >= 3")


def load_grayscale(path: str | Path) -> np.ndarray:
    """Read an image as 8-bit grayscale; raise a useful error on failure."""
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def _threshold(gray: np.ndarray, config: CleaningConfig) -> np.ndarray:
    if gray.ndim != 2:
        raise ValueError("gray image must be a 2D grayscale array")

    if config.threshold_mode == "fixed":
        _, binary = cv2.threshold(gray, config.threshold, 255, cv2.THRESH_BINARY)
    elif config.threshold_mode == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            config.adaptive_block_size,
            config.adaptive_c,
        )
    return binary


def _remove_tiny_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Remove tiny connected components while retaining larger linework."""
    if min_area <= 1:
        return binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def clean_mask(gray: np.ndarray, config: CleaningConfig | None = None) -> np.ndarray:
    """Convert a grayscale drawing layer into a cleaned 0/255 binary mask."""
    config = config or CleaningConfig()
    config.validate()
    if gray is None or not isinstance(gray, np.ndarray) or gray.size == 0:
        raise ValueError("gray must be a non-empty numpy array")
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.ndim != 2:
        raise ValueError("Input must be a grayscale or BGR image")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    binary = _threshold(gray, config)

    # A very small closing operation can reconnect anti-aliased/JPEG-broken
    # strokes. It is optional because it may close intentional tiny gaps.
    if config.close_kernel > 1 and config.close_iterations > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.close_kernel, config.close_kernel)
        )
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel, iterations=config.close_iterations
        )

    binary = _remove_tiny_components(binary, config.min_component_area)
    return np.where(binary > 0, 255, 0).astype(np.uint8)


def save_mask(mask: np.ndarray, path: str | Path) -> Path:
    """Save a binary mask as lossless PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(path), normalized):
        raise OSError(f"Could not write output image: {path}")
    return path


def process_image(
    input_path: str | Path,
    output_path: str | Path,
    config: CleaningConfig | None = None,
) -> dict[str, int | str]:
    """Clean one image, save as PNG, and return basic diagnostics."""
    gray = load_grayscale(input_path)
    mask = clean_mask(gray, config)
    save_mask(mask, output_path)
    foreground = int(cv2.countNonZero(mask))
    total = int(mask.shape[0] * mask.shape[1])
    return {
        "input": str(input_path),
        "output": str(output_path),
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "foreground_pixels": foreground,
        "foreground_percent": round(100.0 * foreground / total, 4),
    }
