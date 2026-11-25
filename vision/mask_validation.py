"""Quality checks for cleaned raster masks before CAD vectorization.

The validator reports measurable raster properties. It does NOT claim that
pixels represent semantically correct CAD objects; that still requires visual
review against the source drawing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import cv2
import numpy as np


@dataclass(frozen=True)
class ValidationConfig:
    # Broad guardrails only. These are warnings, not universal CAD standards.
    min_foreground_percent: float = 0.01
    max_foreground_percent: float = 45.0
    max_small_component_ratio: float = 0.995
    source_support_threshold: int = 20
    min_source_support_ratio: float = 0.80
    max_border_contact_ratio: float = 0.50
    small_component_area: int = 3

    def validate(self) -> None:
        if not (0 <= self.min_foreground_percent < self.max_foreground_percent <= 100):
            raise ValueError("foreground percentage limits must satisfy 0 <= min < max <= 100")
        if not (0 <= self.max_small_component_ratio <= 1):
            raise ValueError("max_small_component_ratio must be in [0, 1]")
        if not (0 <= self.min_source_support_ratio <= 1):
            raise ValueError("min_source_support_ratio must be in [0, 1]")
        if not (0 <= self.max_border_contact_ratio <= 1):
            raise ValueError("max_border_contact_ratio must be in [0, 1]")
        if self.small_component_area < 1:
            raise ValueError("small_component_area must be >= 1")


def _read_gray(path: str | Path) -> np.ndarray:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def _skeleton_metrics(mask: np.ndarray) -> dict[str, int]:
    """Approximate skeleton endpoint/junction counts using iterative thinning.

    Uses cv2.ximgproc.thinning when available; otherwise a conservative
    morphological skeleton implementation is used.
    """
    binary = (mask > 0).astype(np.uint8) * 255
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        skeleton = cv2.ximgproc.thinning(binary)
    else:
        skeleton = np.zeros_like(binary)
        work = binary.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while cv2.countNonZero(work):
            eroded = cv2.erode(work, kernel)
            opened = cv2.dilate(eroded, kernel)
            skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
            work = eroded

    p = (skeleton > 0).astype(np.uint8)
    neighbors = cv2.filter2D(p, cv2.CV_16S, np.ones((3, 3), np.uint8)) - p
    # Count only skeleton pixels, using 8-neighbour connectivity.
    endpoints = int(np.count_nonzero((p == 1) & (neighbors == 1)))
    junctions = int(np.count_nonzero((p == 1) & (neighbors >= 3)))
    return {"skeleton_pixels": int(np.count_nonzero(p)),
            "skeleton_endpoints": endpoints,
            "skeleton_junction_pixels": junctions}


def validate_mask(
    source_path: str | Path,
    mask_path: str | Path,
    config: ValidationConfig | None = None,
) -> dict[str, Any]:
    """Validate one cleaned mask against its source raster and return metrics."""
    cfg = config or ValidationConfig()
    cfg.validate()
    source = _read_gray(source_path)
    mask_raw = _read_gray(mask_path)
    errors: list[str] = []
    warnings: list[str] = []

    same_size = source.shape == mask_raw.shape
    if not same_size:
        errors.append(f"dimension mismatch: source={source.shape[::-1]}, mask={mask_raw.shape[::-1]}")

    values = np.unique(mask_raw)
    is_binary = bool(np.all(np.isin(values, [0, 255])))
    if not is_binary:
        warnings.append("mask is not strictly binary (expected only 0 and 255)")
    mask = np.where(mask_raw >= 128, 255, 0).astype(np.uint8)
    height, width = mask.shape
    total = int(mask.size)
    foreground = int(cv2.countNonZero(mask))
    foreground_percent = 100.0 * foreground / max(total, 1)

    if foreground == 0:
        errors.append("empty foreground mask")

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if component_count > 1 else np.array([], dtype=np.int32)
    components = int(len(areas))
    small_count = int(np.count_nonzero(areas <= cfg.small_component_area))
    small_ratio = small_count / max(components, 1)
    largest_area = int(areas.max()) if components else 0
    largest_ratio = largest_area / max(foreground, 1)

    border = np.zeros_like(mask, dtype=np.uint8)
    border[0, :] = 1; border[-1, :] = 1
    border[:, 0] = 1; border[:, -1] = 1
    border_foreground = int(np.count_nonzero((mask > 0) & (border > 0)))
    border_ratio = border_foreground / max(foreground, 1)

    ys, xs = np.where(mask > 0)
    bbox = None if len(xs) == 0 else {
        "x_min": int(xs.min()), "y_min": int(ys.min()),
        "x_max": int(xs.max()), "y_max": int(ys.max()),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
    }

    source_support_ratio = None
    if same_size and foreground:
        source_support = source >= cfg.source_support_threshold
        source_support_ratio = float(np.count_nonzero((mask > 0) & source_support) / foreground)

    if foreground_percent < cfg.min_foreground_percent or foreground_percent > cfg.max_foreground_percent:
        warnings.append(f"foreground occupancy {foreground_percent:.4f}% is outside configured guardrails")
    if small_ratio > cfg.max_small_component_ratio:
        warnings.append(f"high ratio of tiny connected components: {small_ratio:.3f}")
    if border_ratio > cfg.max_border_contact_ratio:
        warnings.append(f"many foreground pixels touch image border: {border_ratio:.3f}")
    if source_support_ratio is not None and source_support_ratio < cfg.min_source_support_ratio:
        warnings.append(f"low source support ratio: {source_support_ratio:.3f}; inspect threshold/registration")

    skeleton = _skeleton_metrics(mask) if foreground else {
        "skeleton_pixels": 0, "skeleton_endpoints": 0, "skeleton_junction_pixels": 0
    }
    status = "FAIL" if errors else ("REVIEW" if warnings else "PASS")
    return {
        "source": str(source_path), "mask": str(mask_path),
        "status": status, "errors": errors, "warnings": warnings,
        "dimensions": {"width": int(width), "height": int(height), "matches_source": same_size},
        "binary_values": [int(v) for v in values], "is_binary": is_binary,
        "foreground_pixels": foreground,
        "foreground_percent": round(foreground_percent, 5),
        "connected_components": components,
        "small_components_count": small_count,
        "small_components_ratio": round(small_ratio, 5),
        "largest_component_pixels": largest_area,
        "largest_component_ratio": round(largest_ratio, 5),
        "border_foreground_pixels": border_foreground,
        "border_contact_ratio": round(border_ratio, 5),
        "source_support_ratio": None if source_support_ratio is None else round(source_support_ratio, 5),
        "foreground_bbox": bbox,
        **skeleton,
    }


def validate_folder(
    source_dir: str | Path,
    mask_dir: str | Path,
    config: ValidationConfig | None = None,
) -> dict[str, Any]:
    """Validate matching *_clean.png masks against source images by stem."""
    source_dir, mask_dir = Path(source_dir), Path(mask_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory not found: {source_dir}")
    if not mask_dir.is_dir():
        raise NotADirectoryError(f"Mask directory not found: {mask_dir}")

    sources = {p.stem: p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}}
    masks = sorted(mask_dir.glob("*_clean.png"))
    if not masks:
        raise FileNotFoundError(f"No *_clean.png masks found in {mask_dir}")

    results = []
    missing_sources = []
    for mask_path in masks:
        stem = mask_path.stem[:-6] if mask_path.stem.endswith("_clean") else mask_path.stem
        source_path = sources.get(stem)
        if source_path is None:
            missing_sources.append(str(mask_path))
            continue
        results.append(validate_mask(source_path, mask_path, config))

    summary = {
        "source_directory": str(source_dir), "mask_directory": str(mask_dir),
        "config": asdict(config or ValidationConfig()),
        "total_masks": len(masks), "validated": len(results),
        "pass": sum(r["status"] == "PASS" for r in results),
        "review": sum(r["status"] == "REVIEW" for r in results),
        "fail": sum(r["status"] == "FAIL" for r in results),
        "missing_sources": missing_sources,
        "overall_status": "FAIL" if missing_sources or any(r["status"] == "FAIL" for r in results)
                          else ("REVIEW" if any(r["status"] == "REVIEW" for r in results) else "PASS"),
        "results": results,
        "important_note": "PASS means raster checks passed; it does not prove semantic correctness of Gemini layer separation or CAD geometry.",
    }
    return summary


def save_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
