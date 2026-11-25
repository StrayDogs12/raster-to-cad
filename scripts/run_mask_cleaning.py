#!/usr/bin/env python3
"""Batch-clean raster drawing layers into lossless binary PNG masks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow execution as: python scripts/run_mask_cleaning.py ... from project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.mask_cleaning import CleaningConfig, process_image  # noqa: E402

SUPPORTED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean black-background architectural linework into binary masks."
    )
    parser.add_argument("--input", required=True, help="Input image or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--threshold-mode", choices=["fixed", "otsu", "adaptive"], default="fixed")
    parser.add_argument("--threshold", type=int, default=42)
    parser.add_argument("--close-kernel", type=int, default=3, help="Odd size; use 1 to disable")
    parser.add_argument("--close-iterations", type=int, default=1)
    parser.add_argument("--min-component-area", type=int, default=1,
                        help="Default 1 preserves tiny text/symbol details")
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.input)
    output_dir = Path(args.output)
    if not source.exists():
        print(f"ERROR: input does not exist: {source}", file=sys.stderr)
        return 2

    config = CleaningConfig(
        threshold_mode=args.threshold_mode,
        threshold=args.threshold,
        close_kernel=args.close_kernel,
        close_iterations=args.close_iterations,
        min_component_area=args.min_component_area,
    )
    config.validate()

    if source.is_file():
        files = [source] if source.suffix.lower() in SUPPORTED else []
    else:
        pattern = "**/*" if args.recursive else "*"
        files = sorted(p for p in source.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED)

    if not files:
        print("No supported image files found (.png/.jpg/.jpeg/.tif/.tiff/.bmp).", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for file_path in files:
        output_path = output_dir / f"{file_path.stem}_clean.png"
        try:
            report = process_image(file_path, output_path, config)
            reports.append(report)
            print(
                f"OK  {file_path.name} -> {output_path.name} "
                f"[{report['width']}x{report['height']}, "
                f"foreground {report['foreground_percent']}%]"
            )
        except Exception as exc:  # report per-file failure but continue batch
            print(f"FAIL {file_path}: {exc}", file=sys.stderr)

    report_path = output_dir / "cleaning_report.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nProcessed {len(reports)}/{len(files)} file(s). Report: {report_path}")
    return 0 if len(reports) == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
