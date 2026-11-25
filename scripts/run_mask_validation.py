from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.mask_validation import ValidationConfig, validate_folder, save_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate cleaned raster masks before CAD vectorization")
    parser.add_argument("--source", default="input_layers", help="folder containing original source images")
    parser.add_argument("--masks", default="output_masks", help="folder containing *_clean.png masks")
    parser.add_argument("--report", default="validation/validation_report.json", help="JSON report output path")
    parser.add_argument("--min-support", type=float, default=0.80, help="minimum fraction of mask pixels supported by bright source pixels")
    parser.add_argument("--source-threshold", type=int, default=20, help="source grayscale threshold used for support check")
    args = parser.parse_args()

    cfg = ValidationConfig(min_source_support_ratio=args.min_support,
                           source_support_threshold=args.source_threshold)
    report = validate_folder(args.source, args.masks, cfg)
    report_path = save_report(report, args.report)

    print(f"Overall: {report['overall_status']}")
    print(f"Validated: {report['validated']}/{report['total_masks']} | PASS={report['pass']} REVIEW={report['review']} FAIL={report['fail']}")
    for item in report["results"]:
        name = Path(item["mask"]).name
        print(f"[{item['status']}] {name} | fg={item['foreground_percent']:.4f}% | components={item['connected_components']} | endpoints={item['skeleton_endpoints']}")
        for warning in item["warnings"]:
            print(f"  WARNING: {warning}")
        for error in item["errors"]:
            print(f"  ERROR: {error}")
    if report["missing_sources"]:
        print("Masks without matching source:")
        for item in report["missing_sources"]:
            print(f"  {item}")
    print(f"Report: {report_path}")
    return 1 if report["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
