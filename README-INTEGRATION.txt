RASTER-TO-CAD: MASK CLEANING MODULE
==================================

Files:
- vision/mask_cleaning.py
- tests/test_mask_cleaning.py
- scripts/run_mask_cleaning.py
- config/mask_cleaning.yaml

Dependencies (install only if missing):
    python -m pip install opencv-python numpy pyyaml pytest

Run all tests from the repository root:
    python -m pytest tests/ -v

Run cleaning:
    python -m scripts.run_mask_cleaning

Expected input:
    output/segmentation/cyan.png
    output/segmentation/yellow.png
    output/segmentation/red.png
    output/segmentation/magenta.png
    output/segmentation/green.png
    output/segmentation/white_gray.png

Output:
    output/cleaned_masks/<same-filename>.png

Important:
1. Original files in output/segmentation are never overwritten.
2. Default settings are intentionally conservative.
3. Opening and directional bridging are disabled/enabled conservatively because
   aggressive morphology can erase thin details or merge adjacent drafting lines.
4. Skeletonization is exposed as a separate function. Do not replace the cleaned
   mask with skeleton output until you specifically need centerlines.
5. This stage cleans raster masks only. It does not yet infer CAD entities,
   dimensions, architectural semantics, or drawing scale.
