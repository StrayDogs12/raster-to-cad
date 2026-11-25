
from vision.preprocessing import load_image
from vision.color_segmentation import (
    load_color_config,
    segment_colors,
    save_masks,
    mask_statistics,
)

image = load_image("input/waterloo_section.png")

colors = load_color_config("config/colors.yaml")

masks = segment_colors(image, colors)

saved = save_masks(
    masks,
    "output/segmentation"
)

stats = mask_statistics(masks)

print("\nSEGMENTATION RESULTS")

for name, values in stats.items():
    print(
        f"{name:12} "
        f"{values['pixels']:10} pixels "
        f"{values['percentage']:8.4f}%"
    )

print("\nSAVED MASKS")

for name, path in saved.items():
    print(name, "->", path)

print("\nSEGMENTATION COMPLETED")
