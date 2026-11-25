
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_color_config(config_path):
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid color configuration")

    colors = config.get("colors")

    if not isinstance(colors, dict) or not colors:
        raise ValueError("Missing colors configuration")

    for name, settings in colors.items():
        ranges = settings.get("ranges")

        if not isinstance(ranges, list) or not ranges:
            raise ValueError(f"Invalid ranges: {name}")

        for item in ranges:
            lower = np.asarray(item["lower"])
            upper = np.asarray(item["upper"])

            if lower.shape != (3,) or upper.shape != (3,):
                raise ValueError(f"Invalid HSV range: {name}")

            if np.any(lower > upper):
                raise ValueError(f"Lower exceeds upper: {name}")

            if (
                lower[0] < 0 or upper[0] > 179
                or lower[1] < 0 or upper[1] > 255
                or lower[2] < 0 or upper[2] > 255
            ):
                raise ValueError(f"HSV out of bounds: {name}")

    return colors


def segment_colors(image, colors):
    """
    Segmentasi warna berdasarkan HSV.
    Output setiap warna berupa binary mask uint8.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be BGR with 3 channels")

    if image.size == 0:
        raise ValueError("Image is empty")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    masks = {}

    for name, settings in colors.items():
        combined = np.zeros(
            image.shape[:2],
            dtype=np.uint8
        )

        for item in settings["ranges"]:
            lower = np.array(
                item["lower"],
                dtype=np.uint8
            )

            upper = np.array(
                item["upper"],
                dtype=np.uint8
            )

            mask = cv2.inRange(
                hsv,
                lower,
                upper
            )

            combined = cv2.bitwise_or(
                combined,
                mask
            )

        masks[name] = combined

    return masks


def save_masks(masks, output_dir):
    """
    Menyimpan setiap binary mask sebagai PNG.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    saved = {}

    for name, mask in masks.items():
        path = output_dir / f"{name}.png"

        success, encoded = cv2.imencode(
            ".png",
            mask
        )

        if not success:
            raise ValueError(f"Failed to encode {name}")

        encoded.tofile(str(path))
        saved[name] = path

    return saved


def mask_statistics(masks):
    """
    Menghitung jumlah piksel terdeteksi dan persentasenya.
    """

    if not masks:
        raise ValueError("No masks provided")

    first = next(iter(masks.values()))
    total = first.size

    if total == 0:
        raise ValueError("Empty mask")

    stats = {}

    for name, mask in masks.items():
        if mask.shape != first.shape:
            raise ValueError("Mask dimensions do not match")

        pixels = int(np.count_nonzero(mask))

        stats[name] = {
            "pixels": pixels,
            "percentage": round(
                pixels / total * 100, 4
            )
        }

    return stats
