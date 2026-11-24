
from pathlib import Path

import cv2
import numpy as np


def load_image(image_path):
    """
    Load raster image dengan dukungan path Unicode Windows.
    Mengembalikan gambar dalam format BGR.
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image tidak ditemukan: {path}"
        )

    buffer = np.fromfile(str(path), dtype=np.uint8)

    image = cv2.imdecode(
        buffer,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            f"Format gambar tidak valid: {path}"
        )

    return image


def validate_image(image):
    """
    Memeriksa validitas raster image.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image harus berupa NumPy array")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image harus memiliki 3 channel BGR")

    if image.size == 0:
        raise ValueError("Image kosong")

    if not np.isfinite(image).all():
        raise ValueError("Image mengandung nilai tidak valid")

    return True


def resize_image(image, max_dimension=3000):
    """
    Mengubah ukuran gambar tanpa mengubah rasio aspek.
    Tidak memperbesar gambar yang lebih kecil.
    """

    validate_image(image)

    if max_dimension < 1:
        raise ValueError("max_dimension harus positif")

    height, width = image.shape[:2]

    longest = max(height, width)

    if longest <= max_dimension:
        return image.copy()

    scale = max_dimension / longest

    new_width = round(width * scale)
    new_height = round(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def convert_to_hsv(image):
    """
    Mengubah format BGR menjadi HSV
    untuk color segmentation.
    """

    validate_image(image)

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


def save_image(image, output_path):
    """
    Menyimpan hasil preprocessing.
    """

    validate_image(image)

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    extension = path.suffix or ".png"

    success, encoded = cv2.imencode(
        extension,
        image
    )

    if not success:
        raise ValueError("Gagal meng-encode image")

    encoded.tofile(str(path))

    return path
