"""The ONE preprocessing function used by training, evaluation, the model API and
the web app. Keeping it in a single place guarantees that serving sees pixels
prepared exactly like the pixels the model was trained on.

Steps: decode bytes with Pillow -> apply EXIF rotation -> first frame only
(animated GIF) -> force RGB -> centre-crop to a square -> resize to img_size
with bilinear filtering -> float32 array in [0, 255].

Model-specific scaling (e.g. MobileNetV2's [-1, 1]) lives INSIDE the Keras model
as a layer, so callers never have to remember it.
"""
import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from visionguard.config import IMG_SIZE


class ImageDecodeError(ValueError):
    """Raised when bytes cannot be decoded as an image."""


def decode_image(raw: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()  # force full decode; truncated/corrupt files fail here
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError) as exc:
        raise ImageDecodeError(f"cannot decode image: {exc}") from exc
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        # RGBA/LA/P with transparency: composite on white instead of black
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
    return img


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def preprocess_pil(img: Image.Image, img_size: int = IMG_SIZE) -> np.ndarray:
    img = center_crop_square(img)
    img = img.resize((img_size, img_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def preprocess_bytes(raw: bytes, img_size: int = IMG_SIZE) -> np.ndarray:
    """bytes -> float32 array of shape (img_size, img_size, 3), values 0..255."""
    return preprocess_pil(decode_image(raw), img_size)


def preprocess_path(path, img_size: int = IMG_SIZE) -> np.ndarray:
    with open(path, "rb") as f:
        return preprocess_bytes(f.read(), img_size)
