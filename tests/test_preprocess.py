import pytest

from tests.conftest import image_bytes
from visionguard.preprocess import ImageDecodeError, preprocess_bytes


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "BMP", "GIF", "WEBP"])
def test_output_shape_and_range(fmt):
    x = preprocess_bytes(image_bytes(fmt, size=(300, 120)), img_size=224)
    assert x.shape == (224, 224, 3)
    assert x.dtype.name == "float32"
    assert 0 <= x.min() and x.max() <= 255


def test_greyscale_and_alpha_become_rgb():
    assert preprocess_bytes(image_bytes("PNG", mode="L")).shape[-1] == 3
    assert preprocess_bytes(image_bytes("PNG", mode="RGBA")).shape[-1] == 3


def test_garbage_raises():
    with pytest.raises(ImageDecodeError):
        preprocess_bytes(b"definitely not an image")


def test_truncated_jpeg_raises():
    raw = image_bytes("JPEG", size=(400, 400))
    with pytest.raises(ImageDecodeError):
        preprocess_bytes(raw[: len(raw) // 3])
