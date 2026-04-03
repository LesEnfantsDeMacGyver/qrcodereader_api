from __future__ import annotations

import io
from pathlib import Path

import pytest
import qrcode
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent


def _make_qr_image(text: str, image_format: str = "PNG") -> bytes:
    image = qrcode.make(text).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def _make_multi_qr_image(texts: list[str]) -> bytes:
    qr_images = [qrcode.make(text).convert("RGB").resize((180, 180)) for text in texts]
    canvas = Image.new("RGB", (420, 220), color="white")
    canvas.paste(qr_images[0], (10, 20))
    canvas.paste(qr_images[1], (220, 20))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def sample_png_bytes() -> bytes:
    return _make_qr_image("png-qr", image_format="PNG")


@pytest.fixture(scope="session")
def sample_jpeg_bytes() -> bytes:
    return _make_qr_image("jpeg-qr", image_format="JPEG")


@pytest.fixture(scope="session")
def sample_multi_qr_bytes() -> bytes:
    return _make_multi_qr_image(["left-qr", "right-qr"])


@pytest.fixture(scope="session")
def blank_png_bytes() -> bytes:
    image = Image.new("RGB", (200, 200), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
