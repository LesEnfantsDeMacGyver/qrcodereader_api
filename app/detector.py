from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.errors import AppError


@dataclass(slots=True)
class Detection:
    text: str
    points: list[list[float]]


def _normalize_points(points: Any, index: int) -> list[list[float]]:
    if points is None:
        return []

    points_array = np.asarray(points, dtype=np.float32)
    if points_array.ndim == 2 and points_array.shape == (4, 2):
        selected = points_array
    elif points_array.ndim >= 3:
        selected = points_array[index]
    else:
        selected = np.empty((0, 2), dtype=np.float32)
    return [[float(x), float(y)] for x, y in selected.tolist()]


def required_model_paths(model_dir: Path) -> dict[str, Path]:
    return {
        "detect_prototxt": model_dir / "detect.prototxt",
        "detect_caffemodel": model_dir / "detect.caffemodel",
        "sr_prototxt": model_dir / "sr.prototxt",
        "sr_caffemodel": model_dir / "sr.caffemodel",
    }


def create_wechat_detector(model_dir: Path) -> cv2.wechat_qrcode.WeChatQRCode:
    model_paths = required_model_paths(model_dir)
    missing_paths = [str(path) for path in model_paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing model files: {', '.join(missing_paths)}")

    return cv2.wechat_qrcode.WeChatQRCode(
        str(model_paths["detect_prototxt"]),
        str(model_paths["detect_caffemodel"]),
        str(model_paths["sr_prototxt"]),
        str(model_paths["sr_caffemodel"]),
    )


class QRCodeDetectorService:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._detector: cv2.wechat_qrcode.WeChatQRCode | None = None
        self._load_error: str | None = None

    def load(self) -> None:
        try:
            self._detector = create_wechat_detector(self.model_dir)
            self._load_error = None
        except Exception as exc:  # pragma: no cover - exercised via readyz
            self._detector = None
            self._load_error = str(exc)

    @property
    def is_ready(self) -> bool:
        return self._detector is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def detect(self, image_bytes: bytes) -> list[Detection]:
        if self._detector is None:
            raise AppError("service_unavailable", "The QR detector is not ready.", status_code=503)

        image = decode_image_bytes(image_bytes)
        texts, points = self._detector.detectAndDecode(image)

        if not texts:
            return []

        return [
            Detection(text=text, points=_normalize_points(points, index))
            for index, text in enumerate(texts)
        ]


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise AppError("invalid_image", "Could not decode an image from the provided input.", status_code=400)

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise AppError("invalid_image", "Could not decode an image from the provided input.", status_code=400)
    return image
