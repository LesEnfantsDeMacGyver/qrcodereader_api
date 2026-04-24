from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_bool(raw_value: str | None, default: bool) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    model_dir: Path
    max_image_bytes: int
    url_fetch_timeout_seconds: float
    detection_timeout_seconds: float
    max_concurrent_detections: int
    allow_private_urls: bool
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        root_dir = Path(__file__).resolve().parent.parent
        default_model_dir = root_dir / "models"
        return cls(
            model_dir=Path(os.getenv("WECHAT_MODEL_DIR", str(default_model_dir))).resolve(),
            max_image_bytes=int(os.getenv("MAX_IMAGE_BYTES", "10485760")),
            url_fetch_timeout_seconds=float(os.getenv("URL_FETCH_TIMEOUT_SECONDS", "10")),
            detection_timeout_seconds=float(os.getenv("DETECTION_TIMEOUT_SECONDS", "30")),
            max_concurrent_detections=int(os.getenv("MAX_CONCURRENT_DETECTIONS", "1")),
            allow_private_urls=_parse_bool(os.getenv("ALLOW_PRIVATE_URLS"), default=False),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
