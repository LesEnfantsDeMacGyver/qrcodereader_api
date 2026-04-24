from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_readyz_reports_missing_models(tmp_path) -> None:
    settings = Settings(
        model_dir=tmp_path,
        max_image_bytes=10 * 1024 * 1024,
        url_fetch_timeout_seconds=10,
        detection_timeout_seconds=30,
        max_concurrent_detections=1,
        allow_private_urls=False,
        log_level="INFO",
    )
    app = create_app(settings=settings)
    with TestClient(app) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert "Missing model files" in response.json()["error"]


def test_readyz_reports_ready_with_repo_models() -> None:
    settings = Settings(
        model_dir=ROOT_DIR / "models",
        max_image_bytes=10 * 1024 * 1024,
        url_fetch_timeout_seconds=10,
        detection_timeout_seconds=30,
        max_concurrent_detections=1,
        allow_private_urls=False,
        log_level="INFO",
    )
    app = create_app(settings=settings)
    with TestClient(app) as client:
        response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
