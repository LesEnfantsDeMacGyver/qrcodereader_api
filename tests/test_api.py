from __future__ import annotations

import base64
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.fetch import URLFetcher
from app.main import create_app


ROOT_DIR = Path(__file__).resolve().parent.parent


async def _public_resolver(_: str):
    return []


@pytest.fixture
def settings() -> Settings:
    return Settings(
        model_dir=ROOT_DIR / "models",
        max_image_bytes=10 * 1024 * 1024,
        url_fetch_timeout_seconds=10,
        allow_private_urls=False,
        log_level="INFO",
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    app = create_app(settings=settings)
    with TestClient(app) as test_client:
        yield test_client


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_with_models(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_detect_png_multipart_file(client: TestClient, sample_png_bytes: bytes) -> None:
    response = client.post(
        "/v1/qrcodes/detect",
        files={"file": ("qr.png", sample_png_bytes, "image/png")},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["count"] == 1
    assert body["qrcodes"][0]["text"] == "png-qr"
    assert len(body["qrcodes"][0]["points"]) == 4


def test_detect_jpeg_multipart_image_alias(client: TestClient, sample_jpeg_bytes: bytes) -> None:
    response = client.post(
        "/v1/qrcodes/detect",
        files={"image": ("qr.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["count"] == 1
    assert body["qrcodes"][0]["text"] == "jpeg-qr"


def test_detect_json_base64_field(client: TestClient, sample_png_bytes: bytes) -> None:
    response = client.post(
        "/v1/qrcodes/detect",
        json={"base64": base64.b64encode(sample_png_bytes).decode("ascii")},
    )
    assert response.status_code == 200
    assert response.json()["qrcodes"][0]["text"] == "png-qr"


def test_detect_json_data_url(client: TestClient, sample_png_bytes: bytes) -> None:
    payload = f"data:image/png;base64,{base64.b64encode(sample_png_bytes).decode('ascii')}"
    response = client.post("/v1/qrcodes/detect", json={"data_url": payload})
    assert response.status_code == 200
    assert response.json()["qrcodes"][0]["text"] == "png-qr"


def test_detect_json_image_auto_detect_base64(client: TestClient, sample_png_bytes: bytes) -> None:
    response = client.post(
        "/v1/qrcodes/detect",
        json={"image": base64.b64encode(sample_png_bytes).decode("ascii")},
    )
    assert response.status_code == 200
    assert response.json()["qrcodes"][0]["text"] == "png-qr"


def test_detect_json_image_auto_detect_data_url(client: TestClient, sample_png_bytes: bytes) -> None:
    payload = f"data:image/png;base64,{base64.b64encode(sample_png_bytes).decode('ascii')}"
    response = client.post("/v1/qrcodes/detect", json={"image": payload})
    assert response.status_code == 200
    assert response.json()["qrcodes"][0]["text"] == "png-qr"


def test_detect_json_image_auto_detect_url(settings: Settings, sample_png_bytes: bytes) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=sample_png_bytes, request=request)
    )
    fetcher = URLFetcher(settings, transport=transport, resolver=_public_resolver)
    app = create_app(settings=settings, url_fetcher=fetcher)
    with TestClient(app) as client:
        response = client.post("/v1/qrcodes/detect", json={"image": "https://example.com/qr.png"})
    assert response.status_code == 200
    assert response.json()["qrcodes"][0]["text"] == "png-qr"


def test_detect_text_plain_url(settings: Settings, sample_png_bytes: bytes) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=sample_png_bytes, request=request)
    )
    fetcher = URLFetcher(settings, transport=transport, resolver=_public_resolver)
    app = create_app(settings=settings, url_fetcher=fetcher)
    with TestClient(app) as client:
        response = client.post(
            "/v1/qrcodes/detect",
            content="https://example.com/qr.png",
            headers={"content-type": "text/plain"},
        )
    assert response.status_code == 200
    assert response.json()["qrcodes"][0]["text"] == "png-qr"


def test_detect_multiple_qrcodes(client: TestClient, sample_multi_qr_bytes: bytes) -> None:
    response = client.post(
        "/v1/qrcodes/detect",
        files={"file": ("multi.png", sample_multi_qr_bytes, "image/png")},
    )
    assert response.status_code == 200
    texts = sorted(item["text"] for item in response.json()["qrcodes"])
    assert texts == ["left-qr", "right-qr"]


def test_detect_no_qrcodes_returns_empty_list(client: TestClient, blank_png_bytes: bytes) -> None:
    response = client.post(
        "/v1/qrcodes/detect",
        files={"file": ("blank.png", blank_png_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert response.json() == {"count": 0, "qrcodes": []}


def test_detect_invalid_base64_returns_client_error(client: TestClient) -> None:
    response = client.post("/v1/qrcodes/detect", json={"base64": "%%%not-base64%%%"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_base64"


def test_detect_non_image_bytes_returns_client_error(client: TestClient) -> None:
    response = client.post(
        "/v1/qrcodes/detect",
        files={"file": ("bad.bin", b"not-an-image", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_image"


def test_detect_rejects_oversized_fetched_image(settings: Settings, sample_png_bytes: bytes) -> None:
    tiny_settings = Settings(
        model_dir=settings.model_dir,
        max_image_bytes=32,
        url_fetch_timeout_seconds=settings.url_fetch_timeout_seconds,
        allow_private_urls=settings.allow_private_urls,
        log_level=settings.log_level,
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-length": str(len(sample_png_bytes))},
            content=sample_png_bytes,
            request=request,
        )
    )
    fetcher = URLFetcher(tiny_settings, transport=transport, resolver=_public_resolver)
    app = create_app(settings=tiny_settings, url_fetcher=fetcher)
    with TestClient(app) as client:
        response = client.post("/v1/qrcodes/detect", json={"url": "https://example.com/qr.png"})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "image_too_large"


def test_detect_raw_image_bytes(client: TestClient, sample_png_bytes: bytes) -> None:
    response = client.post(
        "/v1/qrcodes/detect",
        content=sample_png_bytes,
        headers={"content-type": "image/png"},
    )
    assert response.status_code == 200
    assert response.json()["qrcodes"][0]["text"] == "png-qr"
