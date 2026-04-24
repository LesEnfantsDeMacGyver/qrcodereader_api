from __future__ import annotations

import ipaddress
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.errors import AppError
from app.fetch import URLFetcher


ROOT_DIR = Path(__file__).resolve().parent.parent


def _settings() -> Settings:
    return Settings(
        model_dir=ROOT_DIR / "models",
        max_image_bytes=10 * 1024 * 1024,
        url_fetch_timeout_seconds=10,
        detection_timeout_seconds=30,
        max_concurrent_detections=1,
        allow_private_urls=False,
        log_level="INFO",
    )


async def _resolver_for(*raw_addresses: str):
    return [ipaddress.ip_address(address) for address in raw_addresses]


@pytest.mark.anyio
async def test_rejects_file_scheme() -> None:
    fetcher = URLFetcher(_settings(), resolver=lambda _: _resolver_for("93.184.216.34"))
    with pytest.raises(AppError) as exc:
        await fetcher.fetch("file:///tmp/test.png")
    assert exc.value.code == "invalid_url"


@pytest.mark.anyio
async def test_rejects_localhost() -> None:
    fetcher = URLFetcher(_settings(), resolver=lambda _: _resolver_for("127.0.0.1"))
    with pytest.raises(AppError) as exc:
        await fetcher.fetch("http://localhost/qr.png")
    assert exc.value.code == "invalid_url"


@pytest.mark.anyio
async def test_rejects_private_ip() -> None:
    fetcher = URLFetcher(_settings(), resolver=lambda _: _resolver_for("10.0.0.12"))
    with pytest.raises(AppError) as exc:
        await fetcher.fetch("https://internal.example/qr.png")
    assert exc.value.code == "invalid_url"


@pytest.mark.anyio
async def test_accepts_public_https() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"image-bytes", request=request)
    )
    fetcher = URLFetcher(
        _settings(),
        transport=transport,
        resolver=lambda _: _resolver_for("93.184.216.34"),
    )
    assert await fetcher.fetch("https://example.com/qr.png") == b"image-bytes"
