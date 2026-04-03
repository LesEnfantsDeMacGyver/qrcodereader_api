from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Awaitable, Callable
from urllib.parse import urljoin, urlparse

import httpx

from app.config import Settings
from app.errors import AppError

Resolver = Callable[[str], Awaitable[list[ipaddress._BaseAddress]]]


def _is_blocked_address(address: ipaddress._BaseAddress) -> bool:
    return any(
        (
            address.is_private,
            address.is_loopback,
            address.is_link_local,
            address.is_multicast,
            address.is_reserved,
            address.is_unspecified,
        )
    )


async def resolve_hostname(hostname: str) -> list[ipaddress._BaseAddress]:
    def _lookup() -> list[ipaddress._BaseAddress]:
        resolved = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        return [ipaddress.ip_address(item[4][0]) for item in resolved]

    return await asyncio.to_thread(_lookup)


class URLFetcher:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: Resolver = resolve_hostname,
        max_redirects: int = 5,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.resolver = resolver
        self.max_redirects = max_redirects

    async def _validate_url(self, raw_url: str) -> str:
        parsed = urlparse(raw_url)
        if parsed.scheme not in {"http", "https"}:
            raise AppError("invalid_url", "Only http and https URLs are allowed.", status_code=400)
        if not parsed.netloc or not parsed.hostname:
            raise AppError("invalid_url", "A valid URL is required.", status_code=400)
        if self.settings.allow_private_urls:
            return raw_url

        try:
            resolved_addresses = await self.resolver(parsed.hostname)
        except OSError as exc:
            raise AppError("invalid_url", "Could not resolve the remote host.", status_code=400) from exc
        if any(_is_blocked_address(address) for address in resolved_addresses):
            raise AppError(
                "invalid_url",
                "Remote URLs must not resolve to private or local network addresses.",
                status_code=400,
            )
        return raw_url

    async def fetch(self, raw_url: str) -> bytes:
        current_url = await self._validate_url(raw_url)
        timeout = httpx.Timeout(self.settings.url_fetch_timeout_seconds)
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=timeout,
            transport=self.transport,
        ) as client:
            for _ in range(self.max_redirects + 1):
                try:
                    response = await client.get(current_url)
                except httpx.HTTPError as exc:
                    raise AppError("invalid_url", "Failed to fetch the remote image.", status_code=400) from exc
                if response.is_redirect:
                    redirect_target = response.headers.get("location")
                    if not redirect_target:
                        raise AppError("invalid_url", "Redirect response did not include a location.", status_code=400)
                    current_url = await self._validate_url(urljoin(current_url, redirect_target))
                    continue

                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise AppError("invalid_url", "Remote URL did not return a successful response.", status_code=400) from exc
                image_bytes = await self._read_limited(response)
                if not image_bytes:
                    raise AppError("invalid_image", "Remote URL returned an empty response body.", status_code=400)
                return image_bytes

        raise AppError("invalid_url", "Too many redirects while fetching the remote image.", status_code=400)

    async def _read_limited(self, response: httpx.Response) -> bytes:
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > self.settings.max_image_bytes:
            raise AppError("image_too_large", "Image exceeds the configured size limit.", status_code=413)

        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > self.settings.max_image_bytes:
                raise AppError("image_too_large", "Image exceeds the configured size limit.", status_code=413)
            chunks.append(chunk)
        return b"".join(chunks)
