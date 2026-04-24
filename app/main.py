from __future__ import annotations

import base64
import binascii
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
from starlette.requests import ClientDisconnect

from app.config import Settings
from app.detector import QRCodeDetectorService
from app.errors import AppError
from app.fetch import URLFetcher
from app.schemas import DetectRequest, DetectResponse, ErrorResponse, ErrorDetail, QRCodeResult


logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _decode_base64_image(raw_value: str) -> bytes:
    try:
        return base64.b64decode(raw_value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AppError("invalid_base64", "The provided base64 image payload is invalid.", status_code=400) from exc


def _decode_data_url(raw_value: str) -> bytes:
    prefix, _, payload = raw_value.partition(",")
    if ";base64" not in prefix or not payload:
        raise AppError("invalid_data_url", "The provided data URL is not a valid base64 image.", status_code=400)
    return _decode_base64_image(payload)


async def _extract_form_image(form_data: Any) -> bytes | str:
    for key in ("file", "image"):
        candidate = form_data.get(key)
        if isinstance(candidate, UploadFile):
            return await candidate.read()
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    for value in form_data.values():
        if isinstance(value, UploadFile):
            return await value.read()

    for key in ("image", "url", "base64", "data_url", "file"):
        candidate = form_data.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    raise AppError("invalid_request", "No image input was found in the multipart form data.", status_code=400)


def _extract_json_source(payload: dict[str, Any]) -> str:
    request_payload = DetectRequest.model_validate(payload)
    for candidate in (request_payload.image, request_payload.url, request_payload.base64, request_payload.data_url):
        if candidate:
            return candidate
    raise AppError("invalid_request", "Provide one of image, url, base64, or data_url.", status_code=400)


async def _read_body(request: Request) -> bytes:
    try:
        return await request.body()
    except ClientDisconnect as exc:
        raise AppError(
            "client_disconnected",
            "The client disconnected before the request body could be read.",
            status_code=400,
        ) from exc


async def _read_request_image(request: Request, fetcher: URLFetcher) -> bytes:
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        try:
            form_data = await request.form()
        except ClientDisconnect as exc:
            raise AppError(
                "client_disconnected",
                "The client disconnected before the form data could be read.",
                status_code=400,
            ) from exc
        source = await _extract_form_image(form_data)
    elif "application/json" in content_type or not content_type:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise AppError("invalid_request", "Request body must be valid JSON, multipart form data, or raw image bytes.", status_code=400) from exc
        except ClientDisconnect as exc:
            raise AppError(
                "client_disconnected",
                "The client disconnected before the JSON body could be read.",
                status_code=400,
            ) from exc
        if not isinstance(payload, dict):
            raise AppError("invalid_request", "JSON request body must be an object.", status_code=400)
        source = _extract_json_source(payload)
    elif content_type.startswith("image/") or "application/octet-stream" in content_type:
        raw_bytes = await _read_body(request)
        if not raw_bytes:
            raise AppError("invalid_request", "Request body did not contain image bytes.", status_code=400)
        return raw_bytes
    elif content_type.startswith("text/plain"):
        raw_text = (await _read_body(request)).decode("utf-8", errors="replace").strip()
        if not raw_text:
            raise AppError("invalid_request", "Request body did not contain image input.", status_code=400)
        source = raw_text
    else:
        raw_bytes = await _read_body(request)
        if raw_bytes:
            logger.warning(
                "Treating unsupported content type as raw image bytes: %s",
                content_type or "<missing>",
            )
            return raw_bytes
        logger.warning("Rejecting unsupported content type with empty body: %s", content_type or "<missing>")
        raise AppError("unsupported_media_type", "Unsupported content type for image input.", status_code=415)

    if isinstance(source, bytes):
        return source

    stripped = source.strip()
    if stripped.startswith(("http://", "https://")):
        return await fetcher.fetch(stripped)
    if stripped.startswith("data:"):
        return _decode_data_url(stripped)
    return _decode_base64_image(stripped)


def _error_response(error: AppError) -> JSONResponse:
    payload = ErrorResponse(error=ErrorDetail(code=error.code, message=error.message))
    return JSONResponse(status_code=error.status_code, content=payload.model_dump())


def create_app(
    settings: Settings | None = None,
    *,
    detector_service: QRCodeDetectorService | None = None,
    url_fetcher: URLFetcher | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    _configure_logging(settings.log_level)
    detector_service = detector_service or QRCodeDetectorService(settings.model_dir)
    url_fetcher = url_fetcher or URLFetcher(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        detector_service.load()
        if detector_service.is_ready:
            logger.info("WeChat QR detector initialized from %s", settings.model_dir)
        else:
            logger.error("WeChat QR detector failed to initialize: %s", detector_service.load_error)
        yield

    app = FastAPI(title="QR Code Reader API", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.detector_service = detector_service
    app.state.url_fetcher = url_fetcher
    app.state.detection_semaphore = asyncio.Semaphore(settings.max_concurrent_detections)

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return _error_response(exc)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", response_model=None)
    async def readyz():
        if detector_service.is_ready:
            return {"status": "ready"}
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "error": detector_service.load_error or "Detector failed to initialize.",
            },
        )

    @app.post(
        "/v1/qrcodes/detect",
        response_model=DetectResponse,
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def detect(request: Request) -> DetectResponse:
        request_id = uuid.uuid4().hex[:12]
        started_at = time.perf_counter()
        content_type = request.headers.get("content-type", "<missing>")
        content_length = request.headers.get("content-length", "<missing>")
        logger.info(
            "QR detect request started id=%s content_type=%s content_length=%s",
            request_id,
            content_type,
            content_length,
        )
        image_bytes = await _read_request_image(request, url_fetcher)
        semaphore: asyncio.Semaphore = request.app.state.detection_semaphore
        if semaphore.locked():
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            logger.warning(
                "QR detect request rejected id=%s code=detection_busy status=429 elapsed_ms=%s",
                request_id,
                elapsed_ms,
            )
            raise AppError(
                "detection_busy",
                "QR detection is already processing another request.",
                status_code=429,
            )

        try:
            async with semaphore:
                detections = await asyncio.to_thread(
                    detector_service.detect_isolated,
                    image_bytes,
                    settings.detection_timeout_seconds,
                )
        except AppError as exc:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            logger.warning(
                "QR detect request failed id=%s code=%s status=%s elapsed_ms=%s",
                request_id,
                exc.code,
                exc.status_code,
                elapsed_ms,
            )
            raise

        results = [QRCodeResult(text=item.text, points=item.points) for item in detections]
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "QR detect request completed id=%s count=%s elapsed_ms=%s",
            request_id,
            len(results),
            elapsed_ms,
        )
        return DetectResponse(count=len(results), qrcodes=results)

    return app


app = create_app()
