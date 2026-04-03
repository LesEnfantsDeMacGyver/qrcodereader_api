from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class QRCodeResult(BaseModel):
    text: str
    points: list[list[float]]


class DetectResponse(BaseModel):
    count: int
    qrcodes: list[QRCodeResult]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    image: str | None = None
    url: str | None = None
    base64: str | None = None
    data_url: str | None = None
