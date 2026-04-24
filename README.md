# QR Code Reader API

Dockerized FastAPI service for reading QR codes with OpenCV WeChat QRCode.

Published image:

```text
ghcr.io/lesenfantsdemacgyver/qrcodereader_api:latest
```

## Run

```bash
cp .env.example .env
docker compose up -d --build
```

Service URL: `http://localhost:8000`

If you do not want to build locally, use the published image in your own compose file:

```yaml
  qrcodereader-api:
    image: ghcr.io/lesenfantsdemacgyver/qrcodereader_api:latest
    container_name: qrcodereader-api
    restart: unless-stopped
```

## Use

Health check:

```bash
curl http://localhost:8000/healthz
```

Detect QR codes from a file:

```bash
curl -X POST http://localhost:8000/v1/qrcodes/detect \
  -F "file=@image.png"
```

Detect QR codes from JSON base64 or URL:

```bash
curl -X POST http://localhost:8000/v1/qrcodes/detect \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/qr.png"}'
```

Response shape:

```json
{
  "count": 1,
  "qrcodes": [
    {
      "text": "hello-world",
      "points": [[14.0, 14.0], [142.0, 14.0], [142.0, 142.0], [14.0, 142.0]]
    }
  ]
}
```

Large images can take time to scan. The service runs one detection at a time by default and returns `429 detection_busy` if another request arrives while a scan is active.

## Stop

```bash
docker compose down
```

## Test

```bash
docker build --target test -t qrcodereader-api-test .
docker run --rm qrcodereader-api-test
```
