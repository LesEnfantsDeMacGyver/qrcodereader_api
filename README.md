# QR Code Reader API

Dockerized FastAPI service for reading QR codes with OpenCV WeChat QRCode.

## Run

```bash
cp .env.example .env
docker compose up -d --build
```

Service URL: `http://localhost:8000`

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

## Stop

```bash
docker compose down
```

## Test

```bash
docker build --target test -t qrcodereader-api-test .
docker run --rm qrcodereader-api-test
```
