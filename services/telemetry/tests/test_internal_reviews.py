import base64
import io
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from app.main import create_app
from app.internal_reviews import Review, private_image, MAX_INTERNAL_BODY_BYTES


def sample():
    return {"review_id": str(uuid4()), "benchmark_event_id": str(uuid4()),
        "caption": "A chair.", "comment": "Missed the table.", "model_version": "test"}


def test_schema():
    Review.model_validate(sample())
    for change in [{"comment": " "}, {"comment": "x" * 5001}, {"extra": "bad"}, {"feedback": {"is_useful": True}}, {"is_useful": True}]:
        with pytest.raises(ValidationError):
            Review.model_validate(sample() | change)


def test_image_validation():
    output = io.BytesIO()
    Image.new("RGB", (10, 10)).save(output, format="JPEG", exif=b"Exif\x00\x00private")
    cleaned = private_image(base64.b64encode(output.getvalue()).decode())
    assert b"private" not in cleaned
    for value in ["bad base64", base64.b64encode(b"not an image").decode()]:
        with pytest.raises(HTTPException):
            private_image(value)


def test_auth_and_limits(monkeypatch):
    monkeypatch.setenv("INGEST_API_KEY", "u" * 32)
    monkeypatch.setenv("DASHBOARD_READ_API_KEY", "r" * 32)
    client = TestClient(create_app())
    assert client.post("/v1/internal/reviews", json=sample()).status_code == 401
    assert client.post("/v1/internal/reviews", headers={"X-Internal-Review-Key": "u" * 32},
                       content=b"x" * (MAX_INTERNAL_BODY_BYTES + 1)).status_code == 413
    path = f"/v1/internal/benchmarks/{uuid4()}"
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"X-Dashboard-Key": "u" * 32}).status_code == 401
    assert client.get(f"/v1/internal/reviews/{uuid4()}/image").status_code == 401


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("INGEST_API_KEY", raising=False)
    client = TestClient(create_app())
    assert client.post("/v1/internal/reviews", json=sample()).status_code == 503
