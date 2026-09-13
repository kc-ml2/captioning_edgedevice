import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import Event, MAX_BODY_BYTES, create_app


def sample():
    return {
        "schema_version": 1, "event_id": str(uuid4()),
        "event_type": "caption_benchmark", "timestamp": "2026-09-13T14:30:00Z",
        "app_version": "1.0", "build_number": "3", "device_model": "iPhone17,3",
        "os_version": "26.5", "model_version": "test-model",
        "metrics": {"is_cold_run": False, "caption_latency_ms": 2450,
                    "generated_tokens": 28, "tokens_per_second": 20.74,
                    "peak_sampled_memory_bytes": 2100000000,
                    "output_characters": 96, "thermal_state": "nominal"},
    }


def test_schema():
    Event.model_validate(sample())
    for change in [{"caption": "private text"}, {"timestamp": "2026-09-13T14:30:00"},
                   {"event_type": "download_failure"}]:
        with pytest.raises(ValidationError):
            Event.model_validate(sample() | change)
    event = sample()
    event["metrics"]["tokens_per_second"] = float("nan")
    with pytest.raises(ValidationError):
        Event.model_validate(event)


def test_failure_payloads():
    event = sample()
    del event["metrics"]
    details = {"elapsed_ms": 100, "error_code": -1001,
               "error_category": "network", "app_state": "active"}
    Event.model_validate(event | {"event_type": "download_failure",
                                 "download_failure": details | {"downloaded_bytes": 1000}})
    caption = event | {"event_type": "caption_failure",
                       "caption_failure": details | {"stage": "inference"}}
    Event.model_validate(caption)
    with pytest.raises(ValidationError):
        Event.model_validate(caption | {"download_failure": details | {"downloaded_bytes": 0}})
    with pytest.raises(ValidationError):
        Event.model_validate(caption | {"caption_failure": details | {"stage": "inference", "message": "private"}})


def test_auth_and_body_limit():
    app = create_app()
    app.state.ingest_key = "a" * 32
    client = TestClient(app)
    assert client.post("/v1/events/batch", json={"events": [sample()]}).status_code == 401
    assert client.post("/v1/events/batch", content=b"x" * (MAX_BODY_BYTES + 1)).status_code == 413
    assert client.post("/v1/events/batch", headers={"X-Ingest-Key": "a" * 32},
                       json={"events": []}).status_code == 422
    assert client.post("/v1/events/batch", headers={"X-Ingest-Key": "a" * 32},
                       json={"events": [sample()] * 101}).status_code == 422


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="Requires disposable PostgreSQL")
def test_postgres_replay():
    os.environ.setdefault("INGEST_API_KEY", "a" * 32)
    with TestClient(create_app()) as client:
        headers = {"X-Ingest-Key": os.environ["INGEST_API_KEY"]}
        event = sample()
        assert client.get("/healthz").status_code == 200
        response = client.post("/v1/events/batch", headers=headers, json={"events": [event, event]})
        assert response.status_code == 200
        assert response.json()["inserted"] == 1
        assert response.json()["duplicates"] == 1
        response = client.post("/v1/events/batch", headers=headers, json={"events": [event]})
        assert response.json()["inserted"] == 0
        assert response.json()["acknowledged_event_ids"] == [event["event_id"]]
