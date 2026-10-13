from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.dashboard import csv_safe
from app.main import create_app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "dashboard-test-password")
    app = create_app()
    row = SimpleNamespace(event_id="test", received_at=datetime(2026, 9, 14, tzinfo=timezone.utc),
                          event_type="caption_benchmark", payload={
                              "event_id": "test", "app_version": "<script>alert(1)</script>",
                              "model_version": "=1+1", "metrics": {"is_cold_run": False,
                              "caption_latency_ms": 2500, "tokens_per_second": 12}})
    session = AsyncMock()
    session.begin = MagicMock()
    session.begin.return_value.__aenter__ = AsyncMock()
    session.begin.return_value.__aexit__ = AsyncMock(return_value=False)
    session.scalar.return_value = 1
    session.execute.return_value = MagicMock()
    session.execute.return_value.all.return_value = [row]
    async def execute(statement):
        if getattr(statement, "_distinct_on", ()):
            result = MagicMock()
            result.all.return_value = []
            return result
        return session.execute.return_value
    session.execute.side_effect = execute
    session.scalars.return_value = MagicMock()
    session.scalars.return_value.all.return_value = ["1.0.0", '<unsafe>"']
    app.state.sessions = MagicMock()
    app.state.sessions.return_value.__aenter__ = AsyncMock(return_value=session)
    app.state.sessions.return_value.__aexit__ = AsyncMock(return_value=False)
    return TestClient(app), session


@pytest.mark.parametrize("path", ["/admin/", "/admin/export?format=csv", "/admin/export?format=jsonl"])
def test_auth(client, monkeypatch, path):
    c, _ = client
    assert c.get(path).status_code == 401
    assert c.get(path, auth=("admin", "wrong")).status_code == 401
    assert c.get(path, headers={"X-Ingest-Key": "dashboard-test-password"}).status_code == 401
    monkeypatch.delenv("DASHBOARD_PASSWORD")
    assert c.get(path).status_code == 503


def test_table_and_exports(client):
    c, session = client
    c.auth = ("admin", "dashboard-test-password")
    response = c.get("/admin/?start=&end=&app_version=&include_tests=true")
    assert response.status_code == 200
    assert "&lt;script&gt;" in response.text
    assert "<script>" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert '<html lang="en">' in response.text
    for label in ("Apply filters", "Reset", "Download CSV", "Download JSONL",
                  "Start date (UTC)", "Event type", "Device model", "Model version"):
        assert label in response.text
    assert not any('\uac00' <= char <= '\ud7a3' for char in response.text)
    for key in ("event_type", "app_version", "device_model", "model_version"):
        assert f'<select name="{key}">' in response.text
    assert 'name="include_tests"' not in response.text
    assert 'No feedback yet' not in response.text
    assert '<tr class="feedback">' not in response.text
    assert '<option value="&lt;unsafe&gt;&quot;">' in response.text
    response = c.get('/admin/?app_version=1.0.0')
    assert '<option value="1.0.0" selected>' in response.text
    response = c.get('/admin/?app_version=missing')
    assert '<option value="missing" selected>' in response.text
    response = c.get("/admin/export?format=csv")
    assert response.status_code == 200
    assert "'=1+1" in response.text
    assert "metrics.caption_latency_ms" in response.text
    response = c.get("/admin/export?format=jsonl")
    assert response.json()["payload"]["metrics"]["caption_latency_ms"] == 2500
    assert c.get("/admin/?start=bad").status_code == 422
    assert c.get("/admin/?start=2026-09-15&end=2026-09-14").status_code == 422
    assert c.get("/admin/?page=0").status_code == 422
    assert c.get("/admin/export?format=bad").status_code == 422
    session.scalar.return_value = 10001
    assert c.get("/admin/export").status_code == 413


def test_linked_feedback(client):
    c, session = client
    c.auth = ("admin", "dashboard-test-password")
    feedback = SimpleNamespace(target="test", received_at=datetime(2026, 9, 15, tzinfo=timezone.utc),
        payload={"caption_feedback": {"benchmark_event_id": "test", "revision": 2,
            "is_useful": True, "reasons": ["overall_situation", "noticing_missed_details"]}})
    async def execute(statement):
        if getattr(statement, "_distinct_on", ()):
            result = MagicMock()
            result.all.return_value = [feedback]
            return result
        return session.execute.return_value
    session.execute.side_effect = execute
    response = c.get('/admin/')
    assert response.status_code == 200
    assert 'Latest feedback' in response.text
    assert 'Revision 2' in response.text
    assert 'Overall situation, Noticing missed details' in response.text
    assert response.text.index('Caption latency (ms)') < response.text.index('Latest feedback')
    row = session.execute.return_value.all.return_value[0]
    row.event_type = 'caption_feedback'
    row.payload = feedback.payload
    response = c.get('/admin/?event_type=caption_feedback')
    assert 'Benchmark ID: test' in response.text
    assert '<strong>Useful</strong>' in response.text
    assert 'Latest feedback ·' not in response.text


def test_csv_formula_protection():
    for value in ["=1+1", "+SUM(1)", "-1+1", "@SUM(1)", "\t=1", "\r=1"]:
        assert csv_safe(value).startswith("'")
    assert csv_safe(-1001) == -1001
    assert csv_safe("normal") == "normal"
