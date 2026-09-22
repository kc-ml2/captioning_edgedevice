"""Write-only telemetry API. Never accept images, captions, or arbitrary logs."""
import hmac
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import DateTime, String, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID, insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from starlette.responses import JSONResponse

MAX_BODY_BYTES = 256 * 1024


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Metrics(StrictModel):
    is_cold_run: bool
    cold_load_ms: float | None = Field(default=None, ge=0)
    caption_latency_ms: float = Field(ge=0)
    time_to_first_token_ms: float | None = Field(default=None, ge=0)
    generation_ms: float | None = Field(default=None, ge=0)
    generated_tokens: int = Field(ge=0)
    tokens_per_second: float | None = Field(default=None, ge=0)
    peak_sampled_memory_bytes: int = Field(ge=0)
    output_words: int | None = Field(default=None, ge=0)
    # Legacy clients and persisted queues still send character counts.
    output_characters: int | None = Field(default=None, ge=0)
    thermal_state: Literal["nominal", "fair", "serious", "critical", "unknown"]

    @model_validator(mode="after")
    def validate_output_length(self):
        if self.output_words is None and self.output_characters is None:
            raise ValueError("output_words or legacy output_characters is required")
        return self


class DownloadFailure(StrictModel):
    elapsed_ms: float = Field(ge=0)
    downloaded_bytes: int = Field(ge=0)
    error_code: int
    http_status: int | None = Field(default=None, ge=100, le=599)
    retry_count: int = Field(default=0, ge=0)
    error_category: Literal["network", "cocoa", "posix", "application"] = "application"
    app_state: Literal["active", "inactive", "background", "unknown"]


class CaptionFailure(StrictModel):
    elapsed_ms: float = Field(ge=0)
    error_code: int
    error_category: Literal["network", "cocoa", "posix", "application"]
    app_state: Literal["active", "inactive", "background", "unknown"]
    stage: Literal["capture", "inference"]


ShortString = Annotated[str, Field(min_length=1, max_length=128)]


class Event(StrictModel):
    schema_version: Literal[1]
    event_id: UUID
    event_type: Literal["caption_benchmark", "download_failure", "caption_failure"]
    timestamp: datetime
    app_version: ShortString
    build_number: ShortString
    device_model: ShortString
    os_version: ShortString
    model_version: ShortString
    metrics: Metrics | None = None
    download_failure: DownloadFailure | None = None
    caption_failure: CaptionFailure | None = None

    @model_validator(mode="after")
    def validate_payload(self):
        if self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        payloads = {"caption_benchmark": self.metrics,
                    "download_failure": self.download_failure,
                    "caption_failure": self.caption_failure}
        if payloads[self.event_type] is None or sum(value is not None for value in payloads.values()) != 1:
            raise ValueError("event_type must match exactly one payload")
        return self


class Batch(StrictModel):
    events: list[Event] = Field(min_length=1, max_length=100)


class Base(DeclarativeBase):
    pass


class StoredEvent(Base):
    __tablename__ = "telemetry_events"
    event_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    payload: Mapped[dict] = mapped_column(JSONB)


class BodyLimitMiddleware:
    """Bound actual streamed bytes, including requests without Content-Length."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            size += len(body)
            if size > MAX_BODY_BYTES:
                return await JSONResponse({"detail": "Request too large"}, 413)(scope, receive, send)
            chunks.append(body)
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": b"".join(chunks), "more_body": False}

        await self.app(scope, bounded_receive, send)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        key = os.environ.get("INGEST_API_KEY", "")
        if len(key) < 32:
            raise RuntimeError("INGEST_API_KEY must be at least 32 characters")
        app.state.ingest_key = key
        engine = create_async_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
        app.state.sessions = async_sessionmaker(engine)
        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            yield
        finally:
            await engine.dispose()

    app = FastAPI(title="SceneSense telemetry", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BodyLimitMiddleware)

    async def authenticate(request: Request, x_ingest_key: str = Header(default="")):
        if not hmac.compare_digest(x_ingest_key.encode(), request.app.state.ingest_key.encode()):
            raise HTTPException(status_code=401, detail="Invalid ingestion key")

    @app.get("/healthz")
    async def health(request: Request):
        async with request.app.state.sessions() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "ok"}

    @app.post("/v1/events/batch", dependencies=[Depends(authenticate)])
    async def ingest(batch: Batch, request: Request):
        # Same UUID is accepted once. Replays never overwrite previously stored data.
        unique = {event.event_id: event for event in batch.events}
        rows = [{"event_id": event.event_id, "event_type": event.event_type,
                 "payload": event.model_dump(mode="json", exclude_none=True)}
                for event in unique.values()]
        statement = insert(StoredEvent).values(rows).on_conflict_do_nothing(
            index_elements=[StoredEvent.event_id]
        ).returning(StoredEvent.event_id)
        async with request.app.state.sessions() as session:
            async with session.begin():
                inserted = len((await session.execute(statement)).all())
        return {"received": len(batch.events), "inserted": inserted,
                "duplicates": len(batch.events) - inserted,
                "acknowledged_event_ids": [str(event_id) for event_id in unique]}

    return app


app = create_app()
