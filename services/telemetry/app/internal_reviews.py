"""Internal-only review collection. Files are never served as public static assets."""
import base64
import hashlib
import hmac
import io
import os
import tempfile
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from PIL import Image, UnidentifiedImageError
from pydantic import Field, model_validator
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from starlette.responses import Response

from .main import Base, StoredEvent, StrictModel
from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_INTERNAL_BODY_BYTES = 6 * 1024 * 1024


class Review(StrictModel):
    review_id: UUID
    benchmark_event_id: UUID
    caption: str = Field(min_length=1, max_length=10000)
    comment: str = Field(min_length=1, max_length=5000)
    model_version: str = Field(min_length=1, max_length=128)
    image_base64: str | None = Field(default=None, max_length=5600000)

    @model_validator(mode="after")
    def nonblank(self):
        if not self.comment.strip() or not self.caption.strip():
            raise ValueError("caption and comment must not be blank")
        return self


class StoredReview(Base):
    __tablename__ = "internal_caption_reviews"
    review_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    benchmark_event_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    received_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now())
    payload: Mapped[dict] = mapped_column(JSONB)
    image_key: Mapped[str | None] = mapped_column(String(80), nullable=True)


def private_image(encoded: str) -> bytes:
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("too large")
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "JPEG" or image.width * image.height > 16000000:
                raise ValueError("unsupported image")
            image.load()
            # Re-encode pixels only: do not retain EXIF/GPS metadata.
            output = io.BytesIO()
            image.convert("RGB").save(output, format="JPEG", quality=85)
            return output.getvalue()
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Expected a JPEG up to 4 MiB and 16 megapixels") from exc


def storage_root() -> Path:
    value = os.environ.get("INTERNAL_REVIEW_STORAGE")
    if not value:
        raise HTTPException(503, "Internal review storage is not configured")
    root = Path(value)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root


def write_image(data: bytes) -> str:
    root = storage_root()
    key = hashlib.sha256(data).hexdigest() + ".jpg"
    fd, temporary = tempfile.mkstemp(dir=root)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temporary, root / key)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return key


def require_key(variable: str, supplied: str):
    expected = os.environ.get(variable, "")
    if len(expected) < 32:
        raise HTTPException(503, "Internal review access is disabled")
    if not hmac.compare_digest(expected.encode(), supplied.encode()):
        raise HTTPException(401, "Invalid credential")


async def upload_auth(x_internal_review_key: str = Header(default="")):
    require_key("INTERNAL_REVIEW_API_KEY", x_internal_review_key)


async def read_auth(x_dashboard_key: str = Header(default="")):
    require_key("DASHBOARD_READ_API_KEY", x_dashboard_key)


router = APIRouter()


@router.post("/v1/internal/reviews", dependencies=[Depends(upload_auth)])
async def create_review(review: Review, request: Request):
    from starlette.concurrency import run_in_threadpool
    image_key = None
    if review.image_base64 is not None:
        data = await run_in_threadpool(private_image, review.image_base64)
        image_key = await run_in_threadpool(write_image, data)
    payload = review.model_dump(mode="json", exclude={"image_base64"})
    payload["source"] = "internal"
    statement = insert(StoredReview).values(
        review_id=review.review_id, benchmark_event_id=review.benchmark_event_id,
        payload=payload, image_key=image_key,
    ).on_conflict_do_nothing(index_elements=[StoredReview.review_id]).returning(StoredReview.review_id)
    async with request.app.state.sessions() as session:
        async with session.begin():
            await session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
                                  {"key": "internal-review:" + str(review.benchmark_event_id)})
            prior = await session.scalar(select(StoredReview.review_id).where(
                StoredReview.benchmark_event_id == review.benchmark_event_id).limit(1))
            if prior is not None and prior != review.review_id:
                raise HTTPException(409, "Internal feedback already submitted for this caption")
            inserted = (await session.execute(statement)).scalar_one_or_none()
            if inserted is None:
                existing = await session.get(StoredReview, review.review_id)
                if existing.payload != payload or existing.image_key != image_key:
                    raise HTTPException(409, "Review ID already has different content")
    return {"review_id": str(review.review_id), "duplicate": inserted is None}


@router.get("/v1/internal/benchmarks/{benchmark_id}", dependencies=[Depends(read_auth)])
async def benchmark_details(benchmark_id: UUID, request: Request):
    async with request.app.state.sessions() as session:
        benchmark = await session.get(StoredEvent, benchmark_id)
        feedback = (await session.scalars(select(StoredEvent).where(
            StoredEvent.event_type == "caption_feedback",
            StoredEvent.payload["caption_feedback"]["benchmark_event_id"].astext == str(benchmark_id),
        ).order_by(StoredEvent.received_at).limit(500))).all()
        reviews = (await session.scalars(select(StoredReview).where(
            StoredReview.benchmark_event_id == benchmark_id,
        ).order_by(StoredReview.received_at).limit(500))).all()
    return {
        "benchmark": benchmark.payload if benchmark and benchmark.event_type == "caption_benchmark" else None,
        "feedback": [item.payload for item in feedback],
        "internal_reviews": [item.payload | {"image_path":
            f"/v1/internal/reviews/{item.review_id}/image" if item.image_key else None} for item in reviews],
    }


@router.get("/v1/internal/reviews/{review_id}/image", dependencies=[Depends(read_auth)])
async def review_image(review_id: UUID, request: Request):
    from starlette.concurrency import run_in_threadpool
    async with request.app.state.sessions() as session:
        review = await session.get(StoredReview, review_id)
    if not review or not review.image_key:
        raise HTTPException(404, "Image not found")
    try:
        data = await run_in_threadpool((storage_root() / review.image_key).read_bytes)
    except FileNotFoundError:
        raise HTTPException(404, "Image not found")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
