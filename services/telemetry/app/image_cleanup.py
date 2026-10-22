"""Remove unreferenced images only, never referenced review attachments."""
import asyncio
import logging
import os
import re
import time
from pathlib import Path

from sqlalchemy import select, text
from starlette.concurrency import run_in_threadpool

from .internal_reviews import StoredReview

log = logging.getLogger(__name__)


def remove_orphans(root: Path, referenced: set[str], cutoff: float) -> int:
    removed = 0
    if not root.exists():
        return removed
    for path in root.iterdir():
        if not re.fullmatch(r"[0-9a-f]{64}\.jpg", path.name):
            continue
        if path.name not in referenced and path.is_file() and path.stat().st_mtime < cutoff:
            path.unlink()
            removed += 1
    return removed


async def collect(sessions) -> int:
    root = os.environ.get("INTERNAL_REVIEW_STORAGE")
    if not root:
        return 0
    async with sessions() as session:
        async with session.begin():
            await session.execute(text("SELECT pg_advisory_xact_lock(712034991)"))
            referenced = set((await session.scalars(select(StoredReview.image_key).where(
                StoredReview.image_key.is_not(None)))).all())
            return await run_in_threadpool(remove_orphans, Path(root), referenced, time.time() - 86400)


async def cleanup_loop(sessions):
    while True:
        try:
            removed = await collect(sessions)
            if removed:
                log.info("Removed %d orphan review images", removed)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Review image cleanup failed")
        await asyncio.sleep(3600)
