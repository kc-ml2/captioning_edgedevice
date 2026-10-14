"""Small, password-protected, read-only telemetry dashboard."""
import csv
import hmac
import io
import json
import os
from datetime import date, datetime, time, timedelta, timezone
from html import escape
from urllib.parse import urlencode
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import Integer, cast, func, select, text
from starlette.responses import HTMLResponse, Response

security = HTTPBasic(auto_error=False)
MAX_EXPORT = 10000
HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
           "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; img-src 'self'; frame-ancestors 'none'"}


def authenticate(credentials: HTTPBasicCredentials | None = Depends(security)):
    password = os.environ.get("DASHBOARD_PASSWORD", "")
    if not password:
        raise HTTPException(503, "Dashboard is disabled")
    if credentials is None or not (
        hmac.compare_digest(credentials.username.encode(), b"admin")
        and hmac.compare_digest(credentials.password.encode(), password.encode())
    ):
        raise HTTPException(401, "Invalid dashboard credentials",
                            headers={"WWW-Authenticate": 'Basic realm="SceneSense dashboard"'})


def filters(event_type: str = "", app_version: str = "", device_model: str = "",
            model_version: str = "", start: str = "", end: str = ""):
    try:
        start = date.fromisoformat(start) if start else None
        end = date.fromisoformat(end) if end else None
    except ValueError:
        raise HTTPException(422, "Dates must use YYYY-MM-DD")
    if start and end and start > end:
        raise HTTPException(422, "Start date must not exceed end date")
    if end == date.max:
        raise HTTPException(422, "End date is out of range")
    return dict(event_type=event_type, app_version=app_version, device_model=device_model,
                model_version=model_version, start=start, end=end)


def conditions(model, values):
    clauses = []
    for key in ("event_type", "app_version", "device_model", "model_version"):
        if values[key]:
            column = model.event_type if key == "event_type" else model.payload[key].astext
            clauses.append(column == values[key])
    clauses.append(model.payload["app_version"].astext != "connectivity-test")
    if values["start"]:
        clauses.append(model.received_at >= datetime.combine(values["start"], time.min, timezone.utc))
    if values["end"]:
        clauses.append(model.received_at < datetime.combine(values["end"] + timedelta(days=1), time.min, timezone.utc))
    return clauses


def flattened(row):
    result = {"received_at": row.received_at.isoformat()}
    for key, value in row.payload.items():
        if isinstance(value, dict):
            result.update({f"{key}.{k}": v for k, v in value.items()})
        else:
            result[key] = value
    return result


def csv_safe(value):
    # Prevent spreadsheet formulas, including those preceded by whitespace.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


REASON_LABELS = {
    "overall_situation": "Overall situation",
    "identifying_objects": "Identifying objects",
    "noticing_missed_details": "Noticing missed details",
}


def feedback_html(payload):
    feedback = payload["caption_feedback"]
    useful = "Useful" if feedback["is_useful"] else "Not useful"
    reasons = ", ".join(REASON_LABELS.get(value, value) for value in feedback.get("reasons", []))
    return (f'<strong>{useful}</strong> · Revision {escape(str(feedback["revision"]))}'
            f'<br>Reasons: {escape(reasons or "None selected")}'
            f'<br>Benchmark ID: {escape(str(feedback["benchmark_event_id"]))}')


def make_router(model):
    router = APIRouter(prefix="/admin", dependencies=[Depends(authenticate)])

    async def load(request, values, limit, offset=0, grouped=False):
        where = conditions(model, values)
        if grouped and not values["event_type"]:
            where.append(model.event_type != "caption_feedback")
        async with request.app.state.sessions() as session:
            async with session.begin():
                await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
                await session.execute(text("SET LOCAL statement_timeout = '30s'"))
                count = await session.scalar(select(func.count()).select_from(model).where(*where))
                if limit == MAX_EXPORT and count > MAX_EXPORT:
                    raise HTTPException(413, f"Export exceeds {MAX_EXPORT} events; narrow the filters")
                rows = (await session.execute(select(model.event_id, model.event_type, model.received_at, model.payload).where(*where)
                        .order_by(model.received_at.desc(), model.event_id.desc())
                        .limit(limit).offset(offset))).all()
        return count, rows

    async def latest_feedback(request, rows):
        ids = [str(row.event_id) for row in rows if row.event_type == "caption_benchmark"]
        if not ids:
            return {}
        target = model.payload["caption_feedback"]["benchmark_event_id"].astext
        revision = cast(model.payload["caption_feedback"]["revision"].astext, Integer)
        # DISTINCT ON selects one latest revision per benchmark, regardless of page/date filters.
        statement = (select(target.label("target"), model.payload, model.received_at)
                     .where(model.event_type == "caption_feedback", target.in_(ids),
                            model.payload["app_version"].astext != "connectivity-test")
                     .distinct(target)
                     .order_by(target, revision.desc(), model.received_at.desc(), model.event_id.desc()))
        async with request.app.state.sessions() as session:
            async with session.begin():
                await session.execute(text("SET TRANSACTION READ ONLY"))
                await session.execute(text("SET LOCAL statement_timeout = '30s'"))
                feedback = (await session.execute(statement)).all()
        return {row.target: row for row in feedback}

    async def linked_reviews(request, rows):
        from app.internal_reviews import StoredReview
        ids = [row.event_id for row in rows if row.event_type == "caption_benchmark"]
        if not ids:
            return {}
        # Display one internal response; retain legacy history without deleting it.
        ranked = select(StoredReview.review_id, StoredReview.benchmark_event_id,
            StoredReview.received_at, StoredReview.payload, StoredReview.image_key,
            func.row_number().over(partition_by=StoredReview.benchmark_event_id,
                order_by=(StoredReview.received_at.desc(), StoredReview.review_id.desc())).label("rank")
        ).where(StoredReview.benchmark_event_id.in_(ids)).subquery()
        async with request.app.state.sessions() as session:
            async with session.begin():
                await session.execute(text("SET TRANSACTION READ ONLY"))
                await session.execute(text("SET LOCAL statement_timeout = '30s'"))
                reviews = (await session.execute(select(ranked).where(ranked.c.rank <= 1)
                    .order_by(ranked.c.received_at.desc()))).all()
        result = {}
        for review in reviews:
            result.setdefault(str(review.benchmark_event_id), []).append(review)
        return result

    async def filter_options(request):
        # Global options stay available when other filters produce no matches.
        options = {}
        async with request.app.state.sessions() as session:
            async with session.begin():
                await session.execute(text("SET TRANSACTION READ ONLY"))
                await session.execute(text("SET LOCAL statement_timeout = '30s'"))
                for key in ("event_type", "app_version", "device_model", "model_version"):
                    column = model.event_type if key == "event_type" else model.payload[key].astext
                    options[key] = (await session.scalars(select(column).where(
                        model.payload["app_version"].astext != "connectivity-test",
                        column.is_not(None), column != "",
                    ).distinct().order_by(column))).all()
        return options

    @router.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request, values: dict = Depends(filters),
                        page: int = Query(default=1, ge=1, le=100000)):
        count, rows = await load(request, values, 50, (page - 1) * 50, grouped=True)
        def url(path, **extra):
            params = {k: str(v).lower() if isinstance(v, bool) else str(v)
                      for k, v in values.items() if v is not None and v != ""}
            params.update(extra)
            return escape(path + "?" + urlencode(params), quote=True)
        inputs = "".join(f'<label>{label}<input name="{key}" type="date" value="{escape(str(values[key] or ""), quote=True)}"></label>'
                         for key, label in [("start", "Start date (UTC)"), ("end", "End date (UTC)")])
        options = await filter_options(request)
        for key, label in [("event_type", "Event type"), ("app_version", "App version"),
                           ("device_model", "Device model"), ("model_version", "Model version")]:
            choices = ['<option value="">All</option>']
            # Preserve stale/bookmarked filters rather than silently showing 'All'.
            if values[key] and values[key] not in options[key]:
                choices.append(f'<option value="{escape(values[key], quote=True)}" selected>{escape(values[key])} (unavailable)</option>')
            for value in options[key]:
                selected = " selected" if value == values[key] else ""
                choices.append(f'<option value="{escape(value, quote=True)}"{selected}>{escape(value)}</option>')
            inputs += f'<label>{label}<select name="{key}">{"".join(choices)}</select></label>'
        linked_feedback = await latest_feedback(request, rows)
        reviews_by_benchmark = await linked_reviews(request, rows)
        body = []
        for row in rows:
            p = row.payload
            m = p.get("metrics", {})
            cells = [row.received_at.isoformat(timespec="seconds"), row.event_type,
                     p.get("app_version", ""), p.get("device_model", ""), p.get("model_version", ""),
                     "Cold" if m.get("is_cold_run") else "Warm" if m else "—",
                     round(m["caption_latency_ms"], 1) if "caption_latency_ms" in m else "—",
                     round(m["tokens_per_second"], 2) if m.get("tokens_per_second") is not None else "—"]
            detail = escape(json.dumps(p, ensure_ascii=False, indent=2))
            feedback_summary = feedback_html(p) if row.event_type == "caption_feedback" else ""
            internal_detail = ""
            if row.event_type == "caption_benchmark":
                reviews = reviews_by_benchmark.get(str(row.event_id), [])
                if reviews:
                    internal_detail = '<strong>Internal feedback</strong>'
                    for review in reviews:
                        internal_detail += (f'<section><p>{escape(review.received_at.isoformat())}</p>'
                            f'<strong>Caption</strong><pre>{escape(str(review.payload.get("caption", "")))}</pre>'
                            f'<strong>Comment</strong><pre>{escape(str(review.payload.get("comment", "")))}</pre>')
                        if review.image_key:
                            internal_detail += (f'<img loading="lazy" src="shares/{review.review_id}/image" '
                                'alt="Caption source photo" style="max-width:100%;max-height:320px">')
                        internal_detail += '</section>'
            body.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in cells)
                        + f"<td>{feedback_summary}<details><summary>JSON</summary><pre>{detail}</pre></details></td></tr>")
            if row.event_type == "caption_benchmark":
                feedback = linked_feedback.get(str(row.event_id))
                if feedback:
                    summary = feedback_html(feedback.payload)
                    received = escape(feedback.received_at.isoformat(timespec="seconds"))
                    raw = escape(json.dumps(feedback.payload, ensure_ascii=False, indent=2))
                    content = (f'Latest feedback · {summary}<br>Received at (UTC): {received}'
                               f'<details><summary>Feedback JSON</summary><pre>{raw}</pre></details>')
                    body.append(f'<tr class="feedback"><td colspan="9">↳ {content}</td></tr>')
                if internal_detail:
                    body.append(f'<tr class="internal-feedback"><td colspan="9">↳ {internal_detail}</td></tr>')
        navigation = (f'<a href="{url("./", page=page-1)}">Previous</a> ' if page > 1 else "")
        if page * 50 < count:
            navigation += f'<a href="{url("./", page=page+1)}">Next</a>'
        html = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>SceneSense telemetry</title>
<style>body{{font:14px system-ui;margin:24px;color:#172033}}form{{display:flex;flex-wrap:wrap;gap:12px;align-items:end}}label{{display:grid;gap:4px}}input,select,button{{padding:8px}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}.feedback td{{background:#f1f5f9;padding-left:28px}}.internal-feedback td{{background:#fff7ed;padding-left:28px}}.table{{overflow:auto}}pre{{max-width:600px;max-height:400px;overflow:auto}}a{{margin-right:16px}}</style>
<h1>SceneSense telemetry</h1><p>Filtered by receipt time · UTC · Connectivity-test events excluded</p>
<form method="get">{inputs}<button>Apply filters</button><a href="./">Reset</a></form>
<p>{count:,} events · Page {page} · 50 parent events per page (linked feedback rows are additional)</p>
<p>Default view groups the latest rating and internal comments beneath each benchmark, including linked records outside the current filters. Select caption_feedback in Event type to inspect individual rating events. Unlinked ratings can also be found there.</p>
<p><a href="{url('export', format='csv')}">Download CSV</a><a href="{url('export', format='jsonl')}">Download JSONL</a>All matching events · Up to {MAX_EXPORT:,} events. Exports contain filtered original events only, not additional linked feedback.</p>
<div class="table"><table><thead><tr><th>Received at (UTC)</th><th>Event type</th><th>App version</th><th>Device model</th><th>Model version</th><th>Run type</th><th>Caption latency (ms)</th><th>Tokens/s</th><th>Details</th></tr></thead><tbody>{''.join(body) or '<tr><td colspan="9">No events found.</td></tr>'}</tbody></table></div>{navigation}</html>'''
        return HTMLResponse(html, headers=HEADERS)

    @router.get("/shares/{review_id}/image")
    async def share_image(review_id: UUID, request: Request):
        # Reuse image retrieval, but authorize via this router's dashboard login.
        from app.internal_reviews import review_image
        response = await review_image(review_id, request)
        response.headers.update(HEADERS)
        return response

    @router.get("/export")
    async def export(request: Request, values: dict = Depends(filters), format: str = "csv"):
        if format not in ("csv", "jsonl"):
            raise HTTPException(422, "Format must be csv or jsonl")
        _, rows = await load(request, values, MAX_EXPORT)
        if format == "jsonl":
            content = "".join(json.dumps({"received_at": r.received_at.isoformat(), "payload": r.payload}, ensure_ascii=False) + "\n" for r in rows)
            media = "application/x-ndjson"
        else:
            data = [flattened(r) for r in rows]
            columns = ["received_at"] + sorted({k for r in data for k in r} - {"received_at"})
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows({k: csv_safe(v) for k, v in r.items()} for r in data)
            content = "\ufeff" + output.getvalue()
            media = "text/csv"
        return Response(content, media_type=media, headers={**HEADERS,
                        "Content-Disposition": f'attachment; filename="scenesense-events.{format}"'})

    return router
