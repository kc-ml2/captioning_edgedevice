# Internal caption reviews

Internal reviews use a separate endpoint and table, but reuse the ordinary
INGEST_API_KEY credential and share benchmark IDs with telemetry. Existing telemetry still rejects captions/images.

## Configuration

Set distinct random credentials of at least 32 characters in server `.env`:

- `INGEST_API_KEY`: shared telemetry and internal-review upload credential
- `DASHBOARD_READ_API_KEY`: dashboard backend only; NEVER put this in the app

Compose provides `INTERNAL_REVIEW_STORAGE` as a persistent private Docker volume.
No public/static file route is configured. Back up this volume together with the
DB. `internal_caption_reviews` is created on API startup (no FK to benchmarks).
Missing credentials disable the corresponding endpoints with 503.
Configure TLS, reverse-proxy upload limits (6 MiB) and rate limiting before use.
The mobile credential is extractable; this is for a trusted internal group, not
an authorization mechanism suitable for public distribution.

## App build

Add `INTERNAL_TESTING` to **Swift Active Compilation Conditions** of the internal
build only, preserving `$(inherited)` and existing conditions. External archives
must not include this flag. Scheme/TestFlight routing is a separate release task.
The app reuses `IngestAPIKey` from its environment-specific secrets plist.
No separate InternalReviewAPIKey is needed. The existing X-Internal-Review-Key
header now carries this shared key; deploy the server change before the new app.
Build flags hide functionality, not API access: any holder of the ingest key can
upload internal reviews. Revisit authorization before public distribution.

Use the separate **Share with team** button next to Read aloud and Feedback.
No rating submission is required. This is a shared result, not a quality vote.
The sheet accepts a comment and optional source-photo attachment.
The caption is always included, and the UI says so. Only internal builds retain
the actual caption input frame in memory. Photos are resized to at most 1600px
and JPEG encoded without location metadata. Server re-encoding also strips EXIF.

Sending is explicit, not automatic. The draft is memory-only; successful sends
close the sheet. Failed sends retain the exact request/UUID for retry and lock
editing to avoid changing a retried payload. Closing discards the draft, and a
previous failed network response does not guarantee the server rejected it.
This is independent of the persistent ordinary-feedback queue.

## Upload API

`POST /v1/internal/reviews`, header `X-Internal-Review-Key`:

```json
{
  "review_id": "dda4cba2-d38c-4612-9d97-fd105724229c",
  "benchmark_event_id": "8d0d494e-197d-427c-9e29-9d3cc7cd1271",
  "caption": "A chair near a table.",
  "comment": "The chair is correct but the table was missed.",
  "model_version": "test-model",
  "image_base64": null
}
```

Optional image: base64 JPEG, at most 4 MiB decoded and 16 megapixels.
Comment: 1–5000 characters, nonblank. Caption: 1–10000, nonblank.
Exact retries return `duplicate: true`; same review_id with different content
returns 409. Photo bytes are stored privately; only the generated file key is
stored in the DB. Review payload has server-assigned `source: internal`.

## Dashboard integration

`GET /v1/internal/benchmarks/{benchmark_event_id}` with `X-Dashboard-Key`
returns `benchmark`, `feedback` (rating history), and `internal_reviews` together.
A missing benchmark returns null, without losing early-arriving reviews.
Each review's `image_path` is a relative protected API path or null. Fetch it
server-side with the same read credential; never embed credentials in image URLs
or expose the dashboard credential to browsers. Image responses use no-store.
The detail endpoint returns up to 500 records of each history ordered by receipt;
it is not a paginated export API. Compute latest feedback by revision, not receipt.

The existing `/admin/` dashboard includes a `/admin/shares` page, with separate
share counts, benchmark filtering and protected image retrieval using the same
DASHBOARD_PASSWORD login. The JSON read API remains available for other integrations.
Render comments/captions as escaped plain text, never raw HTML. Keep internal
reviews distinct from user ratings: internal reviews contain no usefulness rating
or reason tags. Count shared results by unique review_id in internal_caption_reviews;
count rated captions separately from caption_feedback using latest revision per
benchmark_event_id. Do not sum feedback history rows or internal shares as votes. Legacy ordinary events have unknown audience; do not label all as external.

## Operational limits / retention

Before collecting real photos, agree a retention period and implement scheduled
purging of expired review rows and unreferenced files. This initial version has
no automatic retention job. A failed DB transaction or conflicting retry may
leave an unreferenced content-addressed image, so cleanup must handle orphans.
Restrict dashboard access to the team; do not publish this endpoint or credential
through a public dashboard. Storage encryption/access controls and backups are
deployment responsibilities. No server deployment is performed by these changes.
