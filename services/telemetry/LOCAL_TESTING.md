# Local API + dashboard testing

Uses the existing password-protected `/admin/` dashboard, including team shares.
Siri integration is not included.

## Server

Create an ignored `services/telemetry/.env` with local-only credentials:

```dotenv
POSTGRES_PASSWORD=<random-password>
INGEST_API_KEY=<random-32-plus-character-key>
DASHBOARD_READ_API_KEY=<different-random-32-plus-character-key>
DASHBOARD_PASSWORD=<local-dashboard-password>
```

From `services/telemetry`:

```sh
docker compose up --build -d
curl --fail http://127.0.0.1:8000/healthz
```

Do not reuse production credentials or real user photos. Use the compose default
127.0.0.1 bind for simulator-only testing. For a physical device use a trusted LAN,
set API_BIND_ADDRESS=0.0.0.0 in .env, and restrict access with the Mac firewall.

## Local dashboard

Open http://127.0.0.1:8000/admin/ and log in as `admin` with DASHBOARD_PASSWORD.
The existing filters and exports remain available. Follow **Team shares** to see
shared results separately, or a benchmark's link to filter its shares. Photos use
the same dashboard login; no read API key is exposed to the browser.
The temporary Streamlit dashboard has been removed.
To find recent IDs:

```sh
docker compose exec db psql -U telemetry -d telemetry -c \
  "select event_id, received_at from telemetry_events where event_type='caption_benchmark' order by received_at desc limit 10;"
```

## App

For a DEBUG build, add to the ignored bundled
TelemetrySecrets-Debug.plist:

- TelemetryBaseURL: http://localhost:8000 (simulator), or Mac LAN address (device)
- InternalBenchmarkingEnabled: true
- IngestAPIKey: local ingest key
- Internal review uploads reuse IngestAPIKey; no additional upload key is required.

Non-DEBUG builds read the original TelemetrySecrets.plist and always use the
production endpoint. Xcode excludes the other environment's plist from each bundle.
Missing/invalid Debug configuration disables collection instead of falling back to
production. INTERNAL_TESTING controls feature availability, not server selection.
Internal sharing also requires Swift Active Compilation Conditions to include INTERNAL_TESTING.
Use local HTTPS where possible. For plain HTTP on a physical device, configure a
**development-only** Info.plist ATS exception appropriate for your local host and
NSLocalNetworkUsageDescription. Do not add global arbitrary-load exceptions to
shipping builds. The repository does not change shipping ATS policy automatically.
The simulator can generate captions from its existing sample-image mode.

Telemetry queues are partitioned by server URL. Production retains its original
queue filename; local endpoints use a SHA-256-derived suffix. Switching environments
does not send production queued events to the local server.

## Smoke test

1. Generate a caption; check partial text and speech start before completion.
2. Stop speech mid-generation; it must not restart on subsequent token updates.
3. Submit a rating; locate its benchmark ID and load it in the dashboard.
4. In the internal build, Share with team without requiring a rating; attach a
   test photo and comment. Check shared results, not the rating count, increases.
5. Test no-photo submission and a wrong upload/read key.
6. Background the app during speech, then return: no automatic repeated speech.

Streaming uses sentence punctuation or roughly eight completed words per chunk;
short single-sentence captions may still wait until completion. Full generated
text is decoded at every token and already-queued text is never repeated. Speech
stops accepting chunks if the tokenizer revises already-spoken text. Measure
naturalness, VoiceOver overlap and inference throughput on a physical device;
partial decoding and main-actor callbacks add overhead to benchmark timing.
