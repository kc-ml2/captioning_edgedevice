# SceneSense telemetry 수집 서버

FastAPI + PostgreSQL 17. 이벤트 수집과 비밀번호로 보호된 내부용 대시보드를 제공합니다.
앱의 성공한 캡션 벤치마크 계측과 자동 전송을 연결했습니다. 내부 테스트용으로 별도 동의 UI는 없습니다.
캡션 실패(프레임 캡처/추론)와 모델 다운로드·설치 실패도 전송합니다.
사용자 취소(CancellationError 및 URLSession 취소)는 제외하며 앱 실행 이벤트는 없습니다.
오류 원문 대신 allowlist 오류 분류·숫자 코드·경과 시간·앱 상태만 저장합니다.
다운로드 실패에는 받은 바이트 수, 캡션 실패에는 capture/inference 단계가 추가됩니다.
실패 이벤트의 model_version은 현재 설치된 버전이며 설치 전에는 unknown입니다.
크래시·OS 강제 종료는 catch할 수 없으므로 이 수집 범위에 포함되지 않습니다.

**앱 업데이트 전에 서버를 재배포하세요:**
`docker compose up -d --build`. 새 caption_failure 타입 및 download_failure의 error_category 필드를
이전 서버는 422로 거부합니다. 테이블은 JSONB라 이번 변경에 DB 마이그레이션은 필요하지 않습니다.

## 실행

Docker Engine / Docker Desktop과 Compose v2가 필요합니다.

```bash
cd services/telemetry
# 최초 한 번만 생성. 기존 .env는 덮어쓰지 않습니다.
python3 - <<'PY'
import secrets
from pathlib import Path
with Path('.env').open('x') as f:
    f.write(f'POSTGRES_PASSWORD={secrets.token_hex(32)}\n')
    f.write(f'INGEST_API_KEY={secrets.token_hex(32)}\n')
Path('.env').chmod(0o600)
PY
docker compose up -d --build
curl --fail http://127.0.0.1:8000/healthz
```

DB 테이블은 API 최초 시작 시 생성합니다. 향후 스키마 변경에는 마이그레이션을 도입해야 합니다.
PostgreSQL은 외부 포트를 열지 않으며 named volume `postgres_data`에 저장됩니다.
`docker compose down`은 데이터를 보존하지만 **`docker compose down -v`는 삭제합니다.**
볼륨은 백업이 아니므로 운영 시 별도 백업을 설정하세요.
초기화된 DB의 비밀번호는 .env만 수정해도 변경되지 않습니다.

기본 API 바인딩은 127.0.0.1입니다. 실제 앱 연결은 서버 앞단에서 HTTPS를 설정해야 합니다.
테스트용 LAN 노출은 .env에 `API_BIND_ADDRESS=0.0.0.0`을 추가할 수 있지만,
HTTP로 수집 키를 인터넷에 노출하지 마세요. 앱 ATS 예외를 추가하는 대신 HTTPS를 사용하세요.
리버스 프록시에서 요청 속도 제한과 연결/본문 수신 타임아웃을 설정하세요.
이 Compose 자체에는 TLS 및 rate limiting이 포함되지 않습니다.

## 인증 및 수집

`POST /v1/events/batch`, 헤더 `X-Ingest-Key: <INGEST_API_KEY>`.
고정 키는 소규모 TestFlight용 최소 인증입니다. 앱 내 키는 추출 가능하며 사용자 인증이나
정상 앱 검증을 보장하지 않습니다. 키를 git에 커밋하지 마세요. 앱 수집 키로는 관리자 대시보드에 접근할 수 없습니다. 헬스 체크만 인증 없이 접근 가능합니다.

최대 100개 이벤트, 요청 본문 최대 256 KiB. 정의되지 않은 필드는 거부합니다.

```bash
set -a; . ./.env; set +a
curl --fail-with-body http://127.0.0.1:8000/v1/events/batch \
  -H "X-Ingest-Key: $INGEST_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"events":[{
    "schema_version":1,
    "event_id":"cfd76a63-bcae-4e6e-ad49-15dc0a0ea731",
    "event_type":"caption_benchmark",
    "timestamp":"2026-09-13T14:30:00Z",
    "app_version":"1.0","build_number":"3",
    "device_model":"iPhone17,3","os_version":"26.5",
    "model_version":"test-model",
    "metrics":{
      "is_cold_run":false,"caption_latency_ms":2450,
      "time_to_first_token_ms":1100,"generation_ms":1350,
      "generated_tokens":28,"tokens_per_second":20.74,
      "peak_sampled_memory_bytes":2100000000,"output_words":18,
      "thermal_state":"nominal"
    }
  }]}'
```

응답: `received`, `inserted`, `duplicates`, `acknowledged_event_ids`.
동일 event_id 재전송은 성공 응답하며 기존 데이터를 덮어쓰지 않습니다.
이벤트 UUID는 **이벤트 생성 시 한 번만** 만들고 재시도 시 유지하세요.
앱은 acknowledged_event_ids에 있는 로컬 이벤트만 삭제하면 됩니다.
전체 배치 검증/DB 커밋 후 응답하며, 잘못된 이벤트가 있으면 배치 전체를 거부합니다.
401은 키 설정 확인, 413은 배치 축소, 422는 스키마 수정이 필요합니다.
네트워크 오류 및 5xx만 지수 백오프로 재시도하는 것을 권장합니다.

`download_failure` 이벤트는 metrics 대신 다음 객체를 사용합니다:

```json
{"download_failure":{"elapsed_ms":60000,"downloaded_bytes":123456,
 "error_code":-1001,"http_status":null,"retry_count":0,"app_state":"active"}}
```

전체 필드 정의는 `app/main.py`를 참고하세요.
caption_benchmark는 성공한 캡션 결과용입니다.
caption_failure는 metrics 대신 `caption_failure` 객체로 elapsed_ms, error_code,
error_category(network/cocoa/posix/application), app_state, stage(capture/inference)를 보냅니다.

## 측정 의미 (앱 구현 시 준수)

- cold_load_ms: 해당 프로세스의 최초 모델·토크나이저 로딩 구간. 다운로드 제외.
- caption_latency_ms: 입력 이미지 처리 시작부터 최종 문자열까지. 모델 로딩 제외.
  warm 비교는 is_cold_run=false인 이벤트끼리 수행합니다.
- time_to_first_token_ms: caption 시작부터 첫 생성 토큰까지.
- generation_ms: prefill 이후 첫 토큰 선택 시작부터 마지막 생성 토큰 선택까지.
- generated_tokens: EOS를 제외한 생성 토큰 수.
- tokens_per_second: generated_tokens / (generation_ms / 1000). 구간이 0이면 null.
- peak_sampled_memory_bytes: 캡션 실행 구간의 프로세스 메모리 샘플 최대값.
  실제 순간 최대값이나 MLX 할당량과 동일하지 않습니다.
- output_words: 최종 출력을 Swift `Character.isWhitespace` 기준으로 분리한 비어 있지 않은 구간 수. 공백·탭·줄바꿈을 구분자로 사용하며 연속 공백은 무시한다. 구두점은 별도로 분리하지 않는다(예: `it's`, `red-and-white`는 각각 1개). 언어학적 단어 수가 아닌 영어 캡션의 간단한 분량 지표다.
- output_characters: 구버전 호환용 선택 필드. 이전 앱과 로컬 큐의 문자 수를 그대로 보존하며 단어 수로 변환하지 않는다. 새 앱은 `output_words`만 전송한다. 두 필드 중 적어도 하나는 필요하다.

배포 시 서버를 먼저 업데이트한 뒤 앱을 배포한다. 기존 서버는 알 수 없는 `output_words`를 거부한다. 스키마 버전은 1을 유지하며 기존 JSONB 데이터는 수정하지 않는다. 분석 시 구버전 이벤트의 누락된 단어 수를 0으로 취급하지 않는다.

## 캡션 피드백 (내부 테스트)

기존 `POST /v1/events/batch`와 `X-Ingest-Key` 인증을 사용한다.
`schema_version: 1`과 앱·빌드·기기·OS·모델 버전 등의 공통 필드는 그대로 유지하고,
`event_type`을 `caption_feedback`으로 지정한다. `metrics`나 실패 payload와 함께 보내면 거부한다.

```json
{
  "schema_version": 1,
  "event_id": "da34ef1c-9605-4f0a-810a-b6ed26ed4aaf",
  "event_type": "caption_feedback",
  "timestamp": "2026-09-22T12:00:00Z",
  "app_version": "1.0",
  "build_number": "3",
  "device_model": "iPhone17,3",
  "os_version": "26.5",
  "model_version": "test-model",
  "caption_feedback": {
    "benchmark_event_id": "8d0d494e-197d-427c-9e29-9d3cc7cd1271",
    "revision": 2,
    "is_useful": true,
    "reasons": ["overall_situation", "noticing_missed_details"]
  }
}
```

- `benchmark_event_id`: 해당 캡션의 `caption_benchmark` 이벤트 `event_id`를 재사용한다.
  별도 캡션 UUID를 생성하지 않으며, 평가 수정에서도 이 참조는 유지한다.
  UUID 형식만 검증하고 대상 이벤트의 존재 여부는 강제하지 않는다.
- `revision`: 1부터 시작하는 정수. 도움 여부나 이유를 수정할 때마다 클라이언트가 증가시킨다.
- `is_useful`: boolean. false이면 이유는 빈 배열이어야 한다.
- `reasons`: 선택 사항(기본 빈 배열). `overall_situation`(상황 이해),
  `identifying_objects`(물체 식별), `noticing_missed_details`(놓치기 쉬운 부분 발견).
  복수 선택 가능하지만 중복·알 수 없는 값은 거부한다.
- 긍정 응답에 이유가 없어도 유효하다. 피드백이 없는 캡션은 부정 응답으로 취급하지 않는다.
- 이미지·캡션 원문·자유 입력 필드는 허용하지 않는다.

저장은 기존 `telemetry_events.payload` JSONB를 사용하므로 테이블 변경은 필요 없다.
평가 변경마다 **새 event_id**로 전체 평가 상태를 전송하고, 네트워크 재전송은
**동일한 event_id와 payload**를 재사용한다. 같은 ID의 재전송은 기존 데이터에 덮어쓰지 않는다.
서버는 수정 이력을 보존하며 revision 순서를 강제하거나 누락된 revision을 거부하지 않는다.
분석할 때는 수신 순서 대신 benchmark_event_id별 가장 큰 revision의 평가만 집계한다.
클라이언트는 동일 benchmark_event_id/revision에 서로 다른 상태를 생성하지 않아야 한다.

`benchmark_event_id`는 JSONB 내 논리적 참조이며 DB FOREIGN KEY 제약은 걸지 않는다.
피드백이 먼저 도착하거나 벤치마크가 누락되어도 저장한다. 분석 시 벤치마크의
`event_id`와 LEFT JOIN하고 대상의 `event_type = 'caption_benchmark'`를 확인한다.
연결되지 않은 피드백도 보존하며, 성능과 연결된 분석에서는 별도로 구분한다.

서버를 먼저 배포해야 한다. 앱은 벤치마크 event_id와 모델 버전을 파이프라인에서 UI로 전달한다.
피드백은 한 화면에서 선택 후 ‘피드백 보내기’로 제출한다. 안 도움은 false/빈 이유,
안 도움만 단독 선택이다. 전반적으로 도움은 구체적인 세 항목과 함께 선택할 수 있다.
긍정 항목은 복수 선택 가능하며 true/선택한 세부 이유 배열로 매핑한다.
전반적으로 도움 자체는 별도 이유 태그를 추가하지 않으므로, 단독 선택 시 true/빈 이유가 된다. 선택만 하거나 취소하면 전송하지 않는다.
제출된 선택이 변경될 때마다 revision을 증가시키고, 같은 선택을 다시 제출하면
새 이벤트를 만들지 않는다. 다시 열면 마지막으로 제출한 선택을 표시한다.

피드백은 기존 benchmark-queue.json에 원자적으로 저장한 뒤 비동기 전송한다.
UI의 저장 성공은 서버 수신 완료가 아니라 전송 큐 저장 완료를 의미한다.
기존 큐 정책(최대 500개, 재시도, acknowledged_event_ids 기반 삭제)을 공유한다.
이전 프로토타입의 CaptionFeedback/*.json은 벤치마크 참조가 없으므로 자동 전송하지 않는다.

## 내부 테스트용 로컬 키

현재 로컬 환경에는 서버 `.env`의 `INGEST_API_KEY`와 동일한 키를
`apps/ODIC/ODIC/TelemetrySecrets.plist`에 생성했습니다. 두 파일 모두 Git에서 제외됩니다.
앱에서는 `TelemetryConfiguration.ingestAPIKey`로 읽습니다. 키가 없으면 nil을 반환합니다.
다른 개발 머신/CI에는 이 파일을 별도로 제공해야 합니다.
`TelemetrySecrets.plist`의 Boolean `InternalBenchmarkingEnabled`가 true인 경우만 계측·전송합니다.
현재 로컬 내부 테스트 설정은 true입니다. 외부 배포 전 false로 바꾸거나 파일을 제외하세요.
전송 주소는 `https://scenesense.ml2-alpha.com/v1/events/batch`입니다.
로컬 `Application Support/benchmark-queue.json`에 최대 500건을 원자적으로 저장하며,
초과 시 오래된 이벤트부터 삭제합니다. 이 JSON은 전송 대기열이지 영구 벤치마크 기록은 아닙니다.
서버가 확인한 이벤트는 삭제하며 네트워크/5xx/429 실패는 5초부터 최대 5분 간격으로 재시도합니다.
앱 실행·포그라운드 복귀 시 재개하며 OS가 앱을 중단한 동안 백그라운드 전송을 보장하지 않습니다.
서버 인증/스키마 4xx 오류는 해당 전송 루프를 중지합니다.
메모리는 캡션 구간에 100ms 간격으로 physical footprint를 측정합니다. 모델 로딩 중은 제외합니다.
로딩 시간은 파일·토크나이저 로딩 구간이며 MLX 지연 평가 비용은 첫 캡션에 포함될 수 있습니다.
앱 키는 번들에서 추출 가능하므로 운영용 비밀이나 사용자 인증으로 취급하지 마세요.
키 교체 시 서버 설정과 앱을 함께 갱신해야 합니다.

내부 테스트에서는 별도 동의 팝업 없이 수집하는 구성을 검토할 수 있지만,
테스터에게 수집 항목·목적·보관 기간을 사전에 알리고 적용되는 정책을 확인해야 합니다.
이미지·캡션 원문·질문·사용자 식별자는 수집하지 않습니다.
내부 빌드 여부를 TestFlight라는 이유만으로 판별하지 말고, 향후 자동 전송 구현 시
내부 테스트 전용 빌드 설정으로 활성화해야 합니다. 외부 배포에는 재검토가 필요합니다.

## 외부 배포 시 앱 전송 동의 계획

설정에 기본 OFF인 **“성능 및 오류 데이터 공유”** 토글을 둡니다.
처음 안내 시 “공유하기 / 나중에”를 제공하되 다운로드·캡션 사용을 막지 않습니다.

권장 안내 문구:

> 앱 개선을 위해 기기 모델, iOS·앱·모델 버전, 처리 시간, 토큰 수,
> 메모리 사용량 및 다운로드 오류 코드를 서버로 전송합니다.
> 카메라 이미지, 캡션 원문, 질문, 이름 및 계정 식별자는 전송하지 않습니다.
> 설정에서 언제든 끌 수 있습니다.

- 동의 전에는 전송용 이벤트를 수집/저장하지 않습니다.
- 끄면 전송을 중단하고 대기열을 삭제합니다. 다시 켜도 과거 이벤트를 보내지 않습니다.
- 이미 서버에 도착한 데이터가 자동 삭제되는 것은 아닙니다. 이를 정책에 명시하세요.
- 이 서버는 영구 사용자/기기 식별자를 요구하지 않습니다.
- 서버/프록시는 IP를 관찰할 수 있으므로 완전한 익명이라고 약속하지 마세요.
  Uvicorn access log는 껐지만 프록시 로그 정책은 별도 설정해야 합니다.
- 배포 전에 실제 보관 기간·삭제 운영 절차·개인정보 안내를 결정하세요.
  현재 서버는 자동 만료 없이 계속 저장합니다.

## 테스트

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt pytest httpx
.venv/bin/python -m pytest -q
```

PostgreSQL 통합 테스트는 실행 중인 Compose의 일회성 API 컨테이너에서 수행할 수 있습니다.
테스트 이벤트가 DB에 남으므로 테스트용 DB에서 실행하세요.

```bash
docker compose run --rm --user root -v "$PWD/tests:/service/tests:ro" api \
  sh -c 'pip install pytest httpx && python -m pytest -q'
```

## 내부용 대시보드

`.env`에 수집 키와 다른 충분히 긴 `DASHBOARD_PASSWORD`를 설정하고
`docker compose up -d --build`로 반영합니다. 미설정 시 대시보드는 503으로 차단됩니다.
비밀번호를 Git에 커밋하지 마세요.

- 주소: `https://<서비스 도메인>/admin/` (호스트 직접 접속: `http://127.0.0.1:8003/admin/`)
- 브라우저 Basic Auth 로그인: 사용자 이름 `admin`, 비밀번호 `DASHBOARD_PASSWORD`
- nginx가 `/admin/` 및 하위 경로도 API로 전달해야 합니다. 외부 접근은 HTTPS 필수입니다.
- 날짜 필터는 **서버 수신 시각 UTC** 기준이며 시작일/종료일을 모두 포함합니다.
- 이벤트 종류, 앱 버전, 기기 모델, 모델 버전은 드롭다운으로 선택합니다.
  선택지는 날짜/다른 필터와 무관하게 저장된 실제 데이터의 고유 값으로 구성하며, `전체`로 해제합니다.
- `connectivity-test`는 표·다운로드·드롭다운 선택지에서 항상 제외합니다. DB 원본은 유지합니다.
- 최신순 50건 페이지 조회, 각 이벤트 원본 JSON 상세 보기.
- 벤치마크 바로 아래에 가장 높은 revision의 연결된 피드백을 표시합니다. 동일 revision이면
  수신 시각, event_id 내림차순으로 하나를 선택합니다. 추가 행은 페이지 건수에 포함하지 않습니다.
- 연결 피드백은 현재 날짜/필터 범위 밖의 평가도 포함합니다. 피드백이 없으면 추가 행을 표시하지 않습니다.
- 피드백 이벤트 자체에서도 도움 여부, 이유, revision, 대상 벤치마크 ID를 바로 볼 수 있습니다.
  원본 이벤트 목록은 수정 이력을 유지하므로 연결 행과 중복 표시될 수 있습니다.
- 다운로드는 필터에 해당하는 원본 이벤트만 포함하며 추가 연결 행은 자동 포함하지 않습니다.
- CSV/JSONL 다운로드는 현재 필터 전체에 적용됩니다. 최대 10,000건이며 초과 시
  일부만 내보내지 않고 413을 반환합니다. 날짜 범위 등을 좁혀 주세요.
- CSV는 중첩 필드를 점 표기법으로 평탄화하고 UTF-8 BOM을 포함합니다.
  수식으로 해석될 수 있는 문자열에는 작은따옴표를 붙입니다.
- JSONL은 각 줄에 `received_at`과 원본 `payload`를 저장합니다.
- 조회는 기존 DB 연결을 사용하되 read-only 트랜잭션과 30초 SQL 타임아웃을 적용합니다.
  별도 읽기 전용 DB 계정 연결은 현재 구현하지 않았습니다.
- Basic Auth는 브라우저가 자격 증명을 기억할 수 있습니다. 공용 기기에서는 사용하지 마세요.
  nginx에서 로그인 시도 rate limiting을 적용하는 것을 권장합니다.
