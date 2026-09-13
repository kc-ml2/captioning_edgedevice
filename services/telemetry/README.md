# SceneSense telemetry 수집 서버

FastAPI + PostgreSQL 17. 조회/내보내기 API 없이 이벤트를 저장합니다.
앱 계측·전송·동의 UI는 아직 연결되지 않았습니다.

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
정상 앱 검증을 보장하지 않습니다. 키를 git에 커밋하지 마세요. 서버에는 수집 API만 있고
조회 API는 없습니다. 헬스 체크만 인증 없이 접근 가능합니다.

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
      "peak_sampled_memory_bytes":2100000000,"output_characters":96,
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
caption_benchmark는 성공한 캡션 결과용입니다. 캡션 실패 이벤트는 아직 정의하지 않았습니다.

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
- output_characters: 최종 출력 Swift String.count (grapheme clusters).

## 내부 테스트용 로컬 키

현재 로컬 환경에는 서버 `.env`의 `INGEST_API_KEY`와 동일한 키를
`apps/ODIC/ODIC/TelemetrySecrets.plist`에 생성했습니다. 두 파일 모두 Git에서 제외됩니다.
앱에서는 `TelemetryConfiguration.ingestAPIKey`로 읽습니다. 키가 없으면 nil을 반환합니다.
다른 개발 머신/CI에는 이 파일을 별도로 제공해야 합니다. 실제 전송은 아직 구현되지 않았습니다.
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
