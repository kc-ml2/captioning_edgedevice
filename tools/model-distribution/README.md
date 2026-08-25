# ODIC model distribution

모델을 versioned S3 경로에 게시하고 ODIC 앱이 CloudFront HTTPS를 통해 내려받도록 하는 도구입니다.
IAM 자격증명은 AWS CLI credential chain에서만 읽으며 앱이나 저장소에 포함하지 않습니다.

## AWS 인증

조직에서 제공한 방식에 맞춰 AWS CLI를 설정합니다.

```bash
# IAM Identity Center를 사용하는 경우
aws configure sso --profile odic-publisher
aws sso login --profile odic-publisher

# 설정 후 권한 확인 (secret은 출력하지 않음)
aws sts get-caller-identity --profile odic-publisher
```

장기 access key를 저장소의 `.env`, Swift 코드 또는 Xcode 설정에 넣지 마세요.

## Manifest만 생성해 검증

```bash
python tools/model-distribution/publish_model.py \
  --source workspace/mobilevlm-mlx/converted-fp16 \
  --bucket unused \
  --version fp16-v1 \
  --manifest-only
```

## S3 게시

```bash
python tools/model-distribution/publish_model.py \
  --source workspace/mobilevlm-mlx/converted-fp16 \
  --bucket YOUR_PRIVATE_BUCKET \
  --prefix odic/models \
  --version fp16-v1 \
  --profile odic-publisher \
  --region ap-northeast-2 \
  --cloudfront-distribution-id YOUR_DISTRIBUTION_ID
```

게시 순서:

1. 모델 파일을 `releases/<version>/`에 immutable cache policy로 업로드
2. 파일 크기와 SHA-256이 포함된 `manifest.json`을 release commit marker로 업로드
3. 마지막에 `latest.json` 갱신
4. 요청한 경우 CloudFront의 `latest.json`만 invalidation

이미 게시한 version 문자열은 재사용하지 않는 것을 원칙으로 합니다.

## 앱 CDN 주소 설정

Xcode build setting `ODIC_MODEL_BASE_URL`에 CloudFront의 모델 prefix를 지정합니다.
끝의 `/` 유무와 관계없이 동작합니다.

```text
ODIC_MODEL_BASE_URL = https://YOUR_DISTRIBUTION.cloudfront.net/odic/models
```

앱은 다음 순서로 설치합니다.

```text
latest.json → manifest.json → staging 다운로드 → 파일별 크기/SHA-256 검증
→ Library/Application Support/MobileVLM 원자적 교체
```

S3 버킷은 private으로 유지하고 CloudFront Origin Access Control(OAC)만 S3 object를 읽도록 구성하는 것을 권장합니다.
