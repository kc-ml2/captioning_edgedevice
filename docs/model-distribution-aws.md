# ODIC 모델 배포 파이프라인: S3 + CloudFront + iOS

이 문서는 MobileVLM MLX 모델을 private Amazon S3에 게시하고, CloudFront CDN을 통해 ODIC iPhone 앱의 Application Support 디렉터리에 안전하게 설치하는 전체 과정을 설명한다.

> AWS Access Key ID, Secret Access Key, Session Token은 이 문서나 Git 저장소에 기록하지 않는다. IAM 자격증명은 모델 게시용 로컬 AWS CLI에서만 사용하며 iOS 앱에는 포함하지 않는다.

---

## 1. 전체 구조

```text
로컬 변환 모델
  │
  │ publish_model.py (AWS IAM credential 사용)
  ▼
Private S3 bucket
  └── odic/models/
      ├── latest.json
      └── releases/<immutable-version>/
          ├── manifest.json
          ├── model-00001-of-00002.safetensors
          ├── model-00002-of-00002.safetensors
          └── tokenizer 및 설정 파일
  │
  │ CloudFront Origin Access Control (SigV4)
  ▼
CloudFront HTTPS CDN
  │
  │ URLSession (앱에는 IAM credential 없음)
  ▼
iPhone staging 디렉터리
  │ 파일 크기 + SHA-256 검증
  ▼
Library/Application Support/MobileVLM
```

핵심 보안 원칙은 다음과 같다.

- S3 버킷은 public access를 완전히 차단한다.
- S3 object는 지정한 CloudFront distribution만 읽을 수 있다.
- iOS 앱에는 AWS API key를 넣지 않는다.
- 모델 릴리스 경로는 버전별 immutable 경로로 운영한다.
- 변경 가능한 진입점은 `latest.json` 하나뿐이다.
- 다운로드한 모든 파일의 크기와 SHA-256을 검증한다.
- 검증이 완료된 모델만 원자적으로 활성화한다.
- 업데이트 실패 시 기존 정상 모델을 유지한다.

---

## 2. 현재 배포된 리소스

### AWS

| 항목 | 값 |
|---|---|
| Region | `ap-northeast-2` |
| S3 bucket | `odic-models-210499750105-ap-northeast-2` |
| S3 model prefix | `odic/models` |
| CloudFront distribution ID | `E3MHDM0TX47EYC` |
| CloudFront domain | `d3unqpp9xgzmm4.cloudfront.net` |
| CloudFront OAC ID | `E1DKPKJL8094HJ` |
| CloudFront price class | `PriceClass_200` |
| HTTP version | HTTP/2 및 HTTP/3 |

앱이 사용하는 모델 base URL:

```text
https://d3unqpp9xgzmm4.cloudfront.net/odic/models
```

### 현재 모델 릴리스

| 항목 | 값 |
|---|---|
| Version | `mobilevlm-v2-1.7b-fp16-v1` |
| 모델 파일 수 | 10 |
| 모델 전체 크기 | `3,349,107,824 bytes` |
| safetensors shard 수 | 2 |

```text
model-00001-of-00002.safetensors  1,984,063,776 bytes
model-00002-of-00002.safetensors  1,364,259,360 bytes
```

AWS CLI의 multipart upload는 대형 파일을 내부 전송 조각으로 나누는 기능이다. 실제 모델 파일 또는 앱 다운로드 단위가 수백 개로 나뉜 것은 아니며, 앱에는 위 두 shard가 전달된다.

---

## 3. 관련 저장소 파일

```text
tools/model-distribution/
├── publish_model.py
└── README.md

apps/ODIC/
├── Info.plist
└── ODIC/
    ├── ModelStore.swift
    ├── ContentView.swift
    └── CaptionPipeline.swift
```

각 파일의 역할:

- `publish_model.py`
  - 로컬 모델 구조 검사
  - 파일별 SHA-256 계산
  - manifest 생성
  - immutable S3 release 게시
  - `latest.json` 갱신
  - 선택적 CloudFront invalidation
- `ModelStore.swift`
  - CDN metadata 조회
  - 실제 모델 크기 확인
  - 저장 공간 검사
  - 파일 다운로드 및 스트리밍 SHA-256 검증
  - 원자적 설치 및 rollback
- `ContentView.swift`
  - 모델 크기와 설치 진행률 표시
  - 모델 설치와 캡션 생성 UI 전환
- `CaptionPipeline.swift`
  - `Application Support/MobileVLM`에 설치된 모델 로딩

---

## 4. AWS CLI 인증 설정

### 4.1 AWS CLI 설치

macOS/Homebrew:

```bash
brew install awscli
aws --version
```

### 4.2 전용 profile 설정

권장 profile 이름:

```text
odic-publisher
```

장기 IAM access key를 받은 경우 로컬 터미널에서 직접 실행한다.

```bash
aws configure --profile odic-publisher
```

입력 예시:

```text
AWS Access Key ID: <직접 입력>
AWS Secret Access Key: <직접 입력>
Default region name: ap-northeast-2
Default output format: json
```

임시 자격증명의 Access Key ID가 `ASIA`로 시작한다면 Session Token도 필요하다.

```bash
read -s -p "AWS Session Token: " TOKEN
echo
aws configure set aws_session_token "$TOKEN" --profile odic-publisher
unset TOKEN
```

로컬 파일 권한을 제한한다.

```bash
chmod 700 ~/.aws
chmod 600 ~/.aws/credentials ~/.aws/config
```

자격증명 검증:

```bash
aws sts get-caller-identity --profile odic-publisher
```

이 명령은 secret을 출력하지 않는다.

### 4.3 CSV access key를 받은 경우

CSV를 Git 저장소 안으로 복사하지 않는다. 다운로드 파일의 권한도 제한한다.

```bash
chmod 600 ~/Downloads/<access-key-file>.csv
```

가능하면 CSV 내용을 shell argument로 넘기지 말고 `aws configure`에 직접 입력한다. 설정이 끝난 뒤 CSV가 더 필요하지 않다면 안전하게 삭제한다.

```bash
rm ~/Downloads/<access-key-file>.csv
```

### 4.4 필요한 IAM 권한

게시 및 인프라 구성을 모두 수행하려면 대략 다음 권한이 필요하다.

#### S3

```text
s3:ListAllMyBuckets
s3:CreateBucket
s3:DeleteBucket
s3:GetBucketLocation
s3:GetBucketPolicy
s3:PutBucketPolicy
s3:GetBucketPublicAccessBlock
s3:PutBucketPublicAccessBlock
s3:GetEncryptionConfiguration
s3:PutEncryptionConfiguration
s3:GetBucketVersioning
s3:PutBucketVersioning
s3:GetBucketOwnershipControls
s3:PutBucketOwnershipControls
s3:ListBucket
s3:ListBucketVersions
s3:ListBucketMultipartUploads
s3:GetObject
s3:PutObject
s3:DeleteObject
s3:AbortMultipartUpload
```

#### CloudFront

```text
cloudfront:ListDistributions
cloudfront:GetDistribution
cloudfront:GetDistributionConfig
cloudfront:CreateDistribution
cloudfront:UpdateDistribution
cloudfront:CreateInvalidation
cloudfront:ListOriginAccessControls
cloudfront:GetOriginAccessControl
cloudfront:CreateOriginAccessControl
cloudfront:DeleteOriginAccessControl
cloudfront:ListCachePolicies
```

IAM 정책 조회 권한은 파이프라인 실행에 필수는 아니다. 권한이 부족하면 각 AWS API가 `AccessDenied`로 응답한다.

---

## 5. S3 버킷 구성

현재 사용 중인 값:

```bash
export AWS_PROFILE=odic-publisher
export AWS_REGION=ap-northeast-2
export ODIC_BUCKET=odic-models-210499750105-ap-northeast-2
```

### 5.1 버킷 생성

```bash
aws s3api create-bucket \
  --bucket "$ODIC_BUCKET" \
  --region "$AWS_REGION" \
  --create-bucket-configuration LocationConstraint="$AWS_REGION" \
  --profile "$AWS_PROFILE"
```

버킷 이름은 전 세계 S3에서 유일해야 한다.

### 5.2 public access 완전 차단

```bash
aws s3api put-public-access-block \
  --bucket "$ODIC_BUCKET" \
  --profile "$AWS_PROFILE" \
  --public-access-block-configuration \
  'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true'
```

### 5.3 기본 암호화

```bash
aws s3api put-bucket-encryption \
  --bucket "$ODIC_BUCKET" \
  --profile "$AWS_PROFILE" \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"},"BucketKeyEnabled":true}]}'
```

### 5.4 versioning

```bash
aws s3api put-bucket-versioning \
  --bucket "$ODIC_BUCKET" \
  --profile "$AWS_PROFILE" \
  --versioning-configuration Status=Enabled
```

### 5.5 object ownership

```bash
aws s3api put-bucket-ownership-controls \
  --bucket "$ODIC_BUCKET" \
  --profile "$AWS_PROFILE" \
  --ownership-controls 'Rules=[{ObjectOwnership=BucketOwnerEnforced}]'
```

### 5.6 설정 확인

```bash
aws s3api get-public-access-block \
  --bucket "$ODIC_BUCKET" \
  --profile "$AWS_PROFILE"

aws s3api get-bucket-encryption \
  --bucket "$ODIC_BUCKET" \
  --profile "$AWS_PROFILE"

aws s3api get-bucket-versioning \
  --bucket "$ODIC_BUCKET" \
  --profile "$AWS_PROFILE"
```

---

## 6. 모델 release manifest

모델 디렉터리에는 현재 다음 파일이 있다.

```text
workspace/mobilevlm-mlx/converted-fp16/
├── config.json
├── conversion_manifest.json
├── generation_config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── preprocessor_config.json
├── special_tokens_map.json
├── tokenizer.model
└── tokenizer_config.json
```

`manifest.json`은 게시 도구가 생성한다.

개념적 형식:

```json
{
  "schemaVersion": 1,
  "model": "MobileVLM_V2-1.7B-MLX",
  "version": "mobilevlm-v2-1.7b-fp16-v1",
  "createdAt": "<UTC timestamp>",
  "totalSize": 3349107824,
  "files": [
    {
      "path": "model-00001-of-00002.safetensors",
      "size": 1984063776,
      "sha256": "<64-character SHA-256>"
    }
  ]
}
```

`latest.json` 형식:

```json
{
  "schemaVersion": 1,
  "version": "mobilevlm-v2-1.7b-fp16-v1",
  "manifestPath": "releases/mobilevlm-v2-1.7b-fp16-v1/manifest.json"
}
```

manifest가 모델 파일의 단일 기준(source of truth)이다.

- 앱에 모델 크기를 하드코딩하지 않는다.
- 앱에 shard 이름과 개수를 하드코딩하지 않는다.
- manifest의 파일 크기 합계가 `totalSize`와 같아야 한다.
- 각 다운로드 파일의 실제 크기와 SHA-256이 manifest와 같아야 한다.

---

## 7. 모델 게시

### 7.1 manifest만 로컬 생성

AWS 업로드 없이 모델 구조와 SHA-256을 확인할 수 있다.

```bash
python tools/model-distribution/publish_model.py \
  --source workspace/mobilevlm-mlx/converted-fp16 \
  --bucket unused \
  --version local-test \
  --manifest-only
```

생성된 로컬 `manifest.json`은 모델 artifact이므로 현재 `.gitignore` 규칙에 따라 Git에 포함하지 않는다.

### 7.2 실제 S3 게시

```bash
python tools/model-distribution/publish_model.py \
  --source workspace/mobilevlm-mlx/converted-fp16 \
  --bucket odic-models-210499750105-ap-northeast-2 \
  --prefix odic/models \
  --version mobilevlm-v2-1.7b-fp16-v1 \
  --profile odic-publisher \
  --region ap-northeast-2 \
  --cloudfront-distribution-id E3MHDM0TX47EYC
```

도구의 게시 순서:

1. 모델 구조와 safetensors index 검사
2. 모든 파일의 SHA-256 계산
3. `releases/<version>/`에 모델 파일 업로드
4. `manifest.json`을 release commit marker로 업로드
5. `latest.json`을 마지막에 갱신
6. distribution ID가 주어지면 `/odic/models/latest.json` invalidation

대용량 업로드는 `--no-progress`를 사용하므로 multipart 조각별 진행 로그를 출력하지 않는다.

### 7.3 버전 규칙

게시한 version 문자열은 재사용하지 않는 것을 원칙으로 한다.

권장 예시:

```text
mobilevlm-v2-1.7b-fp16-v1
mobilevlm-v2-1.7b-int4-v1
mobilevlm-v2-1.7b-fp16-20260825
```

모델을 수정한 경우 기존 S3 key를 덮어쓰지 않고 새 version 경로에 게시한다. 이 방식은 CloudFront의 1년 immutable 캐시와 충돌하지 않는다.

---

## 8. CloudFront OAC 구성

### 8.1 Origin Access Control 생성

OAC 설정 파일 예시:

```json
{
  "Name": "odic-models-oac",
  "Description": "OAC for private ODIC on-device model releases",
  "SigningProtocol": "sigv4",
  "SigningBehavior": "always",
  "OriginAccessControlOriginType": "s3"
}
```

생성:

```bash
aws cloudfront create-origin-access-control \
  --origin-access-control-config file://oac.json \
  --profile odic-publisher
```

현재 OAC ID:

```text
E1DKPKJL8094HJ
```

### 8.2 Distribution 설정

Origin domain:

```text
odic-models-210499750105-ap-northeast-2.s3.ap-northeast-2.amazonaws.com
```

중요한 distribution 설정:

```text
Enabled: true
ViewerProtocolPolicy: redirect-to-https
AllowedMethods: GET, HEAD
Compress: true
HTTPVersion: http2and3
IsIPV6Enabled: true
PriceClass: PriceClass_200
ViewerCertificate: CloudFront default certificate
OriginAccessControlId: E1DKPKJL8094HJ
```

캐시 정책:

| Path | 정책 | 목적 |
|---|---|---|
| Default/release files | `Managed-CachingOptimized` | immutable 모델 장기 캐시 |
| `/odic/models/latest.json` | `Managed-CachingDisabled` | 즉시 최신 버전 확인 |

AWS managed cache policy ID:

```text
Managed-CachingOptimized: 658327ea-f89d-4fab-a63d-7e88639e58f6
Managed-CachingDisabled:  4135ea2d-6df8-44a3-9df3-4b5a84be39ad
```

현재 distribution:

```text
ID:     E3MHDM0TX47EYC
Domain: d3unqpp9xgzmm4.cloudfront.net
```

### 8.3 S3 bucket policy

CloudFront distribution만 모델 prefix를 읽도록 제한한다.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowCloudFrontReadODICModels",
      "Effect": "Allow",
      "Principal": {
        "Service": "cloudfront.amazonaws.com"
      },
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::odic-models-210499750105-ap-northeast-2/odic/models/*",
      "Condition": {
        "StringEquals": {
          "AWS:SourceArn": "arn:aws:cloudfront::210499750105:distribution/E3MHDM0TX47EYC"
        }
      }
    }
  ]
}
```

적용:

```bash
aws s3api put-bucket-policy \
  --bucket odic-models-210499750105-ap-northeast-2 \
  --policy file://bucket-policy.json \
  --profile odic-publisher
```

---

## 9. CDN 검증

환경 변수:

```bash
export ODIC_CDN=https://d3unqpp9xgzmm4.cloudfront.net/odic/models
```

### 9.1 latest와 manifest

```bash
curl --fail --show-error "$ODIC_CDN/latest.json"

curl --fail --show-error \
  "$ODIC_CDN/releases/mobilevlm-v2-1.7b-fp16-v1/manifest.json"
```

정상 응답은 HTTP 200이다.

### 9.2 대형 shard HEAD

```bash
curl --fail --head \
  "$ODIC_CDN/releases/mobilevlm-v2-1.7b-fp16-v1/model-00001-of-00002.safetensors"
```

확인 항목:

```text
HTTP 200
Content-Length: 1984063776
Cache-Control: public,max-age=31536000,immutable
```

### 9.3 Range request

대형 파일의 부분 전송이 가능한지 검사한다.

```bash
curl --fail \
  --range 0-1023 \
  "$ODIC_CDN/releases/mobilevlm-v2-1.7b-fp16-v1/model-00001-of-00002.safetensors" \
  -o /tmp/odic-range.bin

stat -f '%z' /tmp/odic-range.bin
rm /tmp/odic-range.bin
```

기대 결과:

```text
HTTP 206
1024 bytes
```

### 9.4 S3 직접 접근 차단

```bash
curl -o /dev/null -sS -w '%{http_code}\n' \
  https://odic-models-210499750105-ap-northeast-2.s3.ap-northeast-2.amazonaws.com/odic/models/latest.json
```

기대 결과:

```text
403
```

실제 구성 시 확인된 결과:

```text
CloudFront latest.json:  200
CloudFront manifest:     200
Shard HEAD size:         1,984,063,776 bytes
Range request:           206 / 1,024 bytes
Direct S3 request:       403
Incomplete multipart:   0
```

---

## 10. iOS 앱 설정

### 10.1 CDN base URL 주입

Xcode build setting:

```text
ODIC_MODEL_BASE_URL = https://d3unqpp9xgzmm4.cloudfront.net/odic/models
```

`apps/ODIC/Info.plist`:

```xml
<key>ODICModelBaseURL</key>
<string>$(ODIC_MODEL_BASE_URL)</string>
```

`ModelStore`는 런타임에 다음 값을 읽는다.

```swift
Bundle.main.object(forInfoDictionaryKey: "ODICModelBaseURL")
```

HTTPS URL만 허용한다.

### 10.2 앱 설치 흐름

```text
앱 시작
  ├─ Application Support/MobileVLM 설치 여부 확인
  ├─ 설치됨: 카메라 화면 생성 및 카메라 세션 시작
  └─ 미설치: 카메라 화면/세션을 만들지 않고 전용 모델 준비 화면 표시
       └─ latest.json 조회
            └─ manifest.json 조회
                 ├─ manifest.totalSize를 UI에 표시
                 └─ 사용자 다운로드 요청
                      ├─ 사용 가능한 저장 공간 확인
                      ├─ staging 디렉터리 생성
                      ├─ 파일별 URLSessionDownloadDelegate 다운로드
                      ├─ 전송 바이트/전체 바이트 진행률 표시
                      ├─ 사용자 취소 지원
                      ├─ 파일 크기 검사
                      ├─ 스트리밍 SHA-256 검사
                      ├─ installed-manifest.json 기록
                      ├─ MobileVLM 디렉터리 원자적 교체
                      └─ 설치 완료 후에만 카메라 세션 시작
```

최종 설치 경로:

```text
Library/Application Support/MobileVLM
```

다운로드 staging 예시:

```text
Library/Application Support/.MobileVLM-<version>-download
```

이전 버전 임시 backup:

```text
Library/Application Support/.MobileVLM-previous
```

### 10.3 저장 공간

앱은 manifest의 `totalSize`를 기준으로 저장 공간을 검사한다. 모델 용량을 Swift 코드에 하드코딩하지 않는다.

업데이트 중에는 기존 설치와 새 staging 데이터가 동시에 존재할 수 있다. 따라서 모델 크기 외에 여유 공간을 추가로 확보한다.

### 10.4 SHA-256 메모리 사용

약 2GB shard를 `Data(contentsOf:)`로 전부 메모리에 올리지 않는다. `FileHandle`로 8MB씩 읽어 `CryptoKit.SHA256`에 전달한다.

```swift
var hasher = SHA256()
while true {
    let data = try handle.read(upToCount: 8 * 1024 * 1024) ?? Data()
    if data.isEmpty { break }
    hasher.update(data: data)
}
let digest = hasher.finalize()
```

### 10.5 빌드 검증

```bash
xcodebuild \
  -project apps/ODIC/ODIC.xcodeproj \
  -scheme ODIC \
  -sdk iphonesimulator \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath /tmp/odic-derived \
  CODE_SIGNING_ALLOWED=NO \
  build
```

생성된 endpoint 확인:

```bash
/usr/libexec/PlistBuddy \
  -c 'Print :ODICModelBaseURL' \
  /tmp/odic-derived/Build/Products/Debug-iphonesimulator/ODIC.app/Info.plist
```

기대 결과:

```text
https://d3unqpp9xgzmm4.cloudfront.net/odic/models
```

---

## 11. 새 모델 배포 절차

새 모델 버전을 배포할 때의 체크리스트다.

1. 변환 및 수치 검증 완료
2. 이전과 다른 immutable version 결정
3. manifest-only로 로컬 모델 검사
4. S3 release 게시
5. `latest.json` 갱신
6. CloudFront `latest.json` invalidation
7. CDN에서 latest/manifest 확인
8. shard HEAD 및 Range 확인
9. 테스트 iPhone에서 다운로드 및 SHA-256 설치 확인
10. 캡션 end-to-end 실행

예시:

```bash
VERSION=mobilevlm-v2-1.7b-fp16-v2

python tools/model-distribution/publish_model.py \
  --source workspace/mobilevlm-mlx/converted-fp16 \
  --bucket odic-models-210499750105-ap-northeast-2 \
  --prefix odic/models \
  --version "$VERSION" \
  --profile odic-publisher \
  --region ap-northeast-2 \
  --cloudfront-distribution-id E3MHDM0TX47EYC
```

`latest.json`은 CloudFront 캐시 비활성화 정책을 사용하지만 invalidation까지 실행하면 edge에 남은 응답을 명시적으로 제거할 수 있다.

---

## 12. 중단된 multipart upload 정리

게시 도중 프로세스를 중단하면 대형 shard의 multipart upload가 S3에 남을 수 있다.

목록 확인:

```bash
aws s3api list-multipart-uploads \
  --bucket odic-models-210499750105-ap-northeast-2 \
  --profile odic-publisher
```

개별 정리:

```bash
aws s3api abort-multipart-upload \
  --bucket odic-models-210499750105-ap-northeast-2 \
  --key '<object-key>' \
  --upload-id '<upload-id>' \
  --profile odic-publisher
```

`manifest.json`과 `latest.json`은 모델 파일 업로드가 끝난 뒤 게시되므로, 중단된 release가 앱의 최신 모델로 노출되지는 않는다.

주기적으로 incomplete multipart upload를 자동 삭제하려면 S3 lifecycle rule을 추가하는 것도 권장한다.

예시 정책의 핵심:

```json
{
  "Rules": [
    {
      "ID": "AbortIncompleteODICMultipartUploads",
      "Status": "Enabled",
      "Filter": {"Prefix": "odic/models/"},
      "AbortIncompleteMultipartUpload": {
        "DaysAfterInitiation": 1
      }
    }
  ]
}
```

---

## 13. Release rollback

모델 파일은 immutable이므로 rollback은 `latest.json`이 이전 release manifest를 가리키도록 다시 게시하면 된다.

예시:

```json
{
  "schemaVersion": 1,
  "version": "mobilevlm-v2-1.7b-fp16-v1",
  "manifestPath": "releases/mobilevlm-v2-1.7b-fp16-v1/manifest.json"
}
```

업로드:

```bash
aws s3 cp latest.json \
  s3://odic-models-210499750105-ap-northeast-2/odic/models/latest.json \
  --profile odic-publisher \
  --content-type application/json \
  --cache-control 'no-cache,max-age=0'

aws cloudfront create-invalidation \
  --distribution-id E3MHDM0TX47EYC \
  --paths '/odic/models/latest.json' \
  --profile odic-publisher
```

주의: 현재 앱은 동일한 version이 이미 설치되어 있으면 다시 다운로드하지 않는다. 손상된 동일 버전을 교체해야 하는 경우 version 문자열을 재사용하지 말고 새 release version을 게시한다.

---

## 14. 전체 AWS 리소스 삭제

> 이 절차는 모델 CDN을 완전히 폐기할 때만 사용한다.

삭제 순서는 의존성 때문에 중요하다.

1. CloudFront distribution 비활성화
2. distribution 배포 완료 대기
3. CloudFront distribution 삭제
4. OAC 삭제
5. S3 bucket policy 삭제
6. incomplete multipart upload 취소
7. 모든 S3 object version 및 delete marker 삭제
8. S3 bucket 삭제

### 14.1 Distribution 비활성화와 삭제

현재 config와 ETag 조회:

```bash
aws cloudfront get-distribution-config \
  --id E3MHDM0TX47EYC \
  --profile odic-publisher \
  > /tmp/odic-distribution.json
```

`DistributionConfig.Enabled`를 `false`로 바꾼 config를 `update-distribution`에 전달하고 배포를 기다린다.

```bash
aws cloudfront wait distribution-deployed \
  --id E3MHDM0TX47EYC \
  --profile odic-publisher
```

그 후 최신 ETag로 삭제한다.

```bash
aws cloudfront delete-distribution \
  --id E3MHDM0TX47EYC \
  --if-match '<latest-etag>' \
  --profile odic-publisher
```

### 14.2 S3 삭제 시 versioning 주의

버킷에 versioning이 활성화되어 있으므로 현재 object만 삭제해서는 버킷이 비워지지 않는다. 모든 object version과 delete marker를 명시적으로 제거해야 한다.

기존의 다른 AWS 버킷이나 distribution은 ODIC 리소스가 아니므로 삭제하지 않는다.

---

## 15. 운영상 주의사항

### 비용

- S3에 약 3.35GB가 저장된다.
- 버전별 immutable release를 계속 보관하면 저장 용량이 누적된다.
- CloudFront egress와 request 비용이 발생한다.
- iPhone 한 대가 모델 전체를 새로 받으면 약 3.35GB가 전송된다.
- 사용하지 않는 구버전은 충분한 rollback 기간 후 lifecycle 정책으로 정리할 수 있다.

### 모바일 네트워크

현재 downloader는 foreground `URLSessionDownloadDelegate` 기반이며 다음을 지원한다.

- 파일 내부 전송 바이트 단위 진행률
- 전체 모델의 완료 바이트 및 백분율 표시
- 다운로드 취소 및 staging 정리
- 네트워크 대기
- 다운로드/검증 중 카메라 세션 미사용
- 설치 완료 후에만 카메라 프리뷰 시작

실제 제품 운영에서는 다음 개선을 추가로 고려한다.

- Wi-Fi 다운로드 권장 또는 cellular 확인
- background `URLSession`
- 앱 재시작 후 다운로드 resume
- 네트워크별 안내
- low data mode 대응

### 접근 제어

현재 S3는 비공개지만 CloudFront URL은 공개 HTTPS endpoint다. URL을 아는 사용자는 모델을 다운로드할 수 있다.

모델 자체의 접근 제한이 필요하면 다음 중 하나를 추가한다.

- CloudFront Signed URL
- CloudFront Signed Cookie
- 인증 API가 발급하는 단기 URL
- AWS WAF 기반 제한

IAM access key를 iOS 앱에 넣는 방식은 사용하지 않는다.

### 무결성과 진위성

현재 SHA-256 검증은 전송 중 손상 및 manifest와의 불일치를 탐지한다. CDN 또는 manifest 게시 권한 자체가 침해되는 경우까지 방어하려면 다음 단계로 manifest 전자서명을 추가한다.

예시:

- CI/private key로 manifest 서명
- 앱 bundle에 public key 내장
- manifest signature 검증 후 다운로드
- SHA-256 검증 후 모델 활성화

---

## 16. 빠른 상태 점검

### AWS identity

```bash
aws sts get-caller-identity --profile odic-publisher
```

### Distribution

```bash
aws cloudfront get-distribution \
  --id E3MHDM0TX47EYC \
  --profile odic-publisher \
  --query 'Distribution.{Status:Status,Domain:DomainName,Enabled:DistributionConfig.Enabled}'
```

### 모델 metadata

```bash
curl --fail --show-error \
  https://d3unqpp9xgzmm4.cloudfront.net/odic/models/latest.json
```

### S3 incomplete multipart

```bash
aws s3api list-multipart-uploads \
  --bucket odic-models-210499750105-ap-northeast-2 \
  --profile odic-publisher
```

### 앱 빌드

```bash
xcodebuild \
  -project apps/ODIC/ODIC.xcodeproj \
  -scheme ODIC \
  -sdk iphonesimulator \
  -destination 'generic/platform=iOS Simulator' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

---

## 17. 완료된 검증 요약

현재 구성에서 다음 항목을 실제로 확인했다.

- AWS CLI IAM 인증 성공
- 전용 private S3 bucket 생성
- public access block 적용
- AES-256 기본 암호화 적용
- versioning 적용
- BucketOwnerEnforced 적용
- 모델 10개 파일 업로드
- 파일별 SHA-256 manifest 생성
- 두 개 대형 safetensors multipart 업로드 완료
- incomplete multipart upload 0개
- 원격 S3 object 크기와 manifest 크기 일치
- `latest.json`과 manifest version 일치
- CloudFront OAC 생성
- CloudFront distribution 생성 및 배포
- distribution ARN으로 제한된 S3 bucket policy 적용
- CloudFront latest/manifest HTTP 200
- 대형 shard HEAD Content-Length 일치
- Range 요청 HTTP 206
- S3 직접 익명 접근 HTTP 403
- iOS 앱 CDN endpoint 주입
- iOS Simulator 빌드 성공

현재 파이프라인은 다음 상태다.

```text
모델 변환 → manifest 생성 → private S3 게시 → CloudFront CDN
→ iPhone 다운로드 → 크기/SHA-256 검증 → Application Support 설치
```
