# ODIC TestFlight 배포

## 현재 앱 설정

- Scheme: `ODIC`
- Bundle ID: `com.ml2.scenesense`
- Version: `1.0`
- Build: `1`
- Minimum iOS: `17.0`
- Device family: iPhone
- Signing team: `BZ544YT29Z`
- Model endpoint: `https://d3unqpp9xgzmm4.cloudfront.net/odic/models`

## 최초 업로드 전 준비

1. Apple Developer Program에 가입된 계정을 Xcode에 로그인합니다.
   - Xcode → Settings → Accounts
2. App Store Connect에서 새 앱을 생성합니다.
   - Platform: iOS
   - Bundle ID: `com.ml2.scenesense`
   - SKU: 원하는 고유 값(예: `odic-ios`)
3. `apps/ODIC/ODIC/Assets.xcassets/AppIcon.appiconset`의 앱 아이콘을 확인합니다.
   - `apps/AppIcons/Assets.xcassets/AppIcon.appiconset/1024.png`를 기반으로 적용되어 있습니다.
   - 배포용 이미지는 1024×1024 RGB PNG이며 알파 채널을 제거했습니다. 원본 팩은 그대로 보관합니다.
   - iOS용 단일 이미지 설정으로 필요한 아이콘 크기를 Xcode가 생성합니다.
   - 아이콘 교체 시 투명도(alpha)가 없는 정사각형 이미지를 사용합니다. 모서리 마스킹은 iOS가 처리합니다.
4. Xcode에서 `ODIC` target → Signing & Capabilities를 확인합니다.
   - Automatically manage signing: On
   - Team: 올바른 배포 팀
   - Bundle Identifier: App Store Connect에 등록한 값과 동일
5. 실제 iPhone에서 Release 동작을 확인합니다.
   - 카메라 권한 문구
   - 약 1.39 GB 모델 다운로드
   - 모델 다운로드 취소/재시도
   - 캡션 생성

## Xcode에서 업로드

1. `apps/ODIC/ODIC.xcodeproj`를 엽니다.
2. 실행 대상을 **Any iOS Device (arm64)** 또는 **Generic iOS Device**로 선택합니다.
3. Product → Archive를 실행합니다.
4. Organizer에서 생성된 archive를 선택합니다.
5. Distribute App → App Store Connect → Upload를 선택합니다.
6. 자동 서명 옵션을 유지하고 Validate/Upload를 완료합니다.
7. App Store Connect → TestFlight에서 빌드 처리가 끝날 때까지 기다립니다.

## 테스터 배포

### 내부 테스터

- App Store Connect 사용자만 초대할 수 있습니다.
- 빌드 처리 후 바로 테스트 가능하며 Beta App Review가 일반적으로 필요 없습니다.

### 외부 테스터

- 테스트 정보, 연락처, 로그인 필요 여부를 입력해야 합니다.
- 첫 외부 빌드는 Beta App Review를 거칩니다.
- 심사 메모에 다음 내용을 적는 것을 권장합니다.
  - 앱 실행 후 온디바이스 모델(약 1.39 GB)을 한 번 다운로드해야 함
  - 카메라 영상에서 이미지 설명을 기기 내에서 생성함
  - 모델 파일만 CDN에서 내려받고 캡션 처리는 기기에서 수행함

## 빌드 번호 규칙

같은 마케팅 버전(`1.0`)을 다시 업로드할 때마다 `CURRENT_PROJECT_VERSION`을 증가시켜야 합니다.

예:

- 첫 업로드: Version `1.0`, Build `1`
- 두 번째 업로드: Version `1.0`, Build `2`

Xcode의 target → General → Identity → Build에서 변경할 수 있습니다.

## App Store Connect에 준비할 정보

- 앱 이름과 부제
- 기본 언어
- 카테고리
- 개인정보 처리방침 URL(외부 테스트 및 최종 출시를 고려하면 권장)
- Beta App Description
- Feedback Email
- 심사 연락처
- 암호화 수출 규정 응답
- 테스트용 로그인 정보(로그인이 있는 경우)

이 앱이 HTTPS/CryptoKit의 SHA-256 파일 무결성 확인만 사용하고 별도 암호화 기능을 제공하지 않는다면, App Store Connect의 수출 규정 질문을 실제 구현에 맞게 답변합니다. 법률 판단이 필요한 경우 조직의 담당자에게 확인합니다.

## 로컬 검증 명령

서명 없이 Release 아카이브 컴파일을 확인하려면:

```bash
xcodebuild \
  -project apps/ODIC/ODIC.xcodeproj \
  -scheme ODIC \
  -configuration Release \
  -destination 'generic/platform=iOS' \
  -archivePath /tmp/ODIC.xcarchive \
  CODE_SIGNING_ALLOWED=NO \
  archive
```

실제 TestFlight 업로드용 archive는 Xcode Organizer에서 배포 인증서와 프로비저닝 프로파일을 사용해 생성합니다.
