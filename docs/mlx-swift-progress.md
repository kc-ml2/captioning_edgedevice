# MobileVLM MLX Swift 전환 기록

## 1. 목표

기존 PyTorch, ONNX, Core ML 기반 코드를 직접 수정하지 않고, Apple 플랫폼용 이미지 캡셔닝 파이프라인을 **Swift + MLX Swift로 처음부터 다시 구현**한다.

최종 런타임의 목표 구조는 다음과 같다.

```text
JPEG / UIImage
→ Swift 이미지 전처리
→ MLX Swift CLIP Vision Encoder
→ MLX Swift LDPNetV2 Projector
→ Swift SentencePiece Tokenizer
→ Multimodal Embedding
→ MLX Swift MobileLlama
→ KV-cache Autoregressive Decoding
→ Caption
```

Python은 앱 런타임에 포함하지 않는다. 원본 가중치 변환과 PyTorch 기준 출력 생성에만 사용한다.

---

## 2. 저장소 정리

기존 구현은 삭제하지 않고 정확성 비교를 위한 reference로 이동했다.

```text
.
├── apps/                       # 향후 제품용 MLX Swift 앱
├── docs/                       # 새 프로젝트 문서
├── tools/
│   ├── conversion/             # 가중치 변환 및 PyTorch reference 생성
│   └── swift-validation/       # MLX Swift 수치 검증 및 end-to-end CLI
├── workspace/                  # 모델과 임시 산출물, Git 제외
└── reference/                  # 기존 PyTorch/ONNX/Core ML/iOS 구현
```

작업 브랜치:

```text
feat/mlx-swift
```

기존 코드 이동과 기본 설정은 다음 커밋으로 원격에 push되어 있다.

```text
bbf31d2 chore: move existing implementations to reference
```

`.gitignore`에는 Python/Xcode 생성물과 모델 파일, 로컬 작업공간이 포함되어 있다.

```text
/workspace/
*.safetensors
*.bin
*.onnx
*.mlpackage/
DerivedData/
.build/
__pycache__/
```

---

## 3. 사용 모델

### MobileVLM

```text
mtgv/MobileVLM_V2-1.7B
```

주요 설정:

| 항목 | 값 |
|---|---:|
| Language hidden size | 2048 |
| Intermediate size | 5632 |
| Decoder layers | 24 |
| Attention heads | 16 |
| KV heads | 16 |
| Head dimension | 128 |
| Vocabulary size | 32000 |
| RMSNorm epsilon | 1e-6 |
| RoPE theta | 10000 |
| EOS token | 2 |

### Vision Encoder

```text
openai/clip-vit-large-patch14-336
```

주요 설정:

| 항목 | 값 |
|---|---:|
| Input size | 336 × 336 |
| Patch size | 14 × 14 |
| Patch tokens | 576 |
| Hidden size | 1024 |
| Vision layers | 24 |
| Attention heads | 16 |

MobileVLM은 다음 vision feature를 사용한다.

```text
hidden_states[-2][:, 1:]
```

즉 CLIP encoder layer 22까지 실행한 결과에서 CLS token을 제거하여 `[1, 576, 1024]` feature를 만든다.

### 중요한 발견

MobileVLM checkpoint 안에는 fine-tuned CLIP Vision Tower가 이미 포함되어 있다.

```text
model.vision_tower.vision_tower.vision_model.*
```

이 값은 별도로 받은 OpenAI CLIP 원본과 완전히 같지 않다. 따라서 변환 시:

- Vision weight는 **MobileVLM 내부 checkpoint**에서 가져온다.
- Vision config와 preprocessing metadata만 OpenAI CLIP 저장소에서 가져온다.
- OpenAI CLIP weight로 MobileVLM 내부 vision weight를 덮어쓰지 않는다.

---

## 4. 로컬 모델 작업공간

원본 및 변환 모델은 Git에 넣지 않고 다음 위치에 보관한다.

```text
workspace/mobilevlm-mlx/
├── .venv/
├── source/
│   ├── mtgv--MobileVLM_V2-1.7B/
│   └── openai--clip-vit-large-patch14-336/
├── converted-fp16/
├── validation/
└── xcode-derived/
```

현재 대략적인 사용량:

| 디렉터리 | 크기 |
|---|---:|
| 원본 모델 | 4.7 GB |
| FP16 변환본 | 3.1 GB |
| 검증 fixture | 5.8 GB 이상 |
| Xcode DerivedData | 약 0.9 GB |

검증 fixture는 다시 생성할 수 있으므로 공간이 필요하면 삭제해도 된다.

---

## 5. 가중치 변환

변환 도구:

```text
tools/conversion/convert_mobilevlm.py
tools/conversion/verify_conversion.py
```

Python 환경은 3.10–3.12를 사용한다. 현재 로컬 검증에는 Python 3.12를 사용했다.

```bash
python3.12 -m venv workspace/mobilevlm-mlx/.venv
source workspace/mobilevlm-mlx/.venv/bin/activate
pip install -r tools/conversion/requirements.txt
```

변환 명령:

```bash
workspace/mobilevlm-mlx/.venv/bin/python \
  tools/conversion/convert_mobilevlm.py \
  --mobilevlm workspace/mobilevlm-mlx/source/mtgv--MobileVLM_V2-1.7B \
  --clip workspace/mobilevlm-mlx/source/openai--clip-vit-large-patch14-336 \
  --output workspace/mobilevlm-mlx/converted-fp16 \
  --dtype float16
```

변환 결과:

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

### 변환 규칙

PyTorch key를 다음 namespace로 정리한다.

```text
model.vision_tower.vision_tower.* → vision_model.*
model.mm_projector.*              → projector.*
model.embed_tokens.*              → language_model.embed_tokens.*
model.layers.*                    → language_model.layers.*
model.norm.*                      → language_model.norm.*
lm_head.*                         → lm_head.*
```

PyTorch Conv2D weight는 다음과 같이 변환한다.

```text
PyTorch OIHW: [output, input, height, width]
MLX OHWI:     [output, height, width, input]
```

적용 대상은 CLIP patch embedding과 LDPNetV2 depthwise convolution이다.

### 변환 검증 결과

검증 명령:

```bash
workspace/mobilevlm-mlx/.venv/bin/python \
  tools/conversion/verify_conversion.py \
  --mobilevlm workspace/mobilevlm-mlx/source/mtgv--MobileVLM_V2-1.7B \
  --converted workspace/mobilevlm-mlx/converted-fp16
```

결과:

```json
{
  "checked_tensor_count": 616,
  "converted_tensor_count": 616,
  "expected_tensor_count": 616,
  "failure_count": 0,
  "max_abs_error": 0.0,
  "max_relative_error": 0.0,
  "passed": true
}
```

이 검증은 key, shape, dtype, layout 변환 및 safetensors에 저장된 실제 값이 의도한 변환 결과와 정확히 같음을 증명한다.

---

## 6. MLX Swift 검증 방법

Swift 검증 패키지:

```text
tools/swift-validation/
```

사용 버전:

```text
mlx-swift 0.31.6
swift-sentencepiece 0.0.6
```

MLX Swift의 Metal shader는 일반 `swift run`만으로 준비되지 않는다. 실행 가능한 바이너리는 Xcode로 빌드해야 한다.

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild build \
  -scheme <Scheme> \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath workspace/mobilevlm-mlx/xcode-derived \
  CODE_SIGNING_ALLOWED=NO
```

현재 전역 `xcode-select`가 Command Line Tools를 가리킬 수 있으므로 `DEVELOPER_DIR`를 명시한다.

---

## 7. 단계별 수치 검증 결과

### 7.1 LDPNetV2 Projector

검증 target:

```text
ProjectorValidation
```

검증 범위:

```text
Linear 1024→2048
→ exact GELU
→ Linear 2048→2048
→ 24×24에서 12×12 average pooling
→ depthwise 3×3 Conv2D
→ residual
```

최종 출력:

```text
shape: [1, 144, 2048]
max absolute error: 약 1.0e-6
result: PASS
```

### 7.2 CLIP 첫 Transformer block

검증 target:

```text
VisionBlockValidation
```

검증 범위:

```text
patch embedding
→ CLS/position embedding
→ LayerNorm
→ Q/K/V
→ multi-head attention
→ residual
→ QuickGELU MLP
→ residual
```

주요 결과:

```text
patch tokens max error: 0
block output max error: 약 6.85e-6
result: PASS
```

### 7.3 전체 Vision Encoder와 Projector

검증 target:

```text
FullVisionValidation
```

CLIP encoder layer 0–22를 실행한 뒤 CLS를 제거하고 projector까지 연결했다.

```text
selected vision feature:
  shape: [1, 576, 1024]
  max abs: 0.0062880516
  mean abs: 7.07e-6
  cosine similarity: 약 1.0

projector output:
  shape: [1, 144, 2048]
  max abs: 0.0009797215
  mean abs: 1.95e-6

result: PASS
```

### 7.4 MobileLlama 첫 Decoder block

검증 target:

```text
LanguageBlockValidation
```

검증 범위:

```text
embedding
→ RMSNorm
→ Q/K/V
→ RoPE
→ causal attention
→ residual
→ RMSNorm
→ SwiGLU
→ residual
→ final RMSNorm
→ LM head
```

주요 결과:

```text
block output max error: 약 7.39e-6
logits max error: 약 4.48e-5
last-token argmax: 11687, PyTorch와 일치
result: PASS
```

### 7.5 전체 MobileLlama Prefill

검증 target:

```text
FullLanguageValidation
```

24개 decoder layer 전체와 final RMSNorm, LM head를 비교했다.

```text
hidden:
  max abs: 0.006591797
  mean abs: 9.70e-6
  cosine similarity: 약 1.0

logits:
  max abs: 0.00039672852
  mean abs: 6.97e-5
  cosine similarity: 약 1.0

last-token argmax: 2, PyTorch와 일치
result: PASS
```

### 7.6 Multimodal Prefill과 KV Cache

검증 target:

```text
KVCacheValidation
```

검증 범위:

```text
image sentinel -200 교체
→ multimodal embedding
→ 24-layer prefill
→ 모든 layer의 K/V cache
→ first greedy token
→ 1-token cached decode
→ cache append
→ second greedy token
```

결과:

```text
multimodal embedding max error: 0
prefill hidden max error: 0.006591797
prefill logits max error: 9.92e-5
first token: 13, 일치

decode hidden max error: 0.0011901855
decode logits max error: 0.00045776367
second token: 13, 일치

result: PASS
```

---

## 8. 실제 End-to-End Caption

Swift target:

```text
EndToEndCaption
```

소스:

```text
tools/swift-validation/Sources/EndToEndCaption/main.swift
```

빌드:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild build \
  -scheme EndToEndCaption \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath workspace/mobilevlm-mlx/xcode-derived \
  CODE_SIGNING_ALLOWED=NO
```

실행:

```bash
workspace/mobilevlm-mlx/xcode-derived/Build/Products/Debug/EndToEndCaption \
  workspace/mobilevlm-mlx/converted-fp16 \
  reference/000000000139.jpg
```

기본 질문:

```text
What objects are visible in the scene?
```

네 번째 인자로 질문을 변경할 수 있다.

```bash
.../EndToEndCaption \
  workspace/mobilevlm-mlx/converted-fp16 \
  reference/000000000139.jpg \
  "Describe the image in detail."
```

### Prompt

```text
A chat between a curious user and an artificial intelligence assistant. The
assistant gives helpful, detailed, and polite answers to the user's questions.
USER: <image>
What objects are visible in the scene? ASSISTANT:
```

Prompt token 수는 image sentinel을 포함해 51개이며, 1개의 image token을 144개 image feature로 교체한 최종 multimodal 길이는 194이다.

### 생성 결과

Python reference와 Swift가 생성한 32개 token 전체가 일치했다.

```text
[512, 278, 9088, 29892, 727, 338, 263, 8471,
 5716, 411, 263, 3974, 6689, 29892, 263, 11456,
 29892, 263, 1591, 29892, 521, 7121, 29892, 322,
 263, 6114, 13407, 297, 278, 29181, 29889, 2]
```

최종 caption:

```text
In the scene, there is a living room with a fireplace, a television, a table,
chairs, and a woman standing in the kitchen.
```

Python reference와 Swift의 최종 문자열도 동일하다.

Python 기준 실행:

```bash
workspace/mobilevlm-mlx/.venv/bin/python \
  tools/conversion/run_reference_caption.py \
  --model workspace/mobilevlm-mlx/converted-fp16 \
  --image reference/000000000139.jpg \
  --output workspace/mobilevlm-mlx/validation/reference-caption.json
```

---

## 9. 현재 증명된 것

현재 다음 항목은 실제 비교를 통해 확인됐다.

- PyTorch checkpoint 616개 tensor의 MLX용 변환이 정확하다.
- Swift에서 sharded safetensors를 정상 로드한다.
- CLIP Vision Encoder의 핵심 연산과 전체 선택 feature가 PyTorch와 일치한다.
- LDPNetV2 출력이 PyTorch와 일치한다.
- MobileLlama의 RMSNorm, RoPE, causal attention, SwiGLU가 일치한다.
- 24개 language layer의 prefill logits가 일치한다.
- 모든 layer의 prefill KV cache가 일치한다.
- cached decode 결과와 cache append가 일치한다.
- 실제 SentencePiece prompt tokenization이 Python과 일치한다.
- 실제 샘플 이미지에서 생성 token 전체와 caption이 일치한다.

따라서 현재 prototype은 **모델 정확성 관점에서 동작하는 완전한 MLX Swift MobileVLM 파이프라인**이다.

---

## 10. 현재 한계

`EndToEndCaption`은 빠른 정확성 검증을 위해 대부분의 로직이 한 파일에 들어 있는 prototype이다.

아직 다음 작업은 하지 않았다.

- 제품용 `MobileVLMCore` 모듈 분리
- iOS SwiftUI 앱
- 모델 다운로드와 캐시 관리
- progress/cancellation API
- 동시성 및 background inference 설계
- 메모리 사용량 최적화
- 모델 shard의 선택적/lazy loading
- 4-bit 양자화
- prefill 및 decode 성능 최적화
- 다양한 이미지와 질문에 대한 회귀 테스트
- 이미지 전처리만 별도로 PyTorch와 픽셀 단위 비교하는 자동 테스트

실제 샘플의 최종 token sequence가 일치하므로 현재 이미지 전처리는 이 샘플에서 호환된다. 다만 다양한 이미지 크기와 방향을 다루려면 padding, orientation, resize interpolation에 대한 별도 fixture가 필요하다.

---

## 11. 권장 다음 단계

### 1단계: 현재 검증 도구 커밋

현재 변경 사항을 먼저 독립 커밋으로 남긴다.

```text
feat: add MLX Swift conversion and validation pipeline
```

### 2단계: 제품용 Core 모듈 구성

```text
apps/MobileVLMMLX/
├── Package.swift
├── Sources/
│   ├── MobileVLMCore/
│   │   ├── Configuration/
│   │   ├── ModelLoading/
│   │   ├── Vision/
│   │   ├── Projector/
│   │   ├── Language/
│   │   ├── Tokenizer/
│   │   └── Pipeline/
│   └── MobileVLMCLI/
└── Tests/
```

검증용 `EndToEndCaption` 구현을 그대로 복사하기보다 작은 타입으로 분리한다.

### 3단계: 회귀 테스트

최소한 다음을 자동화한다.

- tokenizer token IDs
- 전처리 tensor
- vision selected feature
- projector output
- prefill first token
- cached decode token
- 샘플 이미지의 전체 generated token sequence

### 4단계: iOS 앱 연결

CLI가 제품 구조로 정리된 후 SwiftUI 앱을 추가한다.

- 사진 선택 및 카메라
- 비동기 모델 로딩
- background inference
- 진행 상태
- 취소
- 메모리 경고 대응

### 5단계: 최적화

FP16 파이프라인을 기준으로 정확성을 보존한 뒤 다음을 검토한다.

- 4-bit language model 양자화
- weight lazy loading 또는 memory mapping
- MLX compilation
- attention 최적화
- KV cache 사전 할당
- token decode latency 측정

---

## 12. 관련 문서와 코드

```text
README.md
reference/README.md
tools/conversion/README.md
tools/swift-validation/README.md
```

핵심 파일:

```text
tools/conversion/convert_mobilevlm.py
tools/conversion/verify_conversion.py
tools/conversion/run_reference_caption.py

tools/swift-validation/Sources/EndToEndCaption/main.swift
tools/swift-validation/Sources/FullVisionValidation/main.swift
tools/swift-validation/Sources/FullLanguageValidation/main.swift
tools/swift-validation/Sources/KVCacheValidation/main.swift
```
