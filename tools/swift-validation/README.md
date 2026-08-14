# MLX Swift numerical validation

This Swift package validates MLX Swift model components against deterministic
PyTorch reference fixtures.

## LDPNetV2 projector

Create the fixture from the converted FP16 checkpoint:

```bash
workspace/mobilevlm-mlx/.venv/bin/python \
  tools/conversion/create_projector_fixture.py \
  --converted workspace/mobilevlm-mlx/converted-fp16 \
  --output workspace/mobilevlm-mlx/validation/projector.safetensors
```

Build with Xcode so that MLX's Metal shaders are compiled, then run the
executable:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild build \
  -scheme MobileVLMMLXValidation \
  -destination 'platform=macOS' \
  -derivedDataPath workspace/mobilevlm-mlx/xcode-derived \
  CODE_SIGNING_ALLOWED=NO

workspace/mobilevlm-mlx/xcode-derived/Build/Products/Debug/ProjectorValidation \
  workspace/mobilevlm-mlx/validation/projector.safetensors
```

`swift run` can compile the Swift package but does not build MLX's Metal shader
library, so it is not sufficient for executing this validation target.

The test compares the first linear layer, exact GELU, second linear layer,
average pooling, depthwise positional convolution, and final projector output.
It exits with a non-zero status if any boundary exceeds its FP16 tolerance.

## CLIP Vision Encoder

Create a fixture for patch embedding and transformer block zero:

```bash
workspace/mobilevlm-mlx/.venv/bin/python \
  tools/conversion/create_vision_fixture.py \
  --converted workspace/mobilevlm-mlx/converted-fp16 \
  --output workspace/mobilevlm-mlx/validation/vision-block.safetensors
```

Build and run `VisionBlockValidation` to compare patch embedding, LayerNorm,
Q/K/V projections, multi-head attention, QuickGELU MLP, and residual outputs.

For the complete vision path, create the full fixture and run the full target:

```bash
workspace/mobilevlm-mlx/.venv/bin/python \
  tools/conversion/create_full_vision_fixture.py \
  --converted workspace/mobilevlm-mlx/converted-fp16 \
  --output workspace/mobilevlm-mlx/validation/full-vision.safetensors

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild build \
  -scheme FullVisionValidation \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath workspace/mobilevlm-mlx/xcode-derived \
  CODE_SIGNING_ALLOWED=NO

workspace/mobilevlm-mlx/xcode-derived/Build/Products/Debug/FullVisionValidation \
  workspace/mobilevlm-mlx/validation/full-vision.safetensors
```

This executes CLIP through encoder layer 22, selects
`hidden_states[-2][:, 1:]`, and runs LDPNetV2. It validates the complete MLX
Swift vision-to-projector path against PyTorch.

## MobileLlama

`LanguageBlockValidation` compares decoder block zero boundary by boundary,
including RMSNorm, RoPE, causal attention, SwiGLU, residuals, final RMSNorm,
and the LM head.

For the complete 24-layer causal prefill path:

```bash
workspace/mobilevlm-mlx/.venv/bin/python \
  tools/conversion/create_full_language_fixture.py \
  --converted workspace/mobilevlm-mlx/converted-fp16 \
  --output workspace/mobilevlm-mlx/validation/full-language.safetensors

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild build \
  -scheme FullLanguageValidation \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath workspace/mobilevlm-mlx/xcode-derived \
  CODE_SIGNING_ALLOWED=NO

workspace/mobilevlm-mlx/xcode-derived/Build/Products/Debug/FullLanguageValidation \
  workspace/mobilevlm-mlx/validation/full-language.safetensors
```

This validates all 24 decoder layers, final RMSNorm, full vocabulary logits,
and the final greedy token.

## Multimodal prefill and KV cache

Create and run the cached-decode fixture:

```bash
workspace/mobilevlm-mlx/.venv/bin/python \
  tools/conversion/create_kv_cache_fixture.py \
  --converted workspace/mobilevlm-mlx/converted-fp16 \
  --output workspace/mobilevlm-mlx/validation/kv-cache.safetensors

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild build \
  -scheme KVCacheValidation \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath workspace/mobilevlm-mlx/xcode-derived \
  CODE_SIGNING_ALLOWED=NO

workspace/mobilevlm-mlx/xcode-derived/Build/Products/Debug/KVCacheValidation \
  workspace/mobilevlm-mlx/validation/kv-cache.safetensors
```

This replaces an image sentinel token with image features, executes the full
24-layer multimodal prefill, validates every layer's key/value cache, selects
the first greedy token, and performs one cached decode step. Prefill/decode
logits and both generated token IDs are compared with PyTorch.

## End-to-end caption

Build and run the complete MLX Swift pipeline with the converted model and a
JPEG image:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild build \
  -scheme EndToEndCaption \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath workspace/mobilevlm-mlx/xcode-derived \
  CODE_SIGNING_ALLOWED=NO

workspace/mobilevlm-mlx/xcode-derived/Build/Products/Debug/EndToEndCaption \
  workspace/mobilevlm-mlx/converted-fp16 \
  reference/000000000139.jpg
```

An optional fourth argument overrides the default question.

The target runs native image loading and CLIP preprocessing, SentencePiece
prompt tokenization, CLIP Vision, LDPNetV2, multimodal embedding, MobileLlama
prefill, KV-cache decoding, and SentencePiece output decoding. For the sample
image, its complete 32-token greedy sequence matches the Python reference and
produces:

```text
In the scene, there is a living room with a fireplace, a television, a table,
chairs, and a woman standing in the kitchen.
```

Generate the independent Python reference with:

```bash
workspace/mobilevlm-mlx/.venv/bin/python \
  tools/conversion/run_reference_caption.py \
  --model workspace/mobilevlm-mlx/converted-fp16 \
  --image reference/000000000139.jpg \
  --output workspace/mobilevlm-mlx/validation/reference-caption.json
```
