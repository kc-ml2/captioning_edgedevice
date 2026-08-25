# On-Device Image Captioning with MLX Swift

This repository is being rebuilt from scratch using [MLX Swift](https://github.com/ml-explore/mlx-swift) for on-device image captioning on Apple platforms.

## Repository layout

```text
.
├── apps/          # MLX Swift applications
├── tools/         # Model conversion and development utilities
└── reference/     # Previous PyTorch, ONNX, Core ML, and iOS implementations
```

The code under `reference/` is retained only as a correctness and architecture reference while the new MLX Swift implementation is developed. See [`reference/README.md`](reference/README.md) for details about the previous implementation.

## Status

A macOS validation CLI now runs the complete MLX Swift MobileVLM pipeline:

```text
JPEG → CLIP preprocessing → CLIP Vision → LDPNetV2 → SentencePiece prompt
→ MobileLlama prefill → KV-cache decode → caption
```

The sample image's full greedy token sequence and caption match the independent
PyTorch reference.

- [MLX Swift 전환 기록](docs/mlx-swift-progress.md)
- [S3 + CloudFront 모델 배포 및 iOS 설치](docs/model-distribution-aws.md)
- [Conversion details](tools/conversion/README.md)
- [Validation and execution](tools/swift-validation/README.md)
