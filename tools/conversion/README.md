# MobileVLM to MLX conversion

These scripts convert the original MobileVLM V2 checkpoint into sharded
`safetensors` suitable for loading from MLX Swift.

## Source models

- `mtgv/MobileVLM_V2-1.7B`
- `openai/clip-vit-large-patch14-336` (configuration and preprocessing metadata)

MobileVLM already embeds a fine-tuned copy of the CLIP vision tower. The
converter intentionally uses those embedded vision weights rather than
replacing them with the base OpenAI CLIP weights.

## Environment

Use Python 3.10–3.12.

```bash
python3.12 -m venv workspace/mobilevlm-mlx/.venv
source workspace/mobilevlm-mlx/.venv/bin/activate
pip install -r tools/conversion/requirements.txt
```

## Convert

```bash
python tools/conversion/convert_mobilevlm.py \
  --mobilevlm workspace/mobilevlm-mlx/source/mtgv--MobileVLM_V2-1.7B \
  --clip workspace/mobilevlm-mlx/source/openai--clip-vit-large-patch14-336 \
  --output workspace/mobilevlm-mlx/converted-fp16 \
  --dtype float16
```

The converter:

- retains all 616 MobileVLM tensors;
- gives the vision, projector, language model, and LM head stable namespaces;
- converts Conv2d weights from PyTorch OIHW to MLX OHWI layout;
- casts floating-point tensors to the selected output dtype;
- copies tokenizer, generation, and image preprocessing metadata;
- creates a conversion manifest and sharded safetensors index.

## Verify checkpoint conversion

```bash
python tools/conversion/verify_conversion.py \
  --mobilevlm workspace/mobilevlm-mlx/source/mtgv--MobileVLM_V2-1.7B \
  --converted workspace/mobilevlm-mlx/converted-fp16 \
  --report workspace/mobilevlm-mlx/validation/conversion-fp16.json
```

Verification reconstructs every expected output tensor from the source,
including dtype and layout transformations, then compares its shape, dtype,
and SHA-256 value with the saved tensor. A successful report has zero missing,
unexpected, or mismatched tensors.

## End-to-end numerical validation

Checkpoint verification proves that conversion and serialization are correct;
it does not prove that a Swift model implementation performs the same
computation. Once the MLX Swift modules exist, validate deterministic outputs
at these boundaries using the same image and prompt:

1. normalized image tensor;
2. CLIP penultimate hidden state after removing the CLS token (`1 x 576 x 1024`);
3. LDPNetV2 projector output (`1 x 144 x 2048`);
4. token IDs and multimodal embeddings;
5. first-token logits and argmax token;
6. per-layer KV-cache shapes and values;
7. complete greedy-generated token sequence.

The first cross-runtime check is implemented in
[`../swift-validation`](../swift-validation/README.md): it compares every
LDPNetV2 projector boundary between PyTorch and MLX Swift.

Use float32 first to isolate implementation errors, then float16. Suggested
initial tolerances are `atol=1e-5, rtol=1e-5` for float32 module outputs and
`atol=1e-2, rtol=1e-2` for float16. End-to-end greedy token IDs should match;
if logits near the maximum are tied, compare top-k IDs and score differences
before treating a token mismatch as a conversion failure.
