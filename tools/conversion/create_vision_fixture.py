#!/usr/bin/env python3
"""Create a deterministic PyTorch fixture for CLIP embeddings and block zero."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def wanted(name: str) -> bool:
    return (
        name.startswith("vision_model.vision_model.embeddings.")
        or name.startswith("vision_model.vision_model.pre_layrnorm.")
        or name.startswith("vision_model.vision_model.encoder.layers.0.")
    )


def load_weights(directory: Path) -> dict[str, torch.Tensor]:
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    names = {name for name in index["weight_map"] if wanted(name)}
    result: dict[str, torch.Tensor] = {}
    by_shard: dict[str, set[str]] = {}
    for name in names:
        by_shard.setdefault(index["weight_map"][name], set()).add(name)
    for shard, shard_names in by_shard.items():
        with safe_open(directory / shard, framework="pt", device="cpu") as handle:
            for name in shard_names:
                result[name] = handle.get_tensor(name)
    return result


def linear(x: torch.Tensor, weights: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    return functional.linear(x, weights[f"{prefix}.weight"].float(), weights[f"{prefix}.bias"].float())


def layer_norm(x: torch.Tensor, weights: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    return functional.layer_norm(
        x,
        (1024,),
        weights[f"{prefix}.weight"].float(),
        weights[f"{prefix}.bias"].float(),
        eps=1e-5,
    )


def main() -> None:
    args = parse_args()
    weights = load_weights(args.converted)
    root = "vision_model.vision_model"
    layer = f"{root}.encoder.layers.0"

    # Deterministic NHWC RGB input in the approximate normalized CLIP range.
    values = torch.arange(336 * 336 * 3, dtype=torch.float32)
    pixels_nhwc = ((values.remainder(1024) - 512) / 256).reshape(1, 336, 336, 3)

    patch_ohwi = weights[f"{root}.embeddings.patch_embedding.weight"].float()
    patch_oihw = patch_ohwi.permute(0, 3, 1, 2).contiguous()
    with torch.inference_mode():
        patch_nchw = functional.conv2d(pixels_nhwc.permute(0, 3, 1, 2), patch_oihw, stride=14)
        patch_tokens = patch_nchw.flatten(2).transpose(1, 2)
        cls = weights[f"{root}.embeddings.class_embedding"].float().reshape(1, 1, 1024)
        embeddings = torch.cat((cls, patch_tokens), dim=1)
        embeddings = embeddings + weights[f"{root}.embeddings.position_embedding.weight"].float()
        pre_norm = layer_norm(embeddings, weights, f"{root}.pre_layrnorm")

        norm1 = layer_norm(pre_norm, weights, f"{layer}.layer_norm1")
        q = linear(norm1, weights, f"{layer}.self_attn.q_proj")
        k = linear(norm1, weights, f"{layer}.self_attn.k_proj")
        v = linear(norm1, weights, f"{layer}.self_attn.v_proj")
        q = q.reshape(1, 577, 16, 64).transpose(1, 2)
        k = k.reshape(1, 577, 16, 64).transpose(1, 2)
        v = v.reshape(1, 577, 16, 64).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(64)
        probabilities = functional.softmax(scores, dim=-1)
        attention = torch.matmul(probabilities, v).transpose(1, 2).reshape(1, 577, 1024)
        attention = linear(attention, weights, f"{layer}.self_attn.out_proj")
        attention_residual = pre_norm + attention

        norm2 = layer_norm(attention_residual, weights, f"{layer}.layer_norm2")
        hidden = linear(norm2, weights, f"{layer}.mlp.fc1")
        hidden = hidden * torch.sigmoid(1.702 * hidden)  # CLIP QuickGELU
        mlp = linear(hidden, weights, f"{layer}.mlp.fc2")
        block_output = attention_residual + mlp

    fixture = {
        "input.pixels_nhwc": pixels_nhwc,
        "expected.patch_tokens": patch_tokens,
        "expected.embeddings": embeddings,
        "expected.pre_norm": pre_norm,
        "expected.norm1": norm1,
        "expected.q": q,
        "expected.k": k,
        "expected.v": v,
        "expected.attention": attention,
        "expected.attention_residual": attention_residual,
        "expected.norm2": norm2,
        "expected.mlp_hidden": hidden,
        "expected.output": block_output,
        **weights,
    }
    fixture = {name: tensor.contiguous() for name, tensor in fixture.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(fixture, args.output, metadata={"reference": "PyTorch 2.6.0 float32 compute"})
    print(f"Saved {len(fixture)} tensors to {args.output}")
    print(f"Embedding shape: {list(pre_norm.shape)}")
    print(f"Block output shape: {list(block_output.shape)}")


if __name__ == "__main__":
    main()
