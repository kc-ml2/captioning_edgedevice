#!/usr/bin/env python3
"""Create a PyTorch reference fixture for MobileVLM's full selected CLIP output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors import safe_open
from safetensors.torch import save_file

SELECTED_LAYER_COUNT = 23  # hidden_states[-2]: output after encoder layer 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def wanted(name: str) -> bool:
    if name.startswith("projector."):
        return True
    root = "vision_model.vision_model."
    if not name.startswith(root):
        return False
    if name.startswith(root + "embeddings.") or name.startswith(root + "pre_layrnorm."):
        return True
    for layer in range(SELECTED_LAYER_COUNT):
        if name.startswith(root + f"encoder.layers.{layer}."):
            return True
    return False


def load_weights(directory: Path) -> dict[str, torch.Tensor]:
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    names = {name for name in index["weight_map"] if wanted(name)}
    by_shard: dict[str, set[str]] = {}
    for name in names:
        by_shard.setdefault(index["weight_map"][name], set()).add(name)
    result: dict[str, torch.Tensor] = {}
    for shard, shard_names in by_shard.items():
        with safe_open(directory / shard, framework="pt", device="cpu") as handle:
            for name in shard_names:
                result[name] = handle.get_tensor(name)
    return result


def linear(x: torch.Tensor, weights: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    return functional.linear(x, weights[f"{prefix}.weight"].float(), weights[f"{prefix}.bias"].float())


def layer_norm(x: torch.Tensor, weights: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    return functional.layer_norm(
        x, (1024,), weights[f"{prefix}.weight"].float(), weights[f"{prefix}.bias"].float(), 1e-5
    )


def encoder_layer(x: torch.Tensor, weights: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    norm = layer_norm(x, weights, f"{prefix}.layer_norm1")
    q = linear(norm, weights, f"{prefix}.self_attn.q_proj").reshape(1, 577, 16, 64).transpose(1, 2)
    k = linear(norm, weights, f"{prefix}.self_attn.k_proj").reshape(1, 577, 16, 64).transpose(1, 2)
    v = linear(norm, weights, f"{prefix}.self_attn.v_proj").reshape(1, 577, 16, 64).transpose(1, 2)
    probabilities = functional.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(64), dim=-1)
    attention = torch.matmul(probabilities, v).transpose(1, 2).reshape(1, 577, 1024)
    x = x + linear(attention, weights, f"{prefix}.self_attn.out_proj")
    norm = layer_norm(x, weights, f"{prefix}.layer_norm2")
    hidden = linear(norm, weights, f"{prefix}.mlp.fc1")
    hidden = hidden * torch.sigmoid(1.702 * hidden)
    return x + linear(hidden, weights, f"{prefix}.mlp.fc2")


def main() -> None:
    args = parse_args()
    weights = load_weights(args.converted)
    root = "vision_model.vision_model"
    values = torch.arange(336 * 336 * 3, dtype=torch.float32)
    pixels = ((values.remainder(1024) - 512) / 256).reshape(1, 336, 336, 3)

    with torch.inference_mode():
        patch_weight = weights[f"{root}.embeddings.patch_embedding.weight"].float()
        patches = functional.conv2d(
            pixels.permute(0, 3, 1, 2), patch_weight.permute(0, 3, 1, 2), stride=14
        ).flatten(2).transpose(1, 2)
        cls = weights[f"{root}.embeddings.class_embedding"].float().reshape(1, 1, 1024)
        hidden = torch.cat((cls, patches), dim=1)
        hidden = hidden + weights[f"{root}.embeddings.position_embedding.weight"].float()
        hidden = layer_norm(hidden, weights, f"{root}.pre_layrnorm")
        for layer in range(SELECTED_LAYER_COUNT):
            print(f"PyTorch vision layer {layer + 1}/{SELECTED_LAYER_COUNT}")
            hidden = encoder_layer(hidden, weights, f"{root}.encoder.layers.{layer}")
        selected_features = hidden[:, 1:].contiguous()

        projected = functional.linear(
            selected_features,
            weights["projector.mlp.mlp.0.weight"].float(),
            weights["projector.mlp.mlp.0.bias"].float(),
        )
        projected = functional.gelu(projected, approximate="none")
        projected = functional.linear(
            projected,
            weights["projector.mlp.mlp.2.weight"].float(),
            weights["projector.mlp.mlp.2.bias"].float(),
        )
        projected_nchw = projected.transpose(1, 2).reshape(1, 2048, 24, 24)
        projected_nchw = functional.adaptive_avg_pool2d(projected_nchw, (12, 12))
        conv_ohwi = weights["projector.peg.peg.0.weight"].float()
        positional = functional.conv2d(
            projected_nchw,
            conv_ohwi.permute(0, 3, 1, 2).contiguous(),
            weights["projector.peg.peg.0.bias"].float(),
            padding=1,
            groups=2048,
        )
        projector_output = (projected_nchw + positional).flatten(2).transpose(1, 2).contiguous()

    fixture = {
        "input.pixels_nhwc": pixels.contiguous(),
        "expected.selected_features": selected_features,
        "expected.projector_output": projector_output,
        **{name: tensor.contiguous() for name, tensor in weights.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        fixture,
        args.output,
        metadata={
            "reference": "PyTorch 2.6.0 float32 compute",
            "vision_select_layer": "-2",
            "executed_encoder_layers": str(SELECTED_LAYER_COUNT),
        },
    )
    print(f"Saved {len(fixture)} tensors to {args.output}")
    print(f"Selected feature shape: {list(selected_features.shape)}")
    print(f"Projector output shape: {list(projector_output.shape)}")


if __name__ == "__main__":
    main()
