#!/usr/bin/env python3
"""Create a PyTorch reference fixture for all 24 MobileLlama decoder layers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors import safe_open
from safetensors.torch import save_file

SEQUENCE_LENGTH = 16
HIDDEN_SIZE = 2048
HEAD_COUNT = 16
HEAD_DIMENSION = 128
LAYER_COUNT = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def wanted(name: str) -> bool:
    return (
        name == "language_model.embed_tokens.weight"
        or name == "language_model.norm.weight"
        or name == "lm_head.weight"
        or name.startswith("language_model.layers.")
    )


def load_weights(directory: Path) -> dict[str, torch.Tensor]:
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    names = {name for name in index["weight_map"] if wanted(name)}
    grouped: dict[str, set[str]] = {}
    for name in names:
        grouped.setdefault(index["weight_map"][name], set()).add(name)
    result: dict[str, torch.Tensor] = {}
    for shard, shard_names in grouped.items():
        with safe_open(directory / shard, framework="pt", device="cpu") as handle:
            for name in shard_names:
                result[name] = handle.get_tensor(name)
    return result


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return functional.linear(x, weight.float())


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight.float()


def apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(SEQUENCE_LENGTH, dtype=torch.float32)
    dimensions = torch.arange(0, HEAD_DIMENSION, 2, dtype=torch.float32)
    inverse_frequency = 1.0 / (10_000.0 ** (dimensions / HEAD_DIMENSION))
    frequencies = torch.outer(positions, inverse_frequency)
    embedding = torch.cat((frequencies, frequencies), -1)[None, None]
    cosine, sine = embedding.cos(), embedding.sin()

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, -1)
        return torch.cat((-second, first), -1)

    return q * cosine + rotate_half(q) * sine, k * cosine + rotate_half(k) * sine


def decoder_layer(x: torch.Tensor, weights: dict[str, torch.Tensor], layer: int) -> torch.Tensor:
    prefix = f"language_model.layers.{layer}"
    norm = rms_norm(x, weights[f"{prefix}.input_layernorm.weight"])
    q = linear(norm, weights[f"{prefix}.self_attn.q_proj.weight"])
    k = linear(norm, weights[f"{prefix}.self_attn.k_proj.weight"])
    v = linear(norm, weights[f"{prefix}.self_attn.v_proj.weight"])
    q = q.reshape(1, SEQUENCE_LENGTH, HEAD_COUNT, HEAD_DIMENSION).transpose(1, 2)
    k = k.reshape(1, SEQUENCE_LENGTH, HEAD_COUNT, HEAD_DIMENSION).transpose(1, 2)
    v = v.reshape(1, SEQUENCE_LENGTH, HEAD_COUNT, HEAD_DIMENSION).transpose(1, 2)
    q, k = apply_rope(q, k)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(HEAD_DIMENSION)
    mask = torch.triu(torch.full((SEQUENCE_LENGTH, SEQUENCE_LENGTH), float("-inf")), 1)
    probabilities = functional.softmax(scores + mask, -1)
    attention = torch.matmul(probabilities, v).transpose(1, 2).reshape(1, SEQUENCE_LENGTH, HIDDEN_SIZE)
    x = x + linear(attention, weights[f"{prefix}.self_attn.o_proj.weight"])
    norm = rms_norm(x, weights[f"{prefix}.post_attention_layernorm.weight"])
    gate = functional.silu(linear(norm, weights[f"{prefix}.mlp.gate_proj.weight"]))
    up = linear(norm, weights[f"{prefix}.mlp.up_proj.weight"])
    return x + linear(gate * up, weights[f"{prefix}.mlp.down_proj.weight"])


def main() -> None:
    args = parse_args()
    weights = load_weights(args.converted)
    token_ids = torch.tensor(
        [1, 319, 13563, 1546, 263, 1967, 29889, 13, 100, 101, 102, 103, 104, 105, 106, 107],
        dtype=torch.int32,
    ).reshape(1, SEQUENCE_LENGTH)
    embedding_rows = weights["language_model.embed_tokens.weight"][token_ids.flatten().long()].float()

    with torch.inference_mode():
        hidden = embedding_rows.reshape(1, SEQUENCE_LENGTH, HIDDEN_SIZE)
        for layer in range(LAYER_COUNT):
            print(f"PyTorch language layer {layer + 1}/{LAYER_COUNT}")
            hidden = decoder_layer(hidden, weights, layer)
        final_norm = rms_norm(hidden, weights["language_model.norm.weight"])
        logits = linear(final_norm, weights["lm_head.weight"])

    fixture = {
        "input.token_ids": token_ids,
        "input.embedding_rows": embedding_rows,
        "expected.hidden": hidden,
        "expected.final_norm": final_norm,
        "expected.logits": logits,
        **{name: tensor for name, tensor in weights.items() if name != "language_model.embed_tokens.weight"},
    }
    fixture = {name: tensor.contiguous() for name, tensor in fixture.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(fixture, args.output, metadata={"reference": "PyTorch 2.6.0 float32 compute"})
    print(f"Saved {len(fixture)} tensors to {args.output}")
    print(f"Hidden shape: {list(hidden.shape)}")
    print(f"Logits shape: {list(logits.shape)}")
    print(f"Last-token argmax: {logits[:, -1].argmax(-1).item()}")


if __name__ == "__main__":
    main()
