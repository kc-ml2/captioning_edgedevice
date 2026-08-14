#!/usr/bin/env python3
"""Create a deterministic PyTorch fixture for MobileLlama decoder block zero."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors import safe_open
from safetensors.torch import save_file

SEQUENCE_LENGTH = 32
HIDDEN_SIZE = 2048
HEAD_COUNT = 16
HEAD_DIMENSION = 128
ROPE_THETA = 10_000.0
RMS_EPSILON = 1e-6


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
        or name.startswith("language_model.layers.0.")
    )


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


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return functional.linear(x, weight.float())


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    normalized = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + RMS_EPSILON)
    return normalized * weight.float()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(SEQUENCE_LENGTH, dtype=torch.float32)
    dimensions = torch.arange(0, HEAD_DIMENSION, 2, dtype=torch.float32)
    inverse_frequency = 1.0 / (ROPE_THETA ** (dimensions / HEAD_DIMENSION))
    frequencies = torch.outer(positions, inverse_frequency)
    embedding = torch.cat((frequencies, frequencies), dim=-1)[None, None, :, :]
    cosine, sine = embedding.cos(), embedding.sin()
    return q * cosine + rotate_half(q) * sine, k * cosine + rotate_half(k) * sine


def main() -> None:
    args = parse_args()
    weights = load_weights(args.converted)
    layer = "language_model.layers.0"
    token_ids = torch.tensor(
        [1, 319, 13563, 1546, 263, 1967, 29889, 13] + list(range(100, 124)),
        dtype=torch.int32,
    ).reshape(1, SEQUENCE_LENGTH)

    with torch.inference_mode():
        embeddings = functional.embedding(token_ids.long(), weights["language_model.embed_tokens.weight"].float())
        norm1 = rms_norm(embeddings, weights[f"{layer}.input_layernorm.weight"])
        q = linear(norm1, weights[f"{layer}.self_attn.q_proj.weight"])
        k = linear(norm1, weights[f"{layer}.self_attn.k_proj.weight"])
        v = linear(norm1, weights[f"{layer}.self_attn.v_proj.weight"])
        q = q.reshape(1, SEQUENCE_LENGTH, HEAD_COUNT, HEAD_DIMENSION).transpose(1, 2)
        k = k.reshape(1, SEQUENCE_LENGTH, HEAD_COUNT, HEAD_DIMENSION).transpose(1, 2)
        v = v.reshape(1, SEQUENCE_LENGTH, HEAD_COUNT, HEAD_DIMENSION).transpose(1, 2)
        q_rope, k_rope = apply_rope(q, k)
        scores = torch.matmul(q_rope, k_rope.transpose(-1, -2)) / math.sqrt(HEAD_DIMENSION)
        causal_mask = torch.triu(
            torch.full((SEQUENCE_LENGTH, SEQUENCE_LENGTH), float("-inf")), diagonal=1
        )
        probabilities = functional.softmax(scores + causal_mask, dim=-1)
        attention = torch.matmul(probabilities, v).transpose(1, 2).reshape(1, SEQUENCE_LENGTH, HIDDEN_SIZE)
        attention = linear(attention, weights[f"{layer}.self_attn.o_proj.weight"])
        attention_residual = embeddings + attention

        norm2 = rms_norm(attention_residual, weights[f"{layer}.post_attention_layernorm.weight"])
        gate = functional.silu(linear(norm2, weights[f"{layer}.mlp.gate_proj.weight"]))
        up = linear(norm2, weights[f"{layer}.mlp.up_proj.weight"])
        mlp = linear(gate * up, weights[f"{layer}.mlp.down_proj.weight"])
        output = attention_residual + mlp

        final_norm = rms_norm(output, weights["language_model.norm.weight"])
        logits = linear(final_norm, weights["lm_head.weight"])

    # The fixture contains only the embedding rows needed by this test, while
    # retaining the full LM head to validate its orientation and output.
    fixture = {
        "input.token_ids": token_ids,
        "input.embedding_rows": weights["language_model.embed_tokens.weight"][token_ids.flatten().long()].float(),
        "expected.embeddings": embeddings,
        "expected.norm1": norm1,
        "expected.q": q,
        "expected.k": k,
        "expected.v": v,
        "expected.q_rope": q_rope,
        "expected.k_rope": k_rope,
        "expected.attention": attention,
        "expected.attention_residual": attention_residual,
        "expected.norm2": norm2,
        "expected.mlp_gate": gate,
        "expected.mlp_up": up,
        "expected.output": output,
        "expected.final_norm": final_norm,
        "expected.logits": logits,
        **{name: tensor for name, tensor in weights.items() if name != "language_model.embed_tokens.weight"},
    }
    fixture = {name: tensor.contiguous() for name, tensor in fixture.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(fixture, args.output, metadata={"reference": "PyTorch 2.6.0 float32 compute"})
    print(f"Saved {len(fixture)} tensors to {args.output}")
    print(f"Block output shape: {list(output.shape)}")
    print(f"Logits shape: {list(logits.shape)}")
    print(f"Last-token argmax: {logits[:, -1].argmax(dim=-1).item()}")


if __name__ == "__main__":
    main()
