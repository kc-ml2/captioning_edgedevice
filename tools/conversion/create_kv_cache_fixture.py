#!/usr/bin/env python3
"""Create a multimodal prefill and one-token KV-cache decode fixture."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

HIDDEN = 2048
HEADS = 16
HEAD_DIM = 128
LAYERS = 24
IMAGE_TOKEN = -200
TEXT_LENGTH = 9
IMAGE_LENGTH = 8
PREFILL_LENGTH = TEXT_LENGTH - 1 + IMAGE_LENGTH  # replace one image token


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
    for shard, names in grouped.items():
        with safe_open(directory / shard, framework="pt", device="cpu") as handle:
            for name in names:
                result[name] = handle.get_tensor(name)
    return result


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x, weight.float())


def norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight.float()


def rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    dimensions = torch.arange(0, HEAD_DIM, 2, dtype=torch.float32)
    inverse = 1.0 / (10_000.0 ** (dimensions / HEAD_DIM))
    frequencies = torch.outer(positions.float(), inverse)
    embedding = torch.cat((frequencies, frequencies), -1)[None, None]
    first, second = x.chunk(2, -1)
    rotated = torch.cat((-second, first), -1)
    return x * embedding.cos() + rotated * embedding.sin()


def layer_forward(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer: int,
    positions: torch.Tensor,
    past: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    prefix = f"language_model.layers.{layer}"
    normalized = norm(x, weights[f"{prefix}.input_layernorm.weight"])
    length = x.shape[1]
    q = linear(normalized, weights[f"{prefix}.self_attn.q_proj.weight"])
    k = linear(normalized, weights[f"{prefix}.self_attn.k_proj.weight"])
    v = linear(normalized, weights[f"{prefix}.self_attn.v_proj.weight"])
    q = q.reshape(1, length, HEADS, HEAD_DIM).transpose(1, 2)
    k = k.reshape(1, length, HEADS, HEAD_DIM).transpose(1, 2)
    v = v.reshape(1, length, HEADS, HEAD_DIM).transpose(1, 2)
    q, k = rope(q, positions), rope(k, positions)

    if past is None:
        all_k, all_v = k, v
        mask = torch.triu(torch.full((length, length), float("-inf")), 1)
    else:
        all_k = torch.cat((past[0], k), dim=2)
        all_v = torch.cat((past[1], v), dim=2)
        mask = 0.0
    scores = torch.matmul(q, all_k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
    probabilities = F.softmax(scores + mask, dim=-1)
    attention = torch.matmul(probabilities, all_v).transpose(1, 2).reshape(1, length, HIDDEN)
    x = x + linear(attention, weights[f"{prefix}.self_attn.o_proj.weight"])
    normalized = norm(x, weights[f"{prefix}.post_attention_layernorm.weight"])
    gate = F.silu(linear(normalized, weights[f"{prefix}.mlp.gate_proj.weight"]))
    up = linear(normalized, weights[f"{prefix}.mlp.up_proj.weight"])
    x = x + linear(gate * up, weights[f"{prefix}.mlp.down_proj.weight"])
    return x, (all_k, all_v)


def logits(hidden: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    return linear(norm(hidden, weights["language_model.norm.weight"]), weights["lm_head.weight"])


def main() -> None:
    args = parse_args()
    weights = load_weights(args.converted)
    embedding = weights["language_model.embed_tokens.weight"].float()
    input_ids = torch.tensor([[1, 319, 13563, IMAGE_TOKEN, 29871, 13, 5618, 29901, 29871]], dtype=torch.int32)
    image_values = torch.arange(IMAGE_LENGTH * HIDDEN, dtype=torch.float32)
    image_features = ((image_values.remainder(4096) - 2048) / 4096).reshape(IMAGE_LENGTH, HIDDEN)
    image_position = int((input_ids[0] == IMAGE_TOKEN).nonzero()[0])
    before = embedding[input_ids[0, :image_position].long()]
    after = embedding[input_ids[0, image_position + 1 :].long()]
    multimodal = torch.cat((before, image_features, after), dim=0).unsqueeze(0)
    assert multimodal.shape == (1, PREFILL_LENGTH, HIDDEN)

    caches: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.inference_mode():
        hidden = multimodal
        positions = torch.arange(PREFILL_LENGTH)
        for layer in range(LAYERS):
            print(f"PyTorch prefill layer {layer + 1}/{LAYERS}")
            hidden, cache = layer_forward(hidden, weights, layer, positions, None)
            caches.append(cache)
        prefill_logits = logits(hidden, weights)
        first_token = prefill_logits[:, -1].argmax(-1)

        decode_hidden = embedding[first_token].unsqueeze(1)
        decode_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(LAYERS):
            print(f"PyTorch decode layer {layer + 1}/{LAYERS}")
            decode_hidden, cache = layer_forward(
                decode_hidden, weights, layer, torch.tensor([PREFILL_LENGTH]), caches[layer]
            )
            decode_caches.append(cache)
        decode_logits = logits(decode_hidden, weights)
        second_token = decode_logits[:, -1].argmax(-1)

    fixture: dict[str, torch.Tensor] = {
        "input.ids": input_ids,
        "input.image_features": image_features,
        "expected.multimodal_embeddings": multimodal,
        "expected.prefill_hidden": hidden,
        "expected.prefill_logits": prefill_logits[:, -1],
        "expected.first_token": first_token.to(torch.int32),
        "expected.decode_hidden": decode_hidden,
        "expected.decode_logits": decode_logits[:, -1],
        "expected.second_token": second_token.to(torch.int32),
        **weights,
    }
    for layer, ((prefill_k, prefill_v), (decode_k, decode_v)) in enumerate(zip(caches, decode_caches)):
        fixture[f"expected.prefill_cache.{layer}.key"] = prefill_k
        fixture[f"expected.prefill_cache.{layer}.value"] = prefill_v
        fixture[f"expected.decode_cache.{layer}.key"] = decode_k
        fixture[f"expected.decode_cache.{layer}.value"] = decode_v

    fixture = {name: tensor.contiguous() for name, tensor in fixture.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(fixture, args.output, metadata={"reference": "PyTorch multimodal prefill + cached decode"})
    print(f"Saved {len(fixture)} tensors to {args.output}")
    print(f"Multimodal shape: {list(multimodal.shape)}")
    print(f"First token: {first_token.item()}; second token: {second_token.item()}")


if __name__ == "__main__":
    main()
