#!/usr/bin/env python3
"""Convert the MobileVLM V2 checkpoint to MLX Swift-friendly safetensors.

The MobileVLM checkpoint already contains its fine-tuned CLIP vision tower. The
separate OpenAI CLIP repository is used only for vision/preprocessor metadata;
its weights must not replace the embedded vision weights.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

MIB = 1024 * 1024
DEFAULT_SHARD_SIZE = 1900 * MIB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobilevlm", type=Path, required=True, help="MobileVLM source directory")
    parser.add_argument("--clip", type=Path, required=True, help="CLIP metadata source directory")
    parser.add_argument("--output", type=Path, required=True, help="Output model directory")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--max-shard-size-mb", type=int, default=1900)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def mapped_name(name: str) -> str:
    mappings = (
        ("model.vision_tower.vision_tower.", "vision_model."),
        ("model.mm_projector.", "projector."),
        ("model.embed_tokens.", "language_model.embed_tokens."),
        ("model.layers.", "language_model.layers."),
        ("model.norm.", "language_model.norm."),
    )
    for source, target in mappings:
        if name.startswith(source):
            return target + name[len(source) :]
    if name.startswith("lm_head."):
        return name
    raise ValueError(f"Unrecognized checkpoint key: {name}")


def convert_tensor(name: str, tensor: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, str]:
    tensor = tensor.detach().cpu()
    transform = "identity"

    # MLX Conv2d uses [output, kernelHeight, kernelWidth, inputPerGroup].
    if name.endswith("embeddings.patch_embedding.weight") or name == "projector.peg.peg.0.weight":
        tensor = tensor.permute(0, 2, 3, 1)
        transform = "conv_oihw_to_ohwi"

    if tensor.is_floating_point():
        tensor = tensor.to(dtype)
    return tensor.contiguous(), transform


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def shard_tensors(tensors: OrderedDict[str, torch.Tensor], limit: int) -> list[OrderedDict[str, torch.Tensor]]:
    shards: list[OrderedDict[str, torch.Tensor]] = []
    current: OrderedDict[str, torch.Tensor] = OrderedDict()
    size = 0
    for name, tensor in tensors.items():
        item_size = tensor_bytes(tensor)
        if current and size + item_size > limit:
            shards.append(current)
            current = OrderedDict()
            size = 0
        current[name] = tensor
        size += item_size
    if current:
        shards.append(current)
    return shards


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    checkpoint = args.mobilevlm / "pytorch_model.bin"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty: {args.output}; pass --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)

    target_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    source = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    manifest_tensors: dict[str, Any] = {}

    for source_name in sorted(source):
        target_name = mapped_name(source_name)
        if target_name in converted:
            raise ValueError(f"Duplicate converted key: {target_name}")
        tensor, transform = convert_tensor(target_name, source[source_name], target_dtype)
        converted[target_name] = tensor
        manifest_tensors[target_name] = {
            "source_name": source_name,
            "source_shape": list(source[source_name].shape),
            "shape": list(tensor.shape),
            "source_dtype": str(source[source_name].dtype).removeprefix("torch."),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "transform": transform,
        }

    shards = shard_tensors(converted, args.max_shard_size_mb * MIB)
    total_shards = len(shards)
    weight_map: dict[str, str] = {}
    for index, shard in enumerate(shards, start=1):
        filename = f"model-{index:05d}-of-{total_shards:05d}.safetensors"
        save_file(shard, args.output / filename, metadata={"format": "pt"})
        weight_map.update({name: filename for name in shard})

    total_size = sum(tensor_bytes(tensor) for tensor in converted.values())
    with (args.output / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, handle, indent=2, sort_keys=True)
        handle.write("\n")

    mobile_config = load_json(args.mobilevlm / "config.json")
    clip_config = load_json(args.clip / "config.json")
    output_config = dict(mobile_config)
    output_config.update(
        {
            "model_type": "mobilevlm_mlx",
            "weight_dtype": args.dtype,
            "vision_config": clip_config["vision_config"],
            "vision_feature_layer": mobile_config.get("mm_vision_select_layer", -2),
            "vision_feature_select_strategy": mobile_config.get("mm_vision_select_feature", "patch"),
        }
    )
    with (args.output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(output_config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for filename in ("generation_config.json", "special_tokens_map.json", "tokenizer.model", "tokenizer_config.json"):
        shutil.copy2(args.mobilevlm / filename, args.output / filename)
    shutil.copy2(args.clip / "preprocessor_config.json", args.output / "preprocessor_config.json")

    manifest = {
        "format_version": 1,
        "source_checkpoint": str(checkpoint),
        "source_tensor_count": len(source),
        "tensor_count": len(converted),
        "weight_dtype": args.dtype,
        "total_size": total_size,
        "shard_count": total_shards,
        "notes": [
            "Vision weights come from the fine-tuned vision tower embedded in MobileVLM.",
            "Conv2d weights are converted from PyTorch OIHW to MLX OHWI layout.",
        ],
        "tensors": manifest_tensors,
    }
    with (args.output / "conversion_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Converted {len(converted)} tensors into {total_shards} shard(s)")
    print(f"Tensor data: {total_size / (1024 ** 3):.2f} GiB")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
