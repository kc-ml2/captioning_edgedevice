#!/usr/bin/env python3
"""Quantize MobileVLM language weights for memory-constrained Apple devices.

Vision and projector tensors remain FP16. Language linear weights, token
embeddings, and the LM head use MLX affine group quantization.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx

MIB = 1024 * 1024


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Converted FP16 model directory")
    parser.add_argument("--output", type=Path, required=True, help="Quantized model directory")
    parser.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 5, 6, 8))
    parser.add_argument("--group-size", type=int, default=64, choices=(32, 64, 128))
    parser.add_argument("--max-shard-size-mb", type=int, default=700)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def should_quantize(name: str, array, group_size: int) -> bool:
    language_weight = name.startswith("language_model.") or name == "lm_head.weight"
    return language_weight and name.endswith(".weight") and array.ndim == 2 and array.shape[-1] % group_size == 0


def main() -> None:
    args = arguments()
    source = args.input.resolve()
    output = args.output.resolve()
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty: {output}; pass --overwrite")
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True)

    # Keep conversion memory bounded while processing multi-gigabyte source shards.
    mx.set_cache_limit(64 * MIB)
    index = json.loads(index_path.read_text())
    source_shards = sorted(set(index["weight_map"].values()))
    limit = args.max_shard_size_mb * MIB
    current = OrderedDict()
    current_size = 0
    temporary_shards: list[tuple[Path, list[str], int]] = []
    quantized_names: list[str] = []
    original_bytes = 0
    output_bytes = 0

    def flush() -> None:
        nonlocal current, current_size
        if not current:
            return
        number = len(temporary_shards) + 1
        path = output / f"model-{number:05d}.safetensors"
        mx.save_safetensors(path, current, metadata={"format": "mlx"})
        temporary_shards.append((path, list(current), current_size))
        print(f"Wrote {path.name}: {current_size / MIB:.1f} MiB, {len(current)} arrays", flush=True)
        current = OrderedDict()
        current_size = 0
        mx.clear_cache()

    for source_shard in source_shards:
        print(f"Loading {source_shard}", flush=True)
        arrays = mx.load(source / source_shard)
        for name in sorted(arrays):
            array = arrays[name]
            original_bytes += array.nbytes
            if should_quantize(name, array, args.group_size):
                weight, scales, biases = mx.quantize(
                    array,
                    group_size=args.group_size,
                    bits=args.bits,
                    mode="affine",
                )
                prefix = name.removesuffix(".weight")
                produced = ((name, weight), (f"{prefix}.scales", scales), (f"{prefix}.biases", biases))
                quantized_names.append(name)
            else:
                produced = ((name, array),)

            for output_name, output_array in produced:
                mx.eval(output_array)
                size = output_array.nbytes
                if current and current_size + size > limit:
                    flush()
                current[output_name] = output_array
                current_size += size
                output_bytes += size
        del arrays
        mx.clear_cache()
    flush()

    shard_count = len(temporary_shards)
    weight_map: dict[str, str] = {}
    for number, (temporary, names, _) in enumerate(temporary_shards, 1):
        final_name = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        temporary.rename(output / final_name)
        weight_map.update({name: final_name for name in names})

    index_output = {"metadata": {"total_size": output_bytes}, "weight_map": weight_map}
    (output / "model.safetensors.index.json").write_text(json.dumps(index_output, indent=2, sort_keys=True) + "\n")

    for filename in (
        "generation_config.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ):
        shutil.copy2(source / filename, output / filename)

    config = json.loads((source / "config.json").read_text())
    quantization = {
        "bits": args.bits,
        "group_size": args.group_size,
        "mode": "affine",
        "scope": "language_model_and_lm_head",
    }
    config.update({"weight_dtype": "mixed_fp16_q4", "quantization": quantization})
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    manifest = {
        "format_version": 2,
        "source_model": str(source),
        "weight_dtype": "mixed_fp16_q4",
        "quantization": quantization,
        "quantized_weight_count": len(quantized_names),
        "quantized_weights": quantized_names,
        "array_count": len(weight_map),
        "shard_count": shard_count,
        "original_tensor_bytes": original_bytes,
        "total_size": output_bytes,
    }
    (output / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Quantized weights: {len(quantized_names)}", flush=True)
    print(f"Tensor bytes: {original_bytes / MIB:.1f} MiB -> {output_bytes / MIB:.1f} MiB", flush=True)
    print(f"Output: {output}", flush=True)


if __name__ == "__main__":
    main()
