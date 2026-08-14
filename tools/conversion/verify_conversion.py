#!/usr/bin/env python3
"""Verify converted MobileVLM safetensors against the source checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from convert_mobilevlm import convert_tensor, mapped_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobilevlm", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def digest(tensor: torch.Tensor) -> str:
    # Flatten first because a tensor can report contiguous while carrying a
    # channels-last stride that cannot be reinterpreted as bytes directly.
    raw = tensor.flatten().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.converted / "conversion_manifest.json").read_text())
    dtype = torch.float16 if manifest["weight_dtype"] == "float16" else torch.float32
    source = torch.load(
        args.mobilevlm / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True
    )

    index = json.loads((args.converted / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    handles = {
        filename: safe_open(args.converted / filename, framework="pt", device="cpu")
        for filename in sorted(set(weight_map.values()))
    }

    failures: list[dict[str, object]] = []
    max_abs_error = 0.0
    max_relative_error = 0.0
    checked = 0

    expected_names = {mapped_name(name) for name in source}
    actual_names = set(weight_map)
    if expected_names != actual_names:
        failures.append(
            {
                "kind": "key_mismatch",
                "missing": sorted(expected_names - actual_names),
                "unexpected": sorted(actual_names - expected_names),
            }
        )

    for source_name, source_tensor in source.items():
        target_name = mapped_name(source_name)
        if target_name not in weight_map:
            continue
        expected, transform = convert_tensor(target_name, source_tensor, dtype)
        actual = handles[weight_map[target_name]].get_tensor(target_name)
        checked += 1

        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            failures.append(
                {
                    "kind": "metadata_mismatch",
                    "name": target_name,
                    "expected_shape": list(expected.shape),
                    "actual_shape": list(actual.shape),
                    "expected_dtype": str(expected.dtype),
                    "actual_dtype": str(actual.dtype),
                }
            )
            continue

        if digest(expected) != digest(actual):
            difference = (expected.float() - actual.float()).abs()
            tensor_max_abs = difference.max().item()
            denominator = expected.float().abs().clamp_min(1e-7)
            tensor_max_relative = (difference / denominator).max().item()
            max_abs_error = max(max_abs_error, tensor_max_abs)
            max_relative_error = max(max_relative_error, tensor_max_relative)
            failures.append(
                {
                    "kind": "value_mismatch",
                    "name": target_name,
                    "transform": transform,
                    "max_abs_error": tensor_max_abs,
                    "max_relative_error": tensor_max_relative,
                }
            )

    report = {
        "passed": not failures,
        "checked_tensor_count": checked,
        "expected_tensor_count": len(source),
        "converted_tensor_count": len(weight_map),
        "max_abs_error": max_abs_error,
        "max_relative_error": max_relative_error,
        "failure_count": len(failures),
        "failures": failures,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
