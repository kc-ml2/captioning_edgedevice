#!/usr/bin/env python3
"""Create deterministic PyTorch reference data for the MLX Swift projector test."""

from __future__ import annotations

import argparse
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


def load_projector_weights(directory: Path) -> dict[str, torch.Tensor]:
    names = {
        "projector.mlp.mlp.0.weight",
        "projector.mlp.mlp.0.bias",
        "projector.mlp.mlp.2.weight",
        "projector.mlp.mlp.2.bias",
        "projector.peg.peg.0.weight",
        "projector.peg.peg.0.bias",
    }
    result: dict[str, torch.Tensor] = {}
    for shard in directory.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in names.intersection(handle.keys()):
                result[name] = handle.get_tensor(name)
    missing = names - result.keys()
    if missing:
        raise KeyError(f"Missing projector weights: {sorted(missing)}")
    return result


def main() -> None:
    args = parse_args()
    weights = load_projector_weights(args.converted)

    # A deterministic, bounded input makes failures reproducible and avoids
    # random-generator differences between PyTorch and MLX.
    values = torch.arange(576 * 1024, dtype=torch.float32)
    input_tensor = ((values.remainder(2048) - 1024) / 1024).reshape(1, 576, 1024)

    w0 = weights["projector.mlp.mlp.0.weight"].float()
    b0 = weights["projector.mlp.mlp.0.bias"].float()
    w2 = weights["projector.mlp.mlp.2.weight"].float()
    b2 = weights["projector.mlp.mlp.2.bias"].float()
    conv_ohwi = weights["projector.peg.peg.0.weight"].float()
    conv_oihw = conv_ohwi.permute(0, 3, 1, 2).contiguous()
    conv_bias = weights["projector.peg.peg.0.bias"].float()

    with torch.inference_mode():
        mlp0 = functional.linear(input_tensor, w0, b0)
        gelu = functional.gelu(mlp0, approximate="none")
        mlp2 = functional.linear(gelu, w2, b2)
        image = mlp2.transpose(1, 2).reshape(1, 2048, 24, 24)
        pooled = functional.adaptive_avg_pool2d(image, (12, 12))
        positional = functional.conv2d(pooled, conv_oihw, conv_bias, padding=1, groups=2048)
        output = (pooled + positional).flatten(2).transpose(1, 2).contiguous()

    fixture = {
        "input": input_tensor,
        "expected.mlp0": mlp0,
        "expected.gelu": gelu,
        "expected.mlp2": mlp2,
        "expected.pooled_nhwc": pooled.permute(0, 2, 3, 1).contiguous(),
        "expected.output": output,
        **weights,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(fixture, args.output, metadata={"reference": "PyTorch 2.6.0 float32 compute"})
    print(f"Saved {len(fixture)} tensors to {args.output}")
    print(f"Input shape: {list(input_tensor.shape)}")
    print(f"Output shape: {list(output.shape)}")


if __name__ == "__main__":
    main()
