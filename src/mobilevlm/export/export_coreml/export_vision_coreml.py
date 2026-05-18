# export_vision_coreml.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import coremltools as ct
import numpy as np

from model.mobilevlm import load_pretrained_model

MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

device = "cpu"

USE_FP16 = False

USE_INT8 = False
USE_INT4 = True

assert not (
    USE_INT8 and USE_INT4
), "Only one quantization mode can be enabled."

# Load model
_, model, _, _ = load_pretrained_model(
    model_path=MODEL_PATH,
    device=device,
)
model.eval()

# Vision tower
vision_tower = model.get_vision_tower()
vision_tower.eval()
vision_tower = vision_tower.to(device)

dummy_input = torch.randn(
    1, 3, 336, 336, 
    dtype=torch.float32
)

# Forward check
with torch.no_grad():
    torch_out = vision_tower(dummy_input)

print("PyTorch output shape:", torch_out.shape)

traced_model = torch.jit.trace(vision_tower, dummy_input)

# -------------------------
# Compute Precision Setting
# -------------------------
if USE_INT8 or USE_INT4:
    compute_precision = ct.precision.FLOAT32

elif USE_FP16:
    compute_precision = ct.precision.FLOAT16

else:
    compute_precision = ct.precision.FLOAT32

# =========================
# Save path
# =========================

if USE_INT8:
    save_path = "VisionEncoder_int8.mlpackage"

elif USE_INT4:
    save_path = "VisionEncoder_int4.mlpackage"

elif USE_FP16:
    save_path = "VisionEncoder_fp16.mlpackage"

else:
    save_path = "VisionEncoder_fp32.mlpackage"

# -------------------------
# CoreML Convert
# -------------------------
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(
            name="pixel_values",
            shape=dummy_input.shape,
            dtype=np.float32
        )
    ],
    outputs=[
        ct.TensorType(
            name="image_features",
            dtype=np.float32
        )
    ],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS18,
    compute_precision=compute_precision
)

# -------------------------
# INT8 Quantization
# -------------------------
if USE_INT8:

    from coremltools.optimize.coreml import (
        linear_quantize_weights,
        OptimizationConfig,
        OpLinearQuantizerConfig
    )

    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=np.int8
        )
    )

    mlmodel = linear_quantize_weights(
        mlmodel,
        config=config
    )

# =========================
# INT4 Palettization
# =========================

elif USE_INT4:

    from coremltools.optimize.coreml import (
        palettize_weights,
        OptimizationConfig,
        OpPalettizerConfig
    )

    config = OptimizationConfig(
        global_config=OpPalettizerConfig(
            mode="kmeans",
            nbits=4
        )
    )

    mlmodel = palettize_weights(
        mlmodel,
        config=config
    )
# -------------------------
# Save
# -------------------------
mlmodel.save(save_path)

print(f"CoreML export done: {save_path}")