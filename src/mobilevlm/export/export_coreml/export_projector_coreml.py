# export_projector_coreml.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import coremltools as ct
import numpy as np

from pytorch.model.mobilevlm import load_pretrained_model

MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

DEVICE = "cpu"

USE_FP16 = False

USE_INT8 = True
USE_INT4 = False

assert not (
    USE_INT8 and USE_INT4
), "Only one quantization mode can be enabled."

# -------------------------
# 1. load model
# -------------------------
_, model, _, _ = load_pretrained_model(
    model_path=MODEL_PATH,
    device=DEVICE,
)
model.eval()

# -------------------------
# 2.load projector
# -------------------------
projector = model.get_model().mm_projector
projector.eval()
projector = projector.to(DEVICE)

mm_hidden_size = model.config.mm_hidden_size  # 1024

# -------------------------
# 3. dummy input (FP32)
# -------------------------
dummy_input = torch.randn(
    1, 
    576, 
    mm_hidden_size, 
    dtype=torch.float32
)

# -------------------------
# 4. forward
# -------------------------
with torch.no_grad():
    torch_out = projector(dummy_input)

print("PyTorch output shape:", torch_out.shape)

# -------------------------
# 5. TorchScript
# -------------------------
traced_model = torch.jit.trace(projector, dummy_input)

# -------------------------
# Compute Precision Setting
# -------------------------
if USE_INT8 or USE_INT4:
    compute_precision = ct.precision.FLOAT32

elif USE_FP16:
    compute_precision = ct.precision.FLOAT16

else:
    compute_precision = ct.precision.FLOAT32

# -------------------------
# Save path
# -------------------------
if USE_INT8:
    save_path = "Projector_int8.mlpackage"

elif USE_INT4:
    save_path = "Projector_int4.mlpackage"

elif USE_FP16:
    save_path = "Projector_fp16.mlpackage"

else:
    save_path = "Projector_fp32.mlpackage"

# -------------------------
# CoreML Convert
# -------------------------
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(
            name="image_features",
            shape=dummy_input.shape,
            dtype=np.float32
        )
    ],
    outputs=[
        ct.TensorType(
            name="projected_features",
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

# -------------------------
# INT4 Palettization
# -------------------------
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