# export_ldpnet_coreml.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import coremltools as ct
import numpy as np

from torch.export import export
from pytorch_modular.pytorch_mm_projector import LDPNetV2Projector


# ============================================================
# Configuration
# ============================================================

USE_FP16 = True
USE_INT8 = False

MODEL_PATH = "../pytorch_modular/mm_projector_weights.pth"


# ============================================================
# Load Model
# ============================================================

mm_projector = LDPNetV2Projector().eval()

mm_projector.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
    )
)


# ============================================================
# Dummy Input
# ============================================================

dummy_input = torch.randn(
    1,
    576,
    1024,
    dtype=torch.float32,
)


# ============================================================
# ExportedProgram
# ============================================================

exported_program = export(
    mm_projector,
    (dummy_input,),
)

exported_program = exported_program.run_decompositions({})

# ============================================================
# Compute Precision
# ============================================================

compute_precision = ct.precision.FLOAT32

if USE_FP16:
    compute_precision = ct.precision.FLOAT16


# ============================================================
# Output File Name
# ============================================================

if USE_FP16:
    save_path = "projector_fp16.mlpackage"

elif USE_INT8:
    save_path = "projector_int8.mlpackage"

else:
    save_path = "projector_fp32.mlpackage"


# ============================================================
# CoreML Conversion
# ============================================================

mlmodel = ct.convert(
    exported_program,
    convert_to="mlprogram",
    inputs=[
        ct.TensorType(
            name="image_features",
            shape=(1, 576, 1024),
            dtype=np.float32,
        )
    ],
    outputs=[
        ct.TensorType(
            name="projected_features",
            dtype=np.float32,
        )
    ],
    minimum_deployment_target=ct.target.iOS18,
    compute_precision=compute_precision,
)


# ============================================================
# INT8 Quantization
# ============================================================

if USE_INT8:

    from coremltools.optimize.coreml import (
        OptimizationConfig,
        OpLinearQuantizerConfig,
        linear_quantize_weights,
    )

    mlmodel = linear_quantize_weights(
        mlmodel,
        config=OptimizationConfig(
            global_config=OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype=np.int8,
            )
        ),
    )


# ============================================================
# Save
# ============================================================

mlmodel.save(save_path)

print("\nCoreML export completed")