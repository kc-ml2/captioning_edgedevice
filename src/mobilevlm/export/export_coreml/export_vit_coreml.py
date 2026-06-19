import numpy as np
import coremltools as ct
import torch
import torch.nn as nn
from transformers import CLIPVisionModel


# ============================================================
# Configuration
# ============================================================

USE_FP16 = False
USE_INT8 = True
USE_INT4 = False

assert not (
    USE_INT8 and USE_INT4
), "Only one quantization mode can be enabled."


# ============================================================
# Model Definition
# ============================================================

class CLIPViT(nn.Module):
    """
    Input:
        pixel_values: [B, 3, 336, 336]

    Output:
        vision_features: [B, 576, 1024]
    """

    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder

    def forward(self, pixel_values):

        outputs = self.vision_encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )

        vision_features = outputs.hidden_states[-2][:, 1:]

        return vision_features


# ============================================================
# Load Vision Encoder
# ============================================================

vision_encoder = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-large-patch14-336",
)

model = CLIPViT(vision_encoder).eval()

dummy_input = torch.randn(
    1, 3, 336, 336
)

exported_program = torch.export.export(
    model,
    (dummy_input,),
)

exported_program = exported_program.run_decompositions({})

# ============================================================
# Compute Precision
# ============================================================


# ============================================================
# Output File Name
# ============================================================

if USE_INT8:
    save_path = "vit_int8.mlpackage"

elif USE_INT4:
    save_path = "vit_int4.mlpackage"

elif USE_FP16:
    save_path = "vit_fp16.mlpackage"

else:
    save_path = "vit_fp32_cp32.mlpackage"


# ============================================================
# CoreML Conversion
# ============================================================

mlmodel = ct.convert(
    exported_program,
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
    compute_precision=ct.precision.FLOAT16
)

# ============================================================
# Optional Weight Compression
# ============================================================

if USE_INT8:

    from coremltools.optimize.coreml import (
        OptimizationConfig,
        OpLinearQuantizerConfig,
        linear_quantize_weights,
    )

    print("Applying INT8 quantization...")

    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=np.int8,
        )
    )

    mlmodel = linear_quantize_weights(
        mlmodel,
        config=config,
    )

elif USE_INT4:

    from coremltools.optimize.coreml import (
        OptimizationConfig,
        OpPalettizerConfig,
        palettize_weights,
    )

    print("Applying INT4 palettization...")

    config = OptimizationConfig(
        global_config=OpPalettizerConfig(
            mode="kmeans",
            nbits=4,
        )
    )

    mlmodel = palettize_weights(
        mlmodel,
        config=config,
    )


# ============================================================
# Save CoreML Model
# ============================================================

mlmodel.save(save_path)

print(f"CoreML export completed: {save_path}")