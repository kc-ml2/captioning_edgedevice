# export_vision_coreml.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import coremltools as ct
import numpy as np

from pytorch.model.mobilevlm import load_pretrained_model

device = torch.device("cpu")

MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

# Load model
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL_PATH,
    device="cpu",
)
model.eval()

# Vision tower
vision_tower = model.get_vision_tower()
vision_tower.eval()
vision_tower = vision_tower.to("cpu")

dummy_input = torch.randn(1, 3, 336, 336, dtype=torch.float32)

# Forward check
with torch.no_grad():
    torch_out = vision_tower(dummy_input)

print("PyTorch output shape:", torch_out.shape)

traced_model = torch.jit.trace(vision_tower, dummy_input)

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
        ct.TensorType(name="image_features", dtype=np.float32)
    ],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS15,
    compute_precision=ct.precision.FLOAT32
)

# Save
mlmodel.save("VisionEncoder_32.mlpackage")

print("CoreML FP32 export done")