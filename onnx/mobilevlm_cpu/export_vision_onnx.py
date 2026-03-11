import torch
import onnx
import onnxruntime as ort
import numpy as np

from model.mobilevlm import load_pretrained_model, build_prompt
from model.mutils import process_images, tokenizer_image_token, print_full_memory_report, to_int8_dynamic


device = torch.device("cpu")

# HF model id or local path
MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL_PATH,
    device="cpu",
)
model.eval()


vision_tower = model.get_vision_tower()
vision_tower.eval()
vision_tower = vision_tower.to("cpu")

dummy_input = torch.randn(1, 3, 336, 336, dtype=torch.float32)

with torch.no_grad():
    torch_out = vision_tower(dummy_input)

print("PyTorch output shape:", torch_out.shape)

onnx_path = "vision_tower.onnx"

torch.onnx.export(
    vision_tower,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["pixel_values"],
    output_names=["image_features"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "image_features": {0: "batch_size"},
    }
)

print(f"Export done: {onnx_path}")