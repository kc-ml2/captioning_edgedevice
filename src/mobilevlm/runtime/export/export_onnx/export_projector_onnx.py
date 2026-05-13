# export_projector_onnx.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import onnx

from pytorch.model.mobilevlm import load_pretrained_model

model_path = "mtgv/MobileVLM_V2-1.7B"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    device="cpu",
)
model.eval()

projector = model.get_model().mm_projector.eval()

mm_hidden_size = model.config.mm_hidden_size
dummy_input = torch.randn(1, 576, mm_hidden_size, dtype=torch.float32)

with torch.no_grad():
    y = projector(dummy_input)
    print("PyTorch output shape:", y.shape)

    torch.onnx.export(
        projector,
        dummy_input,
        "mm_projector.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image_features"],
        output_names=["projected_features"],
    )

onnx_model = onnx.load("mm_projector.onnx")
onnx.checker.check_model(onnx_model)

print("Export done: mm_projector.onnx")