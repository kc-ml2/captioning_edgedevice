# export_vision_onnx.py

import torch
import torch.nn as nn
from transformers import CLIPVisionModel


class ViT(nn.Module):
    def __init__(self):
        super().__init__()

        self.vision_encoder = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14-336",
        ).eval()

    def forward(self, pixel_values):

        outputs = self.vision_encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )
        # remove CLS token
        vision_features = outputs.hidden_states[-2][:, 1:]

        return vision_features


device = torch.device("cpu")

model = ViT().eval().to(device)

dummy_input = torch.randn(
    1,
    3,
    336,
    336,
    dtype=torch.float32,
)

onnx_path = "vision_tower.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["pixel_values"],
    output_names=["image_features"],
    dynamic_axes={
        "pixel_values": {
            0: "batch_size"
        },
        "image_features": {
            0: "batch_size"
        },
    },
)

print(f"Export done: {onnx_path}")