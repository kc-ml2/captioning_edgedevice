import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import onnx

from pytorch_modular.pytorch_mm_projector import LDPNetV2Projector

mm_projector = LDPNetV2Projector().eval()

MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../pytorch_modular/mm_projector_weights.pth"
    )
)

mm_projector.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
    )
)

dummy_input = torch.randn(1, 576, 1024, dtype=torch.float32)

with torch.no_grad():

    torch.onnx.export(
        mm_projector,
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