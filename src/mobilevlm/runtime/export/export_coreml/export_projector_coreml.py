# export_projector_coreml.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import coremltools as ct
import numpy as np

from pytorch.model.mobilevlm import load_pretrained_model

device = torch.device("cpu")

MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

# -------------------------
# 1. 모델 로드
# -------------------------
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL_PATH,
    device="cpu",
)
model.eval()

# -------------------------
# 2. projector 가져오기
# -------------------------
projector = model.get_model().mm_projector
projector.eval()
projector = projector.to("cpu")

mm_hidden_size = model.config.mm_hidden_size  # 1024

# -------------------------
# 3. dummy input (FP32)
# -------------------------
dummy_input = torch.randn(1, 576, mm_hidden_size, dtype=torch.float32)

# -------------------------
# 4. forward 체크
# -------------------------
with torch.no_grad():
    torch_out = projector(dummy_input)

print("PyTorch output shape:", torch_out.shape)

# -------------------------
# 5. TorchScript 변환
# -------------------------
traced_model = torch.jit.trace(projector, dummy_input)

# -------------------------
# 6. CoreML 변환 (FP32 + mlpackage)
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
        ct.TensorType(name="projected_features", dtype=np.float32)
    ],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS15,
    compute_precision=ct.precision.FLOAT32
)

# -------------------------
# 7. 저장
# -------------------------
mlmodel.save("Projector_32.mlpackage")

print("CoreML FP32 export done")