# export_embed_tokens_bin.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from model.mobilevlm import load_pretrained_model

MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

_, model, _, _ = load_pretrained_model(
    model_path=MODEL_PATH,
    device="cpu",
)

lm_weight = (
    model.lm_head.weight
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)


lm_weight.tofile(
    "lm_head_fp32.bin"
)
