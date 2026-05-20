# export_embed_tokens_bin.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from pytorch.model.mobilevlm import load_pretrained_model

MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

USE_INT8 = True

_, model, _, _ = load_pretrained_model(
    model_path=MODEL_PATH,
    device="cpu",
)

embedding_weight = (
    model.get_input_embeddings()
    .weight
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)  # (32000, 2048)

# =========================================
# INT8 Export
# =========================================

if USE_INT8:

    max_abs = np.max(
        np.abs(embedding_weight)
    )

    scale = max_abs / 127.0

    embedding_int8 = np.round(
        embedding_weight / scale
    ).astype(np.int8)

    embedding_int8.tofile(
        "embed_tokens_int8.bin"
    )

    print(f"INT8 export done")
    print(f"scale = {scale}")  # 0.002968134842519685

# =========================================
# FP32 Export
# =========================================

else:

    embedding_weight.tofile(
        "embed_tokens_fp32.bin"
    )

    print("FP32 export done")