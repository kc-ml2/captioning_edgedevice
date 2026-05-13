# export_embed_tokens_bin.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from model.mobilevlm import load_pretrained_model

MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

tokenizer, model, image_processor, context_len = load_pretrained_model(
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
)

print(embedding_weight.shape)
# (32000, 2048)

embedding_weight.tofile("embed_tokens.bin")