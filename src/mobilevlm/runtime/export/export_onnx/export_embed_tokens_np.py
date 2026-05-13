# export_token_embed_np.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from pytorch.model.mobilevlm import load_pretrained_model


# HF model id or local path
MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL_PATH,
    device="cpu",
)
model.eval()  # MobileLlamaForCausalLM


for name, module in model.model.named_modules():
    if "embed_tokens" in name:
        print(name, module)


embedding_weight = (
    model.get_input_embeddings()
    .weight
    .detach()
    .cpu()
    .numpy()
)


np.save("embed_tokens.npy", embedding_weight)
