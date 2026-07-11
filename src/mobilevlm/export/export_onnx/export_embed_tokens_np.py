# export_embed_tokens_np.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pytorch_modular.pytorch_mobilellama import MobileLlamaModel
import numpy as np
import torch

MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

mobilellama = MobileLlamaModel.from_pretrained(
    MODEL_PATH,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32,
).eval()

token_embedding = mobilellama.embed_tokens
token_embedding.eval()

embedding_weight = (
    token_embedding.weight
    .detach()
    .cpu()
    .numpy()
)

np.save("embed_tokens.npy", embedding_weight)
