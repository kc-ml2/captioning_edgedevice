# export_embed_tokens_np.py

from model.mobilevlm import load_pretrained_model
import numpy as np

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
