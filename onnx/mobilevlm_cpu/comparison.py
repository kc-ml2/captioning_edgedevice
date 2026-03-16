# onnx_moblievlm.py

import os, types
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import re, torch, argparse, time, gc
from PIL import Image
from typing import Dict
import numpy as np
import onnxruntime as ort

from model.mobilevlm import load_pretrained_model, build_prompt
from model.mutils import process_images, tokenizer_image_token, print_full_memory_report, to_int8_dynamic


# ---- Default values ---- #
GEN_KWARGS_DEFAULT = dict(
    num_beams=1,
    max_new_tokens=40,
    min_new_tokens=40,
)

device = torch.device("cpu")

# HF model id or local path
MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL_PATH,
    device="cpu",
)
model.eval()


# ---- inference ----
img_path = "000000000139.jpg"
image = Image.open(img_path).convert("RGB")  # (426, 640, 3)

# preprocess image
image_tensor = process_images([image], image_processor, model.config)  # (1, 3, 336, 336)
image_tensor = image_tensor.to(device=device, dtype=torch.float32)

question = "What objects are visible in the image in detail."
prompt = build_prompt(question)
'''
A chat between a curious user and an artificial intelligence assistant. 
The assistant gives helpful, detailed, and polite answers to the user's questions. 
USER: <image> What objects are visible in the image in detail. 
ASSISTANT:
'''

# [1, 319, 13563, ... , -200, ... , 29901] / IMAGE_TOKEN_INDEX (placeholder) = -200
input_ids = tokenizer_image_token(
    prompt,
    tokenizer,  # LlamaTokenizer
    return_tensors="pt",
).unsqueeze(0).to(device)

attention_mask = torch.ones_like(input_ids)

# ONNX
vision_sess = ort.InferenceSession("vision_tower.onnx", providers=["CPUExecutionProvider"])
projector_sess = ort.InferenceSession("mm_projector.onnx", providers=["CPUExecutionProvider"])

pixel_values_np = image_tensor.cpu().numpy().astype(np.float32)
vision_out = vision_sess.run(
    ["image_features"],
    {"pixel_values": pixel_values_np},
)[0]

projector_out = projector_sess.run(
    ["projected_features"],
    {"image_features": vision_out.astype(np.float32)},
)[0]

image_features = torch.from_numpy(projector_out).to(device=device, dtype=model.get_model().embed_tokens.weight.dtype)

############# Above code is same with onnx_mobilevlm.py #############

# -------------------------------
# PyTorch LLM input (vision + projector + text)
# -------------------------------
with torch.no_grad():

    _, _, _, torch_inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=None,
        images=image_tensor,      # PyTorch vision path
        image_features=None
    )

torch_llm_input = torch_inputs_embeds.cpu().numpy()


# -------------------------------
# ONNX LLM input
# -------------------------------
with torch.no_grad():

    _, _, _, onnx_inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=None,
        images=None,
        image_features=image_features   # ONNX vision + projector output
    )

onnx_llm_input = onnx_inputs_embeds.cpu().numpy()


# -------------------------------
# Compare
# -------------------------------
diff = np.abs(torch_llm_input - onnx_llm_input)

print("\n===== LLM Input Comparison =====")

print("shape:", torch_llm_input.shape)   # (1,196,2048)

print("max abs diff :", diff.max())
