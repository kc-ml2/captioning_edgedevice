# onnx_mobilevlm.py

import os, types
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import re, torch, argparse, time, gc
from PIL import Image
from typing import Dict
import numpy as np
import onnxruntime as ort

from model.mobilevlm import load_pretrained_model
from model.mutils import process_images, build_prompt, tokenizer_image_token
from transformers import LlamaTokenizer
from onnx_preprocessor import preprocess_batch
from onnx_multimodal_input import prepare_inputs_labels_for_multimodal_onnx


# ---- Inference ----
# ---- Preprocess ----
img_path = "000000000139.jpg"
image = Image.open(img_path).convert("RGB")

onnx_preprocessor_out = preprocess_batch([image])  # (1, 3, 336, 336)
# np.save("comparison/onnx_preprocessor_out.npy", onnx_preprocessor_out)

# ---- Vision encoder ----
vision_sess = ort.InferenceSession(
    "export_onnx/vision_tower.onnx", 
    providers=["CPUExecutionProvider"]
)

vision_out = vision_sess.run(
    ["image_features"],
    {"pixel_values": onnx_preprocessor_out},
)[0]
# np.save("comparison/onnx_vision_out.npy", vision_out)

# ---- Projector ----
projector_sess = ort.InferenceSession(
    "export_onnx/mm_projector.onnx", 
    providers=["CPUExecutionProvider"]
)

projector_out = projector_sess.run(
    ["projected_features"],
    {"image_features": vision_out},
)[0]
# np.save("comparison/onnx_projector_out.npy", projector_out)

# ---- Multi-modal input ----
tokenizer = LlamaTokenizer.from_pretrained("mtgv/MobileVLM_V2-1.7B", use_fast=False)
question = "What objects are visible in the image in detail."
prompt = build_prompt(question)

input_ids = tokenizer_image_token(
    prompt,
    tokenizer,
    return_tensors="np",    # No use PyTorch
)  # [53]

input_ids = np.expand_dims(input_ids, axis=0)  # [1, 53]

attention_mask = np.ones_like(input_ids)

with torch.no_grad():

    _, onnx_attention_mask_, _, onnx_multimodal_inputs_embeds, _ = \
        prepare_inputs_labels_for_multimodal_onnx(
            input_ids,
            attention_mask,
            past_key_values=None,
            labels=None,
            images=None,
            image_features=projector_out
        )

print(onnx_multimodal_inputs_embeds.shape)

exit()

# ---- LLM prefill ----
sess = ort.InferenceSession(
    "export_onnx/prefill_merged.onnx",
    providers=["CPUExecutionProvider"]
)

onnx_outputs = sess.run(
    None,
    {
        "inputs_embeds": onnx_inputs_embeds.detach().cpu().numpy(),
        "attention_mask": onnx_attention_mask_.cpu().numpy(),
    }
)

onnx_logits = onnx_outputs[0]
onnx_next_logit = onnx_logits[:, -1, :]

# ONNX KV
onnx_kv = onnx_outputs[1:]