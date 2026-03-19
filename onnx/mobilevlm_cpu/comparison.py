# comparison.py

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
image = Image.open(img_path).convert("RGB")

# preprocess image
image_tensor = process_images([image], image_processor, model.config)
image_tensor = image_tensor.to(device=device, dtype=torch.float32)

question = "What objects are visible in the image in detail."
prompt = build_prompt(question)

input_ids = tokenizer_image_token(
    prompt,
    tokenizer,
    return_tensors="pt",
).unsqueeze(0).to(device)

attention_mask = torch.ones_like(input_ids)

# -------------------------------
# PyTorch LLM prefill outputs
# -------------------------------

with torch.no_grad():

    _, torch_attention_mask_, _, torch_inputs_embeds, _ = \
        model.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_mask,
            past_key_values=None,
            labels=None,
            images=image_tensor,
            image_features=None
        )

torch_inputs_embeds = torch_inputs_embeds.to(torch.float32)
torch_attention_mask_ = torch_attention_mask_.to(torch.long)

with torch.no_grad():
    outputs = model.model(                                # MobileLlamaModel
        inputs_embeds=torch_inputs_embeds,
        attention_mask=torch_attention_mask_,
        use_cache=True,
        return_dict=True,                                 # Key: ['last_hidden_state', 'past_key_values']
    )

    pt_logits = model.lm_head(outputs.last_hidden_state)  # [1, 196, 32008]
    pkv = outputs.past_key_values                         # k,v = [1, 16, 196, 128] x 24 layers
    pt_next_logit = pt_logits[:, -1, :]

# PyTorch KV
pt_kv = []
for k, v in pkv:
    pt_kv.append(k.cpu().numpy())
    pt_kv.append(v.cpu().numpy())


# -------------------------------
# ONNX LLM prefill outputs
# -------------------------------

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

onnx_image_features = torch.from_numpy(projector_out).to(device=device, dtype=model.get_model().embed_tokens.weight.dtype)


with torch.no_grad():

    _, onnx_attention_mask_, _, onnx_inputs_embeds, _ = \
        model.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_mask,
            past_key_values=None,
            labels=None,
            images=None,
            image_features=onnx_image_features
        )

sess = ort.InferenceSession(
    "prefill_merged.onnx",
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


# -------------------------------
# Compare
# -------------------------------

print("\n===== LLM Input Comparison =====")

torch_multimodal_input = torch_inputs_embeds.detach().cpu().numpy()
onnx_multimodal_input = onnx_inputs_embeds.detach().cpu().numpy()

diff = np.abs(torch_multimodal_input - onnx_multimodal_input)

print("max abs diff :", diff.max())


print("\n===== LLM Prefill outputs Comparison =====")

logit_diff = np.abs(pt_next_logit.cpu().numpy() - onnx_next_logit)
# Outliers exist among the 32,008 LM head output logits
print("logit max diff:", np.max(logit_diff))
print("logit mean diff:", np.mean(logit_diff))

kv_diffs = []

for i, (pt, ox) in enumerate(zip(pt_kv, onnx_kv)):
    diff = np.max(np.abs(pt - ox))
    kv_diffs.append(diff)

print("KV max diff:", np.max(kv_diffs))

print("\n===== LLM Prefill outputs debugging =====")

pt = pt_next_logit.detach().cpu().numpy()[0]
onnx = onnx_next_logit[0]

# Top-3
pt_top3 = np.argsort(pt)[-3:][::-1]
onnx_top3 = np.argsort(onnx)[-3:][::-1]

print("PT   top3:", pt_top3)
print("ONNX top3:", onnx_top3)
print("Top3 same?", np.array_equal(pt_top3, onnx_top3))