# onnx_mobilevlm.py

import os, types
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import re, torch, argparse, time, gc
from PIL import Image
from typing import Dict
import numpy as np
import onnxruntime as ort

from model.mobilevlm import load_pretrained_model
from model.mutils import process_images, build_prompt, tokenizer_image_token_onnx
from onnx_preprocessor import preprocess_batch
from onnx_multimodal_input import prepare_inputs_labels_for_multimodal_onnx


# ---- Inference ----
img_path = "000000000139.jpg"
image = Image.open(img_path).convert("RGB")


# ---- Image preprocessor ----
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
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load("export_onnx/tokenizer.model")

question = "What objects are visible in the image in detail."
prompt = build_prompt(question)

input_ids = tokenizer_image_token_onnx(
    prompt=prompt,
    tokenizer=sp,
)  # [53]

input_ids = np.expand_dims(input_ids, axis=0)  # [1, 53]

# ---- tokenizer and embedding layer ----
input_ids_clean = input_ids[input_ids >= 0]
embedding_weight = np.load("export_onnx/embed_tokens.npy")  # (32000, 2048)
np_prompt_embedding = embedding_weight[input_ids_clean]
# np.save("comparison/onnx_prompt_embedding.npy", np_prompt_embedding)

attention_mask = np.ones_like(input_ids)

# ---- Multimodal input ----
with torch.no_grad():
    onnx_attention_mask, onnx_multimodal_input_embeds = \
        prepare_inputs_labels_for_multimodal_onnx(
            input_ids,
            attention_mask,
            image_features=projector_out
        )
# np.save("comparison/onnx_multimodal_input.npy", onnx_multimodal_input_embeds)

# ---- LLM prefill ----
sess = ort.InferenceSession(
    "export_onnx/prefill.onnx",
    providers=["CPUExecutionProvider"]
)

onnx_outputs = sess.run(
    None,
    {
        "inputs_embeds": onnx_multimodal_input_embeds,
        "attention_mask": onnx_attention_mask,
    }
)

onnx_logits = onnx_outputs[0]            # (1, 196, 32000)
onnx_next_logit = onnx_logits[:, -1, :]  # (1, 32000)
# np.save("comparison/onnx_next_logit.npy", onnx_next_logit)

# ONNX KV
onnx_kv = onnx_outputs[1:]
# np.savez(
#     "comparison/onnx_kv.npz",
#     *onnx_kv
# )


# ---- LLM decoder ----
eos_token_id = 2
max_new_tokens = 40
generated_tokens = []

cur_token = np.argmax(onnx_next_logit, axis=-1, keepdims=True)  # 512
past_key_values = onnx_kv
cur_len = past_key_values[-1][-1].shape[-2]  # 196
generated_tokens.append(cur_token)


# ---- LLM decoder ONNX session ----
decoder_sess = ort.InferenceSession(
    "export_onnx/decoder.onnx",
    providers=["CPUExecutionProvider"]
)

for step in range(max_new_tokens - 1):

    attention_mask = np.ones((1, cur_len + 1), dtype=np.int64)

    ort_inputs = {
        "input_ids": cur_token.astype(np.int64),
        "attention_mask": attention_mask,
    }

    # ---- KV cache (flatten) ----
    for i in range(len(past_key_values)):
        ort_inputs[f"past_key_values_{i}"] = past_key_values[i]

    # ---- ONNX runtime ----
    outputs = decoder_sess.run(None, ort_inputs)

    # ---- logits ----
    logits = outputs[0]                                                # (1, 1, 32000)
    next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)

    generated_tokens.append(next_token)

    # ---- EOS check ----
    if next_token.item() == eos_token_id:
        break
    
    # ---- KV cache update ----
    past_key_values = list(outputs[1:])

    # ---- next step ----
    cur_token = next_token
    cur_len += 1


generated_tokens = np.concatenate(generated_tokens, axis=1)        # (1, T)
onnx_generated_tokens = generated_tokens.squeeze(0)                # (T,)
np.save("comparison/onnx_generated_tokens.npy", onnx_generated_tokens)
print(onnx_generated_tokens)