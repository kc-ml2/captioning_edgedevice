# pytorch_moblievlm_cpu.py

import re
import time
import torch
import torch.nn as nn

from PIL import Image

from pytorch_utils import process_images, build_multimodal_embeddings, tokenize_image_question, pytorch_zero_kv

# --------------------------------------------------
# Global configuration
# --------------------------------------------------

MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"
VISION_PATH = "openai/clip-vit-large-patch14-336"
IMAGE_PATH = "000000000139.jpg"

device = torch.device("cpu")

total_start_time = time.perf_counter()

# --------------------------------------------------
# Model loading
# --------------------------------------------------

t0 = time.perf_counter()

from transformers import (
    LlamaTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
)

# ---- Tokenizer ----
tokenizer = LlamaTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False,
)

# ---- Image processor ----
image_processor = CLIPImageProcessor.from_pretrained(
    VISION_PATH,
)

# ---- Vision encoder ----
vision_encoder = CLIPVisionModel.from_pretrained(
    VISION_PATH,
).eval()

# ---- MM projector ----
from pytorch_mm_projector import LDPNetV2Projector

mm_projector = LDPNetV2Projector().eval()

mm_projector.load_state_dict(
    torch.load(
        "mm_projector_weights.pth",
        map_location="cpu",
    )
)

# ---- MobileLlama backbone ----
from pytorch_mobilellama import MobileLlamaModel

mobilellama = MobileLlamaModel.from_pretrained(
    MODEL_PATH,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32,
).eval()


# ---- Token embedding ----
token_embedding = mobilellama.embed_tokens

# ---- LM head ----
lm_head = nn.Linear(
    in_features=2048,
    out_features=32000,
    bias=False,
)

lm_head.load_state_dict(
    torch.load(
        "lm_head.pth",
        map_location="cpu",
    )
)

lm_head.eval()

t1 = time.perf_counter()

print(f"[TIME] Model loading: {t1 - t0:.4f} sec")

# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

t0 = time.perf_counter()

image = Image.open(IMAGE_PATH).convert("RGB")

image_tensor = process_images(
    image,
    image_processor,
).unsqueeze(0)

t1 = time.perf_counter()

print(f"[TIME] Preprocess: {t1 - t0:.4f} sec")

# --------------------------------------------------
# Vision encoder + projector
# --------------------------------------------------

t0 = time.perf_counter()

with torch.inference_mode():

    outputs = vision_encoder(
        pixel_values=image_tensor,
        output_hidden_states=True,
    )

    vision_features = outputs.hidden_states[-2][:, 1:]

    projector_features = mm_projector(vision_features)

t1 = time.perf_counter()

print(f"[TIME] Vision + Projector: {t1 - t0:.4f} sec")

# --------------------------------------------------
# Prompt preparation
# --------------------------------------------------

t0 = time.perf_counter()

input_ids = tokenize_image_question(
    question ="What objects are visible in the image.",
    tokenizer=tokenizer
)

with torch.inference_mode():

    multimodal_inputs_embeds = build_multimodal_embeddings(
        input_ids=input_ids,
        image_features=projector_features[0],   # (144, 2048)
        token_embedding=token_embedding,
    )

t1 = time.perf_counter()

print(f"[TIME] Multimodal prepare: {t1 - t0:.4f} sec")

# ---- LLM prefill ----
eos_token_id = 2
max_new_tokens = 40
generated_tokens = []

cur_embed = multimodal_inputs_embeds.to(torch.float32)  # [1, 194, 2048]
torch_kv = pytorch_zero_kv(seq_len=2)   # [24, 2, 1, 16, 2, 128]

cur_len = cur_embed.shape[-2]  # 194

attention_mask = torch.ones(
    (1, cur_len + 2),
    dtype=torch.long,
    device="cpu"
)
attention_mask[:, :2] = 0


with torch.no_grad():
        outputs = mobilellama(              # MobileLlamaModel
            inputs_embeds=cur_embed,        # [1,194,2048]
            attention_mask=attention_mask,  # [1,196]
            past_key_values=torch_kv,       # [24,2,1,16,2,128]
            use_cache=True,
            return_dict=True,  # Key: ['last_hidden_state', 'past_key_values']
        )
pt_next_logit = lm_head(outputs.last_hidden_state[:, -1, :])  # [1, 32000]
cur_token = torch.argmax(pt_next_logit, dim=-1, keepdim=True)


cur_embed = token_embedding(cur_token)  # tensor.Size([1, 1, 2048])
generated_tokens.append(cur_token)      # 512, 278, ...

torch_kv = outputs.past_key_values  # [24,2,1,16,196,128]

cur_len += 2  # 196

print(cur_token)


total_llm_time = 0
# ---- LLM decoder ----
for step in range(max_new_tokens):

    t0 = time.perf_counter()

    max_embedding_input = torch.zeros(
        (1, 2, 2048),
        dtype=torch.float32,
        device="cpu"
    )
    max_embedding_input[:, 1:2, :] = cur_embed

    max_attention_mask = torch.zeros(
        (1, 238),
        dtype=torch.long,
        device="cpu"
    )

    max_attention_mask[:, 2:cur_len] = 1
    max_attention_mask[:, -1] = 1

    max_torch_kv = pytorch_zero_kv(seq_len=236)   # [24, 2, 1, 16, 235, 128]

    for (static_k, static_v), (new_k, new_v) in zip(
        max_torch_kv,
        torch_kv,
    ):
        static_k[:, :, :cur_len, :].copy_(new_k)
        static_v[:, :, :cur_len, :].copy_(new_v)
    
    with torch.no_grad():
        outputs = mobilellama(    # MobileLlamaModel
            inputs_embeds=max_embedding_input,
            attention_mask=max_attention_mask,
            past_key_values=max_torch_kv,
            use_cache=True,
            return_dict=True,     # Key: ['last_hidden_state', 'past_key_values']
        )

    pt_next_logit = lm_head(outputs.last_hidden_state)[:, -1, :]  # [1, 32000]

    torch_kv = tuple(
        (
            torch.cat(
                [
                    k[:, :, :cur_len, :],
                    k[:, :, -1:, :]
                ],
                dim=2
            ),
            torch.cat(
                [
                    v[:, :, :cur_len, :],
                    v[:, :, -1:, :]
                ],
                dim=2
            )
        )
        for k, v in outputs.past_key_values
    )

    cur_token = torch.argmax(
        pt_next_logit,
        dim=-1,
        keepdim=True
    )

    print(f"cur_token: {cur_token}")

    cur_embed = token_embedding(cur_token)  # tensor.Size([1, 1, 2048])
    generated_tokens.append(cur_token)      # 512, 278, ...

    cur_len += 1
    
    if cur_token.item() == eos_token_id:
        break

    t1 = time.perf_counter()
    step_time = t1 - t0
    total_llm_time += step_time

print(f"[TIME] Total LLM decode: {total_llm_time:.4f} sec (Avg per token: {total_llm_time / len(generated_tokens):.4f} sec)")

generated_tokens = torch.cat(generated_tokens, dim=1)  # [1, T]

text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
caption = re.sub(r"\s+", " ", text[0]).strip()
print(f"caption: {caption}")

total_end_time = time.perf_counter()
print(f"[TIME] Total process: {total_end_time - total_start_time:.4f} sec")
