import re, os
os.environ["CUDA_VISIBLE_DEVICES"] = ".."
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
import sentencepiece as spm

from onnx_utils import build_prompt, tokenizer_image_token_onnx, np_empty_kv
from onnx_preprocessor import preprocess_batch
from onnx_multimodal_input import build_multimodal_embeddings


total_start_time = time.perf_counter()

# =========================
# 0. Config
# =========================
IMG_PATH = "../000000000139.jpg"

VISION_MODEL_PATH    = "../export_onnx/vision_tower.onnx"
PROJECTOR_MODEL_PATH = "../export_onnx/mm_projector.onnx"
DECODER_MODEL_PATH   = "../export_onnx/mobilellama.onnx"

TOKENIZER_PATH = "../export_onnx/tokenizer.model"
EMBED_PATH     = "../export_onnx/embed_tokens.npy"

PROVIDERS = ["CPUExecutionProvider"]

QUESTION = "What objects are visible in the scene?"

EOS_TOKEN_ID   = 2
MAX_NEW_TOKENS = 40


# =========================
# 1. ONNX Session Initialization
# =========================
t0 = time.perf_counter()

vision_sess = ort.InferenceSession(
    VISION_MODEL_PATH,
    providers=PROVIDERS
)

projector_sess = ort.InferenceSession(
    PROJECTOR_MODEL_PATH,
    providers=PROVIDERS
)

decoder_sess = ort.InferenceSession(
    DECODER_MODEL_PATH,
    providers=PROVIDERS
)

embedding_weight = np.load(EMBED_PATH)

t1 = time.perf_counter()
print(f"[TIME] Model loading: {t1 - t0:.4f} sec")


# =========================
# 2. Image Loading & Preprocessing
# =========================
t0 = time.perf_counter()

image = Image.open(IMG_PATH).convert("RGB")  # (640, 426)

onnx_preprocessor_out = preprocess_batch([image])  # (1, 3, 336, 336)

t1 = time.perf_counter()
print(f"[TIME] Preprocess: {t1 - t0:.4f} sec")


# =========================
# 3. Vision Encoder Forward Pass
# =========================
t0 = time.perf_counter()

vision_out = vision_sess.run(
    ["image_features"],
    {
        "pixel_values": onnx_preprocessor_out,
    },
)[0]

t1 = time.perf_counter()
print(f"[TIME] Vision encoder: {t1 - t0:.4f} sec")


# =========================
# 4. Projector Forward Pass
# =========================
t0 = time.perf_counter()

projector_out = projector_sess.run(
    ["projected_features"],
    {
        "image_features": vision_out,
    },
)[0]

t1 = time.perf_counter()
print(f"[TIME] Projector: {t1 - t0:.4f} sec")



# =========================
# 5. Tokenization
# =========================
t0 = time.perf_counter()

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)

prompt = build_prompt(QUESTION)

input_ids = tokenizer_image_token_onnx(prompt, sp)
input_ids = np.expand_dims(input_ids, axis=0)


# =========================
# 6. Multimodal Embedding Preparation
# =========================

output = build_multimodal_embeddings(
    input_ids,
    projector_out,
)
output = np.expand_dims(output, axis=0)  # [1,194,2048]


# =========================
# 7. KV Cache Initialization (Flattened)
# =========================
kv_tuple = np_empty_kv(batch_size=1)

past_key_values = [
    arr.astype(np.float32)
    for k, v in kv_tuple
    for arr in (k, v)
]

t1 = time.perf_counter()
print(f"[TIME] Multimodal prepare: {t1 - t0:.4f} sec")


# =========================
# 8. Autoregressive Decoding (LLM)
# =========================
cur_embed = output.astype(np.float32)  # (1, 194, 2048)
cur_len   = cur_embed.shape[1]  # 194

generated_tokens = []

total_llm_time = 0.0

for step in range(MAX_NEW_TOKENS):

    t0 = time.perf_counter()
    
    # Update attention mask for current sequence length
    attention_mask = np.ones(
        (1, cur_len),
        dtype=np.int64
    )

    # Prepare ONNX inputs (including KV cache)
    ort_inputs = {
        "inputs_embeds": cur_embed,
        "attention_mask": attention_mask,
        **{
            f"past_key_values_{i}": past_key_values[i]
            for i in range(len(past_key_values))
        }
    }

    # Run decoder
    outputs = decoder_sess.run(None, ort_inputs)

    # Update KV cache
    past_key_values = list(outputs[1:])

    # Extract logits and select next token (greedy decoding)
    logits = outputs[0]
    next_token = np.argmax(
        logits[:, -1, :],
        axis=-1,
        keepdims=True
    )

    # Convert token to embedding
    cur_embed = embedding_weight[next_token]  # (1, 1, 2048)
    generated_tokens.append(next_token)

    cur_len += 1  # 과거 토큰 + 현재 토큰나

    # Stop if EOS token is generated
    if next_token.item() == EOS_TOKEN_ID:
        break

    t1 = time.perf_counter()
    step_time = t1 - t0
    total_llm_time += step_time


print(f"[TIME] Total LLM decode: {total_llm_time:.4f} sec (Avg per token: {total_llm_time / len(generated_tokens):.4f} sec)")

# =========================
# 9. Decode Generated Tokens
# =========================
generated_tokens = np.concatenate(generated_tokens, axis=1)
onnx_generated_tokens = generated_tokens.squeeze(0)

text = sp.decode(onnx_generated_tokens.tolist())
caption = re.sub(r"\s+", " ", text).strip()

total_end_time = time.perf_counter()
print(f"[TIME] Total process: {total_end_time - total_start_time:.4f} sec")

print(f"caption: {caption}")

