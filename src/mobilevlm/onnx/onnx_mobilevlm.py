import re
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
import sentencepiece as spm

from utils_onnx import build_prompt, tokenizer_image_token_onnx, np_empty_kv
from onnx_preprocessor import preprocess_batch
from onnx_multimodal_input import build_multimodal_embeddings


total_start_time = time.perf_counter()

# =========================
# 0. Config
# =========================
IMG_PATH = "sample.jpg"

VISION_MODEL_PATH    = "vision_tower.onnx"
PROJECTOR_MODEL_PATH = "mm_projector.onnx"
DECODER_MODEL_PATH   = "mobilellama.onnx"

TOKENIZER_PATH = "tokenizer.model"
EMBED_PATH     = "embed_tokens.npy"

PROVIDERS = ["CPUExecutionProvider"]

QUESTION = "What objects are visible in the scene?"

EOS_TOKEN_ID   = 2
MAX_NEW_TOKENS = 40


# =========================
# 1. ONNX Session Initialization
# =========================
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

# =========================
# 2. Image Loading & Preprocessing
# =========================
image = Image.open(IMG_PATH).convert("RGB")  # (640, 426)

onnx_preprocessor_out = preprocess_batch([image])  # (1, 3, 336, 336)

# =========================
# 3. Vision Encoder Forward Pass
# =========================
vision_out = vision_sess.run(
    ["image_features"],
    {
        "pixel_values": onnx_preprocessor_out,
    },
)[0]

# =========================
# 4. Projector Forward Pass
# =========================
projector_out = projector_sess.run(
    ["projected_features"],
    {
        "image_features": vision_out,
    },
)[0]

# =========================
# 5. Tokenization
# =========================
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

# =========================
# 8. Autoregressive Decoding (LLM)
# =========================
cur_embed = output.astype(np.float32)  # (1, 194, 2048)
cur_len   = cur_embed.shape[1]  # 194

generated_tokens = []

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

    cur_len += 1

    # Stop if EOS token is generated
    if next_token.item() == EOS_TOKEN_ID:
        break

# =========================
# 9. Decode Generated Tokens
# =========================
generated_tokens = np.concatenate(generated_tokens, axis=1)
onnx_generated_tokens = generated_tokens.squeeze(0)

text = sp.decode(onnx_generated_tokens.tolist())
caption = re.sub(r"\s+", " ", text).strip()

print(f"caption: {caption}")