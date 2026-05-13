# export_coreml_llm.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import coremltools as ct

from PIL import Image

from model.mobilevlm import load_pretrained_model
from model.mutils import (
    process_images,
    build_prompt,
    tokenizer_image_token,
)

# =========================================================
# Config
# =========================================================

MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

MAX_SEQ_LEN = 256
NUM_LAYERS = 24
NUM_HEADS = 16
HEAD_DIM = 128
HIDDEN_SIZE = 2048
VOCAB_SIZE = 32000

DEVICE = "cpu"

# =========================================================
# Decoder Wrapper
# =========================================================

class DecoderWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()

        self.model = model.model
        self.lm_head = model.lm_head

    def forward(
        self,
        inputs_embeds,
        attention_mask,
        *past_key_values
    ):

        # -----------------------------------------
        # flat -> tuple
        # -----------------------------------------

        pkv = []

        num_layers = len(past_key_values) // 2

        for i in range(num_layers):

            k = past_key_values[2 * i]
            v = past_key_values[2 * i + 1]

            pkv.append((k, v))

        # -----------------------------------------
        # forward
        # -----------------------------------------

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=pkv,
            use_cache=True,
            return_dict=False,
        )

        hidden_states = outputs[0]
        present_kv = outputs[1]

        logits = self.lm_head(hidden_states)
        logits = logits[..., :VOCAB_SIZE]

        # -----------------------------------------
        # tuple -> flat
        # -----------------------------------------

        flat_kv = []

        for k, v in present_kv:
            flat_kv.extend([k, v])

        return (logits, *flat_kv)


# =========================================================
# Main
# =========================================================

def main():

    # =====================================================
    # Load model
    # =====================================================

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=MODEL_PATH,
        device=DEVICE,
    )

    model.eval()

    # =====================================================
    # Build multimodal embed
    # =====================================================

    img_path = "../000000000139.jpg"

    image = Image.open(img_path).convert("RGB")

    image_tensor = process_images(
        [image],
        image_processor,
        model.config,
    )

    image_tensor = image_tensor.to(
        device=DEVICE,
        dtype=torch.float32,
    )

    question = "What objects are visible in the scene?"

    prompt = build_prompt(question)

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        return_tensors="pt",
    ).unsqueeze(0).to(DEVICE)

    attention_mask = torch.ones_like(input_ids)

    (
        input_ids,
        attention_mask_,
        _,
        inputs_embeds,
        _
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids,
        attention_mask,
        past_key_values=None,
        labels=None,
        images=image_tensor,
    )

    inputs_embeds = inputs_embeds.to(torch.float32)
    attention_mask_ = attention_mask_.to(torch.long)

    print("inputs_embeds.shape:", inputs_embeds.shape)

    # =====================================================
    # Run one forward for KV shape extraction
    # =====================================================

    with torch.no_grad():

        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_,
            use_cache=True,
            return_dict=True,
        )

        pkv = outputs.past_key_values

    # expected:
    # [1, 16, seq, 128]

    # =====================================================
    # Wrapper
    # =====================================================

    wrapper = DecoderWrapper(model).eval()

    # =====================================================
    # Dummy inputs
    # =====================================================

    PAST_SEQ = 1
    CURRENT_SEQ = 32
    TOTAL_SEQ = PAST_SEQ + CURRENT_SEQ

    dummy_inputs_embeds = torch.randn(
        1,
        CURRENT_SEQ,
        HIDDEN_SIZE,
        dtype=torch.float32,
    )

    dummy_attention_mask = torch.ones(
        (1, TOTAL_SEQ),
        dtype=torch.float32,
    )

    # -----------------------------------------------------
    # Fully dynamic KV cache
    # -----------------------------------------------------

    dummy_pkv = []

    for _ in range(NUM_LAYERS):

        dummy_k = torch.zeros(
            (
                1,
                NUM_HEADS,
                PAST_SEQ,
                HEAD_DIM,
            ),
            dtype=torch.float32,
        )

        dummy_v = torch.zeros(
            (
                1,
                NUM_HEADS,
                PAST_SEQ,
                HEAD_DIM,
            ),
            dtype=torch.float32,
        )

        dummy_pkv.append(dummy_k)
        dummy_pkv.append(dummy_v)

    # =====================================================
    # Trace
    # =====================================================

    with torch.no_grad():

        traced_model = torch.jit.trace(
            wrapper,
            (
                dummy_inputs_embeds,
                dummy_attention_mask,
                *dummy_pkv,
            ),
            strict=False,
        )

    print("Torch trace complete")

    # =====================================================
    # CoreML input specs
    # =====================================================


    embed_seq_dim = ct.RangeDim(
        lower_bound=1,
        upper_bound=194,
    )

    mask_seq_dim = ct.RangeDim(
        lower_bound=194,
        upper_bound=MAX_SEQ_LEN,
    )

    kv_seq_dim = ct.RangeDim(
        lower_bound=1,
        upper_bound=MAX_SEQ_LEN,
    )

    inputs = []

    # -----------------------------------------------------
    # inputs_embeds
    # -----------------------------------------------------

    inputs.append(
        ct.TensorType(
            name="inputs_embeds",
            shape=(1, embed_seq_dim, HIDDEN_SIZE),
            dtype=np.float32,
        )
    )

    # -----------------------------------------------------
    # attention_mask
    # -----------------------------------------------------

    inputs.append(
        ct.TensorType(
            name="attention_mask",
            shape=(1, mask_seq_dim),
            dtype=np.float32,
        )
    )

    # -----------------------------------------------------
    # KV cache
    # -----------------------------------------------------

    for i in range(NUM_LAYERS):

        inputs.append(
            ct.TensorType(
                name=f"past_key_{i}",
                shape=(
                    1,
                    NUM_HEADS,
                    kv_seq_dim,
                    HEAD_DIM,
                ),
                dtype=np.float32,
            )
        )

        inputs.append(
            ct.TensorType(
                name=f"past_value_{i}",
                shape=(
                    1,
                    NUM_HEADS,
                    kv_seq_dim,
                    HEAD_DIM,
                ),
                dtype=np.float32,
            )
        )
    
    # =====================================================
    # Output specs
    # =====================================================

    outputs = []

    # -----------------------------------------------------
    # logits
    # -----------------------------------------------------

    outputs.append(
        ct.TensorType(
            name="logits",
            dtype=np.float32,
        )
    )

    # -----------------------------------------------------
    # KV cache outputs
    # -----------------------------------------------------

    for i in range(NUM_LAYERS):

        outputs.append(
            ct.TensorType(
                name=f"present_key_{i}",
                dtype=np.float32,
            )
        )

        outputs.append(
            ct.TensorType(
                name=f"present_value_{i}",
                dtype=np.float32,
            )
        )

    # =====================================================
    # Convert
    # =====================================================

    mlmodel = ct.convert(
        traced_model,
        convert_to="mlprogram",   # IMPORTANT
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT32
    )

    # =====================================================
    # Save
    # =====================================================

    SAVE_PATH = "mobilellama_32.mlpackage"

    mlmodel.save(SAVE_PATH)

    print("========================================")
    print("CoreML export complete")
    print(SAVE_PATH)
    print("========================================")


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":
    main()