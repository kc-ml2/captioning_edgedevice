# export_coreml_llm.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import coremltools as ct

from model.mobilevlm import load_pretrained_model


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
USE_INT8_COMPRESSION = True

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

        last_hidden = hidden_states[:, -1:, :]

        logits = self.lm_head(last_hidden)
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

    _, model, _, _ = load_pretrained_model(
        model_path=MODEL_PATH,
        device=DEVICE,
    )

    model.eval()

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

    seq_dim = ct.RangeDim(
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
            shape=(1, seq_dim, HIDDEN_SIZE),
            dtype=np.float32,
        )
    )

    # -----------------------------------------------------
    # attention_mask
    # -----------------------------------------------------

    inputs.append(
        ct.TensorType(
            name="attention_mask",
            shape=(1, seq_dim),
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
                    seq_dim,
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
                    seq_dim,
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
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT32,
    )

    # =====================================================
    # Compression Option
    # =====================================================

    if USE_INT8_COMPRESSION:

        from coremltools.optimize.coreml import (
            linear_quantize_weights,
            OptimizationConfig,
            OpLinearQuantizerConfig
        )

        config = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype=np.int8
            )
        )

        final_model = linear_quantize_weights(
            mlmodel,
            config=config
        )

        SAVE_PATH = "mobilellama_int8.mlpackage"

    else:
        final_model = mlmodel
        SAVE_PATH = "mobilellama_fp32.mlpackage"

    # =====================================================
    # Save
    # =====================================================

    final_model.save(SAVE_PATH)

    print("========================================")
    print(f"CoreML export complete: {SAVE_PATH}")
    print("========================================")


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":
    main()