# export_prefill_coreml.py


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
import coremltools as ct

from torch.export import export
from pytorch_modular.pytorch_mobilellama import MobileLlamaModel


# ============================================================
# Model
# ============================================================

MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

mobilellama = MobileLlamaModel.from_pretrained(
    MODEL_PATH,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32,
).eval()

# ============================================================
# Config
# ============================================================

NUM_LAYERS = 24
NUM_KV_HEADS = 16
HEAD_DIM = 128
HIDDEN_SIZE = 2048

CURRENT_SEQ = 194
PAST_SEQ = 2
TOTAL_SEQ = 196

USE_INT8_COMPRESSION = False

# ============================================================
# Wrapper
# ============================================================

class MobileLlamaWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        inputs_embeds,
        attention_mask,
        kv_cache,
    ):
        """
        inputs_embeds
            (1, 194, 2048)

        attention_mask
            (1, 195)

        kv_cache
            (48, 16, past_seq, 128)

        index:
            layer0 key   -> 0
            layer0 value -> 1
            layer1 key   -> 2
            layer1 value -> 3
            ...
        """

        # ----------------------------------------------------
        # Convert KV cache tensor -> tuple[(k,v), ...]
        # ----------------------------------------------------
        pkv = []

        for layer_idx in range(NUM_LAYERS):

            key = kv_cache[layer_idx * 2]
            value = kv_cache[layer_idx * 2 + 1]

            key = key.unsqueeze(0)
            value = value.unsqueeze(0)

            pkv.append((key, value))

        pkv = tuple(pkv)

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=pkv,
            use_cache=True,
            return_dict=False,
        )

        hidden_states = outputs[0]

        hidden_states = hidden_states[:, -1:, :]

        present_kv = outputs[1]

        # ----------------------------------------------------
        # Convert tuple[(k,v), ...] -> tensor
        # ----------------------------------------------------
        output_kv = []

        for key, value in present_kv:

            output_kv.append(
                key.squeeze(0)
            )

            output_kv.append(
                value.squeeze(0)
            )

        output_kv = torch.stack(
            output_kv,
            dim=0,
        )

        return hidden_states, output_kv


wrapper = MobileLlamaWrapper(
    mobilellama
).eval()

# ============================================================
# Dummy Inputs
# ============================================================

dummy_inputs_embeds = torch.randn(
    1,
    CURRENT_SEQ,
    HIDDEN_SIZE,
    dtype=torch.float32,
)

dummy_attention_mask = torch.ones(
    1,
    TOTAL_SEQ,
    dtype=torch.float32,
)

dummy_kv_cache = torch.zeros(
    NUM_LAYERS * 2,
    NUM_KV_HEADS,
    PAST_SEQ,
    HEAD_DIM,
    dtype=torch.float32,
)


# ============================================================
# Torch Export
# ============================================================

from torch.export import Dim

current_seq = Dim(
    "current_seq",
    min=2,
    max=194,
)

past_seq = Dim(
    "past_seq",
    min=2,
    max=236,
)

total_seq = Dim(
    "total_seq",
    min=196,
    max=238,
)


dynamic_shapes = {

    "inputs_embeds": {
        1: current_seq,
    },

    "attention_mask": {
        1: total_seq,
    },

    "kv_cache": {
        2: past_seq,
    },
}

exported_program = torch.export.export(
    wrapper,
    args=(
        dummy_inputs_embeds,
        dummy_attention_mask,
        dummy_kv_cache,
    ),
    dynamic_shapes=dynamic_shapes,
)

exported_program = exported_program.run_decompositions({})

# ============================================================
# CoreML Inputs
# ============================================================

inputs = [

    ct.TensorType(
        name="inputs_embeds",
        shape=ct.EnumeratedShapes(
            shapes=[
                (1, 194, 2048),
                (1,   2, 2048),
            ]
        ),
        dtype=np.float32,
    ),

    ct.TensorType(
        name="attention_mask",
        shape=ct.EnumeratedShapes(
            shapes=[
                (1, 196),
                (1, 238),
            ]
        ),
        dtype=np.float32,
    ),

    ct.TensorType(
        name="kv_cache",
        shape=ct.EnumeratedShapes(
            shapes=[
                (48, 16,   2, 128),
                (48, 16, 236, 128),
            ]
        ),
        dtype=np.float32,
    ),
]


# ============================================================
# CoreML Outputs
# ============================================================

outputs = [

    ct.TensorType(
        name="hidden_states",
        dtype=np.float32,
    ),

    ct.TensorType(
        name="present_kv",
        dtype=np.float32,
    ),
]


# ============================================================
# Convert
# ============================================================

mlmodel = ct.convert(
    exported_program,
    convert_to="mlprogram",
    inputs=inputs,
    outputs=outputs,
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.ALL,
    compute_precision=ct.precision.FLOAT32,
)

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

    SAVE_PATH = "mobilellama_single_int8_llm.mlpackage"

else:
    final_model = mlmodel
    SAVE_PATH = "mobilellama_single_fp32.mlpackage"

final_model.save(SAVE_PATH)

print("Done")