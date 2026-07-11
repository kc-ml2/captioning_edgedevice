# export_lmhead.py

import torch
import torch.nn as nn
import numpy as np
import coremltools as ct

# =========================================================
# Config
# =========================================================

LMHEAD_PATH = "lm_head_fp32.bin"
lm_head = nn.Linear(
    in_features=2048,
    out_features=32000,
    bias=False,
)

weight = np.fromfile(
    LMHEAD_PATH,
    dtype=np.float32
)

assert weight.size == 32000 * 2048

weight = weight.reshape(32000, 2048)

lm_head.weight.data.copy_(torch.from_numpy(weight))

lm_head.eval()

HIDDEN_SIZE = 2048
VOCAB_SIZE = 32000

DEVICE = "cpu"
USE_INT8_COMPRESSION = True

# =========================================================
# Decoder Wrapper
# =========================================================

class LMHeadWrapper(nn.Module):

    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head

    def forward(
        self, 
        hidden_states
    ):
        return self.lm_head(hidden_states)



# =========================================================
# Main
# =========================================================

wrapper = LMHeadWrapper(
    lm_head
).eval()

# =====================================================
# Dummy inputs
# =====================================================

dummy_hidden_states = torch.randn(
    1,
    1,
    HIDDEN_SIZE,
    dtype=torch.float32,
)

# ============================================================
# Torch Export
# ============================================================

exported_program = torch.export.export(
    wrapper,
    args=(
        dummy_hidden_states,
    ),
)

exported_program = exported_program.run_decompositions({})

# ============================================================
# CoreML Inputs
# ============================================================

inputs = [
    ct.TensorType(
        name="hidden_states",
        shape=(
            1,
            1,
            HIDDEN_SIZE,
        ),
        dtype=np.float32,
    ),
]


# ============================================================
# CoreML Outputs
# ============================================================

outputs = [

    ct.TensorType(
        name="logits",
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
    compute_precision=ct.precision.FLOAT16,
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

    SAVE_PATH = "lm_head_int8.mlpackage"

else:
    final_model = mlmodel
    SAVE_PATH = "lm_head_fp16.mlpackage"

final_model.save(SAVE_PATH)

print("Done")