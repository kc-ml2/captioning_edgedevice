# export_embedding_layer.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
import coremltools as ct

from pytorch_modular.pytorch_mobilellama import MobileLlamaModel


# =========================================================
# Config
# =========================================================

MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

mobilellama = MobileLlamaModel.from_pretrained(
    MODEL_PATH,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32,
).eval()

token_embedding = mobilellama.embed_tokens
token_embedding.eval()

USE_INT8 = True


# ============================================================
# Model Definition
# ============================================================

class TokenEmbeddingModel(nn.Module):
    """
    Input:
        input_ids
        Shape: (batch_size, seq_len)
        Dtype: torch.int64

    Output:
        inputs_embeds
        Shape: (batch_size, seq_len, 2048)
        Dtype: torch.float32
    """

    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input_ids):
        return self.embedding(input_ids)


# ============================================================
# Load Embedding Layer
# ============================================================

embedding_model = TokenEmbeddingModel(
    token_embedding
).eval()

dummy_input = torch.tensor(
    [[1]],
    dtype=torch.long
)

exported_program = torch.export.export(
    embedding_model,
    (dummy_input,),
)

exported_program = exported_program.run_decompositions({})

# ============================================================
# Output File Name
# ============================================================

if USE_INT8:
    save_path = "embeddinglayer_int8.mlpackage"

else:
    save_path = "embeddinglayer_fp16.mlpackage"


# ============================================================
# CoreML Conversion
# ============================================================

mlmodel = ct.convert(
    exported_program,
    inputs=[
        ct.TensorType(
            name="token",
            shape=dummy_input.shape,
            dtype=np.int32
        )
    ],
    outputs=[
        ct.TensorType(
            name="embedding_input",
            dtype=np.float32
        )
    ],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS18,
    compute_precision=ct.precision.FLOAT16
)

# ============================================================
# Optional Weight Compression
# ============================================================

if USE_INT8:

    from coremltools.optimize.coreml import (
        OptimizationConfig,
        OpLinearQuantizerConfig,
        linear_quantize_weights,
    )

    print("Applying INT8 quantization...")

    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=np.int8,
        )
    )

    mlmodel = linear_quantize_weights(
        mlmodel,
        config=config,
    )


# ============================================================
# Save CoreML Model
# ============================================================

mlmodel.save(save_path)

print(f"CoreML export completed: {save_path}")