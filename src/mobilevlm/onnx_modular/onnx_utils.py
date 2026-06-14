# onnx_utils.py

import numpy as np


def build_prompt(question: str) -> str:
    return (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "USER: <image>\n"
        f"{question} ASSISTANT:"
    )

def tokenizer_image_token_onnx(prompt, tokenizer):
    prompt_chunks = [
        [tokenizer.bos_id()] + tokenizer.encode(chunk, out_type=int) 
        for chunk in prompt.split('<image>')
    ]
    # prompt_chunks = [[chunk1], [chunk2]]
    '''
    [
      [1, 319, 13563, ..., 29901, 29871],  # 1 is tokenizer.bos_token_id
      [1, 29871, 13, ..., 13566, 29901]
    ]
    '''
    
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    IMAGE_TOKEN_INDEX = -200
    input_ids = []
    offset = 0
    
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_id():
        offset = 1
        input_ids.append(prompt_chunks[0][0])  # input_ids = [1]
    
    for x in insert_separator(prompt_chunks, [IMAGE_TOKEN_INDEX] * (offset + 1)):
        input_ids.extend(x[offset:])  # input_ids = [1, 319, 13563, ..., -200, 29871, ..., 29901]

    return np.array(input_ids, dtype=np.int64)


def np_empty_kv(batch_size=1, dtype=np.float32):
    """
    Create empty KV cache for decoder (NumPy version, no torch).

    Args:
        model: loaded model (used only for config access)
        batch_size: batch size
        dtype: numpy dtype (e.g., np.float32)

    Returns:
        List of (k, v) tuples for each layer
    """

    # Number of transformer layers
    num_layers = 24


    num_kv_heads = 16

    # Head dimension
    head_dim = 128

    empty_kv = []

    for _ in range(num_layers):
        # Create empty key tensor: (B, num_kv_heads, 0, head_dim)
        k = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim),
            dtype=dtype,
        )

        # Create empty value tensor: (B, num_kv_heads, 0, head_dim)
        v = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim),
            dtype=dtype,
        )

        empty_kv.append((k, v))

    return empty_kv
