# mutils.py

import torch
from PIL import Image
from mobilevlm_cpu.model.constants import IMAGE_TOKEN_INDEX


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    # prompt_chunks = [[chunk1], [chunk2]]
    '''
    [
      [1, 319, 13563, ..., 29901, 29871],  # 1 is tokenizer.bos_token_id
      [1, 29871, 13, ..., 13566, 29901]
    ]
    '''

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])  # input_ids = [1]

    for x in insert_separator(prompt_chunks, [IMAGE_TOKEN_INDEX] * (offset + 1)):
        input_ids.extend(x[offset:])   # input_ids = [1, 319, 13563, ..., -200, 29871, ..., 29901]

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
    return input_ids


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

    input_ids = []
    offset = 0
    
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_id():
        offset = 1
        input_ids.append(prompt_chunks[0][0])  # input_ids = [1]
    
    for x in insert_separator(prompt_chunks, [IMAGE_TOKEN_INDEX] * (offset + 1)):
        input_ids.extend(x[offset:])  # input_ids = [1, 319, 13563, ..., -200, 29871, ..., 29901]

    return np.array(input_ids, dtype=np.int64)


import gc
def to_int8_dynamic(model: torch.nn.Module) -> torch.nn.Module:
    torch.backends.quantized.engine = "qnnpack"

    model_int8 = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    model_int8.eval()

    del model
    gc.collect()

    return model_int8


import os, tempfile, torch, psutil

def _fmt_mib(x_bytes: int) -> str:
    return f"{x_bytes / (1024**2):.2f} MiB"


def get_state_dict_file_bytes(model) -> int:
    """
    Save state_dict temporarily and measure file size.
    -> Pure model weights + persistent buffers only.
    """
    if model is None:
        return 0

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        torch.save(model.state_dict(), path)
        return os.path.getsize(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def print_full_memory_report(tag: str, model=None, pid=None):
    pid = pid or os.getpid()

    rss_bytes = psutil.Process(pid).memory_info().rss
    weight_bytes = get_state_dict_file_bytes(model)

    print(f"\n[MEMORY REPORT] {tag} (pid={pid})")
    print(f"  Process RSS (전체 메모리) : {_fmt_mib(rss_bytes)}")
    print(f"  Model state_dict size     : {_fmt_mib(weight_bytes)}")

    if rss_bytes > 0:
        ratio = weight_bytes / rss_bytes
        print(f"  Weight / RSS ratio        : {ratio:.2%}")


def torch_empty_kv(
    model,
    batch_size=1,
    device="cpu",
    dtype=torch.float32,
):
    num_layers = len(model.model.layers)

    attn = model.model.layers[0].self_attn

    num_kv_heads = getattr(attn, "num_key_value_heads", attn.num_heads)
    head_dim = attn.head_dim

    empty_kv = []

    for _ in range(num_layers):
        k = torch.zeros(
            (batch_size, num_kv_heads, 0, head_dim),
            dtype=dtype,
            device=device,
        )
        v = torch.zeros(
            (batch_size, num_kv_heads, 0, head_dim),
            dtype=dtype,
            device=device,
        )

        empty_kv.append((k, v))   # ⭐ 핵심: tuple로 묶기

    return empty_kv

import numpy as np



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
