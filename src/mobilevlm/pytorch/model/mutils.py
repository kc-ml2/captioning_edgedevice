import torch
import numpy as np
from PIL import Image
from model.vicuan_templete import conv_vicuna_v1
from model.constants import IMAGE_TOKEN_INDEX


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


def build_prompt(question: str) -> str:
    conv = conv_vicuna_v1.copy()
    conv.append_message(conv.roles[0], "<image>" + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


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
        input_ids.extend(x[offset:])  # input_ids = [1, 319, 13563, ..., -200, 29871, ..., 29901]

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
    return input_ids


def pytorch_zero_kv(
    seq_len=1,
    device="cpu",
    dtype=torch.float32,
):
    batch_size = 1
    num_layers = 24
    num_kv_heads = 16
    head_dim = 128

    zero_kv = []

    for _ in range(num_layers):
        k = torch.zeros(
            (batch_size, num_kv_heads, seq_len, head_dim),
            dtype=dtype,
            device=device,
        )
        v = torch.zeros(
            (batch_size, num_kv_heads, seq_len, head_dim),
            dtype=dtype,
            device=device,
        )

        zero_kv.append((k, v))

    return zero_kv