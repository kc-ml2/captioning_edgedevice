import torch
import torch.nn as nn
from PIL import Image


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

def process_images(image, image_processor):

    image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
    image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    return image


def build_multimodal_embeddings(
    input_ids: torch.Tensor,          # (1, L)
    image_features: torch.Tensor,     # (144, 2048)
    token_embedding: nn.Embedding,
):
    IMAGE_TOKEN_INDEX = -200

    # ---- Find image token position ----
    image_pos = (
        input_ids[0] == IMAGE_TOKEN_INDEX
    ).nonzero(as_tuple=True)[0].item()

    # ---- Split text tokens ----
    before_ids = input_ids[0, :image_pos]
    after_ids = input_ids[0, image_pos + 1:]

    output_embeds = []

    # ---- Text before image ----
    if before_ids.numel() > 0:
        output_embeds.append(
            token_embedding(before_ids)
        )

    # ---- Image embedding ----
    output_embeds.append(image_features)

    # ---- Text after image ----
    if after_ids.numel() > 0:
        output_embeds.append(
            token_embedding(after_ids)
        )

    # ---- Concatenate all embeddings ----
    output_embeds = torch.cat(
        output_embeds,
        dim=0
    )

    # ---- Add batch dimension ----
    output_embeds = output_embeds.unsqueeze(0)  # (1, N, 2048)

    return output_embeds

def tokenize_image_question(
    question: str,
    tokenizer,
) -> torch.Tensor:

    BOS_ID = 1
    IMAGE_TOKEN_INDEX = -200

    prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "USER: <image>\n"
        f"{question} ASSISTANT:"
    )

    chunks = prompt.split("<image>")

    input_ids = []

    for chunk_idx, chunk in enumerate(chunks):

        encoded = tokenizer(chunk).input_ids

        if chunk_idx == 0:

            input_ids.extend(encoded)

        else:

            input_ids.append(IMAGE_TOKEN_INDEX)

            # skip duplicated BOS
            if encoded[0] == BOS_ID:
                encoded = encoded[1:]

            input_ids.extend(encoded)

    return torch.tensor(
        input_ids,
        dtype=torch.long
    ).unsqueeze(0)


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