# onnx_multimodal_input.py

import numpy as np

IMAGE_TOKEN_INDEX = -200

embed_path = "embed_tokens.npy"

embedding_weight = np.load(embed_path)  # (32000, 2048)

def build_multimodal_embeddings(
    input_ids,         # (L,)
    image_features,    # (N_img, 144, 2048)
):
    B, _ = input_ids.shape

    output_embeds = []


    current_input_ids = input_ids[0]
    current_image_idx = 0

    while True:

        image_positions = np.where(
            current_input_ids == IMAGE_TOKEN_INDEX
        )[0]

        if len(image_positions) == 0:
            break

        image_start = image_positions[0]

        # text before image token
        before_ids = current_input_ids[:image_start]

        if len(before_ids) > 0:
            before_embeds = embedding_weight[before_ids]
            output_embeds.append(before_embeds)

        # image embeddings
        current_image_embed = image_features[current_image_idx]
        output_embeds.append(current_image_embed)

        current_image_idx += 1

        # remove used image token
        current_input_ids = current_input_ids[image_start + 1:]

        # remain text
        if len(current_input_ids) > 0:
            remain_embeds = embedding_weight[current_input_ids]
            output_embeds.append(remain_embeds)

        # concat all
        output_embeds = np.concatenate(output_embeds, axis=0)

    return output_embeds
