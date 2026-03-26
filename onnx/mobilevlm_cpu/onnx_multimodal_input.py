# onnx_multimodal_prefill.py

import numpy as np
from model.constants import IMAGE_TOKEN_INDEX

embedding_weight = np.load("export_onnx/embed_tokens.npy")  # (32000, 2048)

def prepare_inputs_labels_for_multimodal_onnx(
    input_ids,          # (B, L)
    attention_mask,     # (B, L)
    image_features,     # (B, N_img, D)
):  
    """
    ONNX-only version of MobileVLM multimodal input builder.

    Args:
        input_ids: np.ndarray (B, L)
        attention_mask: np.ndarray (B, L)
        image_features: np.ndarray (B, N_img, D)

    Returns:
        new_inputs_embeds: (B, new_L, D)
        new_attention_mask: (B, new_L)
    """

    B, _ = input_ids.shape

    new_input_embeds = []
    new_attention_masks = []

    for batch_idx in range(B):
        cur_input_ids = input_ids[batch_idx]                          # (53,)
        cur_attention_mask = attention_mask[batch_idx]
        cur_image_idx = 0

        image_token_indices = np.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]  # [35]
        
        cur_new_input_embeds = []
        cur_new_attn_mask = []

        # Replace each <image> token with image features
        while image_token_indices.size > 0:
            image_token_start = image_token_indices[0]                 # 35
            before_ids = cur_input_ids[:image_token_start]
            cur_new_input_embeds.append(embedding_weight[before_ids])  # [[35, 2048]]
            cur_new_attn_mask.append(np.ones(len(before_ids), dtype=cur_attention_mask.dtype))  # [[35]]

            cur_image_features = image_features[cur_image_idx]        # [144, 2048]
            cur_new_input_embeds.append(cur_image_features)           # [[35, 2048], [144, 2048]]
            cur_new_attn_mask.append(np.ones(cur_image_features.shape[0], dtype=cur_attention_mask.dtype))    # [[35], [144]]
            cur_image_idx += 1

            cur_input_ids = cur_input_ids[image_token_start+1:]       # (17,)

            # Update indices of remaining <image> tokens after slicing
            image_token_indices = np.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

        after_ids = cur_input_ids
        cur_new_input_embeds.append(embedding_weight[after_ids])  # [[35, 2048], [144, 2048], [17, 2048]]
        cur_new_attn_mask.append(np.ones(len(after_ids), dtype=cur_attention_mask.dtype))

        # Merge
        cur_new_input_embeds = np.concatenate(cur_new_input_embeds, axis=0)  # (196, 2048)
        cur_new_attn_mask = np.concatenate(cur_new_attn_mask, axis=0)        # (196,)

        new_input_embeds.append(cur_new_input_embeds)
        new_attention_masks.append(cur_new_attn_mask)


    # ------------------------
    # padding (VERY IMPORTANT)
    # ------------------------
    max_len = max(x.shape[0] for x in new_input_embeds)

    padded_embeds = []
    padded_masks = []

    for emb, mask in zip(new_input_embeds, new_attention_masks):
        pad_len = max_len - emb.shape[0]

        if pad_len > 0:
            emb = np.concatenate(
                [emb, np.zeros((pad_len, D), dtype=emb.dtype)],
                axis=0
            )
            mask = np.concatenate(
                [mask, np.zeros(pad_len, dtype=mask.dtype)],
                axis=0
            )

        padded_embeds.append(emb)
        padded_masks.append(mask)

    new_input_embed = np.stack(padded_embeds, axis=0)   # (B, max_len, D)
    new_attention_mask = np.stack(padded_masks, axis=0)  # (B, max_len)

    return new_attention_mask, new_input_embed
