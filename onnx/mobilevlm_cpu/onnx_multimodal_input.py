import numpy as np


def embed_tokens(input_ids, embed_weight):
    # input_ids: [seq]
    return embed_weight[input_ids]  # [seq, hidden_dim]


def prepare_inputs_labels_for_multimodal_onnx(
    input_ids,
    attention_mask,
    past_key_values,
    labels,
    image_features,
    images=None,
    embed_tokens_weight=-None,
    IMAGE_TOKEN_INDEX=None,
    IGNORE_INDEX=None
):
    """
    input_ids: [B, T]
    attention_mask: [B, T]
    image_features: list of [num_patches, hidden_dim]
    """

    B, T = input_ids.shape
    is_decoding_step = (T == 1)

    # -------------------------
    # 1. decoding or text-only skip
    # -------------------------
    if image_features is None or is_decoding_step:
        if (
            past_key_values is not None
            and image_features is not None
            and is_decoding_step
        ):
            kv_len = past_key_values[-1][0].shape[-2]
            attention_mask = np.ones((B, kv_len + 1), dtype=attention_mask.dtype)

        return input_ids, attention_mask, past_key_values, None, labels

    # -------------------------
    # 2. normalize image_features
    # -------------------------
    if isinstance(image_features, np.ndarray):
        image_features = [image_features]

    new_input_embeds = []
    new_labels = [] if labels is not None else None

    cur_image_idx = 0

    # -------------------------
    # 3. batch loop
    # -------------------------
    for b in range(B):
        cur_input_ids = input_ids[b]
        cur_labels = labels[b] if labels is not None else None

        # case: no <image> token
        if np.sum(cur_input_ids == IMAGE_TOKEN_INDEX) == 0:
            half_len = len(cur_input_ids) // 2

            cur_embed_1 = embed_tokens(cur_input_ids[:half_len], embed_tokens_weight)
            cur_embed_2 = embed_tokens(cur_input_ids[half_len:], embed_tokens_weight)

            cur_embed = np.concatenate(
                [cur_embed_1, cur_embed_2], axis=0
            )

            new_input_embeds.append(cur_embed)

            if labels is not None:
                new_labels.append(cur_labels)

            cur_image_idx += 1
            continue

        # -------------------------
        # image token 처리
        # -------------------------
        cur_new_embeds = []
        cur_new_labels = [] if labels is not None else None

        while True:
            image_positions = np.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if len(image_positions) == 0:
                break

            image_token_start = image_positions[0]
            cur_img_feat = image_features[cur_image_idx]  # [P, D]

            # text before image
            if image_token_start > 0:
                text_embed = embed_tokens(
                    cur_input_ids[:image_token_start],
                    embed_tokens_weight
                )
                cur_new_embeds.append(text_embed)

                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])

            # insert image feature
            cur_new_embeds.append(cur_img_feat)

            if labels is not None:
                ignore_block = np.full(
                    (cur_img_feat.shape[0],),
                    IGNORE_INDEX,
                    dtype=cur_labels.dtype
                )
                cur_new_labels.append(ignore_block)

                cur_labels = cur_labels[image_token_start + 1:]

            cur_input_ids = cur_input_ids[image_token_start + 1:]
            cur_image_idx += 1

        # remaining text
        if len(cur_input_ids) > 0:
            text_embed = embed_tokens(cur_input_ids, embed_tokens_weight)
            cur_new_embeds.append(text_embed)

            if labels is not None:
                cur_new_labels.append(cur_labels)

        # concat
        cur_new_embeds = np.concatenate(cur_new_embeds, axis=0)
        new_input_embeds.append(cur_new_embeds)

        if labels is not None:
            cur_new_labels = np.concatenate(cur_new_labels, axis=0)
            new_labels.append(cur_new_labels)

    # -------------------------
    # 4. padding (align batch)
    # -------------------------
    max_len = max(x.shape[0] for x in new_input_embeds)
    hidden_dim = new_input_embeds[0].shape[1]

    padded_embeds = []
    padded_labels = [] if labels is not None else None
    padded_attention = []

    for i in range(B):
        cur = new_input_embeds[i]
        pad_len = max_len - cur.shape[0]

        if pad_len > 0:
            pad = np.zeros((pad_len, hidden_dim), dtype=cur.dtype)
            cur = np.concatenate([cur, pad], axis=0)

        padded_embeds.append(cur)

        if labels is not None:
            cur_lab = new_labels[i]
            if pad_len > 0:
                pad_lab = np.full((pad_len,), IGNORE_INDEX, dtype=cur_lab.dtype)
                cur_lab = np.concatenate([cur_lab, pad_lab], axis=0)
            padded_labels.append(cur_lab)

        # attention mask
        attn = np.concatenate([
            np.ones(cur.shape[0] - pad_len),
            np.zeros(pad_len)
        ])
        padded_attention.append(attn)

    new_input_embeds = np.stack(padded_embeds, axis=0)

    if labels is not None:
        new_labels = np.stack(padded_labels, axis=0)

    attention_mask = np.stack(padded_attention, axis=0)

    return None, attention_mask, past_key_values, new_input_embeds, new_labels