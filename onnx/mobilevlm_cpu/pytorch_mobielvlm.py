# pytorch_moblievlm.py

import os, types
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import re, torch, argparse, time, gc
from PIL import Image
from typing import Dict
import numpy as np
import onnxruntime as ort

from model.mobilevlm import load_pretrained_model
from model.mutils import process_images, build_prompt, tokenizer_image_token, torch_empty_kv


# ---- Default values ---- #
GEN_KWARGS_DEFAULT = dict(
    num_beams=1,
    max_new_tokens=40,
    min_new_tokens=40,
)

device = torch.device("cpu")
MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL_PATH,
    device="cpu",
)
model.eval()  # MobileLlamaForCausalLM


# ---- inference ----
img_path = "000000000139.jpg"
image = Image.open(img_path).convert("RGB")  # (426, 640, 3)


# ---- Image preprocessor ----
image_tensor = process_images([image], image_processor, model.config)  # (1, 3, 336, 336)
# np.save("comparison/pytorch_preprocessor_out.npy", image_tensor.detach().cpu().numpy())


# ---- vision encoder output ----
vision_tower = model.get_model().get_vision_tower()
vision_tower.eval()

with torch.no_grad():
    pytorch_vision_out = vision_tower(image_tensor)
# np.save("comparison/pytorch_vision_out.npy", pytorch_vision_out.cpu().numpy())


# ---- projector output ----
mm_projector = model.get_model().mm_projector
mm_projector.eval()

with torch.no_grad():
    pytorch_projector_out = mm_projector(pytorch_vision_out)
# np.save("comparison/pytorch_projector_out.npy", pytorch_projector_out.cpu().numpy())


# ---- LLM part ----
question = "What objects are visible in the image in detail."
prompt = build_prompt(question)

input_ids = tokenizer_image_token(
    prompt,
    tokenizer,                      # LlamaTokenizer
    return_tensors="pt",
).unsqueeze(0).to(device)


# ---- tokenizer and embedding layer ----
input_ids_clean = input_ids[input_ids >= 0]
with torch.no_grad():
    pt_prompt_embedding = model.model.embed_tokens(input_ids_clean)  # (seq_len, hidden_dim)
# np.save("comparison/pytorch_prompt_embedding.npy", pt_prompt_embedding.cpu().numpy())


# ---- Multimodal input ----
attention_mask = torch.ones_like(input_ids)

with torch.no_grad():
    _, torch_attention_mask_, _, torch_multimodals_inputs_embeds, _ = \
        model.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_mask,
            past_key_values=None,
            labels=None,
            images=image_tensor,
            # image_features=pytorch_projector_out,
        )
# np.save("comparison/pytorch_multimodal_input.npy", torch_multimodals_inputs_embeds.detach().cpu().numpy())

# ---- LLM prefill + decoder ----
eos_token_id = 2
max_new_tokens = GEN_KWARGS_DEFAULT["max_new_tokens"]
generated_tokens = []

cur_embed = torch_multimodals_inputs_embeds.to(torch.float32)
torch_kv = torch_empty_kv(model, batch_size=1, device="cpu")  # [24, 2, 1, 16, 0]

cur_len = cur_embed.shape[-2] - 1


# ---- autoregressive decoding ----
for step in range(max_new_tokens):

    attention_mask = torch.ones(
        (1, cur_len + 1),
        dtype=torch.long,
        device=cur_embed.device
    )

    with torch.no_grad():
        outputs = model.model(    # MobileLlamaModel
            inputs_embeds=cur_embed,
            attention_mask=attention_mask,
            past_key_values=torch_kv,
            use_cache=True,
            return_dict=True,     # Key: ['last_hidden_state', 'past_key_values']
        )

    pt_next_logit = model.lm_head(outputs.last_hidden_state)[:, -1, :]  # [1, 32000]
    torch_kv = outputs.past_key_values

    cur_token = torch.argmax(pt_next_logit, dim=-1, keepdim=True)  # tensor([[512]])
    cur_embed = model.model.embed_tokens(cur_token)                # tensor.Size([1, 1, 2048])

    generated_tokens.append(cur_token)    # 512, 278, ...
    
    if cur_token.item() == eos_token_id:
        break


generated_tokens = torch.cat(generated_tokens, dim=1)  # [1, T]
pt_tokens = generated_tokens.squeeze().cpu().numpy()  # [T]
# np.save("comparison/pytorch_generated_tokens.npy", pt_tokens)
print(pt_tokens)

'''
# ----- original inference code -----
with torch.inference_mode():
    out_ids = model.generate(             # shape: [1,93]
        input_ids,                        # Text prompt token: [1, 319, ... , -200, ... , 29901]
        images=image_tensor,              # Use original PyTorch model
        # image_features=image_features,
        **GEN_KWARGS_DEFAULT,
    )

gen_ids = out_ids[0][input_ids.shape[1] :]  # [512, 278, 1967, 29892, ...]
# text = tokenizer.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=True)[0]
# caption = re.sub(r"\s+", " ", text).strip()

# print(f"caption: {caption}")
'''
