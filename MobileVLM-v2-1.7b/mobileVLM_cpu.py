# mobileVLM_cpu.py

import re
import torch
from PIL import Image

from mobilevlm.conversation import conv_templates
from mobilevlm.constants import DEFAULT_IMAGE_TOKEN
from mobilevlm.model.mobilevlm import load_pretrained_model
import mobilevlm.utils as mutils


# HF model id or local path
MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

# ---- Default values ---- #
GEN_KWARGS_DEFAULT = dict(
    num_beams=1,
    max_new_tokens=80,
    min_new_tokens=40,
)
question = "What objects are visible in the image."

def _build_prompt(question: str) -> str:
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

device = "cpu"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL_PATH,
    load_4bit=False,
    device=str(device)
)
model.eval()

image = Image.open("000000000139.jpg").convert("RGB")
prompt = _build_prompt(question)

# preprocess image
image_tensor = mutils.process_images([image], image_processor, model.config)
image_tensor = image_tensor.to("cpu", dtype=torch.float32)

# tokenize
input_ids = mutils.tokenizer_image_token(
    prompt,
    tokenizer,
    return_tensors="pt",
).unsqueeze(0).to(device)

with torch.inference_mode():
    out_ids = model.generate(
        input_ids,
        images=image_tensor,
        **GEN_KWARGS_DEFAULT,
    )

gen_ids = out_ids[0][input_ids.shape[1] :]
text = tokenizer.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=True)[0]
caption = re.sub(r"\s+", " ", text).strip()

print(caption)
# In the image, there is a living room with a fireplace, a television, a table, 
# chairs, and a woman standing in the kitchen. The room is brightly lit, 
# and the woman is standing in the kitchen.