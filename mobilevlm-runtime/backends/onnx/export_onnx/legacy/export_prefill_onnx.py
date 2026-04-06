# export_prefill_onnx.py

import torch, onnx
import numpy as np
from PIL import Image
import onnxruntime as ort

from mobilevlm_cpu.model.mobilevlm import load_pretrained_model
from mobilevlm_cpu.model.mutils import process_images, build_prompt, tokenizer_image_token


# ===============================
# 1. Prefill Wrapper
# ===============================
class PrefillWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.lm_head = model.lm_head

    def forward(self, inputs_embeds, attention_mask):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=False               # ONNX does not support dictionary outputs
        )

        hidden_states = outputs[0]
        past_key_values = outputs[1]
        logits = self.lm_head(hidden_states)
        logits = logits[..., :32000]
        
        flat_pkv = []
        for k, v in past_key_values:
            flat_pkv.extend([k, v])

        return (logits, *flat_pkv)


# ===============================
# 2. Main
# ===============================
def main():

    device = torch.device("cpu")

    MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=MODEL_PATH,
        device="cpu",
    )
    model.eval()  # MobileLlamaForCausalLM

    img_path = "../000000000139.jpg"
    image = Image.open(img_path).convert("RGB")

    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(device=device, dtype=torch.float32)

    question = "What objects are visible in the image in detail."
    prompt = build_prompt(question)

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        return_tensors="pt",
    ).unsqueeze(0).to(device)

    attention_mask = torch.ones_like(input_ids)

    input_ids, attention_mask_, _, inputs_embeds, _ = \
        model.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_mask,
            past_key_values=None,
            labels=None,
            images=image_tensor,
            image_features=None
        )

    model = model.to(torch.float32)
    inputs_embeds = inputs_embeds.to(torch.float32)
    attention_mask_ = attention_mask_.to(torch.long)

    with torch.no_grad():
        outputs = model.model(                  # MobileLlamaModel
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_,
            use_cache=True,
            return_dict=True,                   # key: ['last_hidden_state', 'past_key_values']
        )

        pkv = outputs.past_key_values           # k,v = [1, 16, 196, 128] x 24 layers
    
    # ===============================
    # 3. ONNX export
    # ===============================

    wrapper = PrefillWrapper(model).eval()      # model size: 6.24 GB

    num_layers = len(pkv)
    output_names = ["logits"]
    for i in range(num_layers):
        output_names.append(f"key_{i}")
        output_names.append(f"value_{i}")


    torch.onnx.export(
        wrapper,
        (inputs_embeds, attention_mask_),
        "prefill.onnx",
        input_names=["inputs_embeds", "attention_mask"],
        output_names=output_names,
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17
    )


    # 1. Load existing ONNX model
    model = onnx.load("prefill.onnx")

    # 2. Save all weights into a single external file
    onnx.save_model(
        model,
        "prefill_merged.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="prefill.weights.bin",
        size_threshold=1024,
    )

if __name__ == "__main__":
    main()