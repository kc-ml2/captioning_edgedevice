# export_decoder_onnx.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch, onnx
import numpy as np
from PIL import Image

from model.mobilevlm import load_pretrained_model
from model.mutils import process_images, build_prompt, tokenizer_image_token


# ===============================
# 1. Decoder Wrapper
# ===============================
class DecoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.lm_head = model.lm_head

    def forward(self, input_ids, attention_mask, *past_key_values):

        # flatten → restore tuple
        pkv = []
        num_layers = len(past_key_values) // 2

        for i in range(num_layers):
            k = past_key_values[2*i]
            v = past_key_values[2*i + 1]
            pkv.append((k, v))


        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=pkv,
            use_cache=True,
            return_dict=False   # for ONNX stability
        )

        hidden_states = outputs[0]
        present_kv = outputs[1]

        logits = self.lm_head(hidden_states)
        logits = logits[..., :32000]

        # flatten KV
        flat_kv = []
        for k, v in present_kv:
            flat_kv.extend([k, v])

        return (logits, *flat_kv)


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
    model.eval()

    # ===============================
    # 3. Run prefill to obtain KV cache
    # ===============================
    img_path = "../../000000000139.jpg"
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
        )

    model = model.to(torch.float32)
    inputs_embeds = inputs_embeds.to(torch.float32)
    attention_mask_ = attention_mask_.to(torch.long)

    with torch.no_grad():
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_,
            use_cache=True,
            return_dict=True,
        )

        pkv = outputs.past_key_values   # list[(k,v)]

    # ===============================
    # 4. Dummy inputs (for decoder)
    # ===============================
    seq_len = pkv[0][0].shape[-2]   # 196

    dummy_input_ids = torch.ones((1, 1), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, seq_len + 1), dtype=torch.long)

    dummy_pkv = []
    for k, v in pkv:
        dummy_pkv.append(k.to(torch.float32))
        dummy_pkv.append(v.to(torch.float32))

    # ===============================
    # 5. Export
    # ===============================
    wrapper = DecoderWrapper(model).eval()

    input_names = ["input_ids", "attention_mask"]
    for i in range(len(dummy_pkv)):
        input_names.append(f"past_key_values_{i}")

    output_names = ["logits"]
    for i in range(len(dummy_pkv)):
        output_names.append(f"present_{i}")

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits": {0: "batch", 1: "seq"},
    }

    # KV dynamic update
    for i in range(len(dummy_pkv)):
        dynamic_axes[f"past_key_values_{i}"] = {2: "past_seq"}
        dynamic_axes[f"present_{i}"] = {2: "past_seq_out"}

    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask, *dummy_pkv),
        "decoders.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17
    )

    # ===============================
    # 6. Save external weights
    # ===============================
    model_onnx = onnx.load("decoders.onnx")

    onnx.save_model(
        model_onnx,
        "decoder.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="decoder.weights.bin",
        size_threshold=1024,
    )


if __name__ == "__main__":
    main()