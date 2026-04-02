# export_llm_onnx.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import onnx
from PIL import Image

from model.mobilevlm import load_pretrained_model
from model.mutils import process_images, build_prompt, tokenizer_image_token


# ===============================
# 1. Decoder Wrapper (단일 ONNX용)
# ===============================
class DecoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.lm_head = model.lm_head

    def forward(self, inputs_embeds, attention_mask, *past_key_values):

        # ---- flat → tuple ----
        pkv = []
        num_layers = len(past_key_values) // 2

        for i in range(num_layers):
            k = past_key_values[2 * i]
            v = past_key_values[2 * i + 1]
            pkv.append((k, v))

        # ---- forward ----
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=pkv,   # 항상 존재 (0-length 포함)
            use_cache=True,
            return_dict=False      # ONNX 안정성
        )

        hidden_states = outputs[0]
        present_kv = outputs[1]

        logits = self.lm_head(hidden_states)
        logits = logits[..., :32000]

        # ---- tuple → flat ----
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
    # 3. Prefill 실행 (shape 확보용)
    # ===============================
    img_path = "../../000000000139.jpg"
    image = Image.open(img_path).convert("RGB")

    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(device=device, dtype=torch.float32)

    question = "Describe the image in detail."
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

    inputs_embeds = inputs_embeds.to(torch.float32)
    attention_mask_ = attention_mask_.to(torch.long)

    # ===============================
    # 4. KV shape 추출
    # ===============================
    with torch.no_grad():
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_,
            use_cache=True,
            return_dict=True,
        )
        pkv = outputs.past_key_values

    num_layers = len(pkv)

    # ===============================
    # 5. Dummy Inputs (핵심)
    # ===============================

    # ✔ decode 기준 (1 token)
    dummy_inputs_embeds = torch.randn(1, 32, inputs_embeds.shape[-1], dtype=torch.float32)

    # ✔ attention_mask는 dynamic
    dummy_attention_mask = torch.ones((1, 32), dtype=torch.long)

    # ✔ 핵심: past_seq = 0
    dummy_pkv = []
    for k, v in pkv:
        shape = list(k.shape)
        shape[-2] = 0   # ← 중요: past_seq = 0

        dummy_k = torch.zeros(shape, dtype=torch.float32)
        dummy_v = torch.zeros(shape, dtype=torch.float32)

        dummy_pkv.append(dummy_k)
        dummy_pkv.append(dummy_v)

    # ===============================
    # 6. Wrapper
    # ===============================
    wrapper = DecoderWrapper(model).eval()

    # ===============================
    # 7. Names
    # ===============================
    input_names = ["inputs_embeds", "attention_mask"]
    for i in range(len(dummy_pkv)):
        input_names.append(f"past_key_values_{i}")

    output_names = ["logits"]
    for i in range(len(dummy_pkv)):
        output_names.append(f"present_{i}")

    # ===============================
    # 8. Dynamic Axes
    # ===============================
    dynamic_axes = {
        "inputs_embeds": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq_total"},
        "logits": {0: "batch", 1: "seq"},
    }

    for i in range(len(dummy_pkv)):
        dynamic_axes[f"past_key_values_{i}"] = {2: "past_seq"}
        dynamic_axes[f"present_{i}"] = {2: "past_seq_out"}

    # ===============================
    # 9. Export
    # ===============================
    torch.onnx.export(
        wrapper,
        (dummy_inputs_embeds, dummy_attention_mask, *dummy_pkv),
        "mobilellamas.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )

    # ===============================
    # 10. External weights
    # ===============================
    model_onnx = onnx.load("mobilellamas.onnx")

    onnx.save_model(
        model_onnx,
        "mobilellama.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="mobilellama.weights.bin",
        size_threshold=1024,
    )

    print("✅ Single ONNX export complete (prefill + decode unified)")


if __name__ == "__main__":
    main()