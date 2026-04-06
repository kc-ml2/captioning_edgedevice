import numpy as np


# ----- Preprocessor comparison -----
onnx_preprocessor_out = np.load("onnx_preprocessor_out.npy")
pytorch_preprocessor_out = np.load("pytorch_preprocessor_out.npy")

preprocessor_diff = np.abs(onnx_preprocessor_out - pytorch_preprocessor_out)

print("max preprocessor_diff :", preprocessor_diff.max())
print("mean preprocessor_diff:", preprocessor_diff.mean())


# ----- Vision encoder comparison -----
onnx_vision_out = np.load("onnx_vision_out.npy")
pytorch_vision_out = np.load("pytorch_vision_out.npy")

vision_diff = np.abs(onnx_vision_out - pytorch_vision_out)

print("max vision_diff :", vision_diff.max())
print("mean vision_diff:", vision_diff.mean())


# ----- Projector comparison -----
onnx_projector_out = np.load("onnx_projector_out.npy")
pytorch_projector_out = np.load("pytorch_projector_out.npy")

projector_diff = np.abs(onnx_projector_out - pytorch_projector_out)

print("max projector_diff :", projector_diff.max())
print("mean projector_diff:", projector_diff.mean())


# ----- LLM comparison -----
# ----- Tokenizer and embedding layer comparison -----
onnx_prompt_emb = np.load("onnx_prompt_embedding.npy")
pt_prompt_emb = np.load("pytorch_prompt_embedding.npy")

prompt_emb_diff = np.abs(onnx_prompt_emb - pt_prompt_emb)

print("Max prompt_emb_diff:", prompt_emb_diff.max())
print("Mean prompt_emb_diff:", prompt_emb_diff.mean())


# ----- Multimodal input comparison -----
pytorch_multimodal_input = np.load("pytorch_multimodal_input.npy")
onnx_multimodal_input = np.load("onnx_multimodal_input.npy")

multimodal_input_diff = np.abs(pytorch_multimodal_input - onnx_multimodal_input)
print("max multimodal_input_diff:", multimodal_input_diff.max())
print("mean multimodal_input_diff:", multimodal_input_diff.mean())


# ----- Prefill comparison -----
pytorch_next_logit = np.load("pytorch_next_logit.npy")
onnx_next_logit = np.load("onnx_next_logit.npy")

logit_diff = np.abs(pytorch_next_logit - onnx_next_logit)

# Outliers exist among the 32,008 LM head output logits
print("logit max diff:", np.max(logit_diff))
print("logit mean diff:", np.mean(logit_diff))

kv_diffs = []

pytorch_kv = np.load("pytorch_kv.npz")
onnx_kv  = np.load("onnx_kv.npz")

for i in range(len(pytorch_kv.files)):
    t = pytorch_kv[f'arr_{i}']
    o = onnx_kv[f'arr_{i}']

    diff = np.abs(t - o)
    kv_diffs.append(diff.max())

print("KV max diff:", np.max(kv_diffs))
print("KV mean diff:", np.mean(kv_diffs))


# ----- decoder comparison -----
onnx_generated_tokens = np.load("onnx_generated_tokens.npy")
pytorch_generated_tokens = np.load("pytorch_generated_tokens.npy")

same = np.array_equal(onnx_generated_tokens, pytorch_generated_tokens)
print(f"onnx_generated_tokens: {onnx_generated_tokens}")
print(f"pytorch_generated_tokens: {pytorch_generated_tokens}")
print("Number of generated_tokens:", len(pytorch_generated_tokens))
print("generated_tokens match?:", same)