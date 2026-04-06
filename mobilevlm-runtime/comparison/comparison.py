import numpy as np


# ----- Image preprocessor comparison -----
onnx_preprocessor_out = np.load("onnx_preprocessor_out.npy")
pytorch_preprocessor_out = np.load("pytorch_preprocessor_out.npy")

preprocessor_diff = np.abs(onnx_preprocessor_out - pytorch_preprocessor_out)

print("Shape preprocessor output :", preprocessor_diff.shape)
print("max preprocessor_diff :", preprocessor_diff.max())
print("mean preprocessor_diff:", preprocessor_diff.mean())
print("------------------------------------------------------")


# ----- Vision encoder comparison -----
onnx_vision_out = np.load("onnx_vision_out.npy")
pytorch_vision_out = np.load("pytorch_vision_out.npy")

vision_diff = np.abs(onnx_vision_out - pytorch_vision_out)

print("Shape vision encoder output :", vision_diff.shape)
print("max vision_diff :", vision_diff.max())
print("mean vision_diff:", vision_diff.mean())
print("------------------------------------------------------")


# ----- Projector comparison -----
onnx_projector_out = np.load("onnx_projector_out.npy")
pytorch_projector_out = np.load("pytorch_projector_out.npy")

projector_diff = np.abs(onnx_projector_out - pytorch_projector_out)

print("Shape projector output :", projector_diff.shape)
print("max projector_diff :", projector_diff.max())
print("mean projector_diff:", projector_diff.mean())
print("------------------------------------------------------")


# ----- LLM comparison -----
# ----- Tokenizer and embedding layer comparison -----
onnx_prompt_emb = np.load("onnx_prompt_embedding.npy")
pt_prompt_emb = np.load("pytorch_prompt_embedding.npy")

prompt_emb_diff = np.abs(onnx_prompt_emb - pt_prompt_emb)

print("Shape prompt_emb output :", prompt_emb_diff.shape)
print("Max prompt_emb_diff:", prompt_emb_diff.max())
print("Mean prompt_emb_diff:", prompt_emb_diff.mean())
print("------------------------------------------------------")


# ----- Multimodal input comparison -----
pytorch_multimodal_input = np.load("pytorch_multimodal_input.npy")
onnx_multimodal_input = np.load("onnx_multimodal_input.npy")

multimodal_input_diff = np.abs(pytorch_multimodal_input - onnx_multimodal_input)

print("Shape multimodal input :", multimodal_input_diff.shape)
print("max multimodal_input_diff:", multimodal_input_diff.max())
print("mean multimodal_input_diff:", multimodal_input_diff.mean())
print("------------------------------------------------------")


# ----- Prefill comparison -----
pytorch_next_logit = np.load("pytorch_next_logit.npy")
onnx_next_logit = np.load("onnx_next_logit.npy")

next_logit_diff = np.abs(pytorch_next_logit - onnx_next_logit)

print("Shape next_logit output :", next_logit_diff.shape)
print("next_logit max diff:", np.max(next_logit_diff))
print("next_logit mean diff:", np.mean(next_logit_diff))
print("------------------------------------------------------")

kv_diffs = []

pytorch_kv = np.load("pytorch_kv.npz")
onnx_kv  = np.load("onnx_kv.npz")

for i in range(len(pytorch_kv.files)):
    t = pytorch_kv[f'arr_{i}']
    o = onnx_kv[f'arr_{i}']

    diff = np.abs(t - o)
    kv_diffs.append(diff.max())

print("Number of KV tensors:", len(kv_diffs))
print("KV max diff:", np.max(kv_diffs))
print("KV mean diff:", np.mean(kv_diffs))
print("------------------------------------------------------")


# ----- decoder comparison -----
onnx_generated_tokens = np.load("onnx_generated_tokens.npy")
pytorch_generated_tokens = np.load("pytorch_generated_tokens.npy")

same = np.array_equal(onnx_generated_tokens, pytorch_generated_tokens)
print(f"onnx_generated_tokens: {onnx_generated_tokens}")
print(f"pytorch_generated_tokens: {pytorch_generated_tokens}")
print("Number of generated_tokens:", len(pytorch_generated_tokens))
print("generated_tokens match?:", same)