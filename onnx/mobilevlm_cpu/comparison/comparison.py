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


# ----- Multi-modal input comparison -----
