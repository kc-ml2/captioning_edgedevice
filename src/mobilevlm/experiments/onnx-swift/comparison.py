import numpy as np



# before PreProcessing
# py = np.fromfile("python_resize.bin", dtype=np.uint8)
# sw = np.fromfile("swift_resize.bin", dtype=np.uint8)

# # PreProcessing
# py = np.fromfile("python_tensor.bin", dtype=np.float32)
# sw = np.fromfile("swift_tensor.bin", dtype=np.float32)

# print("py size:", py.shape)   # size: (640, 426)
# print("sw size:", sw.shape)

# print("Python 30 pixel:", py[0:50])
# print("Swift  30 pixel:", sw[0:50])



# # diff = np.abs(py.astype(np.int32) - sw.astype(np.int32))
# diff = np.abs(py-sw)
# idx = np.argmax(diff)

# print("All close:", np.allclose(py, sw))
# print("Max diff:", np.max(diff))
# print("Mean diff:", np.mean(diff))
# print("Std diff:", np.std(diff))


# print("Max diff index:", idx)
# print("Max diff Python value:", py[idx])
# print("Max diff Swift value:", sw[idx])

# print("py min/max:", py.min(), py.max())
# print("sw min/max:", sw.min(), sw.max())

# Vision output
py = np.fromfile("vision_out_onnx.bin", dtype=np.float32)
sw = np.fromfile("vision_out_coreml.bin", dtype=np.float32)

print("py elements:", py.size)
print("sw elements:", sw.size)

py = py.reshape(1, 576, 1024)
sw = sw.reshape(1, 576, 1024)

print("py size:", py.shape)
print("sw size:", sw.shape)


diff = np.abs(py - sw)

print("Max diff:", np.max(diff))
print("Mean diff:", np.mean(diff))
print("Std diff:", np.std(diff))

diff = np.abs(py - sw)

idx = np.argmax(diff)
print("Worst index:", idx)

i, j, k = np.unravel_index(idx, py.shape)
print("Position:", (i, j, k))

print("py:", py[i, j, k])
print("sw:", sw[i, j, k])

diff = np.abs(py - sw)

idx = np.argmax(diff)
print("Worst index:", idx)

i, j, k = np.unravel_index(idx, py.shape)
print("Position:", (i, j, k))

print("py:", py[i, j, k])
print("sw:", sw[i, j, k])

# 🔥 핵심: cosine similarity
cos_sim = np.dot(py.flatten(), sw.flatten()) / (
    np.linalg.norm(py.flatten()) * np.linalg.norm(sw.flatten())
)
print("Cosine similarity:", cos_sim)

# 🔥 relaxed allclose
print("All close (relaxed):", np.allclose(py, sw, rtol=1e-2, atol=1e-2))