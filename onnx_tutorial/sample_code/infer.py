# infer.py

import numpy as np
import onnxruntime as ort


def main():
    model_path = "model.onnx"

    # Create ONNX Runtime session
    session = ort.InferenceSession(model_path)  # 'CPUExecutionProvider'

    # Model inputs/outputs structure
    inputs = session.get_inputs()  # ['batch_size', 1, 28, 28]
    outputs = session.get_outputs()  # ['batch_size', 10]

    # Prepare a dummy input (batch_size=3)
    input_name = inputs[0].name
    x = np.random.randn(3, 1, 28, 28).astype(np.float32)

    # Run inference
    y = session.run(None, {input_name: x})  # (3, 10)


if __name__ == "__main__":
    main()