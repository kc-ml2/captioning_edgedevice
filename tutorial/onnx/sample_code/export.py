# export.py

import torch
import torch.nn as nn
import onnx


# Define a simple neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        # Flatten input tensor
        x = x.view(x.size(0), -1)
        return self.fc(x)


def main():
    # Create model instance
    model = SimpleNet()

    # Set model to evaluation mode
    model.eval()

    # Create dummy input tensor (batch_size=1, 1x28x28 image)
    dummy_input = torch.randn(1, 1, 28, 28)

    # Export model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17
    )
    print("ONNX model exported successfully: model.onnx created.")

    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")


if __name__ == "__main__":
    main()