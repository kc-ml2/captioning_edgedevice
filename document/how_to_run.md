# On-Device Image Captioning

## Running MobileVLM

### Common Setup

```bash
git clone https://github.com/kc-ml2/captioning_edgedevice
cd captioning_edgedevice

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Note: Python versions lower than 3.12 are recommended.

### Quick Start with PyTorch

```bash
python src/mobilevlm/pytorch/pytorch_mobilevlm.py
```

### ONNX Deployment

Change version for dependencies:
```bash
pip install torch==2.0.1 torchvision==0.15.2 transformers==4.33.1 tokenizers==0.13.3 
```

Export ONNX weights:
```bash
python src/mobilevlm/export/export_onnx/export_vision_onnx.py
python src/mobilevlm/export/export_onnx/export_projector_onnx.py
python src/mobilevlm/export/export_onnx/export_tokenizer.py
python src/mobilevlm/export/export_onnx/export_embed_tokens_np.py
python src/mobilevlm/export/export_onnx/export_llm_onnx.py
```

We need only these weights related ONNX: 

`mm_projector.onnx` , `mobilellama.onnx` , `mobilellama.weights.bin` , `vision_tower.onnx` , `embed_tokens.npy` , `tokenizer.model`

Run inference:
```bash
python src/mobilevlm/onnx/onnx_mobilevlm.py
```

### Core ML / iOS Deployment

You need macOS and Xcode to run the iOS application.

First, export the Core ML model weights.

Re-check for library version dependencies (it is same in requirements.txt):

```bash
pip install torch==2.1.2 torchvision==0.16.2 transformers==4.46.3 tokenizers==0.20.3
```

```bash
python src/mobilevlm/export/export_coreml/export_vision_coreml.py
python src/mobilevlm/export/export_coreml/export_projector_coreml.py
python src/mobilevlm/export/export_coreml/export_embed_tokens_bin.py
python src/mobilevlm/export/export_coreml/export_tokenizer.py
python src/mobilevlm/export/export_coreml/export_llm_coreml.py
```

Second, Open `src/mobilevlm/iOS/mobilevlm/mobilevlm.xcodeproj` in Xcode using **"Open Existing Project..."** and add the following Swift Package dependencies:
- `swift-argument-parser` (1.7.1)
- `swift-sentencepiece` (0.0.6)

How to add the Swift Package:

`File` > `Add Package Dependencies` > Search `swift-sentencepiece` > Add Package

If `swift-sentencepiece` does not appear in the search results:
- Git the repository or download the ZIP file from "github.com/jkrukowski/swift-sentencepiece"
- `File` > `Add Package Dependencies` > `Add Local...` > click `swift-sentencepiece` folder

Third, add the following resources to:

`captioning_edgedevice/src/mobilevlm/iOS/mobilevlm/mobilevlm/Resources`

- Exported Core ML package: `VisionEncoder_32.mlpackage` , `Projector_32.mlpackage` , `embed_tokens.bin` , `tokenizer.model` , `mobilellama_32.mlpackage`

Finally, run this project.
