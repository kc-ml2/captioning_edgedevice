# Captioning Edge Device

Lightweight image captioning system optimized for edge devices, supporting multiple vision-language models with a focus on latency and memory efficiency.

## Overview
This project implements an image captioning pipeline designed for deployment in resource-constrained environments.

- Supports multiple models: BLIP, InstructBLIP (FLAN-T5), MobileVLM (v2-1.7B)
- Supports FP16 and INT4 inference on GPU for BLIP and InstructBLIP
- Supports FP32 ONNX inference on CPU for MobileVLM


## PyTorch model weights

The following pretrained models are used in this project:

- BLIP: "Salesforce/blip-image-captioning-base"
- InstructBLIP: "Salesforce/instructblip-flan-t5-xl"
- MobileVLM: "mtgv/MobileVLM_V2-1.7B"

## ONNX model Weights

Due to their large size (~6GB), ONNX model weights are not included in this repository.

Instead, you can generate them locally using the provided export script:

```bash
python src/mobilevlm/runtime/onnx/export_onnx/export_*.py
```


## Quick Start with MobileVLM (PyTorch)

```bash
git clone https://github.com/kc-ml2/captioning_edgedevice
cd captioning_edgedevice

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
python src/mobilevlm/runtime/pytorch/pytorch_mobilevlm.py
```

### Result

Input
<img src="sample.jpg" width="400"/>

Output
"In the image, there is a living room with a fireplace, a television, a table, chairs, and a woman standing in the kitchen."

## Performance and Latency 

**Load Latency**  
The model server loads the model once at startup and performs an initial warm-up pass.

**VRAM Usage**  
GPU (NVIDIA TITAN V) memory usage is measured after the model is fully loaded and warmed up.

**Inference Latency**  
Image captioning latency per image measured over 500 randomly sampled images from COCO val2017.

### Model Comparison

| Model        | Precision         | Load Latency | VRAM Usage | Inference Latency|
|--------------|-------------------|--------------|------------|------------------|
| BLIP base    | FP16              | 5  s         | 0.9 GiB    | 0.6 s            |
| InstructBLIP | FP16              | 12 s         | 10.7 GiB   | 2.0 s            |
| InstructBLIP | Hybrid (INT4 LLM) | 15 s         | 6.9 GiB    | 2.7 s            |
| InstructBLIP | INT4              | 20 s         | 5.0 GiB    | 2.7 s            |


## Acknowledgement

This project is partially based on the MobileVLM repository:
https://github.com/Meituan-AutoML/MobileVLM

We adapted and modified the original implementation for:
- PyTorch runtime
- ONNX deployment
- Edge-device inference optimization


## License

Copyright (c) ML2.
All rights reserved.

This project includes code derived from the MobileVLM repository,
which is licensed under the Apache License 2.0.

Modifications have been made for PyTorch runtime and edge-device inference.
