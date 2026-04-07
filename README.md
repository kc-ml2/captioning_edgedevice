# Captioning Edge Device

Lightweight image captioning system optimized for edge devices, supporting multiple vision-lanuage models with a focus on latency and memory efficiency.

## Overview
This project implements an image captioning pipeline designed for deployment in resource constrained environments.

- Supports multiple models: **BLIP**, **InstructBLIP**, **MobileVLM**
- Supports FP16 and INT4 inference on GPU for BLIP and InstructBLIP
- Supports FP32 ONNX inference on CPU for MobileVLM

## 📂 Project Structure
```bash
captioning_edgedevice/
├── src/
│   ├── blip/                 # BLIP / InstructBLIP implementations
│   └── mobilevlm/            # MobileVLM runtime
│       ├── runtime/
│       │   ├── pytorch/      # PyTorch implementation
│       │   └── onnx/         # ONNX implementation
│       └── experiments/
├── tutorial/                 # ONNX conversion & usage examples
├── README.md
├── requirements.txt
└── sample.jpg
```


## Models

- BLIP
- InstructBLIP (FLAN-T5)
- MobileVLM (v2-1.7B)


## Quick Start with MobileVLM PyTorch

```bash
git clone https://github.com/kc-ml2/captioning_edgedevice
cd captioning_edgedevice
pip install -r requirements.txt
python run.py
```


## Performance and Latency 

**Load Latency**  
The model server loads the model once at startup and performs an initial warm-up pass.

**VRAM Usage**  
GPU (NVIDIA TITAN V) memory usage is measured after the model is fully loaded and warmed up.

**Inference Latency**  
Image captioning latency per image measured over 500 images COCO val 2017.

### Model Comparison

| Model        | Precision         | Load Latency | VRAM Usage | Inference Latency|
|--------------|-------------------|--------------|------------|------------------|
| BLIP base    | FP16              | 5  s         | 0.9 GiB    | 0.6 s            |
| InstructBLIP | FP16              | 12 s         | 10.7 GiB   | 2.0 s            |
| InstructBLIP | Hybrid (INT4 LLM) | 15 s         | 6.9 GiB    | 2.7 s            |
| InstructBLIP | INT4              | 20 s         | 5.0 GiB    | 2.7 s            |
| MobileVLM-v2 | INT4              | 11 s         | 2.1 GiB    | 2.6 s            |


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

This project includes code adapted from MobileVLM:
https://github.com/Meituan-AutoML/MobileVLM

MobileVLM is licensed under the Apache License, Version 2.0.

Modifications have been made for PyTorch runtime and edge-device inference.
