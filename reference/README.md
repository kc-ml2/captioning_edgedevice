# On-Device Image Captioning (Reference Implementation)

> [!NOTE]
> This is the previous PyTorch, ONNX, Core ML, and iOS implementation. It is retained as a reference while the project is rebuilt with MLX Swift. The new project entry point is the [repository root](../README.md).

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ONNX](https://img.shields.io/badge/ONNX-Supported-green)
![CoreML](https://img.shields.io/badge/CoreML-iOS-orange)


<p align="center">
  <img src="assets/mobilevlm.gif" width="200"/>
</p>

Lightweight on-device image captioning system based on MobileVLM, supporting PyTorch, ONNX, and iOS deployment.

## Overview

This project implements an image captioning pipeline designed for deployment in resource-constrained environments.

- Supports multiple models: BLIP, InstructBLIP, and MobileVLM v2
- Supports PyTorch, ONNX, and Core ML deployment for MobileVLM v2

## Features

- Lightweight on-device image captioning
- MobileVLM v2 support
- PyTorch / ONNX / Core ML inference
- iOS deployment support
- Edge-device optimization

## Model Weights

### PyTorch

The following pretrained models from Hugging Face are used in this project:

- BLIP: "Salesforce/blip-image-captioning-base"
- InstructBLIP: "Salesforce/instructblip-flan-t5-xl"
- MobileVLM: "mtgv/MobileVLM_V2-1.7B"

### ONNX

Due to their large size (~6GB), ONNX model weights are not included in this repository.

Instead, you can generate them locally using the provided export script:

```bash
python src/mobilevlm/export/export_onnx/export_*.py
```

### Core ML

Due to their large size (~6GB), Core ML model weights are not included in this repository.

Instead, you can generate them locally using the provided export script:

```bash
python src/mobilevlm/export/export_coreml/export_*.py
```

## Running MobileVLM

See [document/how_to_run.md](document/how_to_run.md).



## Result

### Input 

`sample.jpg`

<img src="sample.jpg" width="400"/> 

**Question:**  
"What objects are visible in the scene?"

### Generated Caption

"In the image, there is a living room with a fireplace, a television, a table, chairs, and a woman standing in the kitchen."


## Performance and Latency 

**Load Latency**  
The model server loads the model once at startup and performs an initial warm-up pass.

**VRAM Usage**  
GPU (NVIDIA TITAN V) memory usage is measured after the model is fully loaded and warmed up.

**Inference Latency**  
Image captioning latency per image is measured over 500 randomly sampled images from COCO val2017.

### Model Comparison

| Model        | Precision         | Load Latency | VRAM Usage | Inference Latency |
|--------------|-------------------|--------------|------------|-------------------|
| BLIP base    | FP16              | 5  s         | 0.9 GiB    | 0.6 s             |
| InstructBLIP | FP16              | 12 s         | 10.7 GiB   | 2.0 s             |
| InstructBLIP | Hybrid (INT4 LLM) | 15 s         | 6.9 GiB    | 2.7 s             |
| InstructBLIP | INT4              | 20 s         | 5.0 GiB    | 2.7 s             |

## MobileVLM CPU Runtime Breakdown

Latency breakdown for a single image inference.

> The current Xcode/Core ML implementation is not fully optimized.

| Runtime         | Preprocessing | Vision Encoder | Projector  | LLM (40 tkn)  | Total    |
|-----------------|---------------|----------------|------------|---------------|----------|
| GPU             | -             | -              | -          | -             | 1.7 sec  |
| CPU (Python)    | 0.04 sec      | 0.52 sec       | 0.01 sec   | 6.33 sec      | 6.90 sec |
| CPU (ONNX)      | 0.03 sec      | 0.88 sec       | 0.02 sec   | 3.77 sec      | 4.70 sec |
| iOS (Core ML) | 0.12 sec      | 1.27 sec       | 0.02 sec   | 6.10 sec      | 7.51 sec |


## Acknowledgement

This project is partially based on the official MobileVLM repository:
https://github.com/Meituan-AutoML/MobileVLM

We adapted and modified the original implementation for:
- PyTorch runtime
- ONNX deployment
- Core ML deployment
- On-device inference optimization


## License

Copyright (c) ML2.
All rights reserved.

This project includes code derived from the MobileVLM repository,
which is licensed under the Apache License 2.0.

The official PyTorch-based MobileVLM was modified and exported to ONNX and Core ML for on-device inference.
