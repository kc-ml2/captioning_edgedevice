# Captioning Edge Device

Lightweight image captioning system optimized for edge devices using BLIP / InstructBLIP and MobileVLM.  
Supports multiple inference backends including PyTorch and ONNX with quantization (INT4/INT8).


## 📂 Project Structure
```bash
captioning_edgedevice/ 
├── MobileVLM-v2-1.7b/     # MobileVLM (cloned from official repo)
├── blip-serving/          # BLIP / InstructBLIP server implementations 
├── mobilevlm-runtime/     # PyTorch / ONNX runtime implementations 
├── onnx_tutorial/         # ONNX conversion and usage examples 
├── README.md 
└── requirements.txt

git clone https://github.com/Meituan-AutoML/MobileVLM
```


## Models

- BLIP
- InstructBLIP (FLAN-T5)
- MobileVLM (v2-1.7B)


## Performance and Latency 
------------------------------------------------------------

**Load Latency**  
The model server loads the model once at startup and performs an initial warm-up pass.

**VRAM Usage**  
GPU (NVIDIA TITAN V) memory usage is measured after the model is fully loaded and warmed up.

**Inference Latency**  
Image captioning latency per image measured over 500 images.

### Model Comparison

| Model        | Precision         | Load Latency | VRAM Usage | Inference Latency|
|--------------|-------------------|--------------|------------|------------------|
| BLIP base    | FP16              | 5 s          | 0.9 GiB    | 0.6 s            |
| InstructBLIP | FP16              | 12 s         | 10.7 GiB   | 2.0 s            |
| InstructBLIP | Hybrid (INT4 LLM) | 15 s         | 6.9 GiB    | 2.7 s            |
| InstructBLIP | INT4              | 20 s         | 5.0 GiB    | 2.7 s            |
| MobileVLM-v2 | INT4              | 11 s         | 2.1 GiB    | 2.6 s            |


## License
------------------------------------------------------------

Copyright (c) ML2.
All rights reserved.
