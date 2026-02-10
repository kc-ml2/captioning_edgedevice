Image Captioning Client–Server Inference System
==============================================

This repository provides a simple client–server architecture for image captioning with a warm GPU-resident model server.

------------------------------------------------------------
Repository Structure
------------------------------------------------------------
- MobileVLM-v2-1.7b/: git "Meituan-AutoML/MobileVLM"
  - mobileVLM_serve.py: MobileVLM-v2-1.7b model server
  - client.py: Inference client (request images and configuration)

--

- BLIP_serve.py: BLIP model server (loads and serves the BLIP series models)
- HybridBLIP_serve.py: Hybrid InstructBLIP model server (INT4 LLM, FP16 vision/Q-Former)
- INT4BLIP_serve.py: Only INT4 weighted InstructBLIP model server
- client.py: Batch inference client
- config.py: Centralized configuration
- utils.py: Utility functions
- output/: Generated captions and time_debug
  - time_csv.py: time debug and calculate the mean speed
------------------------------------------------------------
HTTP API
------------------------------------------------------------

GET /health
- Health and readiness check endpoint
- Confirms that the model is loaded and the server is running

POST /inference
- Runs caption generation for a single image request
- Request: multipart/form-data (file=image)
- Response: JSON with a generated caption

POST /done
- Optional notification that all inference requests are completed
- Print VRAM memory usage

------------------------------------------------------------
Performance and Latency 
------------------------------------------------------------

**Load Latency**  
The model server loads the model once at startup and performs an initial warm-up pass.

**VRAM Usage**  
GPU (NVIDIA TITAN V) memory usage is measured after the model is fully loaded and warmed up.

**Inference Latency**  
Image captioning latency per image measured over 500 images.

### 📊 Model Comparison

| Model        | Precision         | Load Latency | VRAM Usage | Inference Latency|
|--------------|-------------------|--------------|------------|------------------|
| BLIP base    | FP16              | 5 s          | 0.9 GiB    | 0.6 s            |
| InstructBLIP | FP16              | 12 s         | 10.7 GiB   | 2.0 s            |
| InstructBLIP | Hybrid (INT4 LLM) | 15 s         | 6.9 GiB    | 2.7 s            |
| InstructBLIP | INT4              | 20 s         | 5.0 GiB    | 2.7 s            |
| MobileVLM-v2 | INT4              | 11 s         | 2.1 GiB    | 2.6 s            |


### ⚙️ Generation Configuration

```python
max_new_tokens = 80
min_new_tokens = 40
num_beams = 3

DEFAULT_PROMPT = (
    "Question: Describe this image in two sentences. "
    "Sentence 1 must describe the overall scene and the main objects visible across different regions of the image. "
    "Sentence 2 must describe the spatial layout, mentioning the foreground, background, left, right, and center areas when applicable. "
    "Do not mention 'Question' or 'Answer' in your response. "
    "Answer:"
)
```
The prompt is not used for BLIP base since it does not include an LLM.

------------------------------------------------------------
Notes
------------------------------------------------------------

- Model, prompt, and decoding settings are fixed on the server side.
- The client only sends image files.
- The server is intended to be reused across multiple runs.

------------------------------------------------------------
License
------------------------------------------------------------

Copyright (c) ML2.
All rights reserved.
