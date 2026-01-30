Image Captioning Client–Server Inference System
==============================================

This repository provides a simple client–server architecture for image captioning
with a persistent (warm) model server.

The server loads the model once and keeps it resident in GPU memory, while the client
sends images via HTTP and receives generated captions. This design avoids repeated model
loading and is suitable for batch inference workloads.

------------------------------------------------------------
Repository Structure
------------------------------------------------------------

- BLIP_serve.py: BLIP model server (loads and serves the model)
- HybridBLIP_serve.py: HybridBLIP (INT4 LLM) model server
- client.py: Batch inference client
- config.py: Centralized configuration
- utils.py: Utility functions
- output: 

------------------------------------------------------------
HTTP API
------------------------------------------------------------

GET /health
- Health and readiness check
- Confirms that the model is loaded and the server is running

POST /inference
- Runs caption generation for a single image
- Request: multipart/form-data (file=image)
- Response: JSON with a generated caption

POST /done
- Optional notification that all inference requests are completed

------------------------------------------------------------
Performance and Latency 
------------------------------------------------------------

Boot Latency

The model server loads the model once at startup and performs an initial warm-up pass.

| Model | Precision | Load Latency |
|------|-----------|--------------|
| BLIP | FP16 | 5 s |
| InstructBLIP | FP16 | 12 s |
| InstructBLIP | Hybrid (LLM INT4) | 15 s |
| InstructBLIP | INT4 | 20 s |

------------------------------------------------------------

Captioning Outputs and Timing Records

Image Captioning Latency (100 Images)

All values are reported in milliseconds (ms) and represent per-image average latency.

BLIP
- Preprocess: 7.32 ms
- Forward: 642.52 ms
- Postprocess: 0.73 ms

InstructBLIP
- Preprocess: 5.90 ms
- Forward: 1911.82 ms
- Postprocess: 5.05 ms

------------------------------------------------------------

GPU Memory Usage (Warm State)

GPU memory usage is measured after the model is fully loaded and warmed up.

GPU: NVIDIA TITAN V (12 GiB VRAM)

- BLIP: 920 MiB
- InstructBLIP: 10,682 MiB

------------------------------------------------------------
Notes
------------------------------------------------------------

- Model, prompt, and decoding settings are fixed on the server side
- The client only sends image files
- The server is intended to be reused across multiple runs

------------------------------------------------------------
License
------------------------------------------------------------

Copyright (c) ML2.
All rights reserved.
