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

model_serve.py   Model server (loads and serves the model)
client.py        Batch inference client
config.py        Centralized configuration

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
