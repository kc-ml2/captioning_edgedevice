# model_serve.py
import os, io, time, logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---- configuration ----
from config import (
    HOST,
    PORT,
    ModelFamily,
    MODEL_FAMILY,
    MODEL_ID,
    DTYPE,
    DEFAULT_PROMPT,
    GEN_KWARGS,
)

# ---- logging ----
logger = logging.getLogger("uvicorn.error")

# ---- runtime state ----
_loaded = False  # Ensures _load_once() runs only once per process.

# BLIP runs without a prompt because it has no LLM, while InstructBLIP requires one.
USE_PROMPT = (MODEL_FAMILY == ModelFamily.INSTRUCTBLIP)

# Select torch.dtype for CUDA inference (FP16 or FP32)
TORCH_DTYPE = torch.float16 if DTYPE.lower() == "float16" else torch.float32

# ---- helpers ----
# Prepare model inputs from the image and optional prompt
def _prepare_inputs(img: Image.Image, text: str | None):
    inputs = processor(
        images=img,
        return_tensors="pt",
        **({"text": text} if (USE_PROMPT and text is not None) else {}),
    )
    return {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}


# ---- model initialization ----
# Load the model and processor once and keep them resident on the GPU
def _load_once():
    global processor, model, device, _loaded

    # Guard to ensure the model is loaded exactly once per process
    if _loaded:
        logger.info("[BOOT] _load_once() called again; skipping.")
        return

    # GPU-only execution (CUDA required)
    assert torch.cuda.is_available(), "CUDA is required but not available."
    device = torch.device("cuda")

    # Select the correct processor/model classes
    if MODEL_FAMILY == ModelFamily.INSTRUCTBLIP:
        ProcessorCls = InstructBlipProcessor
        ModelCls = InstructBlipForConditionalGeneration
    else:
        ProcessorCls = BlipProcessor
        ModelCls = BlipForConditionalGeneration

    t0 = time.perf_counter()
    
    # InstructBLIP requires explicitly setting legacy=False
    processor = ProcessorCls.from_pretrained(
        MODEL_ID,
        use_fast=False,
        **({"legacy": False} if ProcessorCls is InstructBlipProcessor else {})
    )
    model = ModelCls.from_pretrained(
        MODEL_ID,
        dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    # Run a dummy forward pass to warm up the model and initialize GPU kernels
    with torch.inference_mode():
        inputs = _prepare_inputs(
            img=Image.new("RGB", (224, 224), (0, 0, 0)), 
            text="Answer:" if USE_PROMPT else None
        )

        with torch.autocast(device_type="cuda", dtype=TORCH_DTYPE):
            _ = model.generate(**inputs, max_new_tokens=2)
        # Ensure all CUDA operations complete before proceeding
        torch.cuda.synchronize()
    
    # Mark as loaded only after successful load + warmup
    _loaded = True

    ds = time.perf_counter() - t0
    dt = ds * 1000
    logger.info(f"[BOOT] Loaded: {MODEL_ID} on {device} in {ds:.1f} s ({dt:,.0f} ms)")


# ---- application lifecycle ----
# Startup lifecycle handler: load the model once and keep it resident on GPU
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_once()
    yield

# ---- application factory ----
# Create the FastAPI application with a custom lifespan for model initialization
app = FastAPI(title="Model Server", lifespan=lifespan)

# ---- API endpoints ----
# Expose runtime and GPU status for health checks and monitoring
@app.get("/health")
def health():
    cuda_mem = {
        "allocated_MB": int(torch.cuda.memory_allocated() / (1024 * 1024)),
        "reserved_MB": int(torch.cuda.memory_reserved() / (1024 * 1024)),
    }
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "device": str(device) if _loaded else "uninitialized",
        "pid": os.getpid(),
        "cuda_memory": cuda_mem,
        "loaded": _loaded
    }


# Handle image upload and run caption generation inference
@app.post("/inference")
async def inference(
    file: UploadFile = File(...),
):
    # 1) bytes -> PIL RGB
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    t0 = time.perf_counter()

    # 2) preprocess
    inputs = _prepare_inputs(
        img=image, 
        text=DEFAULT_PROMPT if USE_PROMPT else None
    )
    t1 = time.perf_counter()

    # 3) forward
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=TORCH_DTYPE):
            out = model.generate(**inputs, **GEN_KWARGS)
    t2 = time.perf_counter()

    # 4) postprocess
    caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    t3 = time.perf_counter()

    timings_ms = {
        "preprocess_ms": (t1 - t0) * 1000,
        "forward_ms": (t2 - t1) * 1000,
        "post_ms": (t3 - t2) * 1000,
    }

    return {"caption": caption, "timings_ms": timings_ms}

# Called by the client once after all inference requests are completed.
@app.post("/done")
def done():
    logger.info("All inference requests have been completed.")
    return {"status": "ok"}


# ---- entrypoint ----
# Launch the FastAPI application using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "model_serve:app", 
        host=HOST, 
        port=PORT, 
        workers=1, 
        log_level="info",
        access_log=False,
    )
