# INT4BLIP_serve.py
import os, io, time, logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import BitsAndBytesConfig

# ---- configuration ----
HOST = "127.0.0.1"
PORT = 8600

# ---- generation config ----
DEFAULT_PROMPT = (
    "Question: Describe this image in two sentences. "
    "Sentence 1 must describe the overall scene and the main objects visible across different regions of the image. "
    "Sentence 2 must describe the spatial layout, mentioning the foreground, background, left, right, and center areas when applicable. "
    "Do not mention 'Question' or 'Answer' in your response. "
    "Answer:"
)

GEN_KWARGS = dict(
    max_new_tokens=80,
    min_new_tokens=40,
    num_beams=3,
    repetition_penalty=1.15,
    no_repeat_ngram_size=3,
    length_penalty=1.1,
)

# ---- logging ----
logger = logging.getLogger("uvicorn.error")

# ---- runtime state ----
_loaded = False  # Ensures _load_once() runs only once per process.

# ---- helpers ----
# Prepare model inputs from the image and optional prompt
def _prepare_inputs(img: Image.Image, text: str | None):
    inputs = processor(images=img, text=text, return_tensors="pt")
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items() if torch.is_tensor(v)}
    return inputs

MODEL_ID = "Salesforce/instructblip-flan-t5-xl"

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

    t0 = time.perf_counter()

    processor = InstructBlipProcessor.from_pretrained(
        MODEL_ID,
        use_fast=False,
        legacy=False,
    )

    # INT4 quantized loading via bitsandbytes (no extra model download)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float16,  # compute in fp16
    )
    model = InstructBlipForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        device_map="cuda:0"
    )
    model.eval()

    # Run a dummy forward pass to warm up the model and initialize GPU kernels
    with torch.inference_mode():
        inputs = _prepare_inputs(
            img=Image.new("RGB", (224, 224), (0, 0, 0)), 
            text="Answer:"
        )
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
        text=DEFAULT_PROMPT
    )

    # 3) forward
    with torch.inference_mode():
        out = model.generate(**inputs, **GEN_KWARGS)

    # 4) postprocess
    caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    latency =  time.perf_counter() - t0

    return {"caption": caption, "latency": latency}
    # return {"caption": caption}

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
        "INT4BLIP_serve:app", 
        host=HOST, 
        port=PORT, 
        workers=1, 
        log_level="info",
        access_log=False,
    )
