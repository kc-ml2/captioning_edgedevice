# INT4BLIP_serve.py
import os, io, time, logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import BitsAndBytesConfig
from src.blip.utils.utils import cuda_mem_mb, ms

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
    device = torch.device("cuda:0")

    t0 = time.perf_counter()

    processor = InstructBlipProcessor.from_pretrained(
        MODEL_ID,
        # True: use a Rust-based tokenizer; False: use a Python-based tokenizer
        use_fast=True,
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
        inputs = processor(
            images=Image.new("RGB", (224, 224), (0, 0, 0)),
            text="Answer:",
            return_tensors="pt"
        ).to(device)
        _ = model.generate(**inputs, max_new_tokens=2)

        # Ensure all CUDA operations complete before proceeding
        torch.cuda.synchronize()
        
    # Mark as loaded only after successful load + warmup
    _loaded = True

    ds = time.perf_counter() - t0
    logger.info(f"[BOOT] Loaded: {MODEL_ID} on {device} in {ds:.1f} s ({ms(ds):,.0f} ms)")

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
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "device": str(device) if _loaded else "uninitialized",
        "pid": os.getpid(),
        "cuda_memory": cuda_mem_mb(),
        "loaded": _loaded
    }


# Handle image upload and run caption generation inference
@app.post("/inference")
async def inference(
    file: UploadFile = File(...),
):
    t0 = time.perf_counter()

    # 1) bytes -> PIL RGB
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    t1 = time.perf_counter()
    # 2) CPU preprocessing (image processing and text tokenization)
    inputs_cpu = processor(
        images=image,
        text=DEFAULT_PROMPT,
        return_tensors="pt"
    )
    t2 = time.perf_counter()

    # Move preprocessed tensors to GPU (H2D transfer)
    inputs = {
        k: v.to(device)
        for k, v in inputs_cpu.items()
        if isinstance(v, torch.Tensor)
    }
    torch.cuda.synchronize()  # Waits for all queued GPU operations to finish

    # 3) forward
    t3 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, **GEN_KWARGS)
    torch.cuda.synchronize()
    t4 = time.perf_counter()

    # 4) postprocess
    caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    t5 = time.perf_counter()

    timings_ms = {
        "preprocess_ms": ms(t2 - t1),
        "forward_ms": ms(t4 - t3),
        "post_ms": ms(t5 - t4),
        "total_ms": ms(t5 - t0),
    }

    return {"caption": caption, "timings_ms": timings_ms}

# Called by the client once after all inference requests are completed.
@app.post("/done")
def done():
    mem = cuda_mem_mb()
    logger.info(
        f"[DONE][VRAM] allocated={mem['allocated_MB']} MB, "
        f"reserved={mem['reserved_MB']} MB, "
        f"nvidia-smi={mem['nvidia_smi_MB']} MB"
    )
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
