# mobileVLM_serve.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import io, time, logging, re
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image

from mobilevlm.conversation import conv_templates
from mobilevlm.constants import DEFAULT_IMAGE_TOKEN
from mobilevlm.model.mobilevlm import load_pretrained_model
import mobilevlm.utils as mutils


# =========================
# ---- configuration ----
# =========================
HOST = "127.0.0.1"
PORT = 8600

# HF model id or local path
MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

# prompt / conversation template
CONV_NAME = "v1"

GEN_KWARGS_DEFAULT = dict(
    num_beams=3,
    max_new_tokens=80,
    min_new_tokens=40,
)

def _build_prompt(question: str) -> str:
    conv = conv_templates[CONV_NAME].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

# logging
logger = logging.getLogger("uvicorn.error")

# runtime state
_loaded = False
tokenizer = None
model = None
image_processor = None
context_len = None
device = None

question = "What objects are visible in the image."


@torch.inference_mode()
def _infer_one(
    image: Image.Image,
    question: str,
    gen_kwargs: Dict[str, Any],
) -> str:
    # 0) prompt
    prompt = _build_prompt(question)

    # 1) preprocess image(s)
    image_tensor = mutils.process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(
        model.device,
        dtype=torch.float16,
    )

    # 2) tokenize (with image token handling)
    input_ids = mutils.tokenizer_image_token(
        prompt,
        tokenizer,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    # 3) forward(generate)
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        **gen_kwargs,
    )

    # 4) postprocess
    gen_ids = output_ids[0][input_ids.shape[1] :]
    out = tokenizer.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=True)[0]
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _warmup():
    # dummy generate로 CUDA kernel/allocator warm
    _ = _infer_one(
        Image.new("RGB", (224, 224), (0, 0, 0)),
        question="Answer is",
               gen_kwargs=dict(num_beams=1, max_new_tokens=8, min_new_tokens=1)
    )
    torch.cuda.synchronize()


def _load_once():
    global tokenizer, model, image_processor, context_len, device, _loaded

    if _loaded:
        logger.info("[BOOT] _load_once() called again; skipping.")
        return

    # GPU-only execution
    assert torch.cuda.is_available(), "CUDA is required but not available."
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()

    # MobileVLM loader (INT4)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL_PATH,
        load_4bit=True,
        device=str(device)
    )
    model.eval()

    # Warm-up
    _warmup()

    _loaded = True
    ds = time.perf_counter() - t0
    logger.info(
        f"[BOOT] Loaded: {MODEL_PATH} on {device} in {ds:.1f} s ({mutils.ms(ds):,.0f} ms), "
    )


# =========================
# ---- app lifecycle ----
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_once()
    yield


app = FastAPI(title="INT4 MobileVLM Server", lifespan=lifespan)


# =========================
# ---- endpoints ----
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "device": str(device) if _loaded else "uninitialized",
        "pid": os.getpid(),
        "cuda_memory": mutils.cuda_mem_mb(),
        "loaded": _loaded,
        "conv": CONV_NAME,
    }


@app.post("/inference")
async def inference(
    file: UploadFile = File(...),
    question_in: str = Form(question),
    num_beams: int = Form(GEN_KWARGS_DEFAULT["num_beams"]),
    max_new_tokens: int = Form(GEN_KWARGS_DEFAULT["max_new_tokens"]),
    min_new_tokens: int = Form(GEN_KWARGS_DEFAULT["min_new_tokens"]),
):
    if not _loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    t0 = time.perf_counter()

    # 1) bytes -> PIL RGB
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    t_pre0 = time.perf_counter()

    prompt = _build_prompt(question_in)

    # preprocess image
    image_tensor = mutils.process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # tokenize
    input_ids = mutils.tokenizer_image_token(
        prompt,
        tokenizer,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    torch.cuda.synchronize()
    t_pre1 = time.perf_counter()

    # 3) forward
    gen_kwargs = dict(
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
    )

    t_fwd0 = time.perf_counter()
    with torch.inference_mode():
        out_ids = model.generate(
            input_ids,
            images=image_tensor,
            **gen_kwargs,
        )
    torch.cuda.synchronize()
    t_fwd1 = time.perf_counter()

    # 4) postprocess
    t_post0 = time.perf_counter()
    gen_ids = out_ids[0][input_ids.shape[1] :]
    text = tokenizer.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=True)[0]
    caption = re.sub(r"\s+", " ", text).strip()
    t_post1 = time.perf_counter()

    t2 = time.perf_counter()

    timings_ms = {
        "preprocess_ms": mutils.ms(t_pre1 - t_pre0),
        "forward_ms": mutils.ms(t_fwd1 - t_fwd0),
        "post_ms": mutils.ms(t_post1 - t_post0),
        "total_ms": mutils.ms(t2 - t0),
    }

    return {
        "caption": caption,
        "timings_ms": timings_ms,
    }


@app.post("/done")
def done():
    mem = mutils.cuda_mem_mb()
    logger.info(
        f"[DONE][VRAM] allocated={mem['allocated_MB']} MB, "
        f"reserved={mem['reserved_MB']} MB, "
        f"nvidia-smi={mem['nvidia_smi_MB']} MB"
    )
    return {"status": "ok"}


# =========================
# ---- entrypoint ----
# =========================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "mobileVLM_serve:app",
        host=HOST,
        port=PORT,
        workers=1,
        log_level="info",
        access_log=False,
    )
