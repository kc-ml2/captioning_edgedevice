# config.py
from enum import Enum
import torch

# ---- server config ----
HOST = "127.0.0.1"
PORT = 8600

# ---- model selection ----
class ModelFamily(str, Enum):
    BLIP = "blip"
    INSTRUCTBLIP = "instructblip"

# Change only this line to switch the active model family
MODEL_FAMILY = ModelFamily.INSTRUCTBLIP

MODEL_ID_MAP = {
    ModelFamily.BLIP: "Salesforce/blip-image-captioning-base",
    ModelFamily.INSTRUCTBLIP: "Salesforce/instructblip-flan-t5-xl",
}
MODEL_ID = MODEL_ID_MAP[MODEL_FAMILY]

# ---- inference precision ----
DTYPE = "int4"  # one of: float32 | float16 | int8 | int4

# ---- inference mode / dtype mapping ----
class InferMode(str, Enum):
    FP = "fp"       # float32 / float16
    INT8 = "int8"   # bitsandbytes 8-bit
    INT4 = "int4"   # bitsandbytes 4-bit (NF4)

DTYPE_CONFIG = {
    "float32": {"mode": InferMode.FP,   "torch_dtype": torch.float32},
    "float16": {"mode": InferMode.FP,   "torch_dtype": torch.float16},
    "int8":    {"mode": InferMode.INT8, "torch_dtype": None},
    "int4":    {"mode": InferMode.INT4, "torch_dtype": None},
}

INFER_MODE = DTYPE_CONFIG[DTYPE]["mode"]
TORCH_DTYPE = DTYPE_CONFIG[DTYPE]["torch_dtype"]


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
