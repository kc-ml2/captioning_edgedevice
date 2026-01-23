# config.py
from enum import Enum

# ---- server config ----
HOST = "127.0.0.1"
PORT = 8600

# ---- model selection ----
class ModelFamily(str, Enum):
    BLIP = "blip"
    INSTRUCTBLIP = "instructblip"

# Change only this line to switch the active model family
MODEL_FAMILY = ModelFamily.BLIP

MODEL_ID_MAP = {
    ModelFamily.BLIP: "Salesforce/blip-image-captioning-base",
    ModelFamily.INSTRUCTBLIP: "Salesforce/instructblip-flan-t5-xl",
}
MODEL_ID = MODEL_ID_MAP[MODEL_FAMILY]

# ---- inference precision ----
DTYPE = "float16"

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
