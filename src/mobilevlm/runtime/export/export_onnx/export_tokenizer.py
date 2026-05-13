from transformers import LlamaTokenizer
import shutil
import os

model_path = "mtgv/MobileVLM_V2-1.7B"

tokenizer = LlamaTokenizer.from_pretrained(
    model_path,
    use_fast=False
)

print("vocab_file:", tokenizer.vocab_file)
print("cwd:", os.getcwd())

save_path = os.path.abspath("tokenizer.model")

shutil.copy(
    tokenizer.vocab_file,
    save_path
)

print("saved:", save_path)
