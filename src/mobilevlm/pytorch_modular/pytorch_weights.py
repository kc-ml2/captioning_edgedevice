import torch

from model.mobilellama import MobileLlamaForCausalLM
from model.mobilevlm import load_pretrained_model

MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"



# ============================================================
# Load Model
# ============================================================

_, model, _, _ = load_pretrained_model(
    model_path=MODEL_PATH,
    device="cpu",
)

model.eval()

# ============================================================
# Get mm_projector
# ============================================================

mm_projector = model.get_model().mm_projector
mm_projector.eval()

torch.save(
    mm_projector.state_dict(),
    "mm_projector_weights.pth"
)

print("mm_projector weights saved: mm_projector_weights.pth")


# ============================================================
# Get LLM components
# ============================================================

model = MobileLlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    low_cpu_mem_usage=True,
)

model.eval()


# ============================================================
# Save Embedding Weight
# ============================================================

torch.save(
    model.model.embed_tokens.state_dict(),
    "embed_tokens.pth"
)


# ============================================================
# Save LM Head Weight
# ============================================================

torch.save(
    model.lm_head.state_dict(),
    "lm_head.pth"
)
