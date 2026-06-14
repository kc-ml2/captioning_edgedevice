import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from model.mobilellama import MobileLlamaForCausalLM

MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"


# ============================================================
# Load Model
# ============================================================

model = MobileLlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    low_cpu_mem_usage=True,
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

# ============================================================
# Save LM Head Weight
# ============================================================

torch.save(
    model.lm_head.state_dict(),
    "lm_head.pth"
)

print("saved weights")
