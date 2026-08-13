# build_multifunction.py

import os
import coremltools as ct


# ============================================================
# Paths
# ============================================================

PREFILL_MODEL = "mobilellama_prefill_fp16.mlpackage"
DECODER_MODEL = "mobilellama_decoder_fp16.mlpackage"

OUTPUT_MODEL = "mobilellama_multifunction_fp16.mlpackage"


# ============================================================
# Check
# ============================================================

assert os.path.exists(PREFILL_MODEL), \
    f"Not found: {PREFILL_MODEL}"

assert os.path.exists(DECODER_MODEL), \
    f"Not found: {DECODER_MODEL}"

print(f"coremltools version : {ct.__version__}")


# ============================================================
# MultiFunction Descriptor
# ============================================================

desc = ct.utils.MultiFunctionDescriptor()


# ============================================================
# Prefill Function
# ============================================================

desc.add_function(
    model_path=PREFILL_MODEL,
    src_function_name="main",
    target_function_name="prefill",
)

print("Added function : prefill")


# ============================================================
# Decode Function
# ============================================================

desc.add_function(
    model_path=DECODER_MODEL,
    src_function_name="main",
    target_function_name="decode",
)

print("Added function : decode")


# ============================================================
# Default Function
# ============================================================

desc.default_function_name = "prefill"


# ============================================================
# Save MultiFunction Model
# ============================================================

ct.utils.save_multifunction(
    desc,
    OUTPUT_MODEL,
)

print()
print("===================================")
print("MultiFunction model created")
print(f"Output : {OUTPUT_MODEL}")
print("Default Function : prefill")
print("Functions :")
print("  - prefill")
print("  - decode")
print("===================================")