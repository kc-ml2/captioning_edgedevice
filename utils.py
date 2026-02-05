# utils.py
import numpy as np
import csv
import torch
import os
import subprocess


def save_timing_debug(
    pre_times,
    forward_times,
    post_times,
    total_times,
    out_csv,
):
    pre = np.asarray(pre_times, dtype=np.float32)
    fwd = np.asarray(forward_times, dtype=np.float32)
    post = np.asarray(post_times, dtype=np.float32)
    total = np.asarray(total_times, dtype=np.float32)

    n = min(len(pre), len(fwd), len(post), len(total))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "idx",
            "preprocess_ms",
            "forward_ms",
            "postprocess_ms",
            "total_ms",
        ])
        for i in range(n):
            w.writerow([
                i,
                float(pre[i]),
                float(fwd[i]),
                float(post[i]),
                float(total[i]),
            ])


def cuda_mem_mb():
    mem = {
        "allocated_MB": 0,
        "reserved_MB": 0,
        "nvidia_smi_MB": None,
    }

    if not torch.cuda.is_available():
        return mem

    # PyTorch memory
    mem["allocated_MB"] = int(torch.cuda.memory_allocated() / (1024 * 1024))
    mem["reserved_MB"] = int(torch.cuda.memory_reserved() / (1024 * 1024))

    # nvidia-smi (process-level GPU memory usage)
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )

        pid = str(os.getpid())
        for line in out.strip().splitlines():
            p, used = [x.strip() for x in line.split(",")]
            if p == pid:
                mem["nvidia_smi_MB"] = int(used)
                break

    except Exception:
        mem["nvidia_smi_MB"] = None

    return mem


def ms(dt_s: float) -> float:
    return round(dt_s * 1000.0, 2)