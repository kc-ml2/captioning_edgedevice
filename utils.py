import numpy as np
import csv

def save_timing_debug(pre_times, forward_times, post_times, out_npz, out_csv):
    pre = np.asarray(pre_times, dtype=np.float32)
    fwd = np.asarray(forward_times, dtype=np.float32)
    post = np.asarray(post_times, dtype=np.float32)

    # NPZ
    np.savez_compressed(
        out_npz,
        pre_ms=pre,
        forward_ms=fwd,
        post_ms=post,
    )

    # CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "preprocess_ms", "forward_ms", "post_ms"])
        for i, (a, b, c) in enumerate(zip(pre, fwd, post)):
            w.writerow([i, float(a), float(b), float(c)])