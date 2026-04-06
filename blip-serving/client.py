# client.py
import os, json, argparse, csv
from typing import List, Dict, Any

import requests
from tqdm import tqdm
from pycocotools.coco import COCO
from utils import save_timing_debug


def main():
    # ---- argument parsing ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", type=str, default="../")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./outputs_inference")
    
    # Prompt and decoding settings are fixed on the model server.
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8600")
    parser.add_argument("--timeout", type=int, default=10)  # seconds

    args = parser.parse_args()

    # ---- dataset paths ----
    img_dir = os.path.join(args.coco_root, "val2017")
    gt_json = os.path.join(args.coco_root, "annotations", "captions_val2017.json")

    os.makedirs(args.save_dir, exist_ok=True)
    pred_json = os.path.join(args.save_dir, "predictions_beam3_fast.json")
    time_csv = os.path.join(args.save_dir, "timing_debug_beam3_fast.csv")
    
    # ---- API endpoint URLs ----
    base = args.server_url.rstrip("/")
    health_url = f"{base}/health"
    inference_url = f"{base}/inference"
    done_url = f"{base}/done"

    # ---- load COCO annotations ----
    coco = COCO(gt_json)
    img_ids = sorted(coco.getImgIds())
    if args.limit and args.limit > 0:
        img_ids = img_ids[: args.limit]

    preds: List[Dict[str, Any]] = []
    timing_records: List[Dict[str, Any]] = []
    session = requests.Session()

    # ---- server health check (ensure model is loaded and warm) ----
    try:
        health = session.get(health_url, timeout=args.timeout).json()
        if not health.get("loaded", False):
            raise RuntimeError("Model server is running but model is not loaded.")
    except Exception as e:
        raise RuntimeError(f"Model server health check failed: {e}")


    # ---- main inference loop ----
    for img_id in tqdm(img_ids, desc="Generating captions (client)"):
        info = coco.loadImgs(img_id)[0]
        path = os.path.join(img_dir, info["file_name"])

        with open(path, "rb") as f:
            files = {"file": (info["file_name"], f)}

            # Server-side policy is fixed: send only the image file
            r = session.post(
                inference_url, 
                files=files, 
                timeout=60
            )
            r.raise_for_status()
            resp = r.json()
            caption = resp["caption"]
            tms = resp["timings_ms"]

        preds.append({
            "image_id": int(img_id),
            "caption": caption
        })

        timing_records.append({
            "image_id": int(img_id),
            "preprocess_ms": float(tms.get("preprocess_ms", -1.0)),
            "forward_ms": float(tms.get("forward_ms", -1.0)),
            "post_ms": float(tms.get("post_ms", -1.0)),
            "total_ms": float(tms.get("total_ms", -1.0)),
        })

    # ---- save predictions ----
    with open(pred_json, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
        f.write("\n")
    
    with open(time_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "preprocess_ms", "forward_ms", "post_ms", "total_ms"])
        for row in timing_records:
            w.writerow([
                row["image_id"],
                row["preprocess_ms"],
                row["forward_ms"],
                row["post_ms"],
                row["total_ms"],
            ])
    
    # ---- notify server that all requests are completed ----
    dr = session.post(done_url, timeout=args.timeout)
    dr.raise_for_status()
    print("[DONE] predictions:", pred_json)
    print("[DONE] time_savings:", time_csv)


# ---- entrypoint ----
if __name__ == "__main__":
    main()
