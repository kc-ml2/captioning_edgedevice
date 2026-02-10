# client.py
import os, json, csv, time, argparse
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm
from pycocotools.coco import COCO


data = {
    "question_in": "What objects are visible in the image.",
    "num_beams": 1,
    "max_new_tokens": 80,
    "min_new_tokens": 40,
}


def main():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--coco_root", type=str, default="../../")
    parser.add_argument("--limit", type=int, default=0)

    # server
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8600")
    parser.add_argument("--timeout", type=int, default=60)  # seconds
    parser.add_argument("--retries", type=int, default=3)

    # outputs
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--log_dir", type=str, default="./logs")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ---- dataset paths (COCO standard) ----
    img_dir = os.path.join(args.coco_root, "val2017")
    gt_json = os.path.join(args.coco_root, "annotations", "captions_val2017.json")

    pred_json = os.path.join(args.out_dir, f"predictions_mobile_b3_short.json")
    time_csv = os.path.join(args.log_dir, f"timing_mobile_b3_short.csv")

    # ---- API endpoint URLs ----
    base = args.server_url.rstrip("/")
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

    # ---- main inference loop ----
    for img_id in tqdm(img_ids, desc="Generating captions (client)"):
        info = coco.loadImgs(img_id)[0]
        path = os.path.join(img_dir, info["file_name"])

        with open(path, "rb") as f:
            files = {"file": (info["file_name"], f, "image/jpeg")}

            r = session.post(inference_url, data=data, files=files, timeout=args.timeout)

            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")

            resp = r.json()

        caption = resp["caption"]
        tms = resp["timings_ms"]

        preds.append({
            "image_id": int(img_id),
            "caption": caption,
        })

        # server keys: preprocess_ms / forward_ms / post_ms / total_ms
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

    # ---- save timings ----
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
    try:
        dr = session.post(done_url, timeout=args.timeout)
        dr.raise_for_status()
    except Exception as e:
        print(f"[WARN] done() failed: {e}")

    print("[DONE] predictions:", pred_json)
    print("[DONE] timings:", time_csv)


if __name__ == "__main__":
    main()
