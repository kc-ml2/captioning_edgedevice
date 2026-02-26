# mobileVLM_cpu.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import re, torch, argparse, time, gc
from PIL import Image
from tqdm import tqdm
from typing import Any, Dict, List
from pycocotools.coco import COCO

from model.vicuan import conv_vicuna_v1
from model.mobilevlm import load_pretrained_model
from model.mutils import process_images, tokenizer_image_token, print_full_memory_report, to_int8_dynamic


def _build_prompt(question: str) -> str:
    conv = conv_vicuna_v1.copy()
    conv.append_message(conv.roles[0], "<image>" + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


# ---- Default values ---- #
GEN_KWARGS_DEFAULT = dict(
    num_beams=1,
    max_new_tokens=40,
    min_new_tokens=40,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", type=str, default="../dataset/coco/", help="Dataset folder path")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--run_name", type=str, default="mobilevlm_cpu_fp32")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ---- dataset paths (COCO standard) ----
    img_dir = os.path.join(args.coco_root, "val2017")
    gt_json = os.path.join(args.coco_root, "annotations", "captions_val2017.json")

    pred_json = os.path.join(args.out_dir, f"{args.run_name}.json")
    time_csv  = os.path.join(args.log_dir, f"{args.run_name}.csv")
    mem_csv   = os.path.join(args.log_dir, f"{args.run_name}_mem.csv")

    # ---- load COCO annotations ----
    coco = COCO(gt_json)
    img_ids = sorted(coco.getImgIds())
    if args.limit and args.limit > 0:
        img_ids = img_ids[: args.limit]

    device = torch.device("cpu")

    # HF model id or local path
    MODEL_PATH = "mtgv/MobileVLM_V2-1.7B"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL_PATH,
        load_8bit=False,
        load_4bit=False,
        device_map=None,
        device="cpu",
    )
    model.eval()

    print_full_memory_report("after fp32 load", model)

    # 2) INT8로 변환 (새 모델 반환)
    # model = to_int8_dynamic(model)
    # print_full_memory_report("after int8 + delete fp32", model)

    question = "What objects are visible in the image in detail."
    prompt = _build_prompt(question)

    preds: List[Dict[str, Any]] = []
    timing_records: List[Dict[str, Any]] = []

    # ---- main inference loop ----
    for img_id in tqdm(img_ids, desc="Generating captions (CPU)"):
        info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, info["file_name"])

        # Load image (PIL)
        image = Image.open(img_path).convert("RGB")

        # preprocess image
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor.to(device=device, dtype=torch.float32)

        # tokenize
        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            return_tensors="pt",
        ).unsqueeze(0).to(device)

        start = time.perf_counter()
        with torch.inference_mode():
            out_ids = model.generate(
                input_ids,
                images=image_tensor,
                **GEN_KWARGS_DEFAULT,
            )
        infer_sec = time.perf_counter() - start

        # Decode caption
        gen_ids = out_ids[0][input_ids.shape[1] :]
        text = tokenizer.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=True)[0]
        caption = re.sub(r"\s+", " ", text).strip()

        # Save predictions (COCO caption eval format commonly uses image_id + caption)
        preds.append({
            "image_id": img_id, 
            "caption": caption
        })
        
        timing_records.append({
            "image_id": img_id,
            "inference_sec": infer_sec,
        })
    
    print_full_memory_report("after generate", model)

    import json, csv
    with open(pred_json, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=4)

    # CSV (timing)
    fieldnames = ["image_id", "inference_sec"]
    with open(time_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timing_records)

    # ---- 평균 추론 시간 계산 ----
    valid_times = []
    for r in timing_records:
        sec = r.get("inference_sec", None)
        valid_times.append(float(sec))
        
    avg_time = (sum(valid_times) / len(valid_times)) if valid_times else 0.0


    print(f"Saved predictions: {pred_json}")
    print(f"Saved timing CSV:  {time_csv}")
    print(f"Average inference time per image: {avg_time:.2f} sec "
          f"(over {len(valid_times)} images)")


if __name__ == "__main__":
    main()