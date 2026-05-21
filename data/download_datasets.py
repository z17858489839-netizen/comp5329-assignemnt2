"""
data/download_datasets.py  —  Step 0
Downloads TextVQA (train + val) and VQAv2 (validation) from HuggingFace,
samples the required number of examples, and saves them as JSON files
in DATA_DIR so all subsequent scripts can load fast without re-downloading.

Runtime: ~15 minutes on Colab (mostly download time).
"""

import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR,
    TEXTVQA_TRAIN_SAMPLES,
    TEXTVQA_VAL_SAMPLES,
    VQAV2_VAL_SAMPLES,
    RANDOM_SEED,
)

from datasets import load_dataset


def sample_and_save(dataset, n: int, out_path: str, task: str):
    random.seed(RANDOM_SEED)
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    samples = []

    for i, idx in enumerate(indices):
        row = dataset[idx]

        if task == "textvqa":
            sample = {
                "id":       str(row.get("question_id", idx)),
                "question": row["question"],
                "answers":  row["answers"],       # list of strings
                "image":    row["image"],          # PIL Image or path
            }
        elif task == "vqav2":
            # VQAv2 val answers come as a list of dicts
            raw_answers = row.get("answers", [])
            if raw_answers and isinstance(raw_answers[0], dict):
                answers = [a["answer"] for a in raw_answers]
            else:
                answers = raw_answers
            sample = {
                "id":       str(row.get("question_id", idx)),
                "question": row["question"],
                "answers":  answers,
                "image":    row["image"],
            }
        else:
            raise ValueError(f"Unknown task: {task}")

        # Save image separately as JPEG to avoid pickle issues
        img_path = os.path.join(DATA_DIR, f"{task}_{sample['id']}.jpg")
        if not os.path.exists(img_path):
            img = sample["image"]
            if hasattr(img, "convert"):
                img.convert("RGB").save(img_path, "JPEG", quality=90)

        # Replace PIL image with path string in JSON record
        sample["image_path"] = img_path
        del sample["image"]
        samples.append(sample)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(indices)}")

    with open(out_path, "w") as f:
        json.dump(samples, f)
    print(f"Saved {len(samples)} samples → {out_path}")
    return samples


def main():
    # ── TextVQA train ─────────────────────────────────────────────────────────
    train_path = os.path.join(DATA_DIR, "textvqa_train.json")
    if os.path.exists(train_path):
        print(f"TextVQA train already cached: {train_path}")
    else:
        print("Downloading TextVQA train ...")
        ds_train = load_dataset("facebook/textvqa", split="train")
        sample_and_save(ds_train, TEXTVQA_TRAIN_SAMPLES, train_path, "textvqa")

    # ── TextVQA val ───────────────────────────────────────────────────────────
    val_path = os.path.join(DATA_DIR, "textvqa_val.json")
    if os.path.exists(val_path):
        print(f"TextVQA val already cached: {val_path}")
    else:
        print("Downloading TextVQA val ...")
        ds_val = load_dataset("facebook/textvqa", split="validation")
        sample_and_save(ds_val, TEXTVQA_VAL_SAMPLES, val_path, "textvqa")

    # ── VQAv2 val ─────────────────────────────────────────────────────────────
    vqa2_path = os.path.join(DATA_DIR, "vqav2_val.json")
    if os.path.exists(vqa2_path):
        print(f"VQAv2 val already cached: {vqa2_path}")
    else:
        print("Downloading VQAv2 validation ...")
        ds_vqa2 = load_dataset(
            "HuggingFaceM4/VQAv2", split="validation", trust_remote_code=True
        )
        sample_and_save(ds_vqa2, VQAV2_VAL_SAMPLES, vqa2_path, "vqav2")

    print("\nAll datasets ready.")


if __name__ == "__main__":
    main()
