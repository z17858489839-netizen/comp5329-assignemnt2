"""
data/generate_pseudo_labels.py  —  Step 1
==========================================
For each (image, query) in the training split:
  1. Run Qwen2.5-VL-3B at resolutions 1/4, 1/2, 1x (stopping early).
  2. Record the LOWEST resolution that yields a correct answer as the
     pseudo label k* ∈ {0=low, 1=mid, 2=full}.
  3. Also extract and cache the query embedding + visual embedding
     (so the training step never needs to reload Qwen).

Checkpoint/resume: progress is saved to LABEL_DIR/progress.json every
SAVE_EVERY samples — safe to interrupt and re-run on Colab or locally.

Estimated runtime on T4: ~60 minutes for 5000 samples.
"""

import json
import os
import sys
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, LABEL_DIR, MODEL_ID,
    RESOLUTION_RATIOS, MAX_NEW_TOKENS, SAVE_EVERY,
    resolve_image_path,
)
from utils.vlm_inference import (
    load_model, resize_image, infer_single,
    extract_embeddings, is_correct,
)


PROGRESS_FILE   = os.path.join(LABEL_DIR, "progress.json")
LABELS_FILE     = os.path.join(LABEL_DIR, "pseudo_labels.json")
EMBEDDINGS_FILE = os.path.join(LABEL_DIR, "embeddings.pt")


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(progress: dict, labels: dict,
                  q_embs: list, v_embs: list) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f)
    if q_embs:
        torch.save(
            {"q_embs": torch.stack(q_embs), "v_embs": torch.stack(v_embs)},
            EMBEDDINGS_FILE,
            pickle_protocol=4,
        )


def main():
    # ── Load training data ────────────────────────────────────────────────────
    train_path = os.path.join(DATA_DIR, "textvqa_train.json")
    assert os.path.exists(train_path), "Run data/download_datasets.py first."

    with open(train_path) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} training samples.")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    progress = load_progress()

    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE) as f:
            labels = json.load(f)
    else:
        labels = {}

    if os.path.exists(EMBEDDINGS_FILE):
        emb_data = torch.load(EMBEDDINGS_FILE, weights_only=False)
        q_embs   = list(emb_data["q_embs"])
        v_embs   = list(emb_data["v_embs"])
    else:
        q_embs = []
        v_embs = []

    already_done = set(progress.keys())
    remaining    = [s for s in samples if str(s["id"]) not in already_done]
    print(f"Already done: {len(already_done)} | Remaining: {len(remaining)}")

    if not remaining:
        print("All samples already processed!")
        return

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor = load_model(MODEL_ID)

    # ── Main loop ─────────────────────────────────────────────────────────────
    stats = {"low": 0, "mid": 0, "full": 0, "none_correct": 0}

    for i, sample in enumerate(tqdm(remaining, desc="Generating pseudo labels")):
        sid        = str(sample["id"])
        query      = sample["question"]
        gt_answers = sample["answers"]
        image_path = resolve_image_path(sample["image_path"])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Cannot open image {image_path}: {e}. Skipping.")
            continue

        # ── Test resolutions from lowest to highest ───────────────────────────
        pseudo_label = 2        # default: full resolution
        for level_idx, ratio in enumerate(RESOLUTION_RATIOS):
            resized = resize_image(image, ratio)
            answer  = infer_single(model, processor, resized, query,
                                   max_new_tokens=MAX_NEW_TOKENS)
            if is_correct(answer, gt_answers):
                pseudo_label = level_idx
                break

        level_name = ["low", "mid", "full"][pseudo_label]
        stats[level_name] += 1

        # ── Extract embeddings ────────────────────────────────────────────────
        q_emb, v_emb = extract_embeddings(model, processor, image, query)
        emb_idx      = len(q_embs)
        q_embs.append(q_emb)
        v_embs.append(v_emb)

        # ── Record ────────────────────────────────────────────────────────────
        labels[sid]   = pseudo_label
        progress[sid] = {"label": pseudo_label, "emb_idx": emb_idx}

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (i + 1) % SAVE_EVERY == 0:
            save_progress(progress, labels, q_embs, v_embs)
            done  = len(already_done) + i + 1
            total = len(samples)
            print(f"\n  Checkpoint saved [{done}/{total}] | "
                  f"low={stats['low']} mid={stats['mid']} full={stats['full']}")

    # ── Final save ────────────────────────────────────────────────────────────
    save_progress(progress, labels, q_embs, v_embs)

    total = sum(stats.values())
    print("\n===== Pseudo Label Distribution =====")
    for k, v in stats.items():
        if total > 0:
            print(f"  {k:12s}: {v:4d}  ({v / total * 100:.1f}%)")
    print(f"  Total       : {total}")
    print(f"\nLabels  → {LABELS_FILE}")
    print(f"Embeds  → {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
