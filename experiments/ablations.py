"""
experiments/ablations.py  —  Step 4
=====================================
Runs all ablation studies by re-training QVFP variants and evaluating them.

Ablation 1 — Fusion mode:
    concat (default) | cross_attn | query_only | image_only

Ablation 2 — Number of resolution levels:
    2-level {1/4, 1x} | 3-level {1/4, 1/2, 1x} | 4-level {1/4, 1/3, 2/3, 1x}

Ablation 3 — Query keyword analysis (no retraining needed):
    Groups test predictions by query keyword category and reports
    level distribution per category.

All results saved to RESULT_DIR/ablation_*.json
Final summary table printed to stdout.
"""

import os, sys, json
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, LABEL_DIR, CKPT_DIR, RESULT_DIR,
    MODEL_ID, RESOLUTION_RATIOS, MAX_NEW_TOKENS,
    ABLATION_FUSION_MODES, ABLATION_LEVEL_COUNTS, NUM_LEVELS,
    resolve_image_path,
)
from models.qvfp import QVFP, build_datasets
from training.train_sl import train_qvfp_sl
from utils.vlm_inference import load_model, resize_image, infer_single, is_correct
from utils.metrics import summarise_results


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_qvfp_from_ckpt(ckpt_path: str) -> QVFP:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    model  = QVFP(
        fusion_mode = ckpt.get("fusion_mode", "concat"),
        num_levels  = ckpt.get("num_levels",  NUM_LEVELS),
        normalize_inputs = ckpt.get("normalize_inputs", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def checkpoint_needs_retrain(ckpt_path: str) -> bool:
    if not os.path.exists(ckpt_path):
        return True
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return not ckpt.get("normalize_inputs", False)


def quick_eval(model, samples, vlm, processor,
               q_embs_all, v_embs_all, id_to_emb):
    """Fast evaluation: returns (vqa_acc, levels_used)."""
    device      = next(model.parameters()).device
    predictions = []
    gt_list     = []
    levels      = []

    for s in tqdm(samples, desc="eval", leave=False):
        sid     = str(s["id"])
        emb_idx = id_to_emb.get(sid)
        if emb_idx is None:
            level = 2
        else:
            q_emb = q_embs_all[emb_idx].unsqueeze(0).to(device)
            v_emb = v_embs_all[emb_idx].unsqueeze(0).to(device)
            with torch.no_grad():
                level = model.predict(q_emb, v_emb).item()

        ratio = RESOLUTION_RATIOS[min(level, len(RESOLUTION_RATIOS) - 1)]
        try:
            img_path = resolve_image_path(s["image_path"])
            img      = Image.open(img_path).convert("RGB")
            pred     = infer_single(vlm, processor,
                                    resize_image(img, ratio), s["question"],
                                    max_new_tokens=MAX_NEW_TOKENS)
        except Exception:
            pred = ""

        predictions.append(pred)
        gt_list.append(s["answers"])
        levels.append(level)

    correct = sum(is_correct(p, g) for p, g in zip(predictions, gt_list))
    acc     = round(correct / len(predictions) * 100, 2)
    return acc, levels


# ── Ablation 1: Fusion mode ───────────────────────────────────────────────────

def ablation_fusion(vlm, processor, val_samples,
                    q_embs, v_embs, id_to_emb):
    print("\n===== ABLATION 1: Fusion Mode =====")
    results = []

    for mode in ABLATION_FUSION_MODES:
        ckpt_path = os.path.join(CKPT_DIR, f"qvfp_sl_{mode}_3lvl_best.pt")

        if checkpoint_needs_retrain(ckpt_path):
            print(f"  Training {mode} ...")
            train_qvfp_sl(fusion_mode=mode, num_levels=3)

        model   = load_qvfp_from_ckpt(ckpt_path)
        acc, lv = quick_eval(model, val_samples, vlm, processor,
                             q_embs, v_embs, id_to_emb)
        res = summarise_results(f"fusion_{mode}", acc, lv)
        results.append(res)
        print(f"  {mode:<14} | acc={acc:.1f}% | "
              f"tokens={res['avg_tokens']:.0f} | reduction={res['token_reduction%']:.1f}%")

    out = os.path.join(RESULT_DIR, "ablation_fusion.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out}")
    return results


# ── Ablation 2: Number of resolution levels ───────────────────────────────────

LEVEL_CONFIGS = {
    2: [0.25, 1.0],
    3: [0.25, 0.5, 1.0],
    4: [0.25, 0.33, 0.67, 1.0],
}


def ablation_levels(vlm, processor, val_samples,
                    q_embs, v_embs, id_to_emb):
    print("\n===== ABLATION 2: Resolution Levels =====")
    results = []

    for n_levels in ABLATION_LEVEL_COUNTS:
        ckpt_path = os.path.join(CKPT_DIR, f"qvfp_sl_concat_{n_levels}lvl_best.pt")

        if checkpoint_needs_retrain(ckpt_path):
            print(f"  Training {n_levels}-level model ...")
            train_qvfp_sl(fusion_mode="concat", num_levels=n_levels)

        model  = load_qvfp_from_ckpt(ckpt_path)
        ratios = LEVEL_CONFIGS[n_levels]
        device = next(model.parameters()).device

        predictions, gt_list, levels = [], [], []

        for s in tqdm(val_samples, desc=f"{n_levels}-level eval", leave=False):
            sid     = str(s["id"])
            emb_idx = id_to_emb.get(sid)
            if emb_idx is None:
                level = n_levels - 1
            else:
                q_emb = q_embs[emb_idx].unsqueeze(0).to(device)
                v_emb = v_embs[emb_idx].unsqueeze(0).to(device)
                with torch.no_grad():
                    level = model.predict(q_emb, v_emb).item()
                level = min(level, n_levels - 1)

            ratio = ratios[level]
            try:
                img_path = resolve_image_path(s["image_path"])
                img      = Image.open(img_path).convert("RGB")
                pred     = infer_single(vlm, processor,
                                        resize_image(img, ratio), s["question"],
                                        max_new_tokens=MAX_NEW_TOKENS)
            except Exception:
                pred = ""

            predictions.append(pred)
            gt_list.append(s["answers"])
            levels.append(level)

        correct   = sum(is_correct(p, g) for p, g in zip(predictions, gt_list))
        acc       = round(correct / len(predictions) * 100, 2)
        token_map = {i: int(ratios[i] ** 2 * 1024) for i in range(n_levels)}
        avg_tok   = round(sum(token_map[l] for l in levels) / len(levels), 1)
        red_pct   = round((1 - avg_tok / 1024) * 100, 1)

        res = {
            "method":           f"{n_levels}_levels",
            "vqa_accuracy":     acc,
            "avg_tokens":       avg_tok,
            "token_reduction%": red_pct,
        }
        results.append(res)
        print(f"  {n_levels}-level | acc={acc:.1f}% | "
              f"tokens={avg_tok:.0f} | reduction={red_pct:.1f}%")

    out = os.path.join(RESULT_DIR, "ablation_levels.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out}")
    return results


# ── Ablation 3: Query keyword analysis ───────────────────────────────────────

KEYWORD_GROUPS = {
    "global_questions": [
        "is there", "are there", "how many", "what color", "what is",
        "where is", "who is", "what kind", "does",
    ],
    "finegrained_text": [
        "read", "what does it say", "what text", "what word",
        "what number", "what letters", "what sign",
    ],
    "counting": ["how many", "count", "number of"],
    "attributes": ["what color", "what shape", "what size", "what type"],
}


def ablation_keyword_analysis(model: QVFP, samples: list,
                               q_embs, v_embs, id_to_emb):
    print("\n===== ABLATION 3: Query Keyword Analysis =====")
    device      = next(model.parameters()).device
    group_levels = {g: [] for g in KEYWORD_GROUPS}

    for s in tqdm(samples, desc="keyword analysis", leave=False):
        sid     = str(s["id"])
        emb_idx = id_to_emb.get(sid)
        if emb_idx is None:
            continue
        q_emb = q_embs[emb_idx].unsqueeze(0).to(device)
        v_emb = v_embs[emb_idx].unsqueeze(0).to(device)
        with torch.no_grad():
            level = model.predict(q_emb, v_emb).item()

        q_lower = s["question"].lower()
        for group, keywords in KEYWORD_GROUPS.items():
            if any(kw in q_lower for kw in keywords):
                group_levels[group].append(level)

    results = {}
    for group, lvls in group_levels.items():
        if not lvls:
            continue
        avg_level = sum(lvls) / len(lvls)
        dist = {
            "low_pct":  round(lvls.count(0) / len(lvls) * 100, 1),
            "mid_pct":  round(lvls.count(1) / len(lvls) * 100, 1),
            "full_pct": round(lvls.count(2) / len(lvls) * 100, 1),
        }
        results[group] = {"avg_level": round(avg_level, 2), **dist, "n": len(lvls)}
        print(f"  {group:<25} | avg_level={avg_level:.2f} | "
              f"low={dist['low_pct']}% mid={dist['mid_pct']}% full={dist['full_pct']}% "
              f"(n={len(lvls)})")

    out = os.path.join(RESULT_DIR, "ablation_keywords.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out}")
    return results


# ── Summary table printer ─────────────────────────────────────────────────────

def print_summary_table():
    print("\n" + "="*80)
    print("SUMMARY OF ALL ABLATION RESULTS")
    print("="*80)
    for fname in ["ablation_fusion.json", "ablation_levels.json",
                  "ablation_keywords.json"]:
        path = os.path.join(RESULT_DIR, fname)
        if not os.path.exists(path):
            continue
        print(f"\n--- {fname} ---")
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for row in data:
                print(f"  {json.dumps(row)}")
        else:
            for k, v in data.items():
                print(f"  {k}: {v}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load VLM
    vlm, processor = load_model(MODEL_ID)

    # Load embeddings (paths come from LABEL_DIR, no cross-module import needed)
    emb_data = torch.load(
        os.path.join(LABEL_DIR, "embeddings.pt"),
        map_location="cpu", weights_only=False,
    )
    q_embs = emb_data["q_embs"].float()
    v_embs = emb_data["v_embs"].float()

    with open(os.path.join(LABEL_DIR, "progress.json")) as f:
        progress  = json.load(f)
    id_to_emb = {sid: info["emb_idx"] for sid, info in progress.items()}

    with open(os.path.join(DATA_DIR, "textvqa_val.json")) as f:
        val_samples = json.load(f)

    # Run ablations
    ablation_fusion(vlm, processor, val_samples, q_embs, v_embs, id_to_emb)
    ablation_levels(vlm, processor, val_samples, q_embs, v_embs, id_to_emb)

    # Keyword analysis — load best SL model
    sl_ckpt = os.path.join(CKPT_DIR, "qvfp_sl_concat_3lvl_best.pt")
    if os.path.exists(sl_ckpt):
        best_model = load_qvfp_from_ckpt(sl_ckpt)
        ablation_keyword_analysis(best_model, val_samples, q_embs, v_embs, id_to_emb)

    print_summary_table()


if __name__ == "__main__":
    main()
