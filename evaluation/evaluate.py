"""
evaluation/evaluate.py  —  Step 3
====================================
Evaluates QADC against all baselines on TextVQA (val) and VQAv2 (val).

Baselines evaluated:
  1. full_res      — always use full resolution (upper bound)
  2. fixed_025     — always use 1/4 resolution  (lower bound efficiency)
  3. fixed_050     — always use 1/2 resolution
  4. oracle        — uses the pseudo-label (theoretical best for QADC)
  5. qadc_sl       — our SL method
  6. qadc_rl_0.1   — RL with λ=0.1
  7. qadc_rl_0.3   — RL with λ=0.3
  8. qadc_rl_0.5   — RL with λ=0.5

Saves a full results table to RESULT_DIR/main_results.json
and prints a formatted summary.
"""

import os, sys, json
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, LABEL_DIR, CKPT_DIR, RESULT_DIR,
    MODEL_ID, RESOLUTION_RATIOS, MAX_NEW_TOKENS,
    RL_LAMBDA_VALUES, NUM_LEVELS,
    resolve_image_path,
)
from models.qvfp import QVFP
from utils.vlm_inference import (
    load_model, resize_image, infer_single, is_correct, extract_embeddings,
)
from utils.metrics import summarise_results, TOKENS_PER_LEVEL

_EVAL_EMB_CACHE = {}


# ── Load embeddings for QVFP baselines ───────────────────────────────────────

def load_embeddings_and_labels(label_dir: str):
    emb_data = torch.load(
        os.path.join(label_dir, "embeddings.pt"),
        map_location="cpu", weights_only=False,
    )
    with open(os.path.join(label_dir, "pseudo_labels.json")) as f:
        labels = json.load(f)
    with open(os.path.join(label_dir, "progress.json")) as f:
        progress = json.load(f)
    id_to_emb = {sid: info["emb_idx"] for sid, info in progress.items()}
    return emb_data["q_embs"].float(), emb_data["v_embs"].float(), labels, id_to_emb


# ── Load QVFP checkpoint ──────────────────────────────────────────────────────

def load_qvfp(ckpt_path: str, fusion_mode: str = "concat") -> QVFP:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not ckpt.get("normalize_inputs", False):
        print(
            f"[WARN] Legacy checkpoint without input normalisation: {ckpt_path}. "
            "Retrain SL/RL checkpoints before trusting QADC results."
        )
    model  = QVFP(
        fusion_mode = ckpt.get("fusion_mode", fusion_mode),
        num_levels  = ckpt.get("num_levels",  NUM_LEVELS),
        normalize_inputs = ckpt.get("normalize_inputs", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ── Evaluate a single method on one dataset split ────────────────────────────

def evaluate_method(
    method_name:   str,
    split_name:    str,
    samples:       list,
    vlm,
    processor,
    get_level_fn,           # callable(split_name, sid, q_emb, v_emb) → int level
    q_embs_all,
    v_embs_all,
    id_to_emb:     dict,
    pseudo_labels: dict,
) -> dict:
    predictions, gt_list, levels_used = [], [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for sample in tqdm(samples, desc=method_name, leave=False):
        sid = str(sample["id"])
        q_emb = v_emb = None

        if q_emb is None and "qadc" in method_name:
            try:
                cache_key = f"{split_name}:{sid}"
                if cache_key not in _EVAL_EMB_CACHE:
                    img_path = resolve_image_path(sample["image_path"])
                    img = Image.open(img_path).convert("RGB")
                    q_new, v_new = extract_embeddings(vlm, processor, img, sample["question"])
                    _EVAL_EMB_CACHE[cache_key] = (q_new.float().cpu(), v_new.float().cpu())
                q_new, v_new = _EVAL_EMB_CACHE[cache_key]
                q_emb = q_new.unsqueeze(0).to(device)
                v_emb = v_new.unsqueeze(0).to(device)
            except Exception:
                q_emb = v_emb = None

        level = get_level_fn(split_name, sid, q_emb, v_emb)
        ratio = RESOLUTION_RATIOS[min(level, len(RESOLUTION_RATIOS) - 1)]

        try:
            img_path = resolve_image_path(sample["image_path"])
            img      = Image.open(img_path).convert("RGB")
            img_r    = resize_image(img, ratio)
            pred     = infer_single(vlm, processor, img_r, sample["question"],
                                    max_new_tokens=MAX_NEW_TOKENS)
        except Exception:
            pred = ""

        predictions.append(pred)
        gt_list.append(sample["answers"])
        levels_used.append(level)

    correct = sum(is_correct(p, g) for p, g in zip(predictions, gt_list))
    acc     = round(correct / len(predictions) * 100, 2)
    return summarise_results(method_name, acc, levels_used)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load VLM ─────────────────────────────────────────────────────────────
    vlm, processor = load_model(MODEL_ID)

    # ── Load embeddings ───────────────────────────────────────────────────────
    q_embs, v_embs, pseudo_labels, id_to_emb = load_embeddings_and_labels(LABEL_DIR)

    # ── Load test splits ──────────────────────────────────────────────────────
    splits = {}
    for split_name, fname in [
        ("textvqa_val", "textvqa_val.json"),
        ("vqav2_val",   "vqav2_val.json"),
    ]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                splits[split_name] = json.load(f)
        else:
            print(f"[WARN] {path} not found, skipping {split_name}.")

    # ── Oracle levels for validation samples ─────────────────────────────────
    oracle_cache_path = os.path.join(LABEL_DIR, "oracle_val_levels.json")
    def oracle_key(split_name: str, sid: str) -> str:
        return f"{split_name}:{sid}"

    if os.path.exists(oracle_cache_path):
        with open(oracle_cache_path) as f:
            oracle_map = json.load(f)
        print(f"Loaded oracle cache ({len(oracle_map)} samples).")
    else:
        oracle_map = {}

    # Legacy caches used bare sample ids and may be incomplete. Fill every
    # missing split:id entry so oracle does not silently fall back to full-res.
    missing_oracle = [
        (split_name, sample)
        for split_name, samples in splits.items()
        for sample in samples
        if oracle_key(split_name, str(sample["id"])) not in oracle_map
    ]

    if missing_oracle:
        print(f"Pre-computing {len(missing_oracle)} missing oracle levels ...")
        for split_name, samples in splits.items():
            for sample in tqdm(samples, desc=f"oracle/{split_name}"):
                sid = str(sample["id"])
                key = oracle_key(split_name, sid)
                if key in oracle_map:
                    continue
                oracle_map[key] = NUM_LEVELS - 1
                try:
                    img_path = resolve_image_path(sample["image_path"])
                    img = Image.open(img_path).convert("RGB")
                    for level, ratio in enumerate(RESOLUTION_RATIOS):
                        pred = infer_single(
                            vlm, processor, resize_image(img, ratio),
                            sample["question"], max_new_tokens=MAX_NEW_TOKENS,
                        )
                        if is_correct(pred, sample["answers"]):
                            oracle_map[key] = level
                            break
                except Exception:
                    oracle_map[key] = NUM_LEVELS - 1
        with open(oracle_cache_path, "w") as f:
            json.dump(oracle_map, f)

    # ── Level functions ───────────────────────────────────────────────────────

    def fixed_level(level: int):
        return lambda split_name, sid, q, v: level

    def oracle_level(split_name, sid, q, v):
        return int(oracle_map.get(oracle_key(split_name, sid), NUM_LEVELS - 1))

    def qvfp_level(model: QVFP, lam: float = 0.10):
        ids = [sid for sid in pseudo_labels.keys() if sid in id_to_emb]
        if ids:
            device = next(model.parameters()).device
            ids = ids[:min(len(ids), 2000)]
            q_cal = torch.stack([q_embs[id_to_emb[sid]] for sid in ids]).to(device)
            v_cal = torch.stack([v_embs[id_to_emb[sid]] for sid in ids]).to(device)
            y_cal = torch.tensor([int(pseudo_labels[sid]) for sid in ids], device=device)
            probs = F.softmax(model(q_cal, v_cal), dim=-1)
            cum = probs.cumsum(dim=-1)
            cost = torch.tensor([64 / 1024, 256 / 1024, 1.0], device=device)[:NUM_LEVELS]
            best_tau, best_score = 0.5, -1e9
            for tau in torch.linspace(0.45, 0.95, 21, device=device):
                pred = (cum >= tau).float().argmax(dim=-1)
                score = ((pred >= y_cal).float() - lam * cost[pred]).mean().item()
                if score > best_score:
                    best_score = score
                    best_tau = float(tau.item())
        else:
            best_tau = None

        def fn(split_name, sid, q_emb, v_emb):
            if q_emb is None:
                return 2
            with torch.no_grad():
                if best_tau is None:
                    return model.predict(q_emb, v_emb).item()
                probs = F.softmax(model(q_emb, v_emb), dim=-1)
                return int((probs.cumsum(dim=-1) >= best_tau).float().argmax(dim=-1).item())
        return fn

    # ── Build method registry ─────────────────────────────────────────────────
    methods = {
        "full_res":  fixed_level(2),
        "fixed_025": fixed_level(0),
        "fixed_050": fixed_level(1),
        "oracle":    oracle_level,
    }

    sl_ckpt = os.path.join(CKPT_DIR, "qvfp_sl_concat_3lvl_best.pt")
    if os.path.exists(sl_ckpt):
        methods["qadc_sl"] = qvfp_level(load_qvfp(sl_ckpt), lam=0.10)
    else:
        print(f"[WARN] SL checkpoint not found: {sl_ckpt}")

    for lam in RL_LAMBDA_VALUES:
        rl_ckpt = os.path.join(CKPT_DIR, f"qvfp_rl_lambda{lam}_best.pt")
        if os.path.exists(rl_ckpt):
            methods[f"qadc_rl_lam{lam}"] = qvfp_level(load_qvfp(rl_ckpt), lam=lam)
        else:
            print(f"[WARN] RL checkpoint not found: {rl_ckpt}")

    # ── Run evaluation ────────────────────────────────────────────────────────
    all_results = {}

    for split_name, samples in splits.items():
        print(f"\n{'='*60}")
        print(f"Evaluating on {split_name} ({len(samples)} samples)")
        print(f"{'='*60}")
        split_results = []

        for method_name, level_fn in methods.items():
            res = evaluate_method(
                method_name, split_name, samples, vlm, processor,
                level_fn, q_embs, v_embs, id_to_emb, pseudo_labels,
            )
            split_results.append(res)
            print(
                f"  {method_name:<22} | "
                f"acc={res['vqa_accuracy']:5.1f}% | "
                f"tokens={res['avg_tokens']:6.0f} | "
                f"reduction={res['token_reduction%']:4.1f}% | "
                f"low={res['low_pct']:4.1f}% mid={res['mid_pct']:4.1f}% full={res['full_pct']:4.1f}%"
            )

        all_results[split_name] = split_results

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(RESULT_DIR, "main_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
