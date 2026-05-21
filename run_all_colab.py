# QADC Master Colab Notebook
# ============================
# Run each cell in order. Each session can start from where you left off.

# ── Cell 1: Install & Mount ───────────────────────────────────────────────────
"""
!pip install transformers datasets torch torchvision \
             qwen-vl-utils accelerate tqdm pillow \
             scikit-learn matplotlib pandas -q

from google.colab import drive
drive.mount('/content/drive')

# Clone or upload your QADC folder to Drive, then:
import sys, os
QADC_PATH = '/content/drive/MyDrive/QADC'
sys.path.insert(0, QADC_PATH)
os.chdir(QADC_PATH)
print("Ready.")
"""

# ── Cell 2: Step 0 — Download datasets ───────────────────────────────────────
"""
%run data/download_datasets.py
# Expected output: 3 JSON files in DATA_DIR (~15 min)
"""

# ── Cell 3: Step 1 — Generate pseudo labels ───────────────────────────────────
"""
%run data/generate_pseudo_labels.py
# Expected output: pseudo_labels.json + embeddings.pt (~60 min)
# SAFE TO INTERRUPT: resumes from last checkpoint on re-run.
"""

# ── Cell 4: Check pseudo label distribution ───────────────────────────────────
"""
import json, os
from config import LABEL_DIR

with open(os.path.join(LABEL_DIR, 'pseudo_labels.json')) as f:
    labels = json.load(f)

from collections import Counter
dist = Counter(labels.values())
total = len(labels)
print(f"Total labelled: {total}")
for k, v in sorted(dist.items()):
    name = ['low (1/4)', 'mid (1/2)', 'full (1x)'][k]
    print(f"  Level {k} ({name}): {v:4d}  ({v/total*100:.1f}%)")
"""

# ── Cell 5: Step 2a — Train QVFP-SL (main method) ────────────────────────────
"""
from training.train_sl import train_qvfp_sl
best_acc, ckpt = train_qvfp_sl(fusion_mode='concat', num_levels=3)
print(f"Best val acc: {best_acc:.1f}%  |  Checkpoint: {ckpt}")
# Expected: ~10 minutes
"""

# ── Cell 6: Step 2b — Train QVFP-RL variants ─────────────────────────────────
"""
from training.train_rl import train_qvfp_rl
for lam in [0.1, 0.3, 0.5]:
    reward, ckpt = train_qvfp_rl(lam=lam, rl_samples=1000)
    print(f"λ={lam}  best_reward={reward:.4f}  ckpt={ckpt}")
# Expected: ~30 minutes total (10 min per λ)
"""

# ── Cell 7: Step 3 — Main evaluation ─────────────────────────────────────────
"""
%run evaluation/evaluate.py
# Prints full results table and saves main_results.json
# Expected: ~40 minutes
"""

# ── Cell 8: Step 4 — Ablation studies ────────────────────────────────────────
"""
%run experiments/ablations.py
# Trains fusion-mode variants, level variants, runs keyword analysis
# Expected: ~60 minutes
"""

# ── Cell 9: Generate all figures ─────────────────────────────────────────────
"""
%run experiments/plot_results.py
# Saves 6 PDF figures to RESULT_DIR/figures/
"""

# ── Cell 10: Quick sanity check — single sample ───────────────────────────────
"""
import torch
from PIL import Image
from config import CKPT_DIR, RESOLUTION_RATIOS, LEVEL_NAMES
from models.qvfp import QVFP
from utils.vlm_inference import load_model, resize_image, infer_single, extract_embeddings

vlm, proc = load_model()

# Load best SL model
import os
ckpt  = torch.load(os.path.join(CKPT_DIR, 'qvfp_sl_concat_3lvl_best.pt'))
model = QVFP(fusion_mode='concat', num_levels=3)
model.load_state_dict(ckpt['model_state'])
model.eval()

# Test on a custom image + query
from datasets import load_dataset
sample = load_dataset('textvqa', split='validation', trust_remote_code=True)[0]
image  = sample['image']
query  = sample['question']
gts    = sample['answers']

q_emb, v_emb = extract_embeddings(vlm, proc, image, query)
level  = model.predict(q_emb.unsqueeze(0), v_emb.unsqueeze(0)).item()
ratio  = RESOLUTION_RATIOS[level]
answer = infer_single(vlm, proc, resize_image(image, ratio), query)

print(f"Query  : {query}")
print(f"GT     : {gts}")
print(f"Level  : {LEVEL_NAMES[level]} (ratio={ratio}, tokens≈{[64,256,1024][level]})")
print(f"Answer : {answer}")
"""

# ── Cell 11: Print final results table ────────────────────────────────────────
"""
import json, os
from config import RESULT_DIR

with open(os.path.join(RESULT_DIR, 'main_results.json')) as f:
    results = json.load(f)

for split, rows in results.items():
    print(f"\n{'='*75}")
    print(f"{'Method':<22} {'Acc%':>6} {'AvgTok':>8} {'Reduction%':>11} "
          f"{'Low%':>6} {'Mid%':>6} {'Full%':>6}")
    print('-'*75)
    for r in rows:
        print(f"{r['method']:<22} {r['vqa_accuracy']:>6.1f} "
              f"{r['avg_tokens']:>8.0f} {r['token_reduction%']:>11.1f} "
              f"{r['low_pct']:>6.1f} {r['mid_pct']:>6.1f} {r['full_pct']:>6.1f}")
"""
