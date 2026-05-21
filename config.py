"""
config.py — All hyperparameters and paths in one place.

Auto-detects the running environment:
  • Google Colab  → uses DRIVE_ROOT = /content/drive/MyDrive/QADC
  • Local machine → uses the directory this file lives in as DRIVE_ROOT

resolve_image_path(path) transparently remaps stale Colab-style paths
to the local DATA_DIR, so JSON files written on Colab work locally
without any manual editing.

HF_ENDPOINT: defaults to hf-mirror.com so model downloads work without
a VPN. Override by setting the environment variable before running:
  HF_ENDPOINT=https://huggingface.co python run_pipeline.py
"""

import os
from pathlib import Path

# ── HuggingFace endpoint (set before any hf imports so downloads work) ────────
# Default: official HuggingFace. Override if you need a mirror, e.g.:
#   HF_ENDPOINT=https://hf-mirror.com python run_pipeline.py
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

# ── Environment detection ─────────────────────────────────────────────────────

def _on_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


ON_COLAB     = _on_colab()
_PROJECT_ROOT = Path(__file__).parent   # always the QADC directory


# ── Paths ─────────────────────────────────────────────────────────────────────

if ON_COLAB:
    DRIVE_ROOT = "/content/drive/MyDrive/QADC"
else:
    DRIVE_ROOT = str(_PROJECT_ROOT)

DATA_DIR   = os.path.join(DRIVE_ROOT, "data")
LABEL_DIR  = os.path.join(DRIVE_ROOT, "pseudo_labels")
CKPT_DIR   = os.path.join(DRIVE_ROOT, "checkpoints")
RESULT_DIR = os.path.join(DRIVE_ROOT, "results")

for _d in [DATA_DIR, LABEL_DIR, CKPT_DIR, RESULT_DIR]:
    os.makedirs(_d, exist_ok=True)


# ── Path resolver (handles stale Colab paths when running locally) ────────────

def resolve_image_path(path: str) -> str:
    """Return path as-is if it exists; otherwise remap basename to DATA_DIR."""
    if os.path.exists(path):
        return path
    local = os.path.join(DATA_DIR, os.path.basename(path))
    return local if os.path.exists(local) else path


# ── Base model ────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


# ── Resolution levels ─────────────────────────────────────────────────────────

RESOLUTION_RATIOS = [0.25, 0.5, 1.0]       # 1/4, 1/2, full
NUM_LEVELS        = 3                        # number of discrete classes
LEVEL_NAMES       = ["low", "mid", "full"]


# ── Dataset sampling ──────────────────────────────────────────────────────────

TEXTVQA_TRAIN_SAMPLES = 5000
TEXTVQA_VAL_SAMPLES   = 500
VQAV2_VAL_SAMPLES     = 300
RANDOM_SEED           = 42


# ── Pseudo label generation ───────────────────────────────────────────────────

INFERENCE_BATCH_SIZE = 1        # T4-safe for 3B model
MAX_NEW_TOKENS       = 20   # VQA answers are 1-4 words; 64 wastes ~3x time
SAVE_EVERY           = 100      # checkpoint frequency (samples)


# ── QVFP architecture ─────────────────────────────────────────────────────────

QUERY_DIM    = 2048             # Qwen2.5-VL-3B text hidden size (embed_tokens output)
VISUAL_DIM   = 2048             # Qwen2.5-VL-3B transformer hidden size (last hidden state)
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 128
DROPOUT      = 0.1


# ── Supervised learning ───────────────────────────────────────────────────────

SL_LR           = 1e-3
SL_EPOCHS       = 20
SL_BATCH_SIZE   = 64
SL_WEIGHT_DECAY = 1e-4


# ── RL (GRPO) ─────────────────────────────────────────────────────────────────

RL_LR            = 5e-5
RL_EPOCHS        = 5
RL_BATCH_SIZE    = 16
RL_GROUP_SIZE    = 4                    # G in GRPO: rollouts per prompt
RL_LAMBDA_VALUES = [0.1, 0.3, 0.5]     # efficiency penalty weights


# ── Ablation variants ─────────────────────────────────────────────────────────

ABLATION_FUSION_MODES = ["concat", "cross_attn", "query_only", "image_only"]
ABLATION_LEVEL_COUNTS = [2, 3, 4]      # number of resolution levels
