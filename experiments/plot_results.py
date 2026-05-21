"""
experiments/plot_results.py
============================
Generates all figures needed for the paper from saved JSON results.

Figures produced:
  Fig 1 — Main results: accuracy vs token reduction (scatter)
  Fig 2 — Training curves (SL loss + val acc)
  Fig 3 — Ablation: fusion mode bar chart
  Fig 4 — Ablation: resolution levels bar chart
  Fig 5 — Keyword analysis: level distribution by query type
  Fig 6 — SL vs RL: reward/accuracy vs λ

All saved to RESULT_DIR/figures/
"""

import os, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULT_DIR

FIG_DIR = os.path.join(RESULT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

STYLE = {
    "full_res":        {"color": "#888780", "marker": "s", "label": "Full-res"},
    "fixed_025":       {"color": "#F09595", "marker": "^", "label": "Fixed 1/4"},
    "fixed_050":       {"color": "#EF9F27", "marker": "^", "label": "Fixed 1/2"},
    "oracle":          {"color": "#1D9E75", "marker": "D", "label": "Oracle"},
    "qadc_sl":         {"color": "#534AB7", "marker": "o", "label": "QADC-SL (ours)"},
    "qadc_rl_lam0.1":  {"color": "#378ADD", "marker": "o", "label": "QADC-RL λ=0.1"},
    "qadc_rl_lam0.3":  {"color": "#185FA5", "marker": "o", "label": "QADC-RL λ=0.3"},
    "qadc_rl_lam0.5":  {"color": "#042C53", "marker": "o", "label": "QADC-RL λ=0.5"},
}


def savefig(name: str):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Fig 1: Accuracy vs Token Reduction ───────────────────────────────────────

def plot_main_results():
    path = os.path.join(RESULT_DIR, "main_results.json")
    if not os.path.exists(path):
        print("[SKIP] main_results.json not found"); return

    with open(path) as f:
        all_results = json.load(f)

    for split_name, rows in all_results.items():
        fig, ax = plt.subplots(figsize=(7, 5))

        for row in rows:
            m    = row["method"]
            st   = STYLE.get(m, {"color": "#aaa", "marker": "x", "label": m})
            size = 120 if "qadc" in m else 70
            ax.scatter(
                row["token_reduction%"],
                row["vqa_accuracy"],
                s=size, color=st["color"], marker=st["marker"],
                zorder=3, edgecolors="white", linewidths=0.8,
            )
            ax.annotate(
                st["label"],
                (row["token_reduction%"], row["vqa_accuracy"]),
                textcoords="offset points", xytext=(6, 3),
                fontsize=8, color=st["color"],
            )

        ax.set_xlabel("Token reduction (%)", fontsize=11)
        ax.set_ylabel("VQA accuracy (%)",    fontsize=11)
        ax.set_title(f"Accuracy vs Efficiency — {split_name}", fontsize=12)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        savefig(f"fig1_acc_vs_tokens_{split_name}.pdf")


# ── Fig 2: Training curves ────────────────────────────────────────────────────

def plot_training_curves():
    path = os.path.join(RESULT_DIR, "sl_curve_concat_3lvl.json")
    if not os.path.exists(path):
        print("[SKIP] sl training curve not found"); return

    with open(path) as f:
        data = json.load(f)
    hist = data["history"]

    epochs     = [h["epoch"]      for h in hist]
    train_loss = [h["train_loss"] for h in hist]
    val_acc    = [h["val_acc"]    for h in hist]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, train_loss, color="#534AB7", linewidth=1.8)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training loss")
    ax1.set_title("QVFP-SL training loss")
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(epochs, val_acc, color="#1D9E75", linewidth=1.8)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation accuracy (%)")
    ax2.set_title("QVFP-SL validation accuracy")
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    savefig("fig2_training_curves.pdf")


# ── Fig 3: Fusion mode ablation ───────────────────────────────────────────────

def plot_fusion_ablation():
    path = os.path.join(RESULT_DIR, "ablation_fusion.json")
    if not os.path.exists(path):
        print("[SKIP] ablation_fusion.json not found"); return

    with open(path) as f:
        rows = json.load(f)

    labels = [r["method"].replace("fusion_", "") for r in rows]
    accs   = [r["vqa_accuracy"]     for r in rows]
    toks   = [r["token_reduction%"] for r in rows]

    x  = np.arange(len(labels))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w/2, accs, w, label="VQA accuracy (%)", color="#534AB7", alpha=0.85)
    b2 = ax.bar(x + w/2, toks, w, label="Token reduction (%)", color="#1D9E75", alpha=0.85)

    ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Percentage (%)"); ax.legend()
    ax.set_title("Ablation: Fusion Mode")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    savefig("fig3_ablation_fusion.pdf")


# ── Fig 4: Resolution levels ablation ────────────────────────────────────────

def plot_levels_ablation():
    path = os.path.join(RESULT_DIR, "ablation_levels.json")
    if not os.path.exists(path):
        print("[SKIP] ablation_levels.json not found"); return

    with open(path) as f:
        rows = json.load(f)

    labels = [r["method"] for r in rows]
    accs   = [r["vqa_accuracy"]     for r in rows]
    toks   = [r["token_reduction%"] for r in rows]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar(x - w/2, accs, w, label="VQA accuracy (%)", color="#534AB7", alpha=0.85)
    b2 = ax.bar(x + w/2, toks, w, label="Token reduction (%)", color="#EF9F27", alpha=0.85)
    ax.bar_label(b1, fmt="%.1f", fontsize=9, padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=9, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Percentage (%)"); ax.legend()
    ax.set_title("Ablation: Number of Resolution Levels")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    savefig("fig4_ablation_levels.pdf")


# ── Fig 5: Keyword analysis ───────────────────────────────────────────────────

def plot_keyword_analysis():
    path = os.path.join(RESULT_DIR, "ablation_keywords.json")
    if not os.path.exists(path):
        print("[SKIP] ablation_keywords.json not found"); return

    with open(path) as f:
        data = json.load(f)

    groups = list(data.keys())
    low    = [data[g]["low_pct"]  for g in groups]
    mid    = [data[g]["mid_pct"]  for g in groups]
    full   = [data[g]["full_pct"] for g in groups]

    x   = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, low,  label="Low (1/4×)",  color="#1D9E75", alpha=0.85)
    ax.bar(x, mid,  bottom=low,          label="Mid (1/2×)",  color="#EF9F27", alpha=0.85)
    ax.bar(x, full, bottom=[l+m for l,m in zip(low, mid)],
           label="Full (1×)", color="#534AB7", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", "\n") for g in groups], fontsize=9)
    ax.set_ylabel("Percentage of queries (%)"); ax.legend(loc="upper right")
    ax.set_title("Resolution Level Distribution by Query Type")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    savefig("fig5_keyword_analysis.pdf")


# ── Fig 6: SL vs RL comparison ────────────────────────────────────────────────

def plot_sl_vs_rl():
    main_path = os.path.join(RESULT_DIR, "main_results.json")
    if not os.path.exists(main_path):
        print("[SKIP] main_results.json not found"); return

    with open(main_path) as f:
        all_results = json.load(f)

    # Use textvqa_val if available, else first split
    rows = all_results.get("textvqa_val", list(all_results.values())[0])
    row_map = {r["method"]: r for r in rows}

    methods = ["qadc_sl"] + [f"qadc_rl_lam{l}" for l in [0.1, 0.3, 0.5]]
    labels  = ["SL", "RL λ=0.1", "RL λ=0.3", "RL λ=0.5"]
    accs    = [row_map.get(m, {}).get("vqa_accuracy",     0) for m in methods]
    toks    = [row_map.get(m, {}).get("token_reduction%", 0) for m in methods]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w/2, accs, w, label="VQA accuracy (%)", color="#534AB7", alpha=0.85)
    b2 = ax.bar(x + w/2, toks, w, label="Token reduction (%)", color="#D85A30", alpha=0.85)
    ax.bar_label(b1, fmt="%.1f", fontsize=9, padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=9, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage (%)"); ax.legend()
    ax.set_title("SL vs RL Training: Accuracy & Efficiency")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    savefig("fig6_sl_vs_rl.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures ...")
    plot_main_results()
    plot_training_curves()
    plot_fusion_ablation()
    plot_levels_ablation()
    plot_keyword_analysis()
    plot_sl_vs_rl()
    print(f"\nAll figures saved to: {FIG_DIR}")
