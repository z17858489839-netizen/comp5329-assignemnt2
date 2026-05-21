"""
training/train_sl.py  —  Step 2a
==================================
Trains QVFP with standard cross-entropy loss on pseudo labels.
This is the main (SL) method.

Saves best checkpoint to CKPT_DIR/qvfp_sl_best.pt
Saves training curves to RESULT_DIR/sl_training_curve.json

Runtime: < 10 minutes on T4 for 5000 samples, 20 epochs.
"""

import os, sys, json, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LABEL_DIR, CKPT_DIR, RESULT_DIR,
    SL_LR, SL_EPOCHS, SL_BATCH_SIZE, SL_WEIGHT_DECAY,
    NUM_LEVELS,
)
from models.qvfp import QVFP, build_datasets


def format_classification_report(y_true, y_pred, labels, target_names) -> str:
    lines = ["              precision    recall  f1-score   support"]
    total_correct = 0
    total_support = 0
    weighted = [0.0, 0.0, 0.0]
    macro = [0.0, 0.0, 0.0]
    for label, name in zip(labels, target_names):
        tp = sum(yt == label and yp == label for yt, yp in zip(y_true, y_pred))
        pred_count = sum(yp == label for yp in y_pred)
        support = sum(yt == label for yt in y_true)
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        lines.append(f"{name:>14} {precision:10.2f} {recall:9.2f} {f1:9.2f} {support:9d}")
        total_correct += tp
        total_support += support
        macro[0] += precision
        macro[1] += recall
        macro[2] += f1
        weighted[0] += precision * support
        weighted[1] += recall * support
        weighted[2] += f1 * support

    n = max(len(labels), 1)
    accuracy = total_correct / total_support if total_support else 0.0
    lines.append("")
    lines.append(f"{'accuracy':>14} {'':10} {'':9} {accuracy:9.2f} {total_support:9d}")
    lines.append(f"{'macro avg':>14} {macro[0]/n:10.2f} {macro[1]/n:9.2f} {macro[2]/n:9.2f} {total_support:9d}")
    if total_support:
        lines.append(f"{'weighted avg':>14} {weighted[0]/total_support:10.2f} {weighted[1]/total_support:9.2f} {weighted[2]/total_support:9.2f} {total_support:9d}")
    return "\n".join(lines)


def remap_dataset_levels(ds, num_levels: int):
    """Map 3-level pseudo labels to the requested ablation label space."""
    if num_levels == NUM_LEVELS:
        return ds

    if num_levels == 2:
        y = torch.where(ds.y_tensor == 0, torch.zeros_like(ds.y_tensor), torch.ones_like(ds.y_tensor))
    elif num_levels == 4:
        mapping = torch.tensor([0, 2, 3], dtype=torch.long)
        y = mapping[ds.y_tensor.clamp(0, 2)]
    else:
        raise ValueError(f"Unsupported num_levels={num_levels}; expected 2, 3, or 4")

    clone = torch.utils.data.TensorDataset(ds.q_tensor, ds.v_tensor, y.long())
    clone.q_tensor = ds.q_tensor
    clone.v_tensor = ds.v_tensor
    clone.y_tensor = y.long()
    return clone


def make_class_weights(labels: torch.Tensor, num_levels: int, device: str) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_levels).float()
    weights = counts.sum() / (num_levels * counts.clamp_min(1))
    weights[counts == 0] = 0.0
    return weights.clamp(max=4.0).to(device)


def balanced_accuracy(preds: list, labels: list, num_levels: int) -> float:
    recalls = []
    for level in range(num_levels):
        support = sum(y == level for y in labels)
        if support == 0:
            continue
        correct = sum(p == level and y == level for p, y in zip(preds, labels))
        recalls.append(correct / support)
    return sum(recalls) / max(len(recalls), 1) * 100


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for q, v, y in loader:
        q, v, y = q.to(device), v.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(q, v)
        loss   = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite SL loss; check embeddings and input normalisation.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total * 100


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for q, v, y in loader:
        q, v, y = q.to(device), v.to(device), y.to(device)
        logits = model(q, v)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total   += len(y)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    return total_loss / total, correct / total * 100, all_preds, all_labels


def train_qvfp_sl(
    fusion_mode: str = "concat",
    num_levels:  int = NUM_LEVELS,
    save_suffix: str = "",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training QVFP-SL | fusion={fusion_mode} | levels={num_levels} | device={device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds, val_ds = build_datasets(LABEL_DIR)
    train_ds = remap_dataset_levels(train_ds, num_levels)
    val_ds = remap_dataset_levels(val_ds, num_levels)
    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=SL_BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=SL_BATCH_SIZE * 2,
                              shuffle=False, num_workers=0, pin_memory=pin)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = QVFP(
        fusion_mode=fusion_mode,
        num_levels=num_levels,
        normalize_inputs=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=SL_LR, weight_decay=SL_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SL_EPOCHS
    )
    criterion = nn.CrossEntropyLoss(
        weight=make_class_weights(train_ds.y_tensor, num_levels, device)
    )

    print(f"QVFP parameters: {model.count_parameters():,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    history   = []
    best_acc  = 0.0
    best_score = -1.0
    ckpt_name = f"qvfp_sl_{fusion_mode}_{num_levels}lvl{save_suffix}_best.pt"
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)

    for epoch in range(1, SL_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, preds, labels = eval_epoch(model, val_loader, criterion, device)
        va_bal = balanced_accuracy(preds, labels, num_levels)
        scheduler.step()

        elapsed = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc":  round(tr_acc,  2),
            "val_loss":   round(va_loss, 4),
            "val_acc":    round(va_acc,  2),
            "val_bal_acc": round(va_bal, 2),
        })

        print(f"Epoch {epoch:02d}/{SL_EPOCHS} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.1f}% | "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.1f}% "
              f"va_bal={va_bal:.1f}% | "
              f"{elapsed:.1f}s")

        if va_bal > best_score:
            best_acc = va_acc
            best_score = va_bal
            torch.save({
                "model_state":  model.state_dict(),
                "fusion_mode":  fusion_mode,
                "num_levels":   num_levels,
                "normalize_inputs": True,
                "epoch":        epoch,
                "val_acc":      va_acc,
                "val_bal_acc":  va_bal,
            }, ckpt_path)
            print(f"  → Saved best checkpoint (val_acc={va_acc:.1f}%, val_bal={va_bal:.1f}%)")

    # ── Final classification report ───────────────────────────────────────────
    print("\n=== Final Validation Classification Report ===")
    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    _, _, preds, labels = eval_epoch(model, val_loader, criterion, device)
    if num_levels == 2:
        target_names = ["low (1/4)", "full (1x)"]
    elif num_levels == 4:
        target_names = ["low (1/4)", "low_mid (1/3)", "mid_high (2/3)", "full (1x)"]
    else:
        target_names = ["low (1/4)", "mid (1/2)", "full (1x)"]
    print(format_classification_report(
        labels, preds, labels=list(range(num_levels)), target_names=target_names,
    ))

    # ── Save history ──────────────────────────────────────────────────────────
    curve_name = f"sl_curve_{fusion_mode}_{num_levels}lvl{save_suffix}.json"
    curve_path = os.path.join(RESULT_DIR, curve_name)
    with open(curve_path, "w") as f:
        json.dump({
            "history": history,
            "best_val_acc": best_acc,
            "best_val_bal_acc": best_score,
        }, f, indent=2)
    print(f"Training curve → {curve_path}")
    print(f"Best val acc   : {best_acc:.1f}%")
    print(f"Best val bal   : {best_score:.1f}%")
    return best_acc, ckpt_path


if __name__ == "__main__":
    # Train main model
    train_qvfp_sl(fusion_mode="concat", num_levels=3)
