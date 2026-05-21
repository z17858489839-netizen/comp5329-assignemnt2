"""
training/train_rl.py  —  Step 2b
==================================
Trains QVFP with Group Relative Policy Optimisation (GRPO).

Key difference from SL: instead of using pre-computed pseudo labels,
the reward is computed ONLINE by running Qwen2.5-VL at the predicted
resolution and checking if the answer is correct, minus an efficiency
penalty proportional to the chosen level.

    R = R_acc - λ * R_cost
    R_acc  = 1 if answer correct else 0
    R_cost = predicted_level / (NUM_LEVELS - 1)   ∈ [0, 1]

GRPO advantage (group-normalised):
    A_i = (R_i - mean(R)) / (std(R) + 1e-8)

Note: RL training is significantly slower than SL because each step
requires running Qwen inference online. Use a smaller subset for RL.

Saves checkpoints to CKPT_DIR/qvfp_rl_lambda{λ}_best.pt
"""

import os, sys, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, LABEL_DIR, CKPT_DIR, RESULT_DIR,
    MODEL_ID, RESOLUTION_RATIOS, MAX_NEW_TOKENS,
    RL_LR, RL_EPOCHS, RL_BATCH_SIZE, RL_GROUP_SIZE,
    RL_LAMBDA_VALUES, NUM_LEVELS,
    resolve_image_path,
)
from models.qvfp import QVFP
from utils.vlm_inference import (
    load_model, resize_image, infer_single,
    is_correct, extract_embeddings,
)

TOKENS_PER_LEVEL = torch.tensor([64, 256, 1024], dtype=torch.float32)


def _normalise_reward_entry(entry):
    if isinstance(entry, list):
        vals = list(entry[:NUM_LEVELS])
        vals += [False] * (NUM_LEVELS - len(vals))
        return [False if v is None else bool(v) for v in vals]
    if isinstance(entry, dict):
        return [bool(entry.get(i, entry.get(str(i), False))) for i in range(NUM_LEVELS)]
    return [False] * NUM_LEVELS


class RewardPolicyDataset(Dataset):
    """Offline reward-cache dataset for stable policy improvement."""
    def __init__(self, samples, id_to_emb, q_embs, v_embs, reward_cache, lam):
        self.rows = []
        cost = (TOKENS_PER_LEVEL[:NUM_LEVELS] / TOKENS_PER_LEVEL[NUM_LEVELS - 1])
        for s in samples:
            sid = str(s["id"])
            if sid not in id_to_emb or sid not in reward_cache:
                continue
            corr = torch.tensor(reward_cache[sid][:NUM_LEVELS], dtype=torch.float32)
            rewards = corr - lam * cost
            target = int(torch.argmax(rewards).item())
            self.rows.append((q_embs[id_to_emb[sid]], v_embs[id_to_emb[sid]], target, rewards))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def reward_collate(batch):
    return {
        "q_embs": torch.stack([b[0] for b in batch]),
        "v_embs": torch.stack([b[1] for b in batch]),
        "targets": torch.tensor([b[2] for b in batch], dtype=torch.long),
        "rewards": torch.stack([b[3] for b in batch]),
    }


def _load_sl_warm_start(device: str) -> dict | None:
    """Use the SL policy as the RL starting point when available."""
    path = os.path.join(CKPT_DIR, f"qvfp_sl_concat_{NUM_LEVELS}lvl_best.pt")
    if not os.path.exists(path):
        print(f"[WARN] SL checkpoint not found, RL will start from scratch: {path}")
        return None
    return torch.load(path, map_location=device, weights_only=False)


@torch.no_grad()
def _eval_reward_policy(model, loader, device):
    model.eval()
    pred_levels, target_levels, rewards = [], [], []
    for batch in loader:
        q = batch["q_embs"].to(device)
        v = batch["v_embs"].to(device)
        logits = model(q, v)
        pred = logits.argmax(dim=-1).cpu()
        pred_levels.extend(pred.tolist())
        target_levels.extend(batch["targets"].tolist())
        rewards.extend(batch["rewards"].gather(1, pred[:, None]).squeeze(1).tolist())
    policy_acc = sum(p == t for p, t in zip(pred_levels, target_levels)) / max(len(target_levels), 1) * 100
    mean_reward = sum(rewards) / max(len(rewards), 1)
    dist = Counter(pred_levels)
    return policy_acc, mean_reward, {
        i: dist.get(i, 0) / max(len(pred_levels), 1) * 100 for i in range(NUM_LEVELS)
    }


def _train_qvfp_rl_from_cache(lam, samples, id_to_emb, q_embs_all, v_embs_all, reward_cache, device):
    dataset = RewardPolicyDataset(samples, id_to_emb, q_embs_all, v_embs_all, reward_cache, lam)
    if len(dataset) == 0:
        print("[WARN] reward cache found but no matching RL samples; falling back to online GRPO.")
        return None

    targets = [row[2] for row in dataset.rows]
    target_counts = Counter(targets)
    print(
        f"Offline RL samples: {len(dataset)} | "
        f"target low={target_counts.get(0,0)} mid={target_counts.get(1,0)} full={target_counts.get(2,0)}"
    )

    loader = DataLoader(dataset, batch_size=RL_BATCH_SIZE, shuffle=True, collate_fn=reward_collate)
    eval_loader = DataLoader(dataset, batch_size=RL_BATCH_SIZE * 2, shuffle=False, collate_fn=reward_collate)

    sl_ckpt = _load_sl_warm_start(device)
    normalize_inputs = True if sl_ckpt is None else sl_ckpt.get("normalize_inputs", False)
    model = QVFP(
        fusion_mode="concat",
        num_levels=NUM_LEVELS,
        normalize_inputs=normalize_inputs,
    ).to(device)
    if sl_ckpt is not None:
        model.load_state_dict(sl_ckpt["model_state"])
        print("Loaded SL checkpoint as RL warm start.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=max(RL_LR, 2e-4), weight_decay=1e-4)
    counts = torch.tensor([target_counts.get(i, 0) for i in range(NUM_LEVELS)], dtype=torch.float32)
    weights = (counts.sum() / (NUM_LEVELS * counts.clamp_min(1))).clamp(max=3.0).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    ckpt_path = os.path.join(CKPT_DIR, f"qvfp_rl_lambda{lam}_best.pt")
    history, best_reward = [], -999

    for epoch in range(1, RL_EPOCHS + 1):
        model.train()
        losses = []
        for batch in tqdm(loader, desc=f"RL offline λ={lam} Epoch {epoch}/{RL_EPOCHS}"):
            q = batch["q_embs"].to(device)
            v = batch["v_embs"].to(device)
            y = batch["targets"].to(device)
            rewards = batch["rewards"].to(device)
            logits = model(q, v)
            probs = F.softmax(logits, dim=-1)
            expected_reward = (probs * rewards).sum(dim=-1).mean()
            loss = criterion(logits, y) - 0.25 * expected_reward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        policy_acc, mean_reward, dist = _eval_reward_policy(model, eval_loader, device)
        mean_loss = sum(losses) / len(losses)
        history.append({
            "epoch": epoch, "loss": round(mean_loss, 4),
            "policy_acc": round(policy_acc, 2), "mean_reward": round(mean_reward, 4),
            "low_pct": round(dist[0], 1), "mid_pct": round(dist[1], 1), "full_pct": round(dist[2], 1),
        })
        print(
            f"Epoch {epoch} | loss={mean_loss:.4f} | policy_acc={policy_acc:.1f}% | "
            f"reward={mean_reward:.4f} | low={dist[0]:.1f}% mid={dist[1]:.1f}% full={dist[2]:.1f}%"
        )

        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save({
                "model_state": model.state_dict(),
                "fusion_mode": "concat",
                "num_levels": NUM_LEVELS,
                "normalize_inputs": normalize_inputs,
                "lambda": lam,
                "mean_reward": mean_reward,
                "policy_acc": policy_acc,
                "target_distribution": dict(target_counts),
            }, ckpt_path)
            print(f"  → Saved best RL checkpoint (reward={mean_reward:.4f})")

    curve_path = os.path.join(RESULT_DIR, f"rl_curve_lambda{lam}.json")
    with open(curve_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"RL curve → {curve_path}")
    return best_reward, ckpt_path


# ── Dataset for RL (needs raw images + answers online) ────────────────────────

class RLDataset(Dataset):
    def __init__(self, samples: list, id_to_emb: dict,
                 q_embs: torch.Tensor, v_embs: torch.Tensor):
        self.samples   = samples
        self.id_to_emb = id_to_emb
        self.q_embs    = q_embs
        self.v_embs    = v_embs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s       = self.samples[idx]
        sid     = str(s["id"])
        emb_idx = self.id_to_emb.get(sid, 0)
        return {
            "q_emb":      self.q_embs[emb_idx],
            "v_emb":      self.v_embs[emb_idx],
            "image_path": resolve_image_path(s["image_path"]),
            "question":   s["question"],
            "answers":    s["answers"],
        }


def rl_collate(batch):
    return {
        "q_embs":      torch.stack([b["q_emb"] for b in batch]),
        "v_embs":      torch.stack([b["v_emb"] for b in batch]),
        "image_paths": [b["image_path"] for b in batch],
        "questions":   [b["question"]   for b in batch],
        "answers":     [b["answers"]    for b in batch],
    }


# ── GRPO update step ──────────────────────────────────────────────────────────

def grpo_step(
    model:         QVFP,
    optimizer:     torch.optim.Optimizer,
    q_embs:        torch.Tensor,
    v_embs:        torch.Tensor,
    rewards:       torch.Tensor,        # (B * G,)
    actions:       torch.Tensor,        # (B * G,)  predicted levels
    lam_kl:        float = 0.01,
    old_log_probs: torch.Tensor = None,
) -> float:
    """Single GRPO gradient step."""
    device  = next(model.parameters()).device
    q_embs  = q_embs.to(device)
    v_embs  = v_embs.to(device)
    rewards = rewards.to(device)
    actions = actions.to(device)

    B_G = rewards.shape[0]
    G   = RL_GROUP_SIZE
    B   = B_G // G

    # rewards is ordered [g0_b0..g0_bB, g1_b0..g1_bB, ...] after torch.cat(all_rewards)
    # Reshape to (B, G) so each row holds G rollout rewards for one sample
    rewards_grouped = rewards.view(G, B).T.contiguous()          # (B, G)
    mean_r          = rewards_grouped.mean(dim=1, keepdim=True)
    std_r           = rewards_grouped.std( dim=1, keepdim=True) + 1e-8
    norm_adv        = (rewards_grouped - mean_r) / std_r         # (B, G)
    # Transpose back so advantages[k] aligns with q_rep[k] (sample k%B, group k//B)
    advantages      = norm_adv.T.contiguous().view(-1)           # (G*B,)

    # repeat(G, 1) gives [b0,b1,...,bB, b0,...] matching the cat order of actions/rewards
    q_rep = q_embs.repeat(G, 1)   # (G*B, D_q)
    v_rep = v_embs.repeat(G, 1)   # (G*B, D_v)

    logits    = model(q_rep, v_rep)
    log_probs = F.log_softmax(logits, dim=-1)
    sel_lp    = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    pg_loss = -(sel_lp * advantages.detach()).mean()
    entropy = -(F.softmax(logits, dim=-1) * log_probs).sum(dim=-1).mean()
    loss    = pg_loss - lam_kl * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


# ── Main RL training loop ─────────────────────────────────────────────────────

def train_qvfp_rl(lam: float = 0.3, rl_samples: int = 1000):
    """
    lam        : efficiency penalty weight λ
    rl_samples : how many training samples to use (keep small: ~1000)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining QVFP-RL | λ={lam} | samples={rl_samples} | device={device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(os.path.join(DATA_DIR, "textvqa_train.json")) as f:
        all_samples = json.load(f)[:rl_samples]

    with open(os.path.join(LABEL_DIR, "progress.json")) as f:
        progress = json.load(f)

    emb_data   = torch.load(
        os.path.join(LABEL_DIR, "embeddings.pt"),
        map_location="cpu", weights_only=False,
    )
    q_embs_all = emb_data["q_embs"].float()
    v_embs_all = emb_data["v_embs"].float()
    id_to_emb  = {sid: info["emb_idx"] for sid, info in progress.items()}

    samples = [s for s in all_samples if str(s["id"]) in id_to_emb]
    print(f"RL training samples: {len(samples)}")

    reward_cache_path = os.path.join(LABEL_DIR, "rl_reward_cache.json")
    if os.path.exists(reward_cache_path):
        with open(reward_cache_path) as f:
            reward_cache = {
                sid: _normalise_reward_entry(vals)
                for sid, vals in json.load(f).items()
            }
        offline_result = _train_qvfp_rl_from_cache(
            lam, samples, id_to_emb, q_embs_all, v_embs_all, reward_cache, device
        )
        if offline_result is not None:
            return offline_result

    dataset = RLDataset(samples, id_to_emb, q_embs_all, v_embs_all)
    loader  = DataLoader(dataset, batch_size=RL_BATCH_SIZE,
                         shuffle=True, collate_fn=rl_collate)

    # ── Load VLM (needed for online reward) ───────────────────────────────────
    vlm, processor = load_model(MODEL_ID)

    # ── QVFP model ────────────────────────────────────────────────────────────
    sl_ckpt = _load_sl_warm_start(device)
    normalize_inputs = True if sl_ckpt is None else sl_ckpt.get("normalize_inputs", False)
    model = QVFP(
        fusion_mode="concat",
        num_levels=NUM_LEVELS,
        normalize_inputs=normalize_inputs,
    ).to(device)
    if sl_ckpt is not None:
        model.load_state_dict(sl_ckpt["model_state"])
        print("Loaded SL checkpoint as RL warm start.")
    optimizer = torch.optim.Adam(model.parameters(), lr=RL_LR)

    history     = []
    best_reward = -999
    ckpt_path   = os.path.join(CKPT_DIR, f"qvfp_rl_lambda{lam}_best.pt")

    for epoch in range(1, RL_EPOCHS + 1):
        epoch_losses, epoch_rewards = [], []

        for batch in tqdm(loader, desc=f"RL Epoch {epoch}/{RL_EPOCHS}"):
            B      = len(batch["questions"])
            q_embs = batch["q_embs"]
            v_embs = batch["v_embs"]

            # ── Sample G rollouts per item ────────────────────────────────────
            all_actions, all_rewards = [], []

            for _ in range(RL_GROUP_SIZE):
                with torch.no_grad():
                    logits  = model(q_embs.to(device), v_embs.to(device))
                    probs   = F.softmax(logits, dim=-1)
                    actions = torch.multinomial(probs, num_samples=1).squeeze(1)

                rewards = []
                for i in range(B):
                    level = actions[i].item()
                    ratio = RESOLUTION_RATIOS[level]
                    try:
                        img   = Image.open(batch["image_paths"][i]).convert("RGB")
                        img_r = resize_image(img, ratio)
                        pred  = infer_single(vlm, processor, img_r,
                                             batch["questions"][i],
                                             max_new_tokens=MAX_NEW_TOKENS)
                        r_acc = float(is_correct(pred, batch["answers"][i]))
                    except Exception:
                        r_acc = 0.0

                    r_cost = level / (NUM_LEVELS - 1)
                    rewards.append(r_acc - lam * r_cost)

                all_actions.append(actions)
                all_rewards.append(torch.tensor(rewards, dtype=torch.float32))

            actions_cat = torch.cat(all_actions, dim=0)   # (B*G,)
            rewards_cat = torch.cat(all_rewards, dim=0)   # (B*G,)

            loss = grpo_step(model, optimizer, q_embs, v_embs,
                             rewards_cat, actions_cat)

            epoch_losses.append(loss)
            epoch_rewards.append(rewards_cat.mean().item())

        mean_loss   = sum(epoch_losses)  / len(epoch_losses)
        mean_reward = sum(epoch_rewards) / len(epoch_rewards)

        history.append({
            "epoch": epoch, "loss": round(mean_loss, 4),
            "mean_reward": round(mean_reward, 4),
        })
        print(f"Epoch {epoch} | loss={mean_loss:.4f} | mean_reward={mean_reward:.4f}")

        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save({
                "model_state": model.state_dict(),
                "fusion_mode": "concat",
                "num_levels":  NUM_LEVELS,
                "normalize_inputs": normalize_inputs,
                "lambda":      lam,
                "epoch":       epoch,
                "mean_reward": mean_reward,
            }, ckpt_path)
            print(f"  → Saved best RL checkpoint (reward={mean_reward:.4f})")

    curve_path = os.path.join(RESULT_DIR, f"rl_curve_lambda{lam}.json")
    with open(curve_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"RL curve → {curve_path}")
    return best_reward, ckpt_path


if __name__ == "__main__":
    for lam in RL_LAMBDA_VALUES:
        train_qvfp_rl(lam=lam, rl_samples=1000)
