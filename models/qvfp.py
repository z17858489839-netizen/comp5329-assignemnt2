"""
models/qvfp.py
==============
Query-Visual Fusion Predictor (QVFP).

Supports four fusion modes (used in ablation studies):
  - "concat"      : concat(q, v) → MLP → 3-way classifier  [default]
  - "cross_attn"  : query attends over visual tokens → MLP → classifier
  - "query_only"  : only query embedding → MLP → classifier
  - "image_only"  : only visual embedding → MLP → classifier

All modes share the same MLP head architecture so comparisons are fair.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    QUERY_DIM, VISUAL_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2,
    DROPOUT, NUM_LEVELS,
)


# ── Shared MLP head ───────────────────────────────────────────────────────────

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = NUM_LEVELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Main QVFP module ──────────────────────────────────────────────────────────

class QVFP(nn.Module):
    """
    Query-Visual Fusion Predictor.

    Input:
        q_emb : (B, QUERY_DIM)   — mean-pooled text embedding
        v_emb : (B, VISUAL_DIM)  — mean-pooled 1/4-res visual embedding

    Output:
        logits : (B, NUM_LEVELS) — unnormalised class scores
    """

    def __init__(
        self,
        fusion_mode: str = "concat",
        num_levels: int = NUM_LEVELS,
        normalize_inputs: bool = True,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.num_levels  = num_levels
        self.normalize_inputs = normalize_inputs

        if fusion_mode == "concat":
            self.head = MLPHead(QUERY_DIM + VISUAL_DIM, num_levels)

        elif fusion_mode == "cross_attn":
            self.q_proj    = nn.Linear(QUERY_DIM,  256)
            self.v_proj    = nn.Linear(VISUAL_DIM, 256)
            self.attn_scale = 256 ** -0.5
            self.head      = MLPHead(256, num_levels)

        elif fusion_mode == "query_only":
            self.head = MLPHead(QUERY_DIM, num_levels)

        elif fusion_mode == "image_only":
            self.head = MLPHead(VISUAL_DIM, num_levels)

        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _clean_norm(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return F.layer_norm(x, (x.shape[-1],))

    def forward(
        self,
        q_emb: torch.Tensor,       # (B, QUERY_DIM)
        v_emb: torch.Tensor,       # (B, VISUAL_DIM)
    ) -> torch.Tensor:             # (B, num_levels)
        if self.normalize_inputs:
            q_emb = self._clean_norm(q_emb)
            v_emb = self._clean_norm(v_emb)

        if self.fusion_mode == "concat":
            fused = torch.cat([q_emb, v_emb], dim=-1)
            return self.head(fused)

        elif self.fusion_mode == "cross_attn":
            q = self.q_proj(q_emb).unsqueeze(1)     # (B, 1, 256)
            v = self.v_proj(v_emb).unsqueeze(1)     # (B, 1, 256)
            attn = torch.bmm(q, v.transpose(1, 2)) * self.attn_scale  # (B, 1, 1)
            attn = F.softmax(attn, dim=-1)
            out  = torch.bmm(attn, v).squeeze(1)    # (B, 256)
            return self.head(out)

        elif self.fusion_mode == "query_only":
            return self.head(q_emb)

        elif self.fusion_mode == "image_only":
            return self.head(v_emb)

    def predict(
        self,
        q_emb: torch.Tensor,
        v_emb: torch.Tensor,
    ) -> torch.Tensor:             # (B,) int64 predicted level
        with torch.no_grad():
            logits = self.forward(q_emb, v_emb)
        return logits.argmax(dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── QVFP Dataset ─────────────────────────────────────────────────────────────

class QVFPDataset(torch.utils.data.Dataset):
    """
    Loads pre-computed embeddings + pseudo labels for QVFP training.
    """

    def __init__(
        self,
        embeddings_file: str,
        labels_file: str,
        sample_ids: list,
        id_to_emb_idx: dict,
    ):
        import json
        emb_data = torch.load(embeddings_file, map_location="cpu", weights_only=False)
        self.q_embs = emb_data["q_embs"].float()    # (N_total, D_q)
        self.v_embs = emb_data["v_embs"].float()    # (N_total, D_v)

        with open(labels_file) as f:
            all_labels = json.load(f)

        # Build aligned lists for the requested split
        self.q_list, self.v_list, self.y_list = [], [], []
        for sid in sample_ids:
            sid_str = str(sid)
            if sid_str not in all_labels or sid_str not in id_to_emb_idx:
                continue
            idx = id_to_emb_idx[sid_str]
            self.q_list.append(self.q_embs[idx])
            self.v_list.append(self.v_embs[idx])
            self.y_list.append(int(all_labels[sid_str]))

        self.q_tensor = torch.stack(self.q_list)   # (N, D_q)
        self.v_tensor = torch.stack(self.v_list)   # (N, D_v)
        self.y_tensor = torch.tensor(self.y_list, dtype=torch.long)

    def __len__(self):
        return len(self.y_tensor)

    def __getitem__(self, idx):
        return self.q_tensor[idx], self.v_tensor[idx], self.y_tensor[idx]


def build_datasets(label_dir: str, train_ratio: float = 0.85):
    """
    Split the pseudo-labelled data into train / val sets.
    Returns (train_dataset, val_dataset).
    """
    import json, random
    from config import RANDOM_SEED

    labels_file     = os.path.join(label_dir, "pseudo_labels.json")
    embeddings_file = os.path.join(label_dir, "embeddings.pt")
    progress_file   = os.path.join(label_dir, "progress.json")

    with open(labels_file)  as f: labels   = json.load(f)
    with open(progress_file) as f: progress = json.load(f)

    id_to_emb_idx = {sid: info["emb_idx"] for sid, info in progress.items()}

    all_ids = list(labels.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(all_ids)

    split = int(len(all_ids) * train_ratio)
    train_ids = all_ids[:split]
    val_ids   = all_ids[split:]

    train_ds = QVFPDataset(embeddings_file, labels_file, train_ids, id_to_emb_idx)
    val_ds   = QVFPDataset(embeddings_file, labels_file, val_ids,   id_to_emb_idx)

    return train_ds, val_ds
