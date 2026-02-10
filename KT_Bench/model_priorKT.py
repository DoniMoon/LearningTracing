from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn


def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


@dataclass
class InitArtifacts:
    pi: torch.Tensor
    num_items: int


def initialize(
    train_path: str,
    num_items: Optional[int] = None,
    user_col: str = "user_id",
    item_col: str = "item_id",
    label_col: str = "correct",
    sep: str = "\t",
) -> InitArtifacts:
    df = pd.read_csv(train_path, sep=sep)

    if num_items is None:
        num_items = int(df[item_col].max() + 1)

    if user_col not in df.columns:
        raise ValueError(f"Column '{user_col}' missing.")
        
    df_first = df.drop_duplicates(subset=[user_col, item_col], keep='first')

    if len(df_first) > 0:
        global_prior = float(df_first[label_col].mean())
    else:
        global_prior = 0.5

    item_means = df_first.groupby(item_col)[label_col].mean().to_dict()

    pi = torch.full((num_items,), global_prior, dtype=torch.float32)

    for i in range(num_items):
        if i in item_means:
            pi[i] = float(item_means[i])

    return InitArtifacts(pi=pi, num_items=num_items)


class PriorKT(nn.Module):
    def __init__(
        self,
        pi: torch.Tensor,
        rank: int = 128,
        pad_id: int = 0,
        init_embed_std: float = 1e-3,
    ):
        super().__init__()
        assert pi.ndim == 1
        self.num_items = int(pi.shape[0])
        self.rank = int(rank)
        self.pad_id = int(pad_id)
        
        self.emb_size = self.num_items + 1

        # Term 1: Fixed Prior logit(p_i)
        self.register_buffer("pi", pi.float(), persistent=True)

        # Term 2: Relevance beta_{ik}
        self.beta_q = nn.Embedding(self.emb_size, self.rank, padding_idx=self.pad_id)
        self.beta_k = nn.Embedding(self.emb_size, self.rank, padding_idx=self.pad_id)

        # Term 3: Evidence Delta
        self.delta_response = nn.Embedding(self.emb_size, self.rank, padding_idx=self.pad_id)
        self.delta_plus_k = nn.Embedding(self.emb_size, self.rank, padding_idx=self.pad_id)
        self.delta_minus_k = nn.Embedding(self.emb_size, self.rank, padding_idx=self.pad_id)

        self._init_parameters(init_embed_std)

    def _init_parameters(self, init_embed_std: float = 1e-3):
        for mod in [self.beta_q, self.beta_k, self.delta_response, self.delta_plus_k, self.delta_minus_k]:
            nn.init.normal_(mod.weight, mean=0.0, std=init_embed_std)

    def forward(
        self,
        hist_indices: torch.LongTensor,
        hist_values: torch.FloatTensor,
        target_items: torch.LongTensor,
    ) -> torch.Tensor:
        
        batch_size, hist_len = hist_indices.shape
        device = target_items.device

        # Mask for valid history
        mask = (hist_indices != 0)

        # 1. Prior Term: logit(p_i)
        # Shift back to 0-index for lookup
        p_i = self.pi[target_items - 1] 
        prior_term = _safe_logit(p_i).to(device) # (B,)

        if hist_len == 0:
            return prior_term

        # 2. Relevance: beta_{ik}
        q_beta = self.beta_q(target_items)       # (B, R)
        k_beta = self.beta_k(hist_indices)       # (B, H, R)

        # scaled-dot product attention -> low-rank adaptation.
        attn_scores = torch.einsum("br,bhr->bh", q_beta, k_beta) / math.sqrt(self.rank)
        attn_scores = attn_scores.masked_fill(~mask, -1e4)
        beta = torch.softmax(attn_scores, dim=1) # (B, H)

        # 3. Evidence: Delta
        q_delta = self.delta_response(target_items) # (B, R)
        
        # Delta+
        k_delta_plus = self.delta_plus_k(hist_indices) 
        delta_plus_val = torch.einsum("br,bhr->bh", q_delta, k_delta_plus) # (B, H)
        
        # Delta-
        k_delta_minus = self.delta_minus_k(hist_indices)
        delta_minus_val = torch.einsum("br,bhr->bh", q_delta, k_delta_minus) # (B, H)

        # 4. Aggregation
        # a_k switching: +1.0 -> Correct, -1.0 -> Wrong, 0.0 -> Pad
        is_correct = (hist_values > 0.5).float()
        is_wrong   = (hist_values < -0.5).float()
        
        # Law of Total Evidence logic
        evidence_term = (is_correct * delta_plus_val) + (is_wrong * delta_minus_val)
        
        history_update = torch.sum(beta * evidence_term, dim=1) # (B,)

        final_logit = prior_term + history_update
        
        return final_logit

    @torch.no_grad()
    def diagnostics(self) -> dict:
        return {
            "num_items": self.num_items,
            "rank": self.rank,
            "architecture": "LogOdds-Additive-Exact-Formula",
        }