from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import NEG_INF, causal_mask, set_random_seed, validate_device

@dataclass
class IRPEAttentionOutput:
    attended_tokens: torch.Tensor
    pair_bucket_ids: torch.Tensor
    pair_last_sales: torch.Tensor
    attention_weights: torch.Tensor

def build_key_history_state(raw_sales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if raw_sales.dim() != 2:
        raise ValueError(f'raw_sales must have shape [B, T], got {tuple(raw_sales.shape)}')
    batch_size, seq_len = raw_sales.shape
    last_nonzero_idx = torch.full((batch_size,), -1, dtype=torch.long, device=raw_sales.device)
    last_nonzero_sales = torch.zeros(batch_size, dtype=raw_sales.dtype, device=raw_sales.device)
    has_history_tensor = torch.empty(batch_size, seq_len, dtype=torch.bool, device=raw_sales.device)
    last_sales_tensor = torch.empty(batch_size, seq_len, dtype=raw_sales.dtype, device=raw_sales.device)
    for t in range(seq_len):
        has_history = last_nonzero_idx >= 0
        has_history_tensor[:, t] = has_history
        last_sales_tensor[:, t] = torch.where(has_history, last_nonzero_sales, torch.zeros_like(last_nonzero_sales))
        current_sales = raw_sales[:, t]
        current_nonzero = current_sales > 0
        last_nonzero_idx = torch.where(current_nonzero, torch.full_like(last_nonzero_idx, t), last_nonzero_idx)
        last_nonzero_sales = torch.where(current_nonzero, current_sales, last_nonzero_sales)
    return (has_history_tensor, last_sales_tensor)

def build_irpe_inputs(raw_sales: torch.Tensor, max_interval: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    has_key_history, key_last_sales = build_key_history_state(raw_sales)
    batch_size, seq_len = raw_sales.shape
    query_pos = torch.arange(seq_len, device=raw_sales.device)[:, None]
    key_pos = torch.arange(seq_len, device=raw_sales.device)[None, :]
    distance = query_pos - key_pos
    mask = causal_mask(seq_len, device=raw_sales.device)
    overflow_bucket = max_interval + 2
    clipped_bucket = distance.clamp(min=0, max=max_interval) + 1
    clipped_bucket = torch.where(distance > max_interval, torch.full_like(clipped_bucket, overflow_bucket), clipped_bucket)
    clipped_bucket = clipped_bucket.to(torch.long)
    pair_bucket = clipped_bucket.unsqueeze(0).expand(batch_size, seq_len, seq_len).clone()
    has_history_pair = has_key_history[:, None, :].expand(batch_size, seq_len, seq_len)
    pair_bucket = torch.where(has_history_pair, pair_bucket, torch.zeros_like(pair_bucket))
    pair_bucket = pair_bucket.masked_fill(~mask[None, :, :], 0)
    pair_last_sales = key_last_sales[:, None, :].expand(batch_size, seq_len, seq_len)
    pair_last_sales = pair_last_sales.masked_fill(~mask[None, :, :], 0.0)
    return (pair_bucket, pair_last_sales, mask)

class IntermittentDemandAttention(nn.Module):

    def __init__(self, token_dim: int, attn_dim: int=64, num_heads: int=4, dropout: float=0.2, max_interval: int=30) -> None:
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError('attn_dim must be divisible by num_heads.')
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.max_interval = max_interval
        self.input_norm = nn.LayerNorm(token_dim)
        self.q_proj = nn.Linear(token_dim, attn_dim)
        self.k_proj = nn.Linear(token_dim, attn_dim)
        self.v_proj = nn.Linear(token_dim, attn_dim)
        self.interval_embedding = nn.Embedding(max_interval + 3, attn_dim)
        self.sales_proj = nn.Sequential(nn.Linear(1, attn_dim), nn.SiLU(), nn.Linear(attn_dim, attn_dim))
        self.rel_proj = nn.Linear(attn_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, raw_sales: torch.Tensor) -> PairwiseIRPEAttentionOutput:
        if tokens.dim() != 3:
            raise ValueError(f'tokens must have shape [B, T, D], got {tuple(tokens.shape)}')
        if raw_sales.shape != tokens.shape[:2]:
            raise ValueError(f'raw_sales must have shape {tuple(tokens.shape[:2])}, got {tuple(raw_sales.shape)}')
        residual = tokens
        tokens = self.input_norm(tokens)
        pair_bucket_ids, pair_last_sales, causal_mask = build_pairwise_irpe_inputs(raw_sales, self.max_interval)
        pair_irpe = self.interval_embedding(pair_bucket_ids) + self.sales_proj(torch.log1p(pair_last_sales).unsqueeze(-1))
        pair_rel = self.rel_proj(pair_irpe)
        q = self.q_proj(tokens)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)
        batch_size, seq_len, _ = q.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        pair_rel = pair_rel.view(batch_size, seq_len, seq_len, self.num_heads, self.head_dim)
        pair_rel = pair_rel.permute(0, 3, 1, 2, 4).contiguous()
        content_scores = torch.matmul(q, k.transpose(-2, -1))
        relative_scores = torch.einsum('bhid,bhijd->bhij', q, pair_rel)
        logits = (content_scores + relative_scores) / self.head_dim ** 0.5
        logits = logits.masked_fill(~causal_mask[None, None, :, :], NEG_INF)
        attention_weights = torch.softmax(logits, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attended = torch.matmul(attention_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attended_tokens = residual + self.dropout(self.out_proj(attended))
        return IRPEAttentionOutput(attended_tokens, pair_bucket_ids, pair_last_sales, attention_weights)


def main() -> None:
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
if __name__ == '__main__':
    main()
