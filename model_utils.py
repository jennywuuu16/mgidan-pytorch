from __future__ import annotations
import random
import sys
from pathlib import Path
import numpy as np
import torch

SEED = 42
NEG_INF = -1000000000.0

def add_repo_paths(current_file):
    script_dir = Path(current_file).parent
    project_root = script_dir.parent
    for path in (project_root, script_dir):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return (project_root, script_dir)

def set_random_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate_device(device):
    if device == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is False')

def causal_mask(seq_len, device=None):
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()

def ensure_nonempty_mask(mask, mode='self'):
    out = mask.clone()
    empty_rows = ~out.any(dim=-1)
    if not empty_rows.any():
        return out
    if out.dim() == 2 and mode == 'self':
        idx = torch.arange(out.shape[0], device=out.device)
        out[empty_rows, idx[empty_rows]] = True
        return out
    if mode == 'last':
        out[empty_rows, -1] = True
        return out
    raise ValueError("mode must be 'self' for square 2D masks or 'last'")

def masked_softmax(logits, mask, dim=-1):
    masked_logits = logits.masked_fill(~mask, NEG_INF)
    weights = torch.softmax(masked_logits, dim=dim)
    weights = weights.masked_fill(~mask, 0.0)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)

def candidate_mask(score, top_k, include_self=False):
    if score.dim() != 2 or score.shape[0] != score.shape[1]:
        raise ValueError(f'score must be square [N, N], got {tuple(score.shape)}')
    n = score.shape[0]
    mask = torch.zeros((n, n), dtype=torch.bool, device=score.device)
    if n == 0:
        return mask
    if top_k <= 0:
        if include_self:
            mask.fill_diagonal_(True)
        return mask
    row_score = score.clone()
    if not include_self:
        row_score.fill_diagonal_(float('-inf'))
    k_eff = min(top_k, n if include_self else max(n - 1, 0))
    if k_eff == 0:
        return mask
    idx = torch.topk(row_score, k=k_eff, dim=1).indices
    mask.scatter_(1, idx, True)
    if not include_self:
        mask.fill_diagonal_(False)
    return mask

