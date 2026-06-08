from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import build_candidate_mask, ensure_nonempty_mask, masked_softmax, set_random_seed, validate_device

@dataclass
class AttentionGraphOutput:
    node_embedding: torch.Tensor
    alpha: torch.Tensor
    neighbor_embedding: torch.Tensor
    graph_embedding: torch.Tensor
    directed_edge_weight: torch.Tensor
    undirected_edge_weight: Optional[torch.Tensor]

class NodeEmbedding(nn.Module):

    def __init__(self, seq_dim: int, static_dim: int, sales_hidden_dim: int=64, char_hidden_dim: int=64, embed_dim: int=32, dropout: float=0.1) -> None:
        super().__init__()
        self.seq_dim = seq_dim
        self.static_dim = static_dim
        self.embed_dim = embed_dim
        self.sales_encoder = nn.GRU(input_size=seq_dim, hidden_size=sales_hidden_dim, batch_first=True)
        self.sales_proj = nn.Sequential(nn.Linear(sales_hidden_dim, embed_dim), nn.ReLU())
        self.has_static = static_dim > 0
        if self.has_static:
            self.char_encoder = nn.Sequential(nn.Linear(static_dim, char_hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(char_hidden_dim, embed_dim), nn.ReLU())
            psi_in_dim = embed_dim * 2
        else:
            self.char_encoder = None
            psi_in_dim = embed_dim
        self.psi = nn.Sequential(nn.Linear(psi_in_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim))

    def forward(self, seq_x: torch.Tensor, static_x: Optional[torch.Tensor]=None) -> torch.Tensor:
        if seq_x.dim() != 3:
            raise ValueError(f'seq_x must have shape [N, T, F], got {tuple(seq_x.shape)}')
        _, h_n = self.sales_encoder(seq_x)
        h_sales = self.sales_proj(h_n[-1])
        if self.has_static:
            if static_x is None:
                raise ValueError('static_x is required when static_dim > 0')
            if static_x.dim() != 2:
                raise ValueError(f'static_x must have shape [N, F_static], got {tuple(static_x.shape)}')
            h_char = self.char_encoder(static_x)
            h = self.psi(torch.cat([h_sales, h_char], dim=-1))
        else:
            h = self.psi(h_sales)
        return h

class AttentionSim(nn.Module):

    def __init__(self, embed_dim: int, attn_dim: Optional[int]=None, negative_slope: float=0.2) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim or embed_dim
        self.query_proj = nn.Linear(embed_dim, self.attn_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, self.attn_dim, bias=False)
        self.attn_vector = nn.Parameter(torch.empty(self.attn_dim * 2))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.attn_vector.unsqueeze(0))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(h)
        k = self.key_proj(h)
        n = h.shape[0]
        qi = q[:, None, :].expand(n, n, self.attn_dim)
        kj = k[None, :, :].expand(n, n, self.attn_dim)
        pair = torch.cat([qi, kj], dim=-1)
        logits = torch.matmul(pair, self.attn_vector)
        return self.leaky_relu(logits)

class AttentionNeighborAggregator(nn.Module):

    def __init__(self, embed_dim: int, graph_dim: Optional[int]=None, dropout: float=0.1, add_self_for_empty_rows: bool=True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.graph_dim = graph_dim or embed_dim
        self.add_self_for_empty_rows = add_self_for_empty_rows
        self.update = nn.Sequential(nn.Linear(embed_dim * 2, self.graph_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(self.graph_dim, self.graph_dim))

    def forward(self, h: torch.Tensor, logits: torch.Tensor, candidate_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if candidate_mask.dtype != torch.bool:
            candidate_mask = candidate_mask.bool()
        if candidate_mask.shape != logits.shape:
            raise ValueError(f'candidate_mask shape {tuple(candidate_mask.shape)} must match logits shape {tuple(logits.shape)}')
        mask = candidate_mask.clone()
        if self.add_self_for_empty_rows:
            mask = ensure_nonempty_mask(mask, mode='self')
        alpha = masked_softmax(logits, mask, dim=1)
        h_neigh = alpha @ h
        h_graph = self.update(torch.cat([h, h_neigh], dim=-1))
        return {'alpha': alpha, 'neighbor_embedding': h_neigh, 'graph_embedding': h_graph}

class AttentionGraphConv(nn.Module):

    def __init__(self, seq_dim: int, static_dim: int, sales_hidden_dim: int=64, char_hidden_dim: int=64, embed_dim: int=32, attn_dim: Optional[int]=None, graph_dim: Optional[int]=None, dropout: float=0.1, normalize_node_embedding: bool=False) -> None:
        super().__init__()
        self.normalize_node_embedding = normalize_node_embedding
        self.node_embedding = PaperNodeEmbedding(seq_dim=seq_dim, static_dim=static_dim, sales_hidden_dim=sales_hidden_dim, char_hidden_dim=char_hidden_dim, embed_dim=embed_dim, dropout=dropout)
        self.attention_scorer = PaperAttentionScorer(embed_dim=embed_dim, attn_dim=attn_dim)
        self.aggregator = AttentionNeighborAggregator(embed_dim=embed_dim, graph_dim=graph_dim, dropout=dropout)

    def forward(self, seq_x: torch.Tensor, static_x: Optional[torch.Tensor], candidate_mask: torch.Tensor, return_undirected_edge_weight: bool=False) -> AttentionGraphOutput:
        h = self.node_embedding(seq_x, static_x)
        if self.normalize_node_embedding:
            h = F.normalize(h, p=2, dim=-1)
        logits = self.attention_scorer(h)
        agg = self.aggregator(h, logits, candidate_mask)
        alpha = agg['alpha']
        undirected = None
        if return_undirected_edge_weight:
            undirected = 0.5 * (alpha + alpha.transpose(0, 1))
        return AttentionGraphOutput(node_embedding=h, alpha=alpha, neighbor_embedding=agg['neighbor_embedding'], graph_embedding=agg['graph_embedding'], directed_edge_weight=alpha, undirected_edge_weight=undirected)


def main() -> None:
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    validate_device(args.device)
if __name__ == '__main__':
    main()
