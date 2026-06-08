from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import build_candidate_mask, ensure_nonempty_mask, masked_softmax, set_random_seed, validate_device
from attention_graph import AttentionGraphOutput, AttentionGraphConv

@dataclass
class MultiGraphFusionOutput:
    fused_embedding: torch.Tensor
    similarity_graph_embedding: torch.Tensor
    similarity_attention: torch.Tensor
    fusion_attention: torch.Tensor
    undirected_similarity_edge_weight: Optional[torch.Tensor]

class SimilarityTemporalMultiHeadAttention(nn.Module):
    def __init__(self, sim_dim: int, temporal_dim: int, fusion_dim: int=64, num_heads: int=4, dropout: float=0.1, use_residual: bool=True, use_layer_norm: bool=True) -> None:
        super().__init__()
        if fusion_dim % num_heads != 0:
            raise ValueError('fusion_dim must be divisible by num_heads')
        self.sim_dim = sim_dim
        self.temporal_dim = temporal_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.head_dim = fusion_dim // num_heads
        self.use_residual = use_residual
        self.q_linear = nn.Linear(sim_dim, fusion_dim)
        self.k_linear = nn.Linear(temporal_dim, fusion_dim)
        self.v_linear = nn.Linear(temporal_dim, fusion_dim)
        self.out_linear = nn.Linear(fusion_dim, fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Linear(sim_dim, fusion_dim) if sim_dim != fusion_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(fusion_dim) if use_layer_norm else nn.Identity()

    def forward(self, similarity_emb: torch.Tensor, temporal_emb: torch.Tensor, temporal_mask: Optional[torch.Tensor]=None) -> tuple[torch.Tensor, torch.Tensor]:
        if similarity_emb.dim() != 2:
            raise ValueError(f'similarity_emb must have shape [N, D], got {tuple(similarity_emb.shape)}')
        if temporal_emb.dim() != 3:
            raise ValueError(f'temporal_emb must have shape [N, T, D], got {tuple(temporal_emb.shape)}')
        if similarity_emb.shape[0] != temporal_emb.shape[0]:
            raise ValueError('similarity_emb and temporal_emb must have the same number of nodes')
        n, time_slices, _ = temporal_emb.shape
        if temporal_mask is None:
            temporal_mask = torch.ones((n, time_slices), dtype=torch.bool, device=temporal_emb.device)
        else:
            temporal_mask = temporal_mask.to(device=temporal_emb.device, dtype=torch.bool)
            if temporal_mask.shape != (n, time_slices):
                raise ValueError(f'temporal_mask must have shape {(n, time_slices)}, got {tuple(temporal_mask.shape)}')
        mask = ensure_nonempty_mask(temporal_mask.clone(), mode='last')
        q = self.q_linear(similarity_emb).view(n, self.num_heads, self.head_dim)
        k = self.k_linear(temporal_emb).view(n, time_slices, self.num_heads, self.head_dim)
        v = self.v_linear(temporal_emb).view(n, time_slices, self.num_heads, self.head_dim)
        scores = torch.einsum('nhd,nthd->nht', q, k) / self.head_dim ** 0.5
        attn_weights = masked_softmax(scores, mask[:, None, :], dim=-1)
        attn_weights = self.dropout(attn_weights)
        head_output = torch.einsum('nht,nthd->nhd', attn_weights, v)
        concat_output = head_output.reshape(n, self.fusion_dim)
        fused = self.out_linear(concat_output)
        if self.use_residual:
            fused = fused + self.residual_proj(similarity_emb)
        fused = self.layer_norm(fused)
        return (fused, attn_weights)

class EndToEndMultiGraphFusion(nn.Module):

    def __init__(self, similarity_seq_dim: int, similarity_static_dim: int, temporal_dim: int, sales_hidden_dim: int=64, char_hidden_dim: int=64, similarity_embed_dim: int=32, similarity_attn_dim: Optional[int]=None, similarity_graph_dim: int=32, fusion_dim: int=64, fusion_heads: int=4, dropout: float=0.1) -> None:
        super().__init__()
        self.similarity_graph = AttentionGraphConv(seq_dim=similarity_seq_dim, static_dim=similarity_static_dim, sales_hidden_dim=sales_hidden_dim, char_hidden_dim=char_hidden_dim, embed_dim=similarity_embed_dim, attn_dim=similarity_attn_dim, graph_dim=similarity_graph_dim, dropout=dropout)
        self.fusion = SimilarityTemporalMultiHeadAttention(sim_dim=similarity_graph_dim, temporal_dim=temporal_dim, fusion_dim=fusion_dim, num_heads=fusion_heads, dropout=dropout)

    def forward(self, similarity_seq_x: torch.Tensor, similarity_static_x: Optional[torch.Tensor], candidate_mask: torch.Tensor, temporal_emb: torch.Tensor, temporal_mask: Optional[torch.Tensor]=None, return_undirected_similarity_edge_weight: bool=False) -> MultiGraphFusionOutput:
        sim_output: AttentionGraphOutput = self.similarity_graph(similarity_seq_x, similarity_static_x, candidate_mask, return_undirected_edge_weight=return_undirected_similarity_edge_weight)
        fused, fusion_attention = self.fusion(similarity_emb=sim_output.graph_embedding, temporal_emb=temporal_emb, temporal_mask=temporal_mask)
        return MultiGraphFusionOutput(fused_embedding=fused, similarity_graph_embedding=sim_output.graph_embedding, similarity_attention=sim_output.alpha, fusion_attention=fusion_attention, undirected_similarity_edge_weight=sim_output.undirected_edge_weight)



def main() -> None:
    parser.add_argument('--device', default='cpu', choices=['cuda'])
    args = parser.parse_args()
    validate_device(args.device)
if __name__ == '__main__':
    main()
