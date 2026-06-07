from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import build_candidate_mask, ensure_nonempty_mask, masked_softmax, set_random_seed, validate_device
from attention_graph import AttentionGraphOutput, EndToEndAttentionGraphConv

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
        self.similarity_graph = EndToEndAttentionGraphConv(seq_dim=similarity_seq_dim, static_dim=similarity_static_dim, sales_hidden_dim=sales_hidden_dim, char_hidden_dim=char_hidden_dim, embed_dim=similarity_embed_dim, attn_dim=similarity_attn_dim, graph_dim=similarity_graph_dim, dropout=dropout)
        self.fusion = SimilarityTemporalMultiHeadAttention(sim_dim=similarity_graph_dim, temporal_dim=temporal_dim, fusion_dim=fusion_dim, num_heads=fusion_heads, dropout=dropout)

    def forward(self, similarity_seq_x: torch.Tensor, similarity_static_x: Optional[torch.Tensor], candidate_mask: torch.Tensor, temporal_emb: torch.Tensor, temporal_mask: Optional[torch.Tensor]=None, return_undirected_similarity_edge_weight: bool=False) -> MultiGraphFusionOutput:
        sim_output: AttentionGraphOutput = self.similarity_graph(similarity_seq_x, similarity_static_x, candidate_mask, return_undirected_edge_weight=return_undirected_similarity_edge_weight)
        fused, fusion_attention = self.fusion(similarity_emb=sim_output.graph_embedding, temporal_emb=temporal_emb, temporal_mask=temporal_mask)
        return MultiGraphFusionOutput(fused_embedding=fused, similarity_graph_embedding=sim_output.graph_embedding, similarity_attention=sim_output.alpha, fusion_attention=fusion_attention, undirected_similarity_edge_weight=sim_output.undirected_edge_weight)

def smoke_test(device: str='cpu') -> None:
    set_random_seed()
    n = 10
    sim_seq_len = 8
    sim_seq_dim = 3
    static_dim = 5
    temporal_slices = 6
    temporal_dim = 12
    fusion_dim = 16
    model = EndToEndMultiGraphFusion(similarity_seq_dim=sim_seq_dim, similarity_static_dim=static_dim, temporal_dim=temporal_dim, sales_hidden_dim=16, char_hidden_dim=12, similarity_embed_dim=10, similarity_attn_dim=8, similarity_graph_dim=14, fusion_dim=fusion_dim, fusion_heads=4, dropout=0.0).to(device)
    similarity_seq_x = torch.randn(n, sim_seq_len, sim_seq_dim, device=device)
    similarity_static_x = torch.randn(n, static_dim, device=device)
    prior = torch.rand(n, n, device=device)
    target = torch.rand(n, n, device=device)
    candidate_mask = build_candidate_mask(prior, target, top_k=3)
    temporal_emb = torch.randn(n, temporal_slices, temporal_dim, device=device)
    temporal_mask = torch.ones(n, temporal_slices, dtype=torch.bool, device=device)
    temporal_mask[0, :2] = False
    temporal_mask[1, :] = False
    output = model(similarity_seq_x=similarity_seq_x, similarity_static_x=similarity_static_x, candidate_mask=candidate_mask, temporal_emb=temporal_emb, temporal_mask=temporal_mask, return_undirected_similarity_edge_weight=True)
    assert output.fused_embedding.shape == (n, fusion_dim)
    assert output.similarity_graph_embedding.shape == (n, 14)
    assert output.similarity_attention.shape == (n, n)
    assert output.fusion_attention.shape == (n, 4, temporal_slices)
    assert output.undirected_similarity_edge_weight is not None
    assert output.undirected_similarity_edge_weight.shape == (n, n)
    sim_row_sums = output.similarity_attention.sum(dim=1)
    fusion_row_sums = output.fusion_attention.sum(dim=-1)
    assert torch.allclose(sim_row_sums, torch.ones_like(sim_row_sums), atol=1e-06)
    assert torch.allclose(fusion_row_sums, torch.ones_like(fusion_row_sums), atol=1e-06)
    effective_temporal_mask = temporal_mask.clone()
    effective_temporal_mask[~effective_temporal_mask.any(dim=1), -1] = True
    assert torch.all(output.fusion_attention.masked_select(~effective_temporal_mask[:, None, :]) == 0)
    forecast_head = nn.Linear(fusion_dim, 1).to(device)
    y = torch.randn(n, 1, device=device)
    pred = forecast_head(output.fused_embedding)
    loss = F.mse_loss(pred, y)
    loss.backward()
    grad_checks = {'similarity_sales_encoder': model.similarity_graph.node_embedding.sales_encoder.weight_ih_l0.grad, 'similarity_attention_query': model.similarity_graph.attention_scorer.query_proj.weight.grad, 'similarity_graph_update': model.similarity_graph.aggregator.update[0].weight.grad, 'fusion_q': model.fusion.q_linear.weight.grad, 'fusion_k': model.fusion.k_linear.weight.grad, 'fusion_v': model.fusion.v_linear.weight.grad, 'fusion_out': model.fusion.out_linear.weight.grad}
    missing = [name for name, grad in grad_checks.items() if grad is None or not torch.isfinite(grad).all()]
    if missing:
        raise RuntimeError(f'Missing or invalid gradients for: {missing}')
    print('smoke_test ok')
    print(f'loss={float(loss.detach().cpu()):.6f}')
    print(f'similarity_alpha_row_sum_min={float(sim_row_sums.min().detach().cpu()):.6f}')
    print(f'similarity_alpha_row_sum_max={float(sim_row_sums.max().detach().cpu()):.6f}')
    print(f'fusion_attention_row_sum_min={float(fusion_row_sums.min().detach().cpu()):.6f}')
    print(f'fusion_attention_row_sum_max={float(fusion_row_sums.max().detach().cpu()):.6f}')

def main() -> None:
    parser = argparse.ArgumentParser(description='Smoke test end-to-end multi-graph fusion.')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    validate_device(args.device)
    smoke_test(device=args.device)
if __name__ == '__main__':
    main()
