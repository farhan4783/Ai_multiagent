"""
Custom edge-attributed Graph Attention Network v2 (GATv2) layers.
Implemented in pure PyTorch for maximum portability.

Supports heterogeneous edge features (e.g., coupling weight, co-churn score)
and computes dynamic spatial attention coefficients α_ij^t.

Reference: Brody et al. "How Attentive are Graph Attention Networks?" (ICLR 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import math


class GATv2Layer(nn.Module):
    """
    Single-head GATv2 attention layer with edge attribute support.

    Computes:
        e_ij = a^T * LeakyReLU(W_s * h_i || W_s * h_j || W_e * e_ij)
        α_ij = softmax_j(e_ij)
        h_i' = σ(Σ_j α_ij * W_s * h_j)

    Args:
        in_features: Input node feature dimension.
        out_features: Output node feature dimension.
        edge_features: Edge attribute dimension (0 to disable edge conditioning).
        dropout: Attention coefficient dropout probability.
        negative_slope: LeakyReLU negative slope.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        edge_features: int = 1,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_features = edge_features
        self.negative_slope = negative_slope

        # Node feature projection
        self.W_src = nn.Linear(in_features, out_features, bias=False)
        self.W_tgt = nn.Linear(in_features, out_features, bias=False)

        # Edge feature projection (if edge features are provided)
        if edge_features > 0:
            self.W_edge = nn.Linear(edge_features, out_features, bias=False)
        else:
            self.W_edge = None

        # Attention vector: maps concatenated projected features to scalar
        attn_input_dim = out_features  # GATv2 uses shared projection space
        self.attn = nn.Linear(attn_input_dim, 1, bias=False)

        self.bias = nn.Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_src.weight)
        nn.init.xavier_uniform_(self.W_tgt.weight)
        if self.W_edge is not None:
            nn.init.xavier_uniform_(self.W_edge.weight)
        nn.init.xavier_uniform_(self.attn.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing GATv2 attention and message passing.

        Args:
            x: Node feature matrix [num_nodes, in_features].
            edge_index: Edge index tensor [2, num_edges] (source -> target).
            edge_attr: Edge attribute tensor [num_edges, edge_features] (optional).

        Returns:
            Tuple of:
                - Updated node features [num_nodes, out_features].
                - Attention weights [num_edges] (α_ij for each edge).
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        src_idx = edge_index[0]  # [num_edges]
        tgt_idx = edge_index[1]  # [num_edges]

        # Project source and target node features
        h_src = self.W_src(x)  # [num_nodes, out_features]
        h_tgt = self.W_tgt(x)  # [num_nodes, out_features]

        # Gather projected features for each edge
        h_src_e = h_src[src_idx]  # [num_edges, out_features]
        h_tgt_e = h_tgt[tgt_idx]  # [num_edges, out_features]

        # GATv2: apply LeakyReLU BEFORE the attention dot product
        # e_ij = a^T * LeakyReLU(W_src * h_i + W_tgt * h_j [+ W_e * e_ij])
        attn_input = h_src_e + h_tgt_e  # [num_edges, out_features]

        if self.W_edge is not None and edge_attr is not None:
            edge_proj = self.W_edge(edge_attr)  # [num_edges, out_features]
            attn_input = attn_input + edge_proj

        attn_input = self.leaky_relu(attn_input)
        e = self.attn(attn_input).squeeze(-1)  # [num_edges]

        # Softmax over incoming edges per target node
        alpha = self._sparse_softmax(e, tgt_idx, num_nodes)  # [num_edges]
        alpha = self.dropout(alpha)

        # Message passing: aggregate weighted source messages to targets
        # h_i' = Σ_j α_ij * W_src * h_j  (messages flow from src to tgt)
        messages = alpha.unsqueeze(-1) * h_src[src_idx]  # [num_edges, out_features]

        # Scatter-add messages to target nodes
        out = torch.zeros(num_nodes, self.out_features, device=x.device, dtype=x.dtype)
        out.scatter_add_(0, tgt_idx.unsqueeze(-1).expand_as(messages), messages)

        out = out + self.bias
        return out, alpha

    def _sparse_softmax(
        self,
        scores: torch.Tensor,
        index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Computes softmax over scores grouped by index (numerically stable).

        Args:
            scores: Raw attention scores [num_edges].
            index: Target node indices [num_edges].
            num_nodes: Total number of nodes.

        Returns:
            Normalized attention weights [num_edges].
        """
        # Subtract max per target for numerical stability
        max_scores = torch.full((num_nodes,), float('-inf'), device=scores.device, dtype=scores.dtype)
        max_scores.scatter_reduce_(0, index, scores, reduce='amax', include_self=True)
        scores_stable = scores - max_scores[index]

        exp_scores = torch.exp(scores_stable)

        # Sum of exps per target
        sum_exp = torch.zeros(num_nodes, device=scores.device, dtype=scores.dtype)
        sum_exp.scatter_add_(0, index, exp_scores)

        # Normalize
        alpha = exp_scores / (sum_exp[index] + 1e-16)
        return alpha


class MultiHeadGATv2Layer(nn.Module):
    """
    Multi-head GATv2 attention layer with edge attributes.

    Concatenates (or averages) outputs from K independent GATv2 heads.

    Args:
        in_features: Input node feature dimension.
        out_features: Output dimension per head.
        num_heads: Number of independent attention heads K.
        edge_features: Edge attribute dimension.
        dropout: Attention dropout probability.
        concat: If True, concatenate heads (output = K * out_features).
                If False, average heads (output = out_features).
        residual: If True, add residual connection.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        edge_features: int = 1,
        dropout: float = 0.1,
        concat: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.concat = concat
        self.residual = residual

        self.heads = nn.ModuleList([
            GATv2Layer(in_features, out_features, edge_features, dropout)
            for _ in range(num_heads)
        ])

        total_out = out_features * num_heads if concat else out_features

        if residual and in_features != total_out:
            self.residual_proj = nn.Linear(in_features, total_out, bias=False)
        elif residual:
            self.residual_proj = None  # Identity
        else:
            self.residual_proj = None

        self.norm = nn.LayerNorm(total_out)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all attention heads.

        Returns:
            Tuple of:
                - Updated node features [num_nodes, K * out_features] or [num_nodes, out_features].
                - Mean attention weights across heads [num_edges].
        """
        head_outputs = []
        head_alphas = []

        for head in self.heads:
            h, alpha = head(x, edge_index, edge_attr)
            head_outputs.append(h)
            head_alphas.append(alpha)

        if self.concat:
            out = torch.cat(head_outputs, dim=-1)  # [N, K * out_features]
        else:
            out = torch.stack(head_outputs, dim=0).mean(dim=0)  # [N, out_features]

        # Residual connection
        if self.residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(x)
            else:
                residual = x
            out = out + residual

        out = self.norm(out)

        # Average attention weights across heads
        avg_alpha = torch.stack(head_alphas, dim=0).mean(dim=0)  # [num_edges]

        return out, avg_alpha


class SpatialGATEncoder(nn.Module):
    """
    Multi-layer spatial graph attention encoder.
    Stacks multiple MultiHeadGATv2Layer blocks with ELU activations.

    Args:
        input_dim: Input node feature dimension.
        hidden_dim: Hidden dimension per head.
        output_dim: Final output embedding dimension.
        num_layers: Number of stacked GAT layers.
        num_heads: Number of attention heads per layer.
        edge_dim: Edge attribute dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        edge_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim * num_heads)

        layers = []
        for i in range(num_layers):
            in_dim = hidden_dim * num_heads
            is_last = (i == num_layers - 1)
            out_dim = output_dim if is_last else hidden_dim
            concat = not is_last  # Last layer averages, others concatenate
            heads = num_heads if not is_last else max(num_heads, 1)

            layers.append(MultiHeadGATv2Layer(
                in_features=in_dim,
                out_features=out_dim,
                num_heads=heads,
                edge_features=edge_dim,
                dropout=dropout,
                concat=concat,
                residual=True,
            ))
            if not is_last:
                # After concatenation the dim becomes hidden_dim * num_heads again
                pass

        self.layers = nn.ModuleList(layers)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes node features through multi-layer spatial graph attention.

        Returns:
            Tuple of:
                - Spatially-encoded node embeddings [num_nodes, output_dim].
                - Attention weights from last layer [num_edges].
        """
        h = self.input_proj(x)
        h = self.activation(h)
        h = self.dropout(h)

        alpha = None
        for layer in self.layers:
            h, alpha = layer(h, edge_index, edge_attr)
            h = self.activation(h)
            h = self.dropout(h)

        return h, alpha
