"""
Unified Spatiotemporal Graph Attention Network (STGAT) model.

Integrates:
    1. SpatialGATEncoder — Multi-head graph attention with edge attributes.
    2. TemporalEncoder — Graph-LSTM or GatedTCN over temporal graph snapshots.
    3. Prediction Heads:
        a. Delay prediction (Gaussian: μ, σ² per task node).
        b. Critical path classification (binary per task node).
        c. Global makespan regression (scalar).
        d. Structural fragility scoring (per task node).

Architecture:
    For each time step t in [t-H+1, ..., t]:
        z_t = SpatialGATEncoder(x_t, edge_index, edge_attr)
    H_t = TemporalEncoder([z_{t-H+1}, ..., z_t])
    predictions = PredictionHeads(H_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from project_marl.models.gat_layers import SpatialGATEncoder
from project_marl.models.temporal_net import TemporalEncoder


class DelayPredictionHead(nn.Module):
    """
    Predicts per-task delay distribution parameters: mean (μ) and log-variance (log σ²).
    Outputs a Gaussian parameterization for P(Δd_i | G_t).

    Args:
        input_dim: Temporal node embedding dimension.
        hidden_dim: MLP hidden dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim // 2, 1)
        self.logvar_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Node embeddings [num_nodes, input_dim].
        Returns:
            (mu, logvar): Predicted delay mean and log-variance [num_nodes, 1] each.
        """
        features = self.mlp(h)
        mu = self.mu_head(features)           # [N, 1]
        logvar = self.logvar_head(features)    # [N, 1]
        return mu, logvar


class CriticalPathHead(nn.Module):
    """
    Binary classifier predicting whether each task node lies on the critical path.

    Args:
        input_dim: Temporal node embedding dimension.
        hidden_dim: MLP hidden dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node embeddings [num_nodes, input_dim].
        Returns:
            logits: Critical path logits [num_nodes, 1].
        """
        return self.classifier(h)


class MakespanHead(nn.Module):
    """
    Predicts global project makespan from a graph-level summary embedding.

    Uses mean-pooling of task node embeddings followed by an MLP.

    Args:
        input_dim: Temporal node embedding dimension.
        hidden_dim: MLP hidden dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, task_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node embeddings [num_nodes, input_dim].
            task_mask: Boolean mask selecting task nodes [num_nodes].
        Returns:
            Predicted makespan scalar [1].
        """
        # Pool only task node embeddings
        task_h = h[task_mask]  # [num_tasks, input_dim]
        if task_h.size(0) == 0:
            return torch.tensor([0.0], device=h.device)
        pooled = task_h.mean(dim=0, keepdim=True)  # [1, input_dim]
        return self.mlp(pooled).squeeze(-1)  # [1]


class STGATModel(nn.Module):
    """
    Complete Spatiotemporal Graph Attention Network for project risk prediction.

    Processes a temporal window of graph snapshots through spatial attention
    and temporal modeling, then outputs:
        1. Per-task delay distribution (μ, log σ²)
        2. Per-task critical path probability
        3. Global project makespan estimate
        4. Per-task structural fragility score (derived from attention)

    Args:
        node_feature_dim: Raw node feature dimension from ProjectGraphBuilder.
        spatial_hidden: Hidden dimension per GAT head.
        spatial_output: Spatial encoder output dimension.
        spatial_layers: Number of stacked GAT layers.
        spatial_heads: Number of attention heads.
        edge_dim: Edge attribute dimension.
        temporal_hidden: Temporal encoder hidden dimension.
        temporal_output: Temporal encoder output dimension.
        temporal_backend: 'lstm' or 'tcn'.
        temporal_layers: Number of temporal layers/blocks.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        node_feature_dim: int = 16,
        spatial_hidden: int = 32,
        spatial_output: int = 64,
        spatial_layers: int = 2,
        spatial_heads: int = 4,
        edge_dim: int = 1,
        temporal_hidden: int = 64,
        temporal_output: int = 64,
        temporal_backend: str = "lstm",
        temporal_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Spatial encoder: processes each graph snapshot
        self.spatial_encoder = SpatialGATEncoder(
            input_dim=node_feature_dim,
            hidden_dim=spatial_hidden,
            output_dim=spatial_output,
            num_layers=spatial_layers,
            num_heads=spatial_heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        # Temporal encoder: processes sequence of spatial embeddings
        self.temporal_encoder = TemporalEncoder(
            input_dim=spatial_output,
            hidden_dim=temporal_hidden,
            output_dim=temporal_output,
            backend=temporal_backend,
            num_layers=temporal_layers,
            dropout=dropout,
        )

        # Prediction heads
        self.delay_head = DelayPredictionHead(temporal_output, hidden_dim=64)
        self.critical_head = CriticalPathHead(temporal_output, hidden_dim=32)
        self.makespan_head = MakespanHead(temporal_output, hidden_dim=64)

    def forward(
        self,
        x_seq: List[torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through STGAT.

        Args:
            x_seq: List of node feature tensors, one per time step.
                   Each tensor: [num_nodes, node_feature_dim].
                   Length = temporal window T.
            edge_index: Edge index [2, num_edges] (assumed static across window).
            edge_attr: Edge attributes [num_edges, edge_dim] (optional).
            task_mask: Boolean mask for task nodes [num_nodes] (optional).

        Returns:
            Dictionary containing:
                'delay_mu': Predicted delay means [num_nodes, 1].
                'delay_logvar': Predicted delay log-variances [num_nodes, 1].
                'critical_logits': Critical path logits [num_nodes, 1].
                'makespan': Predicted makespan scalar [1].
                'attention_weights': Spatial attention from last snapshot [num_edges].
                'node_embeddings': Final temporal node embeddings [num_nodes, temporal_output].
                'fragility_scores': Structural fragility per node [num_nodes].
        """
        T = len(x_seq)
        z_list = []
        last_alpha = None

        # Spatial encoding for each time step
        for t in range(T):
            z_t, alpha_t = self.spatial_encoder(x_seq[t], edge_index, edge_attr)
            z_list.append(z_t)
            last_alpha = alpha_t

        # Stack into temporal sequence [T, num_nodes, spatial_output]
        z_seq = torch.stack(z_list, dim=0)

        # Temporal encoding
        h_T, _ = self.temporal_encoder(z_seq)  # h_T: [num_nodes, temporal_output]

        # Prediction heads
        delay_mu, delay_logvar = self.delay_head(h_T)
        critical_logits = self.critical_head(h_T)

        if task_mask is None:
            task_mask = torch.ones(h_T.size(0), dtype=torch.bool, device=h_T.device)
        makespan = self.makespan_head(h_T, task_mask)

        # Compute structural fragility scores from attention weights
        fragility = self._compute_fragility(
            last_alpha, edge_index, delay_mu, h_T.size(0)
        )

        return {
            "delay_mu": delay_mu,
            "delay_logvar": delay_logvar,
            "critical_logits": critical_logits,
            "makespan": makespan,
            "attention_weights": last_alpha,
            "node_embeddings": h_T,
            "fragility_scores": fragility,
        }

    def forward_single(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method for a single graph snapshot (T=1).
        Wraps x into a single-element sequence.
        """
        return self.forward([x], edge_index, edge_attr, task_mask)

    def _compute_fragility(
        self,
        alpha: torch.Tensor,
        edge_index: torch.Tensor,
        delay_mu: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Computes structural fragility score for each node:
            Φ(v_i) = Σ_{j ∈ Succ(i)} α_ji * μ_i

        Nodes with high predicted delay AND high attention from successors
        represent the most fragile bottleneck points.

        Args:
            alpha: Attention weights [num_edges].
            edge_index: [2, num_edges] (src -> tgt).
            delay_mu: Predicted delay means [num_nodes, 1].
            num_nodes: Total node count.

        Returns:
            Fragility scores [num_nodes].
        """
        if alpha is None:
            return torch.zeros(num_nodes, device=delay_mu.device)

        src = edge_index[0]  # [num_edges]
        # Weight each edge attention by the source node's predicted delay
        weighted = alpha * delay_mu.squeeze(-1)[src].abs()  # [num_edges]

        # Sum weighted attention into source nodes (fragility of being a bottleneck)
        fragility = torch.zeros(num_nodes, device=alpha.device)
        fragility.scatter_add_(0, src, weighted)

        return fragility

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class STGATLoss(nn.Module):
    """
    Composite loss function for STGAT training:

    L = L_delay_MSE(μ̂, d) + λ_NLL * L_variance_NLL + λ_crit * L_BCE(ŷ_crit, y_crit)
        + λ_makespan * L_makespan_MSE

    Components:
        1. Delay MSE: Mean squared error between predicted delay mean and actual delay.
        2. Gaussian NLL: Penalizes both mean error and variance calibration.
        3. Critical Path BCE: Binary cross-entropy for critical path prediction.
        4. Makespan MSE: Mean squared error on global makespan prediction.

    Args:
        lambda_nll: Weight for Gaussian NLL loss.
        lambda_crit: Weight for critical path BCE loss.
        lambda_makespan: Weight for makespan regression loss.
    """

    def __init__(
        self,
        lambda_nll: float = 0.5,
        lambda_crit: float = 1.0,
        lambda_makespan: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_nll = lambda_nll
        self.lambda_crit = lambda_crit
        self.lambda_makespan = lambda_makespan

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        task_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes composite loss.

        Args:
            predictions: Dict from STGATModel.forward().
            targets: Dict containing:
                'delay': Ground truth delays [num_nodes, 1].
                'is_critical': Ground truth critical labels [num_nodes, 1] (float).
                'makespan': Ground truth makespan scalar [1].
            task_mask: Boolean mask for task nodes [num_nodes].

        Returns:
            Dict with 'total', 'delay_mse', 'nll', 'critical_bce', 'makespan_mse'.
        """
        pred_mu = predictions["delay_mu"]
        pred_logvar = predictions["delay_logvar"]
        pred_crit = predictions["critical_logits"]
        pred_makespan = predictions["makespan"]

        gt_delay = targets["delay"]
        gt_crit = targets["is_critical"]
        gt_makespan = targets["makespan"]

        # 1. Delay MSE (only on task nodes)
        task_pred_mu = pred_mu[task_mask]
        task_gt_delay = gt_delay[task_mask]
        delay_mse = F.mse_loss(task_pred_mu, task_gt_delay)

        # 2. Gaussian NLL (calibrated uncertainty)
        task_logvar = pred_logvar[task_mask]
        # NLL = 0.5 * (log σ² + (y - μ)² / σ²)
        variance = torch.exp(task_logvar).clamp(min=1e-6)
        nll = 0.5 * (task_logvar + (task_gt_delay - task_pred_mu) ** 2 / variance)
        nll = nll.mean()

        # 3. Critical Path BCE (only on task nodes)
        task_crit_logits = pred_crit[task_mask]
        task_gt_crit = gt_crit[task_mask]
        critical_bce = F.binary_cross_entropy_with_logits(task_crit_logits, task_gt_crit)

        # 4. Makespan regression MSE
        makespan_mse = F.mse_loss(pred_makespan, gt_makespan)

        # Total loss
        total = (
            delay_mse
            + self.lambda_nll * nll
            + self.lambda_crit * critical_bce
            + self.lambda_makespan * makespan_mse
        )

        return {
            "total": total,
            "delay_mse": delay_mse.detach(),
            "nll": nll.detach(),
            "critical_bce": critical_bce.detach(),
            "makespan_mse": makespan_mse.detach(),
        }
