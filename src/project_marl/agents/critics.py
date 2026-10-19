"""
Centralized Critic Networks for Centralized Training with Decentralized Execution (CTDE).

Provides global state-value estimation V_phi(S_t) evaluating project progress,
resource occupancies, coupling fragility, and multi-objective performance.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from project_marl.core.logging import setup_logger

logger = setup_logger("critics")


def init_weights_orthogonal(module: nn.Module, gain: float = 1.0) -> None:
    """Applies orthogonal initialization with prescribed gain to linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class ResidualBlock(nn.Module):
    """Residual MLP block with LayerNorm and LeakyReLU activation."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        init_weights_orthogonal(self.fc1, gain=np.sqrt(2))
        init_weights_orthogonal(self.fc2, gain=np.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = F.leaky_relu(self.ln1(self.fc1(x)), negative_slope=0.1)
        h = self.dropout(h)
        h = self.ln2(self.fc2(h))
        return F.leaky_relu(residual + h, negative_slope=0.1)


class CentralizedCritic(nn.Module):
    """
    Centralized Value Function V_phi(S_t) evaluating global project state.

    Input:
        Global state vector S_t of shape (batch_size, state_dim) or (state_dim,).
    Output:
        Scalar state-value estimate V(S_t) of shape (batch_size, 1).
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Input projection layer
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.input_ln = nn.LayerNorm(hidden_dim)
        init_weights_orthogonal(self.input_proj, gain=np.sqrt(2))

        # Residual processing layers
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout)
            for _ in range(num_res_blocks)
        ])

        # Value output head (gain 1.0 for value function)
        self.value_head = nn.Linear(hidden_dim, 1)
        init_weights_orthogonal(self.value_head, gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting scalar expected return V(S_t).
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        h = F.leaky_relu(self.input_ln(self.input_proj(state)), negative_slope=0.1)
        for block in self.res_blocks:
            h = block(h)

        values = self.value_head(h)
        return values

    @torch.no_grad()
    def evaluate_state(
        self,
        state: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Helper to evaluate numpy state array or tensor and return 1D value tensor."""
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.float32)
        val = self.forward(state)
        return val.squeeze(-1).detach()


class RoleSpecificCentralizedCritic(nn.Module):
    """
    Centralized Critic with role-specific value heads evaluating distinct utility
    signals for Scheduling, Risk, Developers, and PM agents from the shared global state.
    """

    def __init__(
        self,
        state_dim: int,
        agent_ids: List[str],
        hidden_dim: int = 256,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.agent_ids = agent_ids

        # Shared trunk representation
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.input_ln = nn.LayerNorm(hidden_dim)
        init_weights_orthogonal(self.input_proj, gain=np.sqrt(2))

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])

        # Separate value head per agent
        self.heads = nn.ModuleDict()
        for ag_id in agent_ids:
            head = nn.Linear(hidden_dim, 1)
            init_weights_orthogonal(head, gain=1.0)
            self.heads[ag_id] = head

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes role-specific value predictions from the global state.

        Returns:
            Dict mapping agent_id to value tensor of shape (batch_size, 1).
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        h = F.leaky_relu(self.input_ln(self.input_proj(state)), negative_slope=0.1)
        for block in self.res_blocks:
            h = block(h)

        return {ag_id: self.heads[ag_id](h) for ag_id in self.agent_ids}


class CentralizedSTGATCritic(nn.Module):
    """
    Advanced Centralized Critic fusing topological STGAT graph embeddings
    with physical resource state features for high-fidelity value estimation.
    """

    def __init__(
        self,
        graph_embed_dim: int,
        resource_state_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.joint_dim = graph_embed_dim + resource_state_dim

        self.fusion_fc = nn.Linear(self.joint_dim, hidden_dim)
        self.fusion_ln = nn.LayerNorm(hidden_dim)
        init_weights_orthogonal(self.fusion_fc, gain=np.sqrt(2))

        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)

        self.value_head = nn.Linear(hidden_dim, 1)
        init_weights_orthogonal(self.value_head, gain=1.0)

    def forward(
        self,
        graph_embed: torch.Tensor,
        resource_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuses graph embedding and resource vector to produce value estimate.
        """
        if graph_embed.dim() == 1:
            graph_embed = graph_embed.unsqueeze(0)
        if resource_state.dim() == 1:
            resource_state = resource_state.unsqueeze(0)

        joint = torch.cat([graph_embed, resource_state], dim=-1)
        h = F.leaky_relu(self.fusion_ln(self.fusion_fc(joint)), negative_slope=0.1)
        h = self.res1(h)
        h = self.res2(h)
        return self.value_head(h)
