"""
Temporal modeling modules for the STGAT predictive engine.

Implements:
    1. GraphLSTMCell — LSTM cell operating on graph-structured spatial embeddings.
    2. GatedTemporalConvNet (GatedTCN) — Causal 1D dilated convolutions with gating
       over sequences of node embeddings across time steps.
    3. TemporalEncoder — Wrapper selecting between Graph-LSTM and GatedTCN backends.

These modules process sequences of spatial graph snapshots [z_{t-H+1}, ..., z_t]
and output temporally-aware node representations H_t capturing velocity, delay spikes,
and periodicity in task execution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphLSTMCell(nn.Module):
    """
    LSTM cell that operates on graph-level spatial node embeddings.

    At each time step t, receives the spatially-encoded node features z_t
    and updates per-node hidden and cell states using standard LSTM gating,
    enabling the capture of temporal dependencies (velocity of progress,
    delay evolution, periodic patterns).

    Args:
        input_dim: Dimension of spatial node embeddings z_t.
        hidden_dim: LSTM hidden state dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Combined gate projections: input, forget, cell, output
        self.gates = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Xavier init for gates; bias for forget gate initialized to 1.0
        nn.init.xavier_uniform_(self.gates.weight)
        nn.init.zeros_(self.gates.bias)
        # Set forget gate bias to 1.0 for better gradient flow initially
        hidden_dim = self.hidden_dim
        self.gates.bias.data[hidden_dim:2 * hidden_dim].fill_(1.0)

    def forward(
        self,
        z_t: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        c_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single time step of Graph-LSTM.

        Args:
            z_t: Spatial node embeddings at time t [num_nodes, input_dim].
            h_prev: Previous hidden state [num_nodes, hidden_dim] or None.
            c_prev: Previous cell state [num_nodes, hidden_dim] or None.

        Returns:
            (h_t, c_t): Updated hidden and cell states [num_nodes, hidden_dim].
        """
        num_nodes = z_t.size(0)
        device = z_t.device
        dtype = z_t.dtype

        if h_prev is None:
            h_prev = torch.zeros(num_nodes, self.hidden_dim, device=device, dtype=dtype)
        if c_prev is None:
            c_prev = torch.zeros(num_nodes, self.hidden_dim, device=device, dtype=dtype)

        combined = torch.cat([z_t, h_prev], dim=-1)  # [N, input_dim + hidden_dim]
        gates = self.gates(combined)  # [N, 4 * hidden_dim]

        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(self.layer_norm(c_t))

        return h_t, c_t


class GraphLSTMEncoder(nn.Module):
    """
    Multi-step Graph-LSTM that processes a temporal sequence of spatial embeddings.

    Unrolls the GraphLSTMCell over T time steps and returns the final hidden state.

    Args:
        input_dim: Spatial embedding dimension.
        hidden_dim: LSTM hidden dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout between LSTM layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        cells = []
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            cells.append(GraphLSTMCell(in_dim, hidden_dim))
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z_seq: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a temporal sequence of spatial graph embeddings.

        Args:
            z_seq: Sequence of spatial embeddings [T, num_nodes, input_dim].
            h0: Initial hidden state per layer [num_layers, num_nodes, hidden_dim] or None.
            c0: Initial cell state per layer [num_layers, num_nodes, hidden_dim] or None.

        Returns:
            (h_T, output_seq):
                - h_T: Final hidden state from last layer [num_nodes, hidden_dim].
                - output_seq: Hidden states at each time step from last layer [T, num_nodes, hidden_dim].
        """
        T = z_seq.size(0)
        num_nodes = z_seq.size(1)
        device = z_seq.device
        dtype = z_seq.dtype

        # Initialize hidden/cell states for each layer
        h_states = []
        c_states = []
        for layer_idx in range(self.num_layers):
            if h0 is not None:
                h_states.append(h0[layer_idx])
            else:
                h_states.append(torch.zeros(num_nodes, self.hidden_dim, device=device, dtype=dtype))
            if c0 is not None:
                c_states.append(c0[layer_idx])
            else:
                c_states.append(torch.zeros(num_nodes, self.hidden_dim, device=device, dtype=dtype))

        output_seq = []

        for t in range(T):
            layer_input = z_seq[t]  # [num_nodes, input_dim]

            for layer_idx, cell in enumerate(self.cells):
                h_new, c_new = cell(layer_input, h_states[layer_idx], c_states[layer_idx])
                h_states[layer_idx] = h_new
                c_states[layer_idx] = c_new

                if layer_idx < self.num_layers - 1:
                    layer_input = self.dropout(h_new)
                else:
                    layer_input = h_new

            output_seq.append(h_states[-1])

        output_seq = torch.stack(output_seq, dim=0)  # [T, num_nodes, hidden_dim]
        h_T = h_states[-1]  # [num_nodes, hidden_dim]

        return h_T, output_seq


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution ensuring no information leakage from future time steps.

    Pads only on the left side so that the output at time t depends only on
    inputs at times ≤ t.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        dilation: Dilation factor for dilated convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # We handle padding manually
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch * num_nodes, channels, T]
        Returns:
            Causally convolved output [batch * num_nodes, out_channels, T]
        """
        # Left-pad only
        x_padded = F.pad(x, (self.padding, 0))
        return self.conv(x_padded)


class GatedTCNBlock(nn.Module):
    """
    Single Gated Temporal Convolution block with residual connection.

    Uses a GLU (Gated Linear Unit) activation:
        output = tanh(Conv_filter(x)) ⊙ σ(Conv_gate(x))

    Args:
        channels: Number of input/output channels.
        kernel_size: Temporal kernel size.
        dilation: Dilation factor.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch * num_nodes, channels, T]
        Returns:
            Gated output [batch * num_nodes, channels, T]
        """
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        gated = filter_out * gate_out
        gated = self.dropout(gated)

        # Residual connection
        out = x + gated

        # LayerNorm over channel dimension
        # Transpose: [B*N, C, T] -> [B*N, T, C] -> norm -> [B*N, C, T]
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)

        return out


class GatedTCNEncoder(nn.Module):
    """
    Gated Temporal Convolutional Network with exponentially increasing dilation.

    Processes temporal sequences of per-node spatial embeddings using stacked
    causal dilated convolutions with gating, capturing both short-term and
    long-range temporal patterns without future information leakage.

    Args:
        input_dim: Spatial embedding dimension.
        hidden_dim: Internal channel dimension.
        output_dim: Output temporal embedding dimension.
        num_blocks: Number of stacked GatedTCN blocks.
        kernel_size: Temporal convolution kernel size.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_blocks: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** i  # Exponentially increasing receptive field
            blocks.append(GatedTCNBlock(hidden_dim, kernel_size, dilation, dropout))
        self.blocks = nn.ModuleList(blocks)

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes temporal sequence through gated causal convolutions.

        Args:
            z_seq: [T, num_nodes, input_dim] — sequence of spatial embeddings.

        Returns:
            (h_T, output_seq):
                - h_T: Temporal embedding at last time step [num_nodes, output_dim].
                - output_seq: Full temporal output [T, num_nodes, output_dim].
        """
        T, N, D = z_seq.shape

        # Reshape: [T, N, D] -> [N, D, T] for Conv1d
        x = z_seq.permute(1, 2, 0)  # [N, D, T]

        # Project input dimension to hidden channels
        # [N, D, T] -> [N, T, D] -> proj -> [N, T, hidden] -> [N, hidden, T]
        x = x.permute(0, 2, 1)  # [N, T, D]
        x = self.input_proj(x)  # [N, T, hidden]
        x = x.permute(0, 2, 1)  # [N, hidden, T]

        # Apply gated TCN blocks
        for block in self.blocks:
            x = block(x)

        # Project output
        # [N, hidden, T] -> [N, T, hidden] -> proj -> [N, T, output_dim]
        x = x.permute(0, 2, 1)  # [N, T, hidden]
        x = self.output_proj(x)  # [N, T, output_dim]

        # Reshape back: [N, T, output_dim] -> [T, N, output_dim]
        output_seq = x.permute(1, 0, 2)  # [T, N, output_dim]
        h_T = output_seq[-1]  # [N, output_dim]

        return h_T, output_seq


class TemporalEncoder(nn.Module):
    """
    Wrapper module selecting between Graph-LSTM and GatedTCN temporal backends.

    Args:
        input_dim: Spatial embedding dimension from SpatialGATEncoder.
        hidden_dim: Temporal hidden dimension.
        output_dim: Temporal output dimension.
        backend: 'lstm' for GraphLSTMEncoder, 'tcn' for GatedTCNEncoder.
        num_layers: Number of stacked layers/blocks.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        backend: str = "lstm",
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backend = backend

        if backend == "lstm":
            self.encoder = GraphLSTMEncoder(input_dim, hidden_dim, num_layers, dropout)
            self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        elif backend == "tcn":
            self.encoder = GatedTCNEncoder(input_dim, hidden_dim, output_dim, num_layers, dropout=dropout)
            self.output_proj = nn.Identity()
        else:
            raise ValueError(f"Unknown temporal backend: {backend}. Choose 'lstm' or 'tcn'.")

    def forward(
        self,
        z_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_seq: Temporal sequence of spatial embeddings [T, num_nodes, input_dim].

        Returns:
            (h_T, output_seq):
                - h_T: Final temporal node embedding [num_nodes, output_dim].
                - output_seq: All temporal outputs [T, num_nodes, output_dim].
        """
        h_T, output_seq = self.encoder(z_seq)

        if self.backend == "lstm":
            h_T = self.output_proj(h_T)
            # Apply projection to full sequence too
            T, N, _ = output_seq.shape
            output_seq = self.output_proj(output_seq.reshape(T * N, -1)).reshape(T, N, -1)

        return h_T, output_seq
