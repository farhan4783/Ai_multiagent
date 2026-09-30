"""
Comprehensive unit tests for Phase 2: STGAT Predictive Engine.

Tests cover:
    - GATv2Layer: single-head attention, edge-conditioned attention, sparse softmax.
    - MultiHeadGATv2Layer: multi-head concatenation and averaging.
    - SpatialGATEncoder: multi-layer stacked spatial encoding.
    - GraphLSTMCell and GraphLSTMEncoder: temporal LSTM on graph embeddings.
    - GatedTCNEncoder: causal temporal convolutions.
    - STGATModel: full forward pass with all prediction heads.
    - STGATLoss: composite loss computation.
    - SyntheticProjectGenerator: data generation and tensor export.
"""

import pytest
import torch
import numpy as np

from project_marl.models.gat_layers import (
    GATv2Layer,
    MultiHeadGATv2Layer,
    SpatialGATEncoder,
)
from project_marl.models.temporal_net import (
    GraphLSTMCell,
    GraphLSTMEncoder,
    CausalConv1d,
    GatedTCNBlock,
    GatedTCNEncoder,
    TemporalEncoder,
)
from project_marl.models.stgat import (
    STGATModel,
    STGATLoss,
    DelayPredictionHead,
    CriticalPathHead,
    MakespanHead,
)
from project_marl.sim.generator import (
    SyntheticProjectGenerator,
    collate_samples_to_torch,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def small_graph():
    """A small test graph with 6 nodes, 8 directed edges, 1 edge feature."""
    num_nodes = 6
    x = torch.randn(num_nodes, 16)
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 3, 3, 4],
        [1, 2, 3, 4, 4, 4, 5, 5],
    ], dtype=torch.long)
    edge_attr = torch.rand(8, 1)
    task_mask = torch.tensor([True, True, True, True, False, False])
    return x, edge_index, edge_attr, task_mask


# ============================================================================
# GATv2 Layer Tests
# ============================================================================

class TestGATv2Layer:
    def test_basic_forward(self, small_graph):
        x, edge_index, edge_attr, _ = small_graph
        layer = GATv2Layer(in_features=16, out_features=8, edge_features=1)
        out, alpha = layer(x, edge_index, edge_attr)

        assert out.shape == (6, 8)
        assert alpha.shape == (8,)
        # Attention weights should be non-negative and sum ~1 per target
        assert (alpha >= 0).all()

    def test_without_edge_features(self, small_graph):
        x, edge_index, _, _ = small_graph
        layer = GATv2Layer(in_features=16, out_features=8, edge_features=0)
        out, alpha = layer(x, edge_index, edge_attr=None)

        assert out.shape == (6, 8)
        assert alpha.shape == (8,)

    def test_attention_sums_to_one(self, small_graph):
        x, edge_index, edge_attr, _ = small_graph
        layer = GATv2Layer(in_features=16, out_features=8, edge_features=1, dropout=0.0)
        layer.eval()
        _, alpha = layer(x, edge_index, edge_attr)

        # Check attention sums per target node
        tgt_idx = edge_index[1]
        for node in tgt_idx.unique():
            mask = tgt_idx == node
            attn_sum = alpha[mask].sum().item()
            assert abs(attn_sum - 1.0) < 1e-4, f"Node {node} attention sum: {attn_sum}"

    def test_gradient_flow(self, small_graph):
        x, edge_index, edge_attr, _ = small_graph
        x.requires_grad_(True)
        layer = GATv2Layer(in_features=16, out_features=8, edge_features=1)
        out, _ = layer(x, edge_index, edge_attr)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestMultiHeadGATv2:
    def test_concat_mode(self, small_graph):
        x, edge_index, edge_attr, _ = small_graph
        layer = MultiHeadGATv2Layer(
            in_features=16, out_features=8, num_heads=4,
            edge_features=1, concat=True,
        )
        out, alpha = layer(x, edge_index, edge_attr)
        assert out.shape == (6, 32)  # 4 heads * 8

    def test_average_mode(self, small_graph):
        x, edge_index, edge_attr, _ = small_graph
        layer = MultiHeadGATv2Layer(
            in_features=16, out_features=8, num_heads=4,
            edge_features=1, concat=False,
        )
        out, alpha = layer(x, edge_index, edge_attr)
        assert out.shape == (6, 8)  # averaged


class TestSpatialGATEncoder:
    def test_encoder_output_shape(self, small_graph):
        x, edge_index, edge_attr, _ = small_graph
        encoder = SpatialGATEncoder(
            input_dim=16, hidden_dim=8, output_dim=32,
            num_layers=2, num_heads=4, edge_dim=1,
        )
        out, alpha = encoder(x, edge_index, edge_attr)
        assert out.shape == (6, 32)
        assert alpha.shape == (8,)


# ============================================================================
# Temporal Module Tests
# ============================================================================

class TestGraphLSTM:
    def test_cell_single_step(self):
        cell = GraphLSTMCell(input_dim=32, hidden_dim=24)
        z_t = torch.randn(6, 32)
        h, c = cell(z_t)
        assert h.shape == (6, 24)
        assert c.shape == (6, 24)

    def test_cell_with_previous_state(self):
        cell = GraphLSTMCell(input_dim=32, hidden_dim=24)
        z_t = torch.randn(6, 32)
        h0 = torch.randn(6, 24)
        c0 = torch.randn(6, 24)
        h, c = cell(z_t, h0, c0)
        assert h.shape == (6, 24)

    def test_encoder_sequence(self):
        encoder = GraphLSTMEncoder(input_dim=32, hidden_dim=24, num_layers=2)
        z_seq = torch.randn(5, 6, 32)  # T=5, N=6, D=32
        h_T, out_seq = encoder(z_seq)
        assert h_T.shape == (6, 24)
        assert out_seq.shape == (5, 6, 24)


class TestGatedTCN:
    def test_causal_conv(self):
        conv = CausalConv1d(32, 32, kernel_size=3, dilation=1)
        x = torch.randn(6, 32, 8)  # [N, C, T]
        out = conv(x)
        assert out.shape == (6, 32, 8)  # Same temporal dimension

    def test_tcn_block(self):
        block = GatedTCNBlock(channels=32, kernel_size=3, dilation=2)
        x = torch.randn(6, 32, 8)
        out = block(x)
        assert out.shape == (6, 32, 8)

    def test_tcn_encoder(self):
        encoder = GatedTCNEncoder(
            input_dim=32, hidden_dim=24, output_dim=16,
            num_blocks=3,
        )
        z_seq = torch.randn(5, 6, 32)  # [T, N, D]
        h_T, out_seq = encoder(z_seq)
        assert h_T.shape == (6, 16)
        assert out_seq.shape == (5, 6, 16)


class TestTemporalEncoder:
    def test_lstm_backend(self):
        enc = TemporalEncoder(
            input_dim=32, hidden_dim=24, output_dim=16,
            backend="lstm",
        )
        z_seq = torch.randn(5, 6, 32)
        h_T, out_seq = enc(z_seq)
        assert h_T.shape == (6, 16)

    def test_tcn_backend(self):
        enc = TemporalEncoder(
            input_dim=32, hidden_dim=24, output_dim=16,
            backend="tcn",
        )
        z_seq = torch.randn(5, 6, 32)
        h_T, out_seq = enc(z_seq)
        assert h_T.shape == (6, 16)


# ============================================================================
# STGAT Model Tests
# ============================================================================

class TestSTGATModel:
    def test_forward_single_snapshot(self, small_graph):
        x, edge_index, edge_attr, task_mask = small_graph
        model = STGATModel(
            node_feature_dim=16,
            spatial_hidden=8, spatial_output=16,
            spatial_layers=2, spatial_heads=4, edge_dim=1,
            temporal_hidden=16, temporal_output=16,
            temporal_backend="lstm", temporal_layers=1,
        )
        preds = model.forward_single(x, edge_index, edge_attr, task_mask)

        assert preds["delay_mu"].shape == (6, 1)
        assert preds["delay_logvar"].shape == (6, 1)
        assert preds["critical_logits"].shape == (6, 1)
        assert preds["makespan"].shape == (1,)
        assert preds["attention_weights"].shape == (8,)
        assert preds["node_embeddings"].shape == (6, 16)
        assert preds["fragility_scores"].shape == (6,)

    def test_forward_temporal_sequence(self, small_graph):
        x, edge_index, edge_attr, task_mask = small_graph
        model = STGATModel(
            node_feature_dim=16,
            spatial_hidden=8, spatial_output=16,
            spatial_layers=1, spatial_heads=2, edge_dim=1,
            temporal_hidden=16, temporal_output=16,
            temporal_backend="lstm", temporal_layers=1,
        )
        x_seq = [x + torch.randn_like(x) * 0.1 for _ in range(4)]
        preds = model(x_seq, edge_index, edge_attr, task_mask)

        assert preds["delay_mu"].shape == (6, 1)
        assert preds["makespan"].shape == (1,)

    def test_tcn_backend(self, small_graph):
        x, edge_index, edge_attr, task_mask = small_graph
        model = STGATModel(
            node_feature_dim=16,
            spatial_hidden=8, spatial_output=16,
            spatial_layers=1, spatial_heads=2, edge_dim=1,
            temporal_hidden=16, temporal_output=16,
            temporal_backend="tcn", temporal_layers=2,
        )
        x_seq = [x + torch.randn_like(x) * 0.1 for _ in range(4)]
        preds = model(x_seq, edge_index, edge_attr, task_mask)
        assert preds["delay_mu"].shape == (6, 1)

    def test_parameter_count(self, small_graph):
        model = STGATModel(
            node_feature_dim=16,
            spatial_hidden=8, spatial_output=16,
            spatial_layers=1, spatial_heads=2, edge_dim=1,
            temporal_hidden=16, temporal_output=16,
        )
        n_params = model.count_parameters()
        assert n_params > 0
        assert isinstance(n_params, int)

    def test_backward_pass(self, small_graph):
        x, edge_index, edge_attr, task_mask = small_graph
        model = STGATModel(
            node_feature_dim=16,
            spatial_hidden=8, spatial_output=16,
            spatial_layers=1, spatial_heads=2, edge_dim=1,
            temporal_hidden=16, temporal_output=16,
        )
        preds = model.forward_single(x, edge_index, edge_attr, task_mask)
        loss = preds["delay_mu"].sum() + preds["makespan"].sum()
        loss.backward()

        # Check gradients exist on model parameters
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0


# ============================================================================
# Loss Function Tests
# ============================================================================

class TestSTGATLoss:
    def test_loss_computation(self, small_graph):
        x, edge_index, edge_attr, task_mask = small_graph
        model = STGATModel(
            node_feature_dim=16,
            spatial_hidden=8, spatial_output=16,
            spatial_layers=1, spatial_heads=2, edge_dim=1,
            temporal_hidden=16, temporal_output=16,
        )
        preds = model.forward_single(x, edge_index, edge_attr, task_mask)

        targets = {
            "delay": torch.randn(6, 1),
            "is_critical": torch.tensor([[1.0], [0.0], [1.0], [0.0], [0.0], [0.0]]),
            "makespan": torch.tensor([25.0]),
        }

        criterion = STGATLoss(lambda_nll=0.5, lambda_crit=1.0, lambda_makespan=0.1)
        losses = criterion(preds, targets, task_mask)

        assert "total" in losses
        assert "delay_mse" in losses
        assert "nll" in losses
        assert "critical_bce" in losses
        assert "makespan_mse" in losses

        # Total loss should be finite
        assert torch.isfinite(losses["total"])
        assert losses["total"].item() > 0


# ============================================================================
# Synthetic Generator Tests
# ============================================================================

class TestSyntheticGenerator:
    def test_generate_one(self):
        gen = SyntheticProjectGenerator(
            min_tasks=5, max_tasks=10,
            min_modules=3, max_modules=6,
            seed=123,
        )
        sample = gen.generate_one()

        assert sample.num_tasks >= 5
        assert sample.num_modules >= 3
        assert sample.x.shape[0] == sample.num_tasks + sample.num_modules
        assert sample.x.shape[1] == 16
        assert sample.edge_index.shape[0] == 2
        assert sample.task_mask.sum() == sample.num_tasks
        assert sample.makespan > 0
        assert len(sample.critical_path) > 0

    def test_generate_batch(self):
        gen = SyntheticProjectGenerator(
            min_tasks=5, max_tasks=8,
            min_modules=2, max_modules=4,
            seed=456,
        )
        samples = gen.generate_batch(10)
        assert len(samples) == 10

    def test_collate_to_torch(self):
        gen = SyntheticProjectGenerator(seed=789)
        samples = gen.generate_batch(3)
        torch_data = collate_samples_to_torch(samples)

        assert len(torch_data) == 3
        for td in torch_data:
            assert isinstance(td["x"], torch.Tensor)
            assert td["x"].dtype == torch.float32
            assert td["edge_index"].dtype == torch.int64
            assert "delay" in td["targets"]

    def test_temporal_sequence_generation(self):
        gen = SyntheticProjectGenerator(seed=321)
        seq = gen.generate_temporal_sequence(window_size=5)

        assert len(seq["x_seq"]) == 5
        assert seq["edge_index"].shape[0] == 2
        assert "delay" in seq["targets"]

    def test_end_to_end_with_model(self):
        """Test that generated data flows through the STGAT model."""
        gen = SyntheticProjectGenerator(
            min_tasks=5, max_tasks=8,
            min_modules=2, max_modules=4,
            seed=999,
        )
        samples = gen.generate_batch(2)
        torch_data = collate_samples_to_torch(samples)

        model = STGATModel(
            node_feature_dim=16,
            spatial_hidden=8, spatial_output=16,
            spatial_layers=1, spatial_heads=2, edge_dim=1,
            temporal_hidden=16, temporal_output=16,
        )
        model.eval()

        for td in torch_data:
            preds = model.forward_single(
                td["x"], td["edge_index"], td["edge_attr"], td["task_mask"],
            )
            assert torch.isfinite(preds["delay_mu"]).all()
            assert torch.isfinite(preds["makespan"]).all()
