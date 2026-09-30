"""
Training pipeline for the STGAT predictive engine.

Supports:
    - Training on synthetic procedurally generated project DAGs.
    - Composite loss (delay MSE + Gaussian NLL + critical path BCE + makespan MSE).
    - Temporal sequence training (window of graph snapshots).
    - Validation with accuracy and calibration metrics.
    - Checkpoint saving and learning rate scheduling.
    - Console logging with per-epoch metric summaries.
"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from project_marl.models.stgat import STGATModel, STGATLoss
from project_marl.sim.generator import (
    SyntheticProjectGenerator,
    ProjectSample,
    collate_samples_to_torch,
)
from project_marl.core.logging import setup_logger

logger = setup_logger("train_stgat")


class STGATTrainer:
    """
    Training harness for STGATModel.

    Args:
        model: STGATModel instance.
        lr: Learning rate.
        weight_decay: L2 regularization weight.
        lambda_nll: Gaussian NLL loss weight.
        lambda_crit: Critical path BCE loss weight.
        lambda_makespan: Makespan MSE loss weight.
        device: 'cpu' or 'cuda'.
        checkpoint_dir: Directory for saving model checkpoints.
    """

    def __init__(
        self,
        model: STGATModel,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_nll: float = 0.5,
        lambda_crit: float = 1.0,
        lambda_makespan: float = 0.1,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = STGATLoss(lambda_nll, lambda_crit, lambda_makespan)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6,
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_history: List[Dict[str, float]] = []
        self.val_history: List[Dict[str, float]] = []
        self.best_val_loss = float("inf")

    def train_epoch(
        self,
        train_data: List[Dict[str, torch.Tensor]],
        temporal_window: int = 3,
    ) -> Dict[str, float]:
        """
        Trains one epoch over the provided data.

        Args:
            train_data: List of sample dicts from collate_samples_to_torch.
            temporal_window: Number of time steps for temporal sequence generation.

        Returns:
            Dict of averaged loss components for the epoch.
        """
        self.model.train()
        epoch_losses = {
            "total": 0.0,
            "delay_mse": 0.0,
            "nll": 0.0,
            "critical_bce": 0.0,
            "makespan_mse": 0.0,
        }
        n_samples = 0

        for sample in train_data:
            x = sample["x"].to(self.device)
            edge_index = sample["edge_index"].to(self.device)
            edge_attr = sample["edge_attr"].to(self.device)
            task_mask = sample["task_mask"].to(self.device)
            targets = {
                k: v.to(self.device) for k, v in sample["targets"].items()
            }

            # Generate temporal sequence by adding small noise perturbations
            x_seq = self._create_temporal_views(x, temporal_window)

            self.optimizer.zero_grad()

            predictions = self.model(x_seq, edge_index, edge_attr, task_mask)
            losses = self.criterion(predictions, targets, task_mask)

            losses["total"].backward()

            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            n_samples += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(n_samples, 1)

        self.train_history.append(epoch_losses)
        return epoch_losses

    @torch.no_grad()
    def validate(
        self,
        val_data: List[Dict[str, torch.Tensor]],
        temporal_window: int = 3,
    ) -> Dict[str, float]:
        """
        Validates the model on held-out data.

        Returns:
            Dict of averaged loss components + accuracy metrics.
        """
        self.model.eval()
        epoch_losses = {
            "total": 0.0,
            "delay_mse": 0.0,
            "nll": 0.0,
            "critical_bce": 0.0,
            "makespan_mse": 0.0,
        }
        crit_correct = 0
        crit_total = 0
        delay_abs_errors = []
        makespan_abs_errors = []
        n_samples = 0

        for sample in val_data:
            x = sample["x"].to(self.device)
            edge_index = sample["edge_index"].to(self.device)
            edge_attr = sample["edge_attr"].to(self.device)
            task_mask = sample["task_mask"].to(self.device)
            targets = {
                k: v.to(self.device) for k, v in sample["targets"].items()
            }

            x_seq = self._create_temporal_views(x, temporal_window)
            predictions = self.model(x_seq, edge_index, edge_attr, task_mask)
            losses = self.criterion(predictions, targets, task_mask)

            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()

            # Critical path accuracy
            pred_crit = (predictions["critical_logits"][task_mask] > 0.0).float()
            gt_crit = targets["is_critical"][task_mask]
            crit_correct += (pred_crit == gt_crit).sum().item()
            crit_total += gt_crit.numel()

            # Delay MAE
            pred_delay = predictions["delay_mu"][task_mask]
            gt_delay = targets["delay"][task_mask]
            delay_abs_errors.append((pred_delay - gt_delay).abs().mean().item())

            # Makespan MAE
            pred_ms = predictions["makespan"]
            gt_ms = targets["makespan"]
            makespan_abs_errors.append((pred_ms - gt_ms).abs().item())

            n_samples += 1

        for key in epoch_losses:
            epoch_losses[key] /= max(n_samples, 1)

        epoch_losses["critical_accuracy"] = crit_correct / max(crit_total, 1)
        epoch_losses["delay_mae"] = float(np.mean(delay_abs_errors)) if delay_abs_errors else 0.0
        epoch_losses["makespan_mae"] = float(np.mean(makespan_abs_errors)) if makespan_abs_errors else 0.0

        self.val_history.append(epoch_losses)
        return epoch_losses

    def fit(
        self,
        num_epochs: int = 50,
        train_samples: int = 200,
        val_samples: int = 50,
        temporal_window: int = 3,
        generator: Optional[SyntheticProjectGenerator] = None,
        log_interval: int = 5,
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Full training loop: generates data, trains, validates, saves checkpoints.

        Args:
            num_epochs: Number of training epochs.
            train_samples: Number of synthetic training projects.
            val_samples: Number of synthetic validation projects.
            temporal_window: Temporal sequence length.
            generator: SyntheticProjectGenerator instance (created if None).
            log_interval: Print metrics every N epochs.

        Returns:
            Dict with 'train_history' and 'val_history'.
        """
        if generator is None:
            generator = SyntheticProjectGenerator(seed=42)

        logger.info(f"Generating {train_samples} training + {val_samples} validation samples...")
        train_raw = generator.generate_batch(train_samples)
        val_raw = generator.generate_batch(val_samples)

        train_data = collate_samples_to_torch(train_raw)
        val_data = collate_samples_to_torch(val_raw)

        logger.info(
            f"STGAT Training: {num_epochs} epochs, {len(train_data)} train, "
            f"{len(val_data)} val, T={temporal_window}, "
            f"params={self.model.count_parameters():,}"
        )

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            train_losses = self.train_epoch(train_data, temporal_window)
            val_losses = self.validate(val_data, temporal_window)

            self.scheduler.step(val_losses["total"])
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Save best model
            if val_losses["total"] < self.best_val_loss:
                self.best_val_loss = val_losses["total"]
                self._save_checkpoint("best_stgat.pt", epoch, val_losses)

            if epoch % log_interval == 0 or epoch == 1:
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch {epoch:>3d}/{num_epochs} | "
                    f"Train Loss: {train_losses['total']:.4f} | "
                    f"Val Loss: {val_losses['total']:.4f} | "
                    f"Crit Acc: {val_losses.get('critical_accuracy', 0):.3f} | "
                    f"Delay MAE: {val_losses.get('delay_mae', 0):.4f} | "
                    f"MS MAE: {val_losses.get('makespan_mae', 0):.2f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )

        # Save final checkpoint
        final_val = self.validate(val_data, temporal_window)
        self._save_checkpoint("final_stgat.pt", num_epochs, final_val)

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time:.1f}s. Best val loss: {self.best_val_loss:.4f}")

        return {
            "train_history": self.train_history,
            "val_history": self.val_history,
        }

    def _create_temporal_views(
        self,
        x: torch.Tensor,
        window_size: int,
    ) -> List[torch.Tensor]:
        """
        Creates temporal sequence by applying small noise perturbations
        to the base features, simulating project progression.
        """
        x_seq = []
        for t in range(window_size):
            noise = torch.randn_like(x) * 0.02 * (t + 1)
            x_t = x + noise
            # Clamp to reasonable range
            x_t = x_t.clamp(min=-1.0, max=10.0)
            x_seq.append(x_t)
        return x_seq

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        val_metrics: Dict[str, float],
    ) -> None:
        """Saves model checkpoint with metadata."""
        filepath = self.checkpoint_dir / filename
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_metrics": val_metrics,
            "model_config": {
                "params": self.model.count_parameters(),
            },
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> Dict:
        """Loads a saved checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from {filepath} (epoch {checkpoint['epoch']})")
        return checkpoint


def main() -> None:
    """CLI entry point for STGAT training."""
    parser = argparse.ArgumentParser(description="STGAT Training Pipeline")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--train-samples", type=int, default=150, help="Training set size")
    parser.add_argument("--val-samples", type=int, default=40, help="Validation set size")
    parser.add_argument("--temporal-window", type=int, default=3, help="Temporal window T")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--spatial-hidden", type=int, default=24, help="GAT hidden dim per head")
    parser.add_argument("--spatial-heads", type=int, default=4, help="GAT attention heads")
    parser.add_argument("--temporal-backend", choices=["lstm", "tcn"], default="lstm")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = STGATModel(
        node_feature_dim=16,
        spatial_hidden=args.spatial_hidden,
        spatial_output=48,
        spatial_layers=2,
        spatial_heads=args.spatial_heads,
        edge_dim=1,
        temporal_hidden=48,
        temporal_output=48,
        temporal_backend=args.temporal_backend,
        temporal_layers=2,
        dropout=0.1,
    )

    trainer = STGATTrainer(
        model=model,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )

    generator = SyntheticProjectGenerator(
        min_tasks=5,
        max_tasks=25,
        min_modules=3,
        max_modules=15,
        edge_density=0.3,
        delay_noise_sigma=0.4,
        seed=args.seed,
    )

    history = trainer.fit(
        num_epochs=args.epochs,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        temporal_window=args.temporal_window,
        generator=generator,
        log_interval=5,
    )

    # Save training history
    history_file = Path(args.checkpoint_dir) / "training_history.json"
    serializable = {
        "train": history["train_history"],
        "val": history["val_history"],
    }
    with open(history_file, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Training history saved to {history_file}")


if __name__ == "__main__":
    main()
