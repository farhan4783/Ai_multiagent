"""
Multi-Agent PPO (MAPPO) with Centralized Training and Decentralized Execution (CTDE).

Implements:
    - MAPPORolloutBuffer: multi-agent trajectory storage with Generalized Advantage Estimation (GAE).
    - MAPPOTrainer: actor-critic optimization with PPO clipping, value clipping, and action masking.
"""

from typing import Dict, List, Optional, Tuple, Any, Generator, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from project_marl.agents.policies import MultiAgentActorSwarm
from project_marl.agents.critics import CentralizedCritic
from project_marl.core.logging import setup_logger

logger = setup_logger("mappo")


class MAPPORolloutBuffer:
    """
    Trajectory rollout buffer storing experience tuples across discrete steps
    for all agents in the decentralized swarm and the centralized global state.
    """

    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        agent_obs_dims: Dict[str, int],
        agent_action_dims: Dict[str, int],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.agent_obs_dims = agent_obs_dims
        self.agent_action_dims = agent_action_dims
        self.agents = list(agent_obs_dims.keys())
        self.device = device

        self.step_idx = 0
        self.full = False

        # Centralized state buffer
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32, device=device)

        # Agent-specific observation, action, mask, and log_prob buffers
        self.obs: Dict[str, torch.Tensor] = {}
        self.actions: Dict[str, torch.Tensor] = {}
        self.action_masks: Dict[str, torch.Tensor] = {}
        self.log_probs: Dict[str, torch.Tensor] = {}

        for ag in self.agents:
            self.obs[ag] = torch.zeros((buffer_size, agent_obs_dims[ag]), dtype=torch.float32, device=device)
            self.actions[ag] = torch.zeros(buffer_size, dtype=torch.long, device=device)
            self.action_masks[ag] = torch.zeros((buffer_size, agent_action_dims[ag]), dtype=torch.float32, device=device)
            self.log_probs[ag] = torch.zeros(buffer_size, dtype=torch.float32, device=device)

    def insert(
        self,
        state: Union[np.ndarray, torch.Tensor],
        obs_dict: Dict[str, Union[np.ndarray, torch.Tensor]],
        actions_dict: Dict[str, int],
        log_probs_dict: Dict[str, Union[float, torch.Tensor]],
        value: Union[float, torch.Tensor],
        reward: float,
        done: bool,
        action_masks_dict: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
    ) -> None:
        """Inserts one discrete multi-agent step transition into the buffer."""
        idx = self.step_idx

        # Global state and value
        if isinstance(state, np.ndarray):
            self.states[idx] = torch.from_numpy(state).float().to(self.device)
        else:
            self.states[idx] = state.detach().float().to(self.device)

        val_tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float32)
        self.values[idx] = val_tensor.detach().squeeze().to(self.device)
        self.rewards[idx] = float(reward)
        self.dones[idx] = 1.0 if done else 0.0

        # Agent specific transitions
        for ag in self.agents:
            if ag in obs_dict:
                o = obs_dict[ag]
                if isinstance(o, np.ndarray):
                    self.obs[ag][idx] = torch.from_numpy(o).float().to(self.device)
                else:
                    self.obs[ag][idx] = o.detach().float().to(self.device)

            if ag in actions_dict:
                self.actions[ag][idx] = int(actions_dict[ag])

            if ag in log_probs_dict:
                lp = log_probs_dict[ag]
                lp_tensor = lp if isinstance(lp, torch.Tensor) else torch.tensor(lp, dtype=torch.float32)
                self.log_probs[ag][idx] = lp_tensor.detach().squeeze().to(self.device)

            if action_masks_dict is not None and ag in action_masks_dict:
                m = action_masks_dict[ag]
                if isinstance(m, np.ndarray):
                    self.action_masks[ag][idx] = torch.from_numpy(m).float().to(self.device)
                else:
                    self.action_masks[ag][idx] = m.detach().float().to(self.device)
            else:
                self.action_masks[ag][idx].fill_(1.0)

        self.step_idx += 1
        if self.step_idx >= self.buffer_size:
            self.full = True

    @torch.no_grad()
    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Computes Generalized Advantage Estimation (GAE-lambda) and discounted returns.
        """
        num_steps = self.step_idx
        if num_steps == 0:
            return

        last_val = float(last_value)
        last_gae = 0.0

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_val = last_val
            else:
                next_non_terminal = 1.0 - self.dones[t + 1].item()
                next_val = self.values[t + 1].item()

            delta = self.rewards[t].item() + gamma * next_val * next_non_terminal - self.values[t].item()
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:num_steps] = self.advantages[:num_steps] + self.values[:num_steps]

        # Normalize advantages across the rollout batch for numerical stability
        adv_slice = self.advantages[:num_steps]
        adv_mean = adv_slice.mean()
        adv_std = adv_slice.std() + 1e-8
        self.advantages[:num_steps] = (adv_slice - adv_mean) / adv_std

    def get_batches(
        self,
        batch_size: int,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Yields shuffled mini-batches over the filled rollout buffer.
        """
        num_steps = self.step_idx
        indices = np.random.permutation(num_steps)

        for start in range(0, num_steps, batch_size):
            end = min(start + batch_size, num_steps)
            batch_idx = indices[start:end]
            batch_tensor_idx = torch.as_tensor(batch_idx, dtype=torch.long, device=self.device)

            batch = {
                "states": self.states[batch_tensor_idx],
                "values": self.values[batch_tensor_idx],
                "returns": self.returns[batch_tensor_idx],
                "advantages": self.advantages[batch_tensor_idx],
                "obs": {ag: self.obs[ag][batch_tensor_idx] for ag in self.agents},
                "actions": {ag: self.actions[ag][batch_tensor_idx] for ag in self.agents},
                "log_probs": {ag: self.log_probs[ag][batch_tensor_idx] for ag in self.agents},
                "action_masks": {ag: self.action_masks[ag][batch_tensor_idx] for ag in self.agents},
            }
            yield batch

    def clear(self) -> None:
        """Resets buffer index for next rollout cycle."""
        self.step_idx = 0
        self.full = False


class MAPPOTrainer:
    """
    Multi-Agent PPO (MAPPO) Trainer implementing Centralized Training with
    Decentralized Execution (CTDE) for the project scheduling agent swarm.
    """

    def __init__(
        self,
        swarm: MultiAgentActorSwarm,
        critic: CentralizedCritic,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.swarm = swarm.to(device)
        self.critic = critic.to(device)
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.swarm.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

    def train_step(self, buffer: MAPPORolloutBuffer) -> Dict[str, float]:
        """
        Executes PPO update epochs using trajectories stored in the rollout buffer.

        Returns:
            Dict containing mean training loss metrics.
        """
        if buffer.step_idx == 0:
            return {}

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        approx_kl_total = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            for batch in buffer.get_batches(self.batch_size):
                b_states = batch["states"]
                b_values = batch["values"]
                b_returns = batch["returns"]
                b_advantages = batch["advantages"]

                # 1. Update Centralized Critic
                val_preds = self.critic(b_states).squeeze(-1)

                # Clipped value objective
                val_clipped = b_values + (val_preds - b_values).clamp(-self.clip_eps, self.clip_eps)
                val_loss_unclipped = (val_preds - b_returns) ** 2
                val_loss_clipped = (val_clipped - b_returns) ** 2
                critic_loss = 0.5 * torch.mean(torch.max(val_loss_unclipped, val_loss_clipped))

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # 2. Update Decentralized Actor Policies
                batch_actor_loss = torch.tensor(0.0, device=self.device)
                batch_entropy = torch.tensor(0.0, device=self.device)
                batch_kl = 0.0

                for ag in self.swarm.agents:
                    ag_obs = batch["obs"][ag]
                    ag_actions = batch["actions"][ag]
                    ag_old_log_probs = batch["log_probs"][ag]
                    ag_masks = batch["action_masks"][ag]

                    curr_log_prob, entropy = self.swarm.evaluate_agent_actions(
                        ag, ag_obs, ag_actions, ag_masks
                    )

                    ratio = torch.exp(curr_log_prob - ag_old_log_probs)
                    surr1 = ratio * b_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages

                    ag_actor_loss = -torch.min(surr1, surr2).mean()
                    batch_actor_loss = batch_actor_loss + ag_actor_loss
                    batch_entropy = batch_entropy + entropy.mean()

                    with torch.no_grad():
                        kl = (ag_old_log_probs - curr_log_prob).mean().item()
                        batch_kl += kl

                actor_total_loss = batch_actor_loss - self.entropy_coeff * batch_entropy

                self.actor_optimizer.zero_grad()
                actor_total_loss.backward()
                nn.utils.clip_grad_norm_(self.swarm.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                total_actor_loss += batch_actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += batch_entropy.item()
                approx_kl_total += batch_kl / max(len(self.swarm.agents), 1)
                num_updates += 1

        metrics = {
            "actor_loss": total_actor_loss / max(num_updates, 1),
            "critic_loss": total_critic_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "approx_kl": approx_kl_total / max(num_updates, 1),
        }
        return metrics

    def save_checkpoint(self, filepath: str) -> None:
        """Saves weights of actor swarm and centralized critic."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(
            {
                "swarm_state_dict": self.swarm.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_opt_state_dict": self.actor_optimizer.state_dict(),
                "critic_opt_state_dict": self.critic_optimizer.state_dict(),
            },
            filepath,
        )
        logger.info("Saved MAPPO checkpoint to %s", filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Loads weights from checkpoint file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.swarm.load_state_dict(checkpoint["swarm_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        if "actor_opt_state_dict" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_opt_state_dict"])
        if "critic_opt_state_dict" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_opt_state_dict"])
        logger.info("Loaded MAPPO checkpoint from %s", filepath)
