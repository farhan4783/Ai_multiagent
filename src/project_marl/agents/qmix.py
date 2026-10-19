"""
QMIX Value Decomposition Network for Cooperative Multi-Agent Reinforcement Learning.

Implements:
    - AgentQNetwork: Individual utility network Q_i(o_i, a_i).
    - QMixer: Monotonic mixing network driven by hypernetworks conditioned on global state S_t.
    - MultiAgentReplayBuffer: Experience replay buffer for joint transitions.
    - QMIXTrainer: Multi-agent Q-learning with target networks and action masking.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from project_marl.core.logging import setup_logger

logger = setup_logger("qmix")


class AgentQNetwork(nn.Module):
    """
    Individual Agent Utility Network Q_i(o_i, a_i).
    Maps local observation to Q-values for all discrete actions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        layers: List[nn.Module] = []
        prev_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev_dim = h

        self.encoder = nn.Sequential(*layers)
        self.q_head = nn.Linear(prev_dim, action_dim)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes Q-values for all discrete actions.
        Invalid actions are masked with -1e9.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            if action_mask is not None and action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

        feats = self.encoder(obs)
        q_values = self.q_head(feats)

        if action_mask is not None:
            mask_bool = action_mask.bool()
            all_masked = (~mask_bool).all(dim=-1, keepdim=True)
            safe_mask = torch.where(all_masked, torch.ones_like(mask_bool), mask_bool)
            q_values = torch.where(safe_mask, q_values, torch.full_like(q_values, -1e9))

        return q_values


class QMixer(nn.Module):
    """
    QMIX Monotonic Mixing Network.
    Computes Q_tot = f_mix(Q_1, ..., Q_N; S_t).

    Hypernetworks conditioned on the global state S_t generate the non-negative
    weights W1 >= 0 and W2 >= 0, guaranteeing the monotonicity constraint:
        dQ_tot / dQ_i >= 0  for all agents i.
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        mixing_embed_dim: int = 64,
        hypernet_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim

        # Hypernetwork 1: generates W1 of shape (num_agents, embed_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, num_agents * mixing_embed_dim),
        )

        # Hypernetwork bias 1: generates b1 of shape (1, embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)

        # Hypernetwork 2: generates W2 of shape (embed_dim, 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, mixing_embed_dim),
        )

        # State bias 2: state projection to scalar bias
        self.state_bias2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1),
        )

    def forward(
        self,
        agent_qs: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass computing Q_tot.

        Args:
            agent_qs: Tensor of shape (batch_size, num_agents) containing chosen action Q-values.
            state: Tensor of shape (batch_size, state_dim) containing global state S_t.

        Returns:
            Q_tot: Tensor of shape (batch_size, 1).
        """
        batch_size = agent_qs.size(0)
        q_vals = agent_qs.view(batch_size, 1, self.num_agents)

        # Generate W1 >= 0 using torch.abs
        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)

        # Layer 1 message passing with ELU activation
        hidden = F.elu(torch.bmm(q_vals, w1) + b1)

        # Generate W2 >= 0
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.embed_dim, 1)
        b2 = self.state_bias2(state).view(batch_size, 1, 1)

        # Output Q_tot
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch_size, 1)


class MultiAgentReplayBuffer:
    """Replay buffer for transition tuples in cooperative multi-agent environments."""

    def __init__(
        self,
        capacity: int,
        agents: List[str],
    ) -> None:
        self.capacity = capacity
        self.agents = agents
        self.buffer: List[Dict[str, Any]] = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        actions: Dict[str, int],
        reward: float,
        done: bool,
        action_masks: Dict[str, np.ndarray],
        next_action_masks: Dict[str, np.ndarray],
    ) -> None:
        transition = {
            "state": state,
            "next_state": next_state,
            "obs": obs,
            "next_obs": next_obs,
            "actions": actions,
            "reward": reward,
            "done": done,
            "action_masks": action_masks,
            "next_action_masks": next_action_masks,
        }
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, Any]:
        transitions = random.sample(self.buffer, batch_size)
        batch: Dict[str, Any] = {
            "states": np.array([t["state"] for t in transitions], dtype=np.float32),
            "next_states": np.array([t["next_state"] for t in transitions], dtype=np.float32),
            "rewards": np.array([t["reward"] for t in transitions], dtype=np.float32),
            "dones": np.array([t["done"] for t in transitions], dtype=np.float32),
            "obs": {
                ag: np.array([t["obs"][ag] for t in transitions], dtype=np.float32)
                for ag in self.agents
            },
            "next_obs": {
                ag: np.array([t["next_obs"][ag] for t in transitions], dtype=np.float32)
                for ag in self.agents
            },
            "actions": {
                ag: np.array([t["actions"][ag] for t in transitions], dtype=np.int64)
                for ag in self.agents
            },
            "action_masks": {
                ag: np.array([t["action_masks"][ag] for t in transitions], dtype=np.float32)
                for ag in self.agents
            },
            "next_action_masks": {
                ag: np.array([t["next_action_masks"][ag] for t in transitions], dtype=np.float32)
                for ag in self.agents
            },
        }
        return batch

    def __len__(self) -> int:
        return len(self.buffer)


class QMIXTrainer:
    """
    QMIX Trainer coordinating individual agent Q-networks, mixing network,
    target networks, and epsilon-greedy exploration.
    """

    def __init__(
        self,
        agents: List[str],
        state_dim: int,
        agent_obs_dims: Dict[str, int],
        agent_action_dims: Dict[str, int],
        mixing_embed_dim: int = 64,
        lr: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.agents = agents
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Individual Q-networks
        self.q_nets = nn.ModuleDict({
            ag: AgentQNetwork(agent_obs_dims[ag], agent_action_dims[ag])
            for ag in agents
        }).to(device)

        # Mixer network
        self.mixer = QMixer(len(agents), state_dim, mixing_embed_dim).to(device)

        # Target networks
        self.target_q_nets = nn.ModuleDict({
            ag: AgentQNetwork(agent_obs_dims[ag], agent_action_dims[ag])
            for ag in agents
        }).to(device)
        self.target_mixer = QMixer(len(agents), state_dim, mixing_embed_dim).to(device)

        # Initialize target networks with matching weights
        self.target_q_nets.load_state_dict(self.q_nets.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # Parameters & Optimizer
        params = list(self.q_nets.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, eps=1e-5)

    def select_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        action_masks_dict: Dict[str, np.ndarray],
        epsilon: float = 0.05,
    ) -> Dict[str, int]:
        """Epsilon-greedy action selection for each agent with action masking."""
        actions: Dict[str, int] = {}
        for ag in self.agents:
            obs = torch.as_tensor(obs_dict[ag], dtype=torch.float32, device=self.device).unsqueeze(0)
            mask = torch.as_tensor(action_masks_dict[ag], dtype=torch.float32, device=self.device).unsqueeze(0)

            if random.random() < epsilon:
                valid_acts = np.where(action_masks_dict[ag] > 0.5)[0]
                if len(valid_acts) > 0:
                    actions[ag] = int(random.choice(valid_acts))
                else:
                    actions[ag] = 0
            else:
                with torch.no_grad():
                    q_vals = self.q_nets[ag](obs, mask)
                    actions[ag] = int(torch.argmax(q_vals, dim=-1).item())

        return actions

    def train_step(self, batch: Dict[str, Any]) -> float:
        """
        Executes one Q-learning optimization step over a batch of transitions.
        """
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device).unsqueeze(-1)

        batch_size = states.size(0)

        # 1. Current Q-values for chosen actions
        chosen_qs: List[torch.Tensor] = []
        for ag in self.agents:
            obs = torch.as_tensor(batch["obs"][ag], dtype=torch.float32, device=self.device)
            acts = torch.as_tensor(batch["actions"][ag], dtype=torch.long, device=self.device).unsqueeze(-1)
            masks = torch.as_tensor(batch["action_masks"][ag], dtype=torch.float32, device=self.device)

            q_all = self.q_nets[ag](obs, masks)
            chosen_q = q_all.gather(1, acts)
            chosen_qs.append(chosen_q)

        chosen_qs_tensor = torch.cat(chosen_qs, dim=-1)  # (batch_size, num_agents)
        q_tot = self.mixer(chosen_qs_tensor, states)     # (batch_size, 1)

        # 2. Target Q-values: max Q over next valid actions
        with torch.no_grad():
            target_max_qs: List[torch.Tensor] = []
            for ag in self.agents:
                next_o = torch.as_tensor(batch["next_obs"][ag], dtype=torch.float32, device=self.device)
                next_m = torch.as_tensor(batch["next_action_masks"][ag], dtype=torch.float32, device=self.device)

                target_q_all = self.target_q_nets[ag](next_o, next_m)
                target_max_q = target_q_all.max(dim=-1, keepdim=True)[0]
                target_max_qs.append(target_max_q)

            target_max_qs_tensor = torch.cat(target_max_qs, dim=-1)
            target_q_tot = self.target_mixer(target_max_qs_tensor, next_states)
            y = rewards + self.gamma * (1.0 - dones) * target_q_tot

        # 3. Loss & Backprop
        loss = F.mse_loss(q_tot, y)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.q_nets.parameters()) + list(self.mixer.parameters()), 1.0)
        self.optimizer.step()

        # 4. Soft update target networks
        for ag in self.agents:
            for p, tp in zip(self.q_nets[ag].parameters(), self.target_q_nets[ag].parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        for p, tp in zip(self.mixer.parameters(), self.target_mixer.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return float(loss.item())

    def save_checkpoint(self, filepath: str) -> None:
        """Saves QMIX weights."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(
            {
                "q_nets": self.q_nets.state_dict(),
                "mixer": self.mixer.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str) -> None:
        """Loads QMIX weights."""
        ckpt = torch.load(filepath, map_location=self.device)
        self.q_nets.load_state_dict(ckpt["q_nets"])
        self.mixer.load_state_dict(ckpt["mixer"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
