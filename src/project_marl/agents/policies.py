"""
Decentralized Actor Policies with specialized observation encoders and action masking
for the STGAT-MARL autonomous project management framework.

Agents:
    - SchedulingActor: Selects ready task-developer pairings or NO_OP with strict action masking.
    - RiskActor: Selects pending fragile tasks for buffer margin injection or NO_OP.
    - DeveloperActor: Selects execution mode (0: Aggressive, 1: Balanced, 2: Conservative).
    - PMActor: Selects multi-objective reward trade-off weight presets (0: Speed, 1: Balanced, 2: Quality, 3: Efficiency).
    - MultiAgentActorSwarm: Unified coordinator managing all decentralized actor policies.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from project_marl.core.logging import setup_logger

logger = setup_logger("policies")


def init_weights_orthogonal(module: nn.Module, gain: float = 1.0) -> None:
    """Applies orthogonal initialization with prescribed gain to linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class BaseActorPolicy(nn.Module):
    """
    Base decentralized actor policy with MLP observation encoder, action masking,
    and categorical distribution sampling.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128),
        gain: float = 0.01,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Multi-layer feature encoder with LayerNorm and Tanh activation
        layers: List[nn.Module] = []
        prev_dim = obs_dim
        for h_dim in hidden_dims:
            lin = nn.Linear(prev_dim, h_dim)
            init_weights_orthogonal(lin, gain=np.sqrt(2))
            layers.append(lin)
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.Tanh())
            prev_dim = h_dim

        self.encoder = nn.Sequential(*layers)

        # Action logits head (orthogonal init with small gain for uniform start)
        self.action_head = nn.Linear(prev_dim, action_dim)
        init_weights_orthogonal(self.action_head, gain=gain)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Categorical:
        """
        Forward pass returning a Categorical distribution over discrete actions.

        Args:
            obs: Observation tensor of shape (batch_size, obs_dim) or (obs_dim,).
            action_mask: Binary mask of shape (batch_size, action_dim) where 1.0 is valid, 0.0 is invalid.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            if action_mask is not None and action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

        feats = self.encoder(obs)
        logits = self.action_head(feats)

        if action_mask is not None:
            # Mask out invalid actions by assigning large negative logits (-1e9)
            # Ensure at least one valid action exists per row
            mask_bool = action_mask.bool()
            # If all are masked, fallback to allow all
            all_masked = (~mask_bool).all(dim=-1, keepdim=True)
            safe_mask = torch.where(all_masked, torch.ones_like(mask_bool), mask_bool)
            logits = torch.where(safe_mask, logits, torch.full_like(logits, -1e9))

        return Categorical(logits=logits)

    def sample(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples an action from the policy.

        Returns:
            (action, log_prob, entropy)
        """
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if isinstance(action_mask, np.ndarray):
            action_mask = torch.as_tensor(action_mask, dtype=torch.float32)

        dist = self.forward(obs, action_mask)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates log probability and entropy for a given batch of actions.

        Returns:
            (log_probs, entropy)
        """
        dist = self.forward(obs, action_mask)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy


class SchedulingActor(BaseActorPolicy):
    """
    Decentralized Scheduling Policy.
    Maps scheduling observation to eligible task-developer dispatch actions or NO_OP.
    Action index: task_idx * num_devs + dev_idx, or max_tasks * num_devs (NO_OP).
    """

    def __init__(
        self,
        obs_dim: int,
        max_tasks: int,
        num_developers: int,
        hidden_dims: Tuple[int, ...] = (256, 128),
    ) -> None:
        action_dim = max_tasks * num_developers + 1
        super().__init__(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=hidden_dims, gain=0.01)
        self.max_tasks = max_tasks
        self.num_developers = num_developers
        self.noop_action = max_tasks * num_developers


class RiskActor(BaseActorPolicy):
    """
    Decentralized Risk Mitigation Policy.
    Selects pending fragile tasks for dynamic buffer injection or NO_OP.
    Action index: target_task_idx in [0, max_tasks - 1], or max_tasks (NO_OP).
    """

    def __init__(
        self,
        obs_dim: int,
        max_tasks: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
    ) -> None:
        action_dim = max_tasks + 1
        super().__init__(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=hidden_dims, gain=0.01)
        self.max_tasks = max_tasks
        self.noop_action = max_tasks


class DeveloperActor(BaseActorPolicy):
    """
    Decentralized Developer Policy.
    Selects execution mode:
        0: Aggressive (fast commit, higher conflict/failure risk)
        1: Balanced (standard velocity)
        2: Conservative (heavy unit tests, low risk)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__(obs_dim=obs_dim, action_dim=3, hidden_dims=hidden_dims, gain=0.01)


class PMActor(BaseActorPolicy):
    """
    Decentralized PM Policy.
    Selects multi-objective reward trade-off weights:
        0: Speed Priority
        1: Balanced
        2: Quality & Safety Priority
        3: Resource Efficiency Priority
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__(obs_dim=obs_dim, action_dim=4, hidden_dims=hidden_dims, gain=0.01)


class MultiAgentActorSwarm(nn.Module):
    """
    Unified swarm of decentralized actor policies coordinating across all roles.
    Manages scheduling, risk mitigation, developer execution, and PM trade-off policies.
    """

    def __init__(
        self,
        sched_obs_dim: int,
        risk_obs_dim: int,
        dev_obs_dim: int,
        pm_obs_dim: int,
        num_developers: int = 3,
        max_tasks: int = 30,
        share_dev_params: bool = True,
    ) -> None:
        super().__init__()
        self.num_developers = num_developers
        self.max_tasks = max_tasks
        self.share_dev_params = share_dev_params

        self.sched_agent_id = "scheduling_agent"
        self.risk_agent_id = "risk_agent"
        self.dev_agent_ids = [f"developer_agent_{m}" for m in range(num_developers)]
        self.pm_agent_id = "pm_agent"

        self.agents = [self.sched_agent_id, self.risk_agent_id, *self.dev_agent_ids, self.pm_agent_id]

        # Individual actor policies
        self.sched_actor = SchedulingActor(
            obs_dim=sched_obs_dim,
            max_tasks=max_tasks,
            num_developers=num_developers,
        )
        self.risk_actor = RiskActor(
            obs_dim=risk_obs_dim,
            max_tasks=max_tasks,
        )

        if share_dev_params:
            # Single shared developer policy conditioned on one-hot dev ID
            self.shared_dev_actor = DeveloperActor(obs_dim=dev_obs_dim)
            self.dev_actors = nn.ModuleDict(
                {agent_id: self.shared_dev_actor for agent_id in self.dev_agent_ids}
            )
        else:
            self.dev_actors = nn.ModuleDict(
                {agent_id: DeveloperActor(obs_dim=dev_obs_dim) for agent_id in self.dev_agent_ids}
            )

        self.pm_actor = PMActor(obs_dim=pm_obs_dim)

    def get_actor(self, agent_id: str) -> BaseActorPolicy:
        """Returns the specific actor policy module for the given agent."""
        if agent_id == self.sched_agent_id:
            return self.sched_actor
        elif agent_id == self.risk_agent_id:
            return self.risk_actor
        elif agent_id in self.dev_agent_ids:
            return self.dev_actors[agent_id]
        elif agent_id == self.pm_agent_id:
            return self.pm_actor
        else:
            raise KeyError(f"Unknown agent: {agent_id}")

    def act(
        self,
        observations: Dict[str, Union[np.ndarray, torch.Tensor]],
        action_masks: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, int], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Samples joint actions across all agents in the swarm.

        Args:
            observations: Mapping from agent ID to observation tensor / ndarray.
            action_masks: Optional mapping from agent ID to binary action mask.
            deterministic: If True, selects argmax action.

        Returns:
            (actions_dict, log_probs_dict, entropies_dict)
        """
        actions: Dict[str, int] = {}
        log_probs: Dict[str, torch.Tensor] = {}
        entropies: Dict[str, torch.Tensor] = {}

        for agent_id, obs in observations.items():
            if agent_id not in self.agents:
                continue
            actor = self.get_actor(agent_id)
            mask = action_masks.get(agent_id) if action_masks is not None else None
            act, lp, ent = actor.sample(obs, mask, deterministic=deterministic)

            actions[agent_id] = int(act.item()) if act.numel() == 1 else int(act[0].item())
            log_probs[agent_id] = lp.detach().squeeze()
            entropies[agent_id] = ent.detach().squeeze()

        return actions, log_probs, entropies

    def evaluate_agent_actions(
        self,
        agent_id: str,
        obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        action_masks_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates batch log probabilities and entropy for a specific agent."""
        actor = self.get_actor(agent_id)
        return actor.evaluate_actions(obs_batch, actions_batch, action_masks_batch)
