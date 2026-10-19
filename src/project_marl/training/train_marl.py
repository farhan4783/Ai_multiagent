"""
End-to-End Multi-Agent RL Training Pipeline using Centralized Training and
Decentralized Execution (MAPPO) with baseline evaluation against CPM and SPT.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import os
import time
import torch
import numpy as np

from project_marl.sim.project_env import ProjectSchedulingEnv
from project_marl.agents.policies import MultiAgentActorSwarm
from project_marl.agents.critics import CentralizedCritic
from project_marl.agents.mappo import MAPPORolloutBuffer, MAPPOTrainer
from project_marl.baselines.classical_solvers import CPMPolicy, SPTPolicy
from project_marl.core.logging import setup_logger

logger = setup_logger("train_marl")


@dataclass
class MARLTrainingConfig:
    """Hyperparameters and configuration for MAPPO training."""
    num_episodes: int = 60
    rollout_steps: int = 120
    num_developers: int = 3
    max_tasks: int = 20
    max_modules: int = 15
    max_steps_per_episode: int = 100
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    batch_size: int = 64
    ppo_epochs: int = 4
    eval_interval: int = 15
    num_eval_episodes: int = 5
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    device: str = "cpu"


def evaluate_marl_policy(
    env: ProjectSchedulingEnv,
    swarm: MultiAgentActorSwarm,
    num_episodes: int = 5,
    seed: int = 999,
) -> Dict[str, float]:
    """
    Evaluates the current MAPPO actor swarm deterministically on the scheduling environment.

    Returns:
        Dict containing mean makespan, reward, conflict count, test pass rate, and completion rate.
    """
    total_makespan = 0.0
    total_reward = 0.0
    total_conflicts = 0.0
    total_completed = 0.0
    total_cpm_makespan = 0.0

    swarm.eval()

    for ep in range(num_episodes):
        ep_seed = seed + ep * 13
        obs, _ = env.reset(seed=ep_seed)
        cpm_makespan = env.cpm_makespan
        total_cpm_makespan += cpm_makespan

        ep_reward = 0.0
        done = False

        while not done:
            masks = env.get_action_masks()
            with torch.no_grad():
                actions, _, _ = swarm.act(obs, masks, deterministic=True)

            obs, rewards, terminations, truncations, _ = env.step(actions)
            step_reward = float(np.mean(list(rewards.values())))
            ep_reward += step_reward

            done = any(terminations.values()) or any(truncations.values())

        total_makespan += env.current_time
        total_reward += ep_reward
        total_conflicts += env.total_conflicts
        total_completed += len(env.completed_tasks) / max(env.num_tasks_actual, 1)

    swarm.train()

    mean_makespan = total_makespan / num_episodes
    mean_cpm = total_cpm_makespan / num_episodes
    reduction_vs_cpm = ((mean_cpm - mean_makespan) / max(mean_cpm, 1e-4)) * 100.0

    return {
        "eval_makespan": round(mean_makespan, 2),
        "eval_cpm_makespan": round(mean_cpm, 2),
        "eval_reward": round(total_reward / num_episodes, 2),
        "eval_conflicts": round(total_conflicts / num_episodes, 2),
        "eval_completion_rate": round(total_completed / num_episodes, 3),
        "makespan_reduction_pct": round(reduction_vs_cpm, 2),
    }


def train_marl(config: Optional[MARLTrainingConfig] = None) -> Dict[str, List[float]]:
    """
    Executes the MAPPO multi-agent reinforcement learning training loop.
    """
    cfg = config or MARLTrainingConfig()
    device = torch.device(cfg.device)

    # Set seeds for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Initialize Environment
    env = ProjectSchedulingEnv(
        num_developers=cfg.num_developers,
        max_tasks=cfg.max_tasks,
        max_modules=cfg.max_modules,
        max_steps=cfg.max_steps_per_episode,
        seed=cfg.seed,
    )

    # Dimensionalities
    sched_obs_dim = env.observation_space("scheduling_agent").shape[0]
    risk_obs_dim = env.observation_space("risk_agent").shape[0]
    dev_obs_dim = env.observation_space("developer_agent_0").shape[0]
    pm_obs_dim = env.observation_space("pm_agent").shape[0]
    state_dim = env.state_size

    agent_obs_dims = {
        "scheduling_agent": sched_obs_dim,
        "risk_agent": risk_obs_dim,
        **{ag: dev_obs_dim for ag in env.dev_agent_ids},
        "pm_agent": pm_obs_dim,
    }
    agent_action_dims = {
        ag: env.action_space(ag).n for ag in env.possible_agents
    }

    # Initialize Decentralized Swarm & Centralized Critic
    swarm = MultiAgentActorSwarm(
        sched_obs_dim=sched_obs_dim,
        risk_obs_dim=risk_obs_dim,
        dev_obs_dim=dev_obs_dim,
        pm_obs_dim=pm_obs_dim,
        num_developers=cfg.num_developers,
        max_tasks=cfg.max_tasks,
        share_dev_params=True,
    ).to(device)

    critic = CentralizedCritic(
        state_dim=state_dim,
        hidden_dim=256,
        num_res_blocks=2,
    ).to(device)

    # Trainer and Rollout Buffer
    trainer = MAPPOTrainer(
        swarm=swarm,
        critic=critic,
        lr_actor=cfg.lr_actor,
        lr_critic=cfg.lr_critic,
        clip_eps=cfg.clip_eps,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        entropy_coeff=cfg.entropy_coeff,
        value_loss_coeff=cfg.value_loss_coeff,
        batch_size=cfg.batch_size,
        ppo_epochs=cfg.ppo_epochs,
        device=device,
    )

    buffer = MAPPORolloutBuffer(
        buffer_size=cfg.rollout_steps,
        state_dim=state_dim,
        agent_obs_dims=agent_obs_dims,
        agent_action_dims=agent_action_dims,
        device=device,
    )

    logger.info(
        "Starting MARL MAPPO training: %d episodes, rollout_steps=%d, state_dim=%d",
        cfg.num_episodes,
        cfg.rollout_steps,
        state_dim,
    )

    history: Dict[str, List[float]] = {
        "episode_rewards": [],
        "actor_losses": [],
        "critic_losses": [],
        "eval_makespans": [],
    }

    best_eval_makespan = float("inf")
    episodes_completed = 0
    obs, _ = env.reset(seed=cfg.seed)
    curr_state = env.state()
    ep_reward_accum = 0.0

    start_time = time.time()

    while episodes_completed < cfg.num_episodes:
        # Collect rollout transition
        masks = env.get_action_masks()

        with torch.no_grad():
            actions, log_probs, _ = swarm.act(obs, masks, deterministic=False)
            val = critic.evaluate_state(curr_state)

        next_obs, rewards, terminations, truncations, _ = env.step(actions)
        next_state = env.state()

        # Mean reward across agents (cooperative multi-objective setting)
        mean_reward = float(np.mean(list(rewards.values())))
        ep_reward_accum += mean_reward

        done = any(terminations.values()) or any(truncations.values())

        # Insert transition into buffer
        buffer.insert(
            state=curr_state,
            obs_dict=obs,
            actions_dict=actions,
            log_probs_dict=log_probs,
            value=val,
            reward=mean_reward,
            done=done,
            action_masks_dict=masks,
        )

        obs = next_obs
        curr_state = next_state

        if done:
            episodes_completed += 1
            history["episode_rewards"].append(round(ep_reward_accum, 2))
            ep_reward_accum = 0.0
            obs, _ = env.reset(seed=cfg.seed + episodes_completed * 7)
            curr_state = env.state()

        # When rollout buffer is full, trigger MAPPO training update
        if buffer.step_idx >= cfg.rollout_steps:
            with torch.no_grad():
                last_val = critic.evaluate_state(curr_state).item()

            buffer.compute_returns_and_advantages(
                last_value=last_val,
                last_done=done,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )

            metrics = trainer.train_step(buffer)
            buffer.clear()

            if metrics:
                history["actor_losses"].append(metrics.get("actor_loss", 0.0))
                history["critic_losses"].append(metrics.get("critic_loss", 0.0))

        # Periodic Evaluation
        if episodes_completed > 0 and episodes_completed % cfg.eval_interval == 0 and done:
            eval_metrics = evaluate_marl_policy(
                env=env,
                swarm=swarm,
                num_episodes=cfg.num_eval_episodes,
                seed=cfg.seed + 1000,
            )
            history["eval_makespans"].append(eval_metrics["eval_makespan"])
            logger.info(
                "Episode %d/%d | Eval Makespan: %.2f (CPM: %.2f, Delta: %.1f%%) | Reward: %.2f",
                episodes_completed,
                cfg.num_episodes,
                eval_metrics["eval_makespan"],
                eval_metrics["eval_cpm_makespan"],
                eval_metrics["makespan_reduction_pct"],
                eval_metrics["eval_reward"],
            )

            # Checkpoint best model
            if eval_metrics["eval_makespan"] < best_eval_makespan:
                best_eval_makespan = eval_metrics["eval_makespan"]
                trainer.save_checkpoint(os.path.join(cfg.checkpoint_dir, "mappo_best.pt"))

    # Save final model
    trainer.save_checkpoint(os.path.join(cfg.checkpoint_dir, "mappo_latest.pt"))
    elapsed = time.time() - start_time
    logger.info(
        "MARL Training complete in %.2f seconds. Best Eval Makespan: %.2f",
        elapsed,
        best_eval_makespan,
    )

    return history


if __name__ == "__main__":
    train_marl()
