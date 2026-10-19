"""
Ray RLlib Distributed CTDE MARL Training Integration for ProjectSchedulingEnv.

Provides multi-agent policy mapping and cluster scaling adapters for PettingZoo
environments when running distributed rollouts across Ray worker nodes.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import os
import sys

from project_marl.core.logging import setup_logger

logger = setup_logger("train_marl_ray")


def env_creator(config: Optional[Dict[str, Any]] = None) -> Any:
    """Creates a PettingZoo ProjectSchedulingEnv instance wrapped for RLlib."""
    from project_marl.sim.project_env import ProjectSchedulingEnv

    cfg = config or {}
    num_devs = cfg.get("num_developers", 3)
    max_tasks = cfg.get("max_tasks", 20)
    seed = cfg.get("seed", 42)

    return ProjectSchedulingEnv(
        num_developers=num_devs,
        max_tasks=max_tasks,
        seed=seed,
    )


def policy_mapping_fn(agent_id: str, episode: Any = None, worker: Any = None, **kwargs: Any) -> str:
    """
    Maps agent IDs to specialized policy names:
        - "scheduling_agent" -> "sched_policy"
        - "risk_agent" -> "risk_policy"
        - "developer_agent_*" -> "dev_policy"
        - "pm_agent" -> "pm_policy"
    """
    if agent_id == "scheduling_agent":
        return "sched_policy"
    elif agent_id == "risk_agent":
        return "risk_policy"
    elif agent_id.startswith("developer_agent_"):
        return "dev_policy"
    elif agent_id == "pm_agent":
        return "pm_policy"
    else:
        return "default_policy"


def build_rllib_config(
    num_workers: int = 2,
    num_gpus: int = 0,
    num_developers: int = 3,
    max_tasks: int = 20,
) -> Any:
    """
    Builds Ray RLlib PPO / MAPPO multi-agent configuration dictionary.
    """
    try:
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.tune.registry import register_env
        from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

        register_env(
            "project_scheduling_env",
            lambda cfg: PettingZooEnv(env_creator(cfg)),
        )

        dummy_env = env_creator({"num_developers": num_developers, "max_tasks": max_tasks})
        policies = {
            "sched_policy": (
                None,
                dummy_env.observation_space("scheduling_agent"),
                dummy_env.action_space("scheduling_agent"),
                {},
            ),
            "risk_policy": (
                None,
                dummy_env.observation_space("risk_agent"),
                dummy_env.action_space("risk_agent"),
                {},
            ),
            "dev_policy": (
                None,
                dummy_env.observation_space("developer_agent_0"),
                dummy_env.action_space("developer_agent_0"),
                {},
            ),
            "pm_policy": (
                None,
                dummy_env.observation_space("pm_agent"),
                dummy_env.action_space("pm_agent"),
                {},
            ),
        }

        config = (
            PPOConfig()
            .environment("project_scheduling_env", env_config={"num_developers": num_developers, "max_tasks": max_tasks})
            .framework("torch")
            .resources(num_gpus=num_gpus)
            .rollouts(num_rollout_workers=num_workers)
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
            )
            .training(
                gamma=0.99,
                lr=3e-4,
                clip_param=0.2,
                entropy_coeff=0.01,
            )
        )
        return config

    except ImportError:
        logger.warning("Ray / RLlib is not installed. Use project_marl.training.train_marl for native PyTorch training.")
        return None


def run_ray_marl(
    num_iterations: int = 50,
    num_workers: int = 2,
) -> None:
    """Entry point for executing Ray RLlib distributed MARL training."""
    try:
        import ray
        ray.init(ignore_reinit_error=True)
        config = build_rllib_config(num_workers=num_workers)
        if config is None:
            return
        algo = config.build()
        for i in range(num_iterations):
            result = algo.train()
            logger.info("Ray Iteration %d: Episode Reward Mean = %.2f", i, result.get("episode_reward_mean", 0.0))
    except ImportError:
        logger.error("Ray / RLlib is required to execute distributed cluster training.")
