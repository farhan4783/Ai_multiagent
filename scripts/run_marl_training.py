#!/usr/bin/env python3
"""
CLI runner for Multi-Agent Reinforcement Learning (MAPPO) training and baseline benchmarking.

Usage:
    python scripts/run_marl_training.py --episodes 50 --devs 3 --tasks 15
    python scripts/run_marl_training.py --eval-only --checkpoint checkpoints/mappo_best.pt
"""

import argparse
import os
import sys
import torch

# Add src to pythonpath if run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from project_marl.training.train_marl import (
    MARLTrainingConfig,
    train_marl,
    evaluate_marl_policy,
)
from project_marl.sim.project_env import ProjectSchedulingEnv
from project_marl.agents.policies import MultiAgentActorSwarm
from project_marl.agents.critics import CentralizedCritic
from project_marl.agents.mappo import MAPPOTrainer
from project_marl.core.logging import setup_logger

logger = setup_logger("run_marl_training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STGAT-MARL Multi-Agent Reinforcement Learning Training")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--rollout-steps", type=int, default=100, help="Rollout steps per PPO iteration")
    parser.add_argument("--devs", type=int, default=3, help="Number of parallel developer agents")
    parser.add_argument("--tasks", type=int, default=15, help="Maximum number of project tasks")
    parser.add_argument("--modules", type=int, default=10, help="Maximum number of codebase modules")
    parser.add_argument("--lr-actor", type=float, default=3e-4, help="Actor learning rate")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="Critic learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only using checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.eval_only:
        ckpt_path = args.checkpoint or os.path.join(args.checkpoint_dir, "mappo_best.pt")
        if not os.path.exists(ckpt_path):
            logger.error("Checkpoint not found at %s", ckpt_path)
            sys.exit(1)

        env = ProjectSchedulingEnv(
            num_developers=args.devs,
            max_tasks=args.tasks,
            max_modules=args.modules,
            seed=args.seed,
        )

        sched_obs_dim = env.observation_space("scheduling_agent").shape[0]
        risk_obs_dim = env.observation_space("risk_agent").shape[0]
        dev_obs_dim = env.observation_space("developer_agent_0").shape[0]
        pm_obs_dim = env.observation_space("pm_agent").shape[0]

        swarm = MultiAgentActorSwarm(
            sched_obs_dim=sched_obs_dim,
            risk_obs_dim=risk_obs_dim,
            dev_obs_dim=dev_obs_dim,
            pm_obs_dim=pm_obs_dim,
            num_developers=args.devs,
            max_tasks=args.tasks,
        )
        critic = CentralizedCritic(state_dim=env.state_size)
        trainer = MAPPOTrainer(swarm=swarm, critic=critic)
        trainer.load_checkpoint(ckpt_path)

        metrics = evaluate_marl_policy(env, swarm, num_episodes=10, seed=args.seed + 500)
        print("\n" + "=" * 60)
        print("          MARL POLICY EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Mean Makespan:               {metrics['eval_makespan']} hrs")
        print(f"  CPM Makespan Baseline:       {metrics['eval_cpm_makespan']} hrs")
        print(f"  Makespan Reduction vs CPM:   {metrics['makespan_reduction_pct']}%")
        print(f"  Mean Cumulative Reward:      {metrics['eval_reward']}")
        print(f"  Mean Merge Conflicts:        {metrics['eval_conflicts']}")
        print(f"  Task Completion Rate:        {metrics['eval_completion_rate'] * 100:.1f}%")
        print("=" * 60 + "\n")
        return

    config = MARLTrainingConfig(
        num_episodes=args.episodes,
        rollout_steps=args.rollout_steps,
        num_developers=args.devs,
        max_tasks=args.tasks,
        max_modules=args.modules,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )

    print("\n" + "=" * 60)
    print("      STGAT-MARL MULTI-AGENT REINFORCEMENT LEARNING")
    print("=" * 60)
    print(f"  Episodes:          {config.num_episodes}")
    print(f"  Rollout Steps:     {config.rollout_steps}")
    print(f"  Developer Agents:  {config.num_developers}")
    print(f"  Max Tasks:         {config.max_tasks}")
    print(f"  Checkpoint Dir:    {config.checkpoint_dir}")
    print("=" * 60 + "\n")

    history = train_marl(config)
    print("\n[SUCCESS] MARL Training completed successfully.")


if __name__ == "__main__":
    main()
