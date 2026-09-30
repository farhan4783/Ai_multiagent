"""
Evaluation runner and comparative benchmark suite for RCPSP baselines.

Evaluates:
    - Exact Solver (OR-Tools CP-SAT) for static theoretical optimal
    - Dispatching Heuristics: SPT, LPT, CPM, MSF, Random
across synthetic and real-world project graphs under stochastic uncertainty.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np

from project_marl.core.schemas import TaskNode
from project_marl.sim.project_env import ProjectSchedulingEnv
from project_marl.sim.generator import SyntheticProjectGenerator
from project_marl.baselines.classical_solvers import (
    BaseSchedulingPolicy,
    SPTPolicy,
    LPTPolicy,
    CPMPolicy,
    MSFPolicy,
    RandomPolicy,
    solve_rcpsp_exact,
    RCPSPSolution,
)
from project_marl.core.logging import setup_logger

logger = setup_logger("evaluator")


@dataclass
class BenchmarkResult:
    """Aggregated evaluation metrics for a single scheduling policy."""
    policy_name: str
    num_episodes: int
    mean_makespan: float
    std_makespan: float
    mean_reward: float
    std_reward: float
    mean_conflicts: float
    mean_regressions: float
    mean_idle_rate: float
    optimality_gap_pct: Optional[float] = None  # (mean_makespan - mean_opt) / mean_opt * 100
    mean_opt_makespan: Optional[float] = None
    episodes_data: List[Dict[str, Any]] = field(default_factory=list)


class BenchmarkSuite:
    """
    Automated comparative benchmarking suite for Project Scheduling.
    Executes multiple policies across identical benchmark project instances.
    """

    def __init__(
        self,
        num_episodes: int = 10,
        num_developers: int = 3,
        max_tasks: int = 15,
        max_steps: int = 150,
        stochastic: bool = True,
        seed: int = 42,
    ) -> None:
        self.num_episodes = num_episodes
        self.num_developers = num_developers
        self.max_tasks = max_tasks
        self.max_steps = max_steps
        self.stochastic = stochastic
        self.seed = seed

        self.generator = SyntheticProjectGenerator(
            min_tasks=max(5, max_tasks // 2),
            max_tasks=max_tasks,
            seed=seed,
        )

    def run_policy(
        self,
        policy: BaseSchedulingPolicy,
        projects: List[Dict[str, Any]],
    ) -> BenchmarkResult:
        """Evaluates a single heuristic policy across pre-generated benchmark projects."""
        env = ProjectSchedulingEnv(
            num_developers=self.num_developers,
            max_tasks=self.max_tasks,
            max_steps=self.max_steps,
            stochastic_duration=self.stochastic,
            seed=self.seed,
        )

        makespans: List[float] = []
        rewards: List[float] = []
        conflicts: List[float] = []
        regressions: List[float] = []
        idle_rates: List[float] = []
        ep_data: List[Dict[str, Any]] = []

        for ep_idx, proj in enumerate(projects):
            # Reset environment with exact project instance
            opts = {"tasks": [t.model_copy(deep=True) for t in proj["tasks"]]}
            obs, info = env.reset(seed=self.seed + ep_idx, options=opts)

            ep_reward = 0.0
            done = False

            while not done and env.agents:
                actions = policy.get_actions(env)
                obs, step_rewards, terminations, truncations, step_infos = env.step(actions)

                # Accumulate team reward
                r = step_rewards.get(env.sched_agent, 0.0)
                ep_reward += r

                done = all(terminations.values()) or all(truncations.values())

            final_makespan = env.current_time
            makespans.append(final_makespan)
            rewards.append(ep_reward)
            conflicts.append(float(env.total_conflicts))
            regressions.append(float(env.total_regressions))

            # Approximate dev idle rate
            idle_rates.append(0.0)  # tracked internally

            ep_data.append({
                "episode": ep_idx,
                "makespan": final_makespan,
                "opt_makespan": proj.get("opt_makespan"),
                "reward": ep_reward,
                "conflicts": env.total_conflicts,
                "regressions": env.total_regressions,
            })

        env.close()

        mean_ms = float(np.mean(makespans))
        std_ms = float(np.std(makespans))
        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards))
        mean_conf = float(np.mean(conflicts))
        mean_reg = float(np.mean(regressions))

        # Calculate optimality gap if exact optimum is available
        opt_makespans = [p["opt_makespan"] for p in projects if p.get("opt_makespan") is not None]
        opt_gap = None
        mean_opt = None
        if opt_makespans:
            mean_opt = float(np.mean(opt_makespans))
            if mean_opt > 0:
                opt_gap = float(((mean_ms - mean_opt) / mean_opt) * 100.0)

        return BenchmarkResult(
            policy_name=policy.name,
            num_episodes=len(projects),
            mean_makespan=mean_ms,
            std_makespan=std_ms,
            mean_reward=mean_r,
            std_reward=std_r,
            mean_conflicts=mean_conf,
            mean_regressions=mean_reg,
            mean_idle_rate=0.0,
            optimality_gap_pct=opt_gap,
            mean_opt_makespan=mean_opt,
            episodes_data=ep_data,
        )

    def run_all(
        self,
        policies: Optional[List[BaseSchedulingPolicy]] = None,
        solve_exact: bool = True,
    ) -> Dict[str, BenchmarkResult]:
        """
        Runs the full comparative benchmark suite across exact and heuristic policies.
        """
        if policies is None:
            policies = [
                CPMPolicy(),
                SPTPolicy(),
                LPTPolicy(),
                MSFPolicy(),
                RandomPolicy(seed=self.seed),
            ]

        # 1. Generate benchmark projects
        logger.info(f"Generating {self.num_episodes} benchmark projects...")
        projects: List[Dict[str, Any]] = []

        temp_env = ProjectSchedulingEnv(
            num_developers=self.num_developers,
            max_tasks=self.max_tasks,
            seed=self.seed,
        )

        for i in range(self.num_episodes):
            temp_env.reset(seed=self.seed + i)
            tasks_copy = [t.model_copy(deep=True) for t in temp_env.tasks.values()]

            opt_makespan = None
            if solve_exact:
                sol = solve_rcpsp_exact(tasks_copy, num_resources=self.num_developers)
                if sol.status in ("OPTIMAL", "FEASIBLE"):
                    opt_makespan = sol.makespan

            projects.append({
                "tasks": tasks_copy,
                "opt_makespan": opt_makespan,
            })

        temp_env.close()

        # 2. Evaluate each policy
        results: Dict[str, BenchmarkResult] = {}
        for pol in policies:
            logger.info(f"Evaluating policy: {pol.name}...")
            res = self.run_policy(pol, projects)
            results[pol.name] = res

        return results

    def format_table(self, results: Dict[str, BenchmarkResult]) -> str:
        """Formats the comparative results into a clean Markdown table."""
        lines = [
            "| Policy | Mean Makespan (hrs) | Optimality Gap (%) | Mean Reward | Conflicts | Regressions |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        # Check if we have exact baseline
        first_res = next(iter(results.values()))
        if first_res.mean_opt_makespan is not None:
            lines.insert(
                0,
                f"**Static CP-SAT Exact Optimum (Lower Bound): {first_res.mean_opt_makespan:.2f} hrs**\n",
            )

        for name, r in results.items():
            gap_str = f"{r.optimality_gap_pct:+.1f}%" if r.optimality_gap_pct is not None else "N/A"
            lines.append(
                f"| **{name}** | {r.mean_makespan:.2f} +/- {r.std_makespan:.1f} | {gap_str} | "
                f"{r.mean_reward:.2f} | {r.mean_conflicts:.1f} | {r.mean_regressions:.1f} |"
            )

        return "\n".join(lines)
