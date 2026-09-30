"""
PettingZoo ParallelEnv implementation for Stochastic RCPSP in STGAT-MARL.

Simulates project scheduling with:
    - Activity-On-Node (AON) precedence constraints
    - Stochastic LogNormal task durations
    - Multi-agent roles: SchedulingAgent, RiskAgent, DeveloperAgents, PMAgent
    - External stochastic events: developer crashes, flaky tests, and merge conflicts
      derived from codebase coupling and graph attention weights
    - Multi-objective global reward balancing makespan, risk, conflicts, idle time, and quality.
"""

import copy
import math
import random
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

from project_marl.core.constants import (
    ExecutionStatus,
    EdgeType,
    AgentRole,
    DEFAULT_FEATURE_DIM,
)
from project_marl.core.schemas import TaskNode, CodebaseModuleNode, DependencyEdge
from project_marl.graph.aon_builder import AONGraphBuilder
from project_marl.sim.generator import SyntheticProjectGenerator, ProjectSample
from project_marl.core.logging import setup_logger

logger = setup_logger("project_env")


# Developer execution mode presets: (duration_multiplier, failure_prob_per_hr, conflict_multiplier)
DEV_MODE_CONFIGS = {
    0: (0.80, 0.18, 1.50),   # Aggressive: fast-commit, higher risk
    1: (1.00, 0.08, 1.00),   # Balanced: standard pace
    2: (1.25, 0.02, 0.60),   # Conservative: heavy unit tests, low risk
}

# PM Agent reward trade-off weight presets: [makespan, risk, conflict, idle, quality]
PM_WEIGHT_PRESETS = {
    0: np.array([2.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),  # Speed Priority
    1: np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),  # Balanced
    2: np.array([0.5, 1.5, 1.5, 0.5, 2.0], dtype=np.float32),  # Quality & Safety Priority
    3: np.array([0.8, 0.5, 0.5, 2.0, 0.8], dtype=np.float32),  # Resource Efficiency Priority
}


class ProjectSchedulingEnv(ParallelEnv):
    """
    PettingZoo Parallel Environment for Stochastic Resource-Constrained Project Scheduling.

    Agents:
        - "scheduling_agent": Dispatches ready tasks to idle developer worktrees.
        - "risk_agent": Injects buffer margins to fragile tasks and triggers preflight tests.
        - "developer_agent_{m}": Selects execution mode (aggressive, balanced, conservative).
        - "pm_agent": Arbitrates multi-objective reward trade-off weights.
    """

    metadata = {"render_modes": ["ansi", "human"], "name": "project_scheduling_v0"}

    def __init__(
        self,
        num_developers: int = 3,
        max_tasks: int = 30,
        max_modules: int = 20,
        max_steps: int = 150,
        time_step_size: float = 1.0,
        stochastic_duration: bool = True,
        delay_noise_sigma: float = 0.3,
        failure_penalty_hours: float = 2.0,
        flaky_test_prob: float = 0.04,
        deadline_factor: float = 1.35,
        generator: Optional[SyntheticProjectGenerator] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.num_developers = num_developers
        self.max_tasks = max_tasks
        self.max_modules = max_modules
        self.max_steps = max_steps
        self.time_step_size = time_step_size
        self.stochastic_duration = stochastic_duration
        self.delay_noise_sigma = delay_noise_sigma
        self.failure_penalty_hours = failure_penalty_hours
        self.flaky_test_prob = flaky_test_prob
        self.deadline_factor = deadline_factor

        self.generator = generator or SyntheticProjectGenerator(
            min_tasks=max(5, max_tasks // 2),
            max_tasks=max_tasks,
            min_modules=max(3, max_modules // 2),
            max_modules=max_modules,
            seed=seed,
        )

        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Agent IDs
        self.sched_agent = "scheduling_agent"
        self.risk_agent_id = "risk_agent"
        self.dev_agent_ids = [f"developer_agent_{m}" for m in range(self.num_developers)]
        self.pm_agent_id = "pm_agent"

        self.possible_agents = [self.sched_agent, self.risk_agent_id, *self.dev_agent_ids, self.pm_agent_id]
        self.agents: List[str] = copy.copy(self.possible_agents)

        # Define Observation Spaces
        # 1. Scheduling Agent:
        #    - 6 summary floats: (pct_pending, pct_in_prog, pct_completed, pct_failed, pct_critical_ready, avg_progress)
        #    - max_tasks floats: ready mask (1.0 = ready, 0.0 = not ready)
        #    - max_tasks floats: normalized estimated durations
        #    - max_tasks floats: is_critical flags
        #    - max_tasks floats: normalized total slack
        #    - 5 * num_devs floats: (is_idle, mode/2.0, health, assigned_task_norm, remaining_work_norm)
        #    - 3 global floats: (current_time/deadline, completed_ratio, active_conflicts/5.0)
        self._sched_obs_dim = 6 + 4 * self.max_tasks + 5 * self.num_developers + 3

        # 2. Risk Agent:
        #    - max_tasks: is_critical flags
        #    - max_tasks: normalized slack
        #    - max_tasks: failure rates
        #    - max_tasks: buffers injected (normalized)
        #    - max_tasks: coupling intensity
        #    - 3 global floats: (risk_index, active_conflicts/5.0, crit_completed_ratio)
        self._risk_obs_dim = 5 * self.max_tasks + 3

        # 3. Developer Agent:
        #    - num_developers: one-hot agent ID
        #    - 3 status floats: (is_idle, mode/2.0, health)
        #    - 6 current task floats: (has_task, norm_duration, remaining_ratio, is_critical, failure_rate, impacted_modules_norm)
        #    - 3 history floats: (conflicts_count/5.0, tasks_done/10.0, recent_fail_flag)
        self._dev_obs_dim = self.num_developers + 3 + 6 + 3

        # 4. PM Agent:
        #    - 3 progress floats: (completed_ratio, current_time/deadline, is_overdue)
        #    - 3 quality floats: (tests_passed/max_tasks, regressions/max_tasks, conflicts/10.0)
        #    - 2 resource floats: (idle_rate, avg_health)
        #    - 5 weight floats: active lambda weights
        self._pm_obs_dim = 3 + 3 + 2 + 5

        self.observation_spaces = {
            self.sched_agent: spaces.Box(low=-np.inf, high=np.inf, shape=(self._sched_obs_dim,), dtype=np.float32),
            self.risk_agent_id: spaces.Box(low=-np.inf, high=np.inf, shape=(self._risk_obs_dim,), dtype=np.float32),
            **{
                agent_id: spaces.Box(low=-np.inf, high=np.inf, shape=(self._dev_obs_dim,), dtype=np.float32)
                for agent_id in self.dev_agent_ids
            },
            self.pm_agent_id: spaces.Box(low=-np.inf, high=np.inf, shape=(self._pm_obs_dim,), dtype=np.float32),
        }

        # Define Action Spaces
        # Scheduling Agent: Discrete action:
        # Index in [0, max_tasks * num_developers - 1] -> task_idx = a // num_devs, dev_idx = a % num_devs
        # Index max_tasks * num_developers -> NO_OP
        self.action_spaces = {
            self.sched_agent: spaces.Discrete(self.max_tasks * self.num_developers + 1),
            # Risk Agent: index in [0, max_tasks - 1] -> inject buffer (+2.0 hrs) to task, index max_tasks -> NO_OP
            self.risk_agent_id: spaces.Discrete(self.max_tasks + 1),
            # Developer Agents: 0 = Aggressive, 1 = Balanced, 2 = Conservative
            **{agent_id: spaces.Discrete(3) for agent_id in self.dev_agent_ids},
            # PM Agent: 0 = Speed, 1 = Balanced, 2 = Quality, 3 = Resource Efficiency
            self.pm_agent_id: spaces.Discrete(4),
        }

        # Simulation dynamic state variables
        self.tasks: Dict[str, TaskNode] = {}
        self.task_list: List[str] = []
        self.num_tasks_actual: int = 0
        self.task_remaining_work: Dict[str, float] = {}
        self.task_buffers: Dict[str, float] = {}
        self.task_actual_durations: Dict[str, float] = {}
        self.completed_tasks: Set[str] = set()

        self.dev_current_task: List[Optional[str]] = [None] * self.num_developers
        self.dev_modes: List[int] = [1] * self.num_developers  # default Balanced
        self.dev_health: List[float] = [1.0] * self.num_developers
        self.dev_tasks_completed: List[int] = [0] * self.num_developers
        self.dev_recent_fails: List[float] = [0.0] * self.num_developers

        self.coupling_matrix: np.ndarray = np.zeros((self.max_tasks, self.max_tasks), dtype=np.float32)

        self.current_time: float = 0.0
        self.step_count: int = 0
        self.cpm_makespan: float = 0.0
        self.deadline: float = 0.0
        self.previous_estimated_makespan: float = 0.0

        self.active_conflicts_count: int = 0
        self.total_conflicts: int = 0
        self.total_tests_passed: int = 0
        self.total_regressions: int = 0
        self.active_weights: np.ndarray = PM_WEIGHT_PRESETS[1].copy()

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """Resets the environment to start a new project scheduling episode."""
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_rng = np.random.RandomState(seed)

        self.agents = copy.copy(self.possible_agents)
        self.current_time = 0.0
        self.step_count = 0
        self.completed_tasks = set()
        self.active_conflicts_count = 0
        self.total_conflicts = 0
        self.total_tests_passed = 0
        self.total_regressions = 0
        self.active_weights = PM_WEIGHT_PRESETS[1].copy()

        # Initialize Developer States
        self.dev_current_task = [None] * self.num_developers
        self.dev_modes = [1] * self.num_developers
        self.dev_health = [1.0] * self.num_developers
        self.dev_tasks_completed = [0] * self.num_developers
        self.dev_recent_fails = [0.0] * self.num_developers

        # Load or generate project DAG
        if options and "tasks" in options:
            self._load_custom_project(options["tasks"], options.get("modules", []))
        else:
            self._generate_project()

        # Compute initial CPM and deadlines
        aon = AONGraphBuilder()
        aon.add_tasks(list(self.tasks.values()))
        self.cpm_makespan, _ = aon.compute_cpm()
        self.deadline = self.cpm_makespan * self.deadline_factor
        self.previous_estimated_makespan = self.cpm_makespan

        # Initialize remaining work and sampled durations
        self.task_remaining_work = {}
        self.task_buffers = {t_id: 0.0 for t_id in self.task_list}
        self.task_actual_durations = {}

        for t_id, task in self.tasks.items():
            task.status = ExecutionStatus.PENDING
            task.elapsed_duration = 0.0
            task.assigned_agent_id = None
            if self.stochastic_duration:
                # LogNormal distribution: E[X] = est_duration
                sigma = self.delay_noise_sigma
                mu = math.log(max(task.estimated_duration, 0.5)) - 0.5 * (sigma ** 2)
                sampled_dur = float(self.np_rng.lognormal(mean=mu, sigma=sigma))
                sampled_dur = max(0.5, round(sampled_dur, 2))
            else:
                sampled_dur = task.estimated_duration
            self.task_actual_durations[t_id] = sampled_dur
            self.task_remaining_work[t_id] = sampled_dur

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(
        self,
        actions: Dict[str, Any],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """
        Executes one discrete step in the simulation:
            1. Updates PM trade-off weights.
            2. Injects Risk buffer margins.
            3. Sets Developer execution modes.
            4. Dispatches Scheduling actions.
            5. Advances time by time_step_size, simulating progress, failures, and merge conflicts.
            6. Computes multi-objective rewards and check termination.
        """
        self.step_count += 1
        conflicts_this_step = 0
        failures_this_step = 0
        completed_this_step = 0

        # 1. PM Agent Action: Selects reward trade-off weights
        if self.pm_agent_id in actions:
            pm_action = int(actions[self.pm_agent_id])
            if pm_action in PM_WEIGHT_PRESETS:
                self.active_weights = PM_WEIGHT_PRESETS[pm_action].copy()

        # 2. Risk Agent Action: Inject buffer to a specific task
        if self.risk_agent_id in actions:
            risk_act = int(actions[self.risk_agent_id])
            if 0 <= risk_act < self.num_tasks_actual:
                target_task_id = self.task_list[risk_act]
                if self.tasks[target_task_id].status == ExecutionStatus.PENDING:
                    self.task_buffers[target_task_id] += 2.0  # +2 hours buffer protection

        # 3. Developer Agents Actions: Select execution mode
        for m, dev_id in enumerate(self.dev_agent_ids):
            if dev_id in actions:
                mode = int(actions[dev_id])
                if mode in (0, 1, 2):
                    self.dev_modes[m] = mode

        # 4. Scheduling Agent Action: Task-Agent dispatch
        if self.sched_agent in actions:
            sched_act = int(actions[self.sched_agent])
            if 0 <= sched_act < self.max_tasks * self.num_developers:
                task_idx = sched_act // self.num_developers
                dev_idx = sched_act % self.num_developers

                if task_idx < self.num_tasks_actual and 0 <= dev_idx < self.num_developers:
                    t_id = self.task_list[task_idx]
                    task = self.tasks[t_id]
                    # Check eligibility: task is ready and developer is idle
                    if task.is_ready(self.completed_tasks) and self.dev_current_task[dev_idx] is None:
                        task.status = ExecutionStatus.IN_PROGRESS
                        task.assigned_agent_id = self.dev_agent_ids[dev_idx]
                        self.dev_current_task[dev_idx] = t_id

        # 5. Advance Simulation Time
        self.current_time += self.time_step_size

        # Check for merge conflicts among concurrently running tasks
        active_task_indices = [
            self.task_list.index(t_id)
            for t_id in self.dev_current_task
            if t_id is not None
        ]

        if len(active_task_indices) >= 2:
            for i in range(len(active_task_indices)):
                for j in range(i + 1, len(active_task_indices)):
                    idx_a = active_task_indices[i]
                    idx_b = active_task_indices[j]
                    coupling = self.coupling_matrix[idx_a, idx_b]
                    if coupling > 0.15:
                        dev_a_idx = self._find_dev_for_task(self.task_list[idx_a])
                        dev_b_idx = self._find_dev_for_task(self.task_list[idx_b])
                        mode_factor = 1.0
                        if dev_a_idx is not None and dev_b_idx is not None:
                            mode_factor = (
                                DEV_MODE_CONFIGS[self.dev_modes[dev_a_idx]][2]
                                * DEV_MODE_CONFIGS[self.dev_modes[dev_b_idx]][2]
                            )
                        conflict_prob = min(0.60, coupling * 0.40 * mode_factor)
                        if self.rng.random() < conflict_prob:
                            # Merge conflict triggered: adds delay to both tasks
                            self.task_remaining_work[self.task_list[idx_a]] += 1.0
                            self.task_remaining_work[self.task_list[idx_b]] += 1.0
                            conflicts_this_step += 1
                            self.total_conflicts += 1

        self.active_conflicts_count = conflicts_this_step

        # Process Developer Work & External Failures
        idle_dev_count = 0
        for m in range(self.num_developers):
            t_id = self.dev_current_task[m]
            if t_id is None:
                idle_dev_count += 1
                # Idle recovery for dev health
                self.dev_health[m] = min(1.0, self.dev_health[m] + 0.05)
                self.dev_recent_fails[m] = max(0.0, self.dev_recent_fails[m] - 0.2)
                continue

            task = self.tasks[t_id]
            mode = self.dev_modes[m]
            speed_mult, fail_rate, _ = DEV_MODE_CONFIGS[mode]

            # Check for failure / bug crash
            effective_fail_rate = fail_rate * (1.0 + (1.0 - self.dev_health[m]))
            if self.rng.random() < effective_fail_rate * self.time_step_size:
                # Failure event: rework penalty
                self.task_remaining_work[t_id] += self.failure_penalty_hours
                self.dev_health[m] = max(0.2, self.dev_health[m] - 0.15)
                self.dev_recent_fails[m] = 1.0
                failures_this_step += 1
                self.total_regressions += 1
                task.status = ExecutionStatus.BLOCKED
                continue

            # Advance work on task
            task.status = ExecutionStatus.IN_PROGRESS
            work_done = self.time_step_size * speed_mult
            self.task_remaining_work[t_id] -= work_done
            task.elapsed_duration += self.time_step_size

            # Check if task is finished
            if self.task_remaining_work[t_id] <= 0:
                # Check for flaky test detection
                if self.rng.random() < self.flaky_test_prob:
                    # Flaky test detected: requires rerun
                    self.task_remaining_work[t_id] = 1.0
                    self.total_regressions += 1
                else:
                    # Completed successfully!
                    task.status = ExecutionStatus.COMPLETED
                    self.completed_tasks.add(t_id)
                    self.dev_current_task[m] = None
                    self.dev_tasks_completed[m] += 1
                    completed_this_step += 1
                    self.total_tests_passed += 1

        # 6. Multi-Objective Reward Calculation
        # Estimate dynamic makespan
        current_est_makespan = self._estimate_current_makespan()
        makespan_delta = current_est_makespan - self.previous_estimated_makespan
        self.previous_estimated_makespan = current_est_makespan

        # R_makespan: penalize makespan expansion and deadline overrun
        overdue_penalty = 2.0 if self.current_time > self.deadline else 0.0
        r_makespan = -max(0.0, makespan_delta) - overdue_penalty

        # R_risk: structural fragility of remaining tasks
        r_risk = -self._compute_current_fragility()

        # R_conflict: penalize merge disputes
        r_conflict = -1.5 * conflicts_this_step

        # R_idle: penalize idle developer capacity when tasks are ready
        num_ready = len(self._get_ready_tasks())
        r_idle = -float(idle_dev_count) if num_ready > 0 else 0.0

        # R_quality: reward completed tasks and penalize failures
        r_quality = (1.5 * completed_this_step) - (2.0 * failures_this_step)

        # Composite Global Reward
        w = self.active_weights
        global_reward = float(
            w[0] * r_makespan
            + w[1] * r_risk
            + w[2] * r_conflict
            + w[3] * r_idle
            + w[4] * r_quality
        )

        # 7. Check Terminations & Truncations
        terminated = len(self.completed_tasks) >= self.num_tasks_actual
        truncated = self.step_count >= self.max_steps

        # PettingZoo standard: dicts for all currently active agents
        rewards = {agent: global_reward for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

        infos = {
            agent: {
                "current_time": self.current_time,
                "completed_tasks": len(self.completed_tasks),
                "total_tasks": self.num_tasks_actual,
                "makespan_estimate": current_est_makespan,
                "deadline": self.deadline,
                "conflicts_step": conflicts_this_step,
                "failures_step": failures_this_step,
                "active_weights": self.active_weights.tolist(),
            }
            for agent in self.agents
        }

        # If terminated or truncated, clear self.agents per PettingZoo ParallelEnv standard
        if terminated or truncated:
            self.agents = []

        observations = {agent: self._get_observation(agent) for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def render(self, mode: str = "ansi") -> str:
        """Renders the current scheduling state as an ASCII dashboard."""
        lines = [
            f"=== Project Scheduling Sim (Step {self.step_count}, T={self.current_time:.1f}h / Deadline={self.deadline:.1f}h) ===",
            f"Progress: {len(self.completed_tasks)}/{self.num_tasks_actual} tasks completed | Conflicts: {self.total_conflicts} | Regressions: {self.total_regressions}",
            f"Active Weights: Speed={self.active_weights[0]:.1f}, Risk={self.active_weights[1]:.1f}, Conflict={self.active_weights[2]:.1f}, Idle={self.active_weights[3]:.1f}, Quality={self.active_weights[4]:.1f}",
            "Developer Worktrees:",
        ]
        for m in range(self.num_developers):
            curr = self.dev_current_task[m]
            mode_name = ["Aggressive", "Balanced", "Conservative"][self.dev_modes[m]]
            task_info = f"Working on {curr} (rem={self.task_remaining_work.get(curr, 0.0):.1f}h)" if curr else "IDLE"
            lines.append(f"  [Dev {m}] Mode: {mode_name:<12} Health: {self.dev_health[m]:.2f} | {task_info}")
        
        ready = [t.task_id for t in self._get_ready_tasks()]
        lines.append(f"Ready Tasks ({len(ready)}): {', '.join(ready[:6])}{'...' if len(ready) > 6 else ''}")
        rendered_str = "\n".join(lines)
        if mode == "human":
            print(rendered_str)
        return rendered_str

    def close(self) -> None:
        pass

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _get_observation(self, agent: str) -> np.ndarray:
        """Constructs fixed-size observation vector for a given agent."""
        if agent == self.sched_agent:
            return self._get_sched_observation()
        elif agent == self.risk_agent_id:
            return self._get_risk_observation()
        elif agent in self.dev_agent_ids:
            dev_idx = self.dev_agent_ids.index(agent)
            return self._get_dev_observation(dev_idx)
        elif agent == self.pm_agent_id:
            return self._get_pm_observation()
        else:
            raise KeyError(f"Unknown agent: {agent}")

    def _get_sched_observation(self) -> np.ndarray:
        obs = np.zeros(self._sched_obs_dim, dtype=np.float32)
        n = max(self.num_tasks_actual, 1)

        # 6 summary floats
        pending_cnt = sum(1 for t in self.tasks.values() if t.status == ExecutionStatus.PENDING)
        in_prog_cnt = sum(1 for t in self.tasks.values() if t.status == ExecutionStatus.IN_PROGRESS)
        failed_cnt = sum(1 for t in self.tasks.values() if t.status == ExecutionStatus.BLOCKED)
        crit_ready_cnt = sum(1 for t in self._get_ready_tasks() if t.is_critical)
        avg_progress = (
            np.mean([
                1.0 - (self.task_remaining_work.get(t_id, 0.0) / max(self.task_actual_durations.get(t_id, 1.0), 0.1))
                for t_id in self.task_list
            ])
            if self.task_list
            else 0.0
        )

        obs[0] = pending_cnt / n
        obs[1] = in_prog_cnt / n
        obs[2] = len(self.completed_tasks) / n
        obs[3] = failed_cnt / n
        obs[4] = crit_ready_cnt / n
        obs[5] = float(avg_progress)

        idx = 6
        ready_set = {t.task_id for t in self._get_ready_tasks()}

        # max_tasks floats: ready mask
        for i in range(self.max_tasks):
            if i < self.num_tasks_actual and self.task_list[i] in ready_set:
                obs[idx + i] = 1.0
        idx += self.max_tasks

        # max_tasks floats: durations
        for i in range(self.max_tasks):
            if i < self.num_tasks_actual:
                obs[idx + i] = min(self.tasks[self.task_list[i]].estimated_duration / 40.0, 3.0)
        idx += self.max_tasks

        # max_tasks floats: is_critical
        for i in range(self.max_tasks):
            if i < self.num_tasks_actual and self.tasks[self.task_list[i]].is_critical:
                obs[idx + i] = 1.0
        idx += self.max_tasks

        # max_tasks floats: slack
        for i in range(self.max_tasks):
            if i < self.num_tasks_actual:
                obs[idx + i] = min(self.tasks[self.task_list[i]].total_slack / 20.0, 3.0)
        idx += self.max_tasks

        # 5 * num_devs floats: developer states
        for m in range(self.num_developers):
            curr = self.dev_current_task[m]
            obs[idx] = 1.0 if curr is None else 0.0
            obs[idx + 1] = self.dev_modes[m] / 2.0
            obs[idx + 2] = self.dev_health[m]
            obs[idx + 3] = (self.task_list.index(curr) / self.max_tasks) if curr in self.task_list else -1.0
            obs[idx + 4] = min(self.task_remaining_work.get(curr, 0.0) / 20.0, 3.0) if curr else 0.0
            idx += 5

        # 3 global floats
        obs[idx] = min(self.current_time / max(self.deadline, 1.0), 3.0)
        obs[idx + 1] = len(self.completed_tasks) / n
        obs[idx + 2] = min(self.active_conflicts_count / 5.0, 2.0)

        return obs

    def _get_risk_observation(self) -> np.ndarray:
        obs = np.zeros(self._risk_obs_dim, dtype=np.float32)
        n = max(self.num_tasks_actual, 1)

        idx = 0
        # 1. Criticality
        for i in range(self.max_tasks):
            if i < self.num_tasks_actual and self.tasks[self.task_list[i]].is_critical:
                obs[idx + i] = 1.0
        idx += self.max_tasks

        # 2. Slack
        for i in range(self.max_tasks):
            if i < self.num_tasks_actual:
                obs[idx + i] = min(self.tasks[self.task_list[i]].total_slack / 20.0, 3.0)
        idx += self.max_tasks

        # 3. Failure rates
        for i in range(self.max_tasks):
            if i < self.num_tasks_actual:
                obs[idx + i] = self.tasks[self.task_list[i]].historical_failure_rate
        idx += self.max_tasks

        # 4. Buffers injected
        for i in range(self.max_tasks):
            if i < self.num_tasks_actual:
                obs[idx + i] = min(self.task_buffers.get(self.task_list[i], 0.0) / 10.0, 2.0)
        idx += self.max_tasks

        # 5. Coupling intensity
        for i in range(self.max_tasks):
            if i < self.num_tasks_actual:
                obs[idx + i] = min(float(np.sum(self.coupling_matrix[i, :])) / 5.0, 3.0)
        idx += self.max_tasks

        # 3 global floats
        crit_tasks = [t for t in self.tasks.values() if t.is_critical]
        crit_done = sum(1 for t in crit_tasks if t.task_id in self.completed_tasks)
        obs[idx] = min(self._compute_current_fragility() / 10.0, 3.0)
        obs[idx + 1] = min(self.active_conflicts_count / 5.0, 2.0)
        obs[idx + 2] = crit_done / max(len(crit_tasks), 1)

        return obs

    def _get_dev_observation(self, dev_idx: int) -> np.ndarray:
        obs = np.zeros(self._dev_obs_dim, dtype=np.float32)

        # Developer identity one-hot
        if dev_idx < self.num_developers:
            obs[dev_idx] = 1.0
        idx = self.num_developers

        # 3 status floats
        curr = self.dev_current_task[dev_idx]
        obs[idx] = 1.0 if curr is None else 0.0
        obs[idx + 1] = self.dev_modes[dev_idx] / 2.0
        obs[idx + 2] = self.dev_health[dev_idx]
        idx += 3

        # 6 current task floats
        if curr and curr in self.tasks:
            task = self.tasks[curr]
            tot_dur = max(self.task_actual_durations.get(curr, 1.0), 0.1)
            rem = self.task_remaining_work.get(curr, 0.0)
            obs[idx] = 1.0
            obs[idx + 1] = min(task.estimated_duration / 40.0, 3.0)
            obs[idx + 2] = min(rem / tot_dur, 1.0)
            obs[idx + 3] = 1.0 if task.is_critical else 0.0
            obs[idx + 4] = task.historical_failure_rate
            obs[idx + 5] = min(len(task.impacted_modules) / 5.0, 2.0)
        idx += 6

        # 3 history floats
        obs[idx] = min(self.total_conflicts / 10.0, 2.0)
        obs[idx + 1] = min(self.dev_tasks_completed[dev_idx] / 10.0, 3.0)
        obs[idx + 2] = self.dev_recent_fails[dev_idx]

        return obs

    def _get_pm_observation(self) -> np.ndarray:
        obs = np.zeros(self._pm_obs_dim, dtype=np.float32)
        n = max(self.num_tasks_actual, 1)

        # 3 progress floats
        obs[0] = len(self.completed_tasks) / n
        obs[1] = min(self.current_time / max(self.deadline, 1.0), 3.0)
        obs[2] = 1.0 if self.current_time > self.deadline else 0.0

        # 3 quality floats
        obs[3] = min(self.total_tests_passed / n, 2.0)
        obs[4] = min(self.total_regressions / n, 2.0)
        obs[5] = min(self.total_conflicts / 10.0, 2.0)

        # 2 resource floats
        idle_count = sum(1 for curr in self.dev_current_task if curr is None)
        obs[6] = idle_count / max(self.num_developers, 1)
        obs[7] = float(np.mean(self.dev_health))

        # 5 active lambda weights
        obs[8:13] = self.active_weights[:5]

        return obs

    def _get_ready_tasks(self) -> List[TaskNode]:
        """Returns pending tasks whose predecessor dependencies are fully satisfied."""
        return [
            t for t in self.tasks.values()
            if t.is_ready(self.completed_tasks)
        ]

    def _find_dev_for_task(self, task_id: str) -> Optional[int]:
        for m, t in enumerate(self.dev_current_task):
            if t == task_id:
                return m
        return None

    def _compute_current_fragility(self) -> float:
        """Computes structural risk index based on remaining critical path and coupling."""
        risk = 0.0
        for t_id in self.task_list:
            if t_id not in self.completed_tasks:
                task = self.tasks[t_id]
                crit_factor = 2.0 if task.is_critical else 1.0
                fail_factor = 1.0 + task.historical_failure_rate
                risk += crit_factor * fail_factor * (1.0 / (1.0 + self.task_buffers.get(t_id, 0.0)))
        return min(risk / max(self.num_tasks_actual, 1), 5.0)

    def _estimate_current_makespan(self) -> float:
        """Estimates remaining project makespan based on remaining work and CPM."""
        remaining_tasks = {
            t_id: TaskNode(
                task_id=t_id,
                title=t.title,
                estimated_duration=max(self.task_remaining_work.get(t_id, t.estimated_duration), 0.1),
                predecessors=[p for p in t.predecessors if p not in self.completed_tasks],
            )
            for t_id, t in self.tasks.items()
            if t_id not in self.completed_tasks
        }
        if not remaining_tasks:
            return self.current_time

        builder = AONGraphBuilder()
        builder.add_tasks(list(remaining_tasks.values()))
        rem_makespan, _ = builder.compute_cpm()
        return self.current_time + rem_makespan

    def _generate_project(self) -> None:
        """Generates a synthetic project using the generator."""
        sample: ProjectSample = self.generator.generate_one()
        # Parse tasks from sample
        num_tasks = sample.num_tasks
        self.num_tasks_actual = min(num_tasks, self.max_tasks)
        self.task_list = [f"TASK-{i:03d}" for i in range(self.num_tasks_actual)]
        self.tasks = {}

        # Reconstruct tasks
        for i, t_id in enumerate(self.task_list):
            dur = float(sample.x[i, 0]) * 40.0
            dur = max(dur, 1.0)
            self.tasks[t_id] = TaskNode(
                task_id=t_id,
                title=f"Synthetic Task {i}",
                estimated_duration=dur,
                historical_failure_rate=float(sample.x[i, 10]) if sample.x.shape[1] > 10 else 0.05,
                priority=max(1, min(5, int(float(sample.x[i, 7]) * 5.0))),
                impacted_modules=[f"module_{i % 5}"],
            )

        # Extract edges from sample edge_index
        src_nodes = sample.edge_index[0]
        dst_nodes = sample.edge_index[1]
        for src, dst in zip(src_nodes, dst_nodes):
            if src < self.num_tasks_actual and dst < self.num_tasks_actual and src != dst:
                src_id = self.task_list[src]
                dst_id = self.task_list[dst]
                if src_id not in self.tasks[dst_id].predecessors:
                    self.tasks[dst_id].predecessors.append(src_id)

        # Build coupling matrix from task overlaps
        self.coupling_matrix = np.zeros((self.max_tasks, self.max_tasks), dtype=np.float32)
        for i in range(self.num_tasks_actual):
            for j in range(i + 1, self.num_tasks_actual):
                mods_i = set(self.tasks[self.task_list[i]].impacted_modules)
                mods_j = set(self.tasks[self.task_list[j]].impacted_modules)
                overlap = len(mods_i.intersection(mods_j))
                if overlap > 0:
                    coupling_val = min(1.0, 0.4 * overlap)
                    self.coupling_matrix[i, j] = coupling_val
                    self.coupling_matrix[j, i] = coupling_val

    def _load_custom_project(self, tasks: List[TaskNode], modules: List[CodebaseModuleNode]) -> None:
        """Loads user-supplied tasks and modules."""
        self.num_tasks_actual = min(len(tasks), self.max_tasks)
        self.tasks = {t.task_id: t for t in tasks[:self.num_tasks_actual]}
        self.task_list = list(self.tasks.keys())

        # Build coupling matrix based on shared modules
        self.coupling_matrix = np.zeros((self.max_tasks, self.max_tasks), dtype=np.float32)
        for i, t_i in enumerate(self.task_list):
            for j in range(i + 1, self.num_tasks_actual):
                t_j = self.task_list[j]
                mods_i = set(self.tasks[t_i].impacted_modules)
                mods_j = set(self.tasks[t_j].impacted_modules)
                overlap = len(mods_i.intersection(mods_j))
                if overlap > 0:
                    c_val = min(1.0, 0.5 * overlap)
                    self.coupling_matrix[i, j] = c_val
                    self.coupling_matrix[j, i] = c_val
