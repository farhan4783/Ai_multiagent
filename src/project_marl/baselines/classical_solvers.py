"""
Classical optimization baselines and heuristic dispatching rules for RCPSP.

Includes:
    - Heuristic Dispatching Rules:
        * SPT (Shortest Processing Time)
        * LPT (Longest Processing Time)
        * CPM (Critical Path Method priority / Most Work Remaining)
        * MSF (Minimum Slack First)
        * Random baseline
    - Exact Solver:
        * Google OR-Tools CP-SAT RCPSP exact solver (cumulative global scheduling).
"""

import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from ortools.sat.python import cp_model

from project_marl.core.schemas import TaskNode
from project_marl.core.constants import ExecutionStatus
from project_marl.graph.aon_builder import AONGraphBuilder
from project_marl.sim.project_env import ProjectSchedulingEnv
from project_marl.core.logging import setup_logger

logger = setup_logger("classical_solvers")


@dataclass
class RCPSPSolution:
    """Represents the schedule solution output from an exact or heuristic solver."""
    status: str                         # "OPTIMAL", "FEASIBLE", "INFEASIBLE", "HEURISTIC"
    makespan: float                     # Total project makespan in hours
    task_schedule: Dict[str, Dict[str, float]] = field(default_factory=dict) # task_id -> {start, end, duration}
    solve_time_seconds: float = 0.0
    solver_name: str = "ExactCPSATSolver"


def solve_rcpsp_exact(
    tasks: List[TaskNode],
    num_resources: int,
    time_limit_seconds: float = 10.0,
    time_scale: int = 10,
) -> RCPSPSolution:
    """
    Solves the static Resource-Constrained Project Scheduling Problem (RCPSP)
    to exact mathematical optimality using Google OR-Tools CP-SAT.

    Formulation:
        Min makespan M
        s.t.
            Start_j >= End_i,                     ∀ (i, j) ∈ Precedences
            End_i = Start_i + Duration_i,         ∀ i ∈ Tasks
            CumulativeUsage(t) <= num_resources,  ∀ t
            M >= End_i,                           ∀ i ∈ Tasks

    Args:
        tasks: List of TaskNode instances with precedence declarations and estimated durations.
        num_resources: Number of available parallel developer agents.
        time_limit_seconds: CP-SAT search timeout in seconds.
        time_scale: Resolution factor for fractional hours (default 10 = 0.1 hr resolution).

    Returns:
        RCPSPSolution containing optimal makespan and scheduled start/end intervals.
    """
    start_cpu = time.perf_counter()

    if not tasks:
        return RCPSPSolution(status="OPTIMAL", makespan=0.0, solve_time_seconds=0.0)

    model = cp_model.CpModel()
    task_map = {t.task_id: t for t in tasks}

    # Upper bound horizon: sum of all durations
    total_dur_scaled = sum(max(1, int(round(t.estimated_duration * time_scale))) for t in tasks)
    horizon = total_dur_scaled + 100

    start_vars = {}
    end_vars = {}
    intervals = {}

    for t in tasks:
        dur_scaled = max(1, int(round(t.estimated_duration * time_scale)))
        s_var = model.NewIntVar(0, horizon, f"start_{t.task_id}")
        e_var = model.NewIntVar(0, horizon, f"end_{t.task_id}")
        interval = model.NewIntervalVar(s_var, dur_scaled, e_var, f"interval_{t.task_id}")

        start_vars[t.task_id] = s_var
        end_vars[t.task_id] = e_var
        intervals[t.task_id] = interval

    # 1. Precedence Constraints
    for t in tasks:
        for pred_id in t.predecessors:
            if pred_id in task_map:
                model.Add(start_vars[t.task_id] >= end_vars[pred_id])

    # 2. Cumulative Resource Constraints
    demands = [1] * len(tasks)
    model.AddCumulative(list(intervals.values()), demands, num_resources)

    # 3. Objective: Minimize Makespan
    makespan_var = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan_var, list(end_vars.values()))
    model.Minimize(makespan_var)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_workers = 1  # Deterministic single-thread solve

    status_code = solver.Solve(model)
    solve_duration = time.perf_counter() - start_cpu

    status_str_map = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    status_str = status_str_map.get(status_code, "UNKNOWN")

    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        optimal_makespan = solver.Value(makespan_var) / float(time_scale)
        schedule = {}
        for t in tasks:
            s_val = solver.Value(start_vars[t.task_id]) / float(time_scale)
            e_val = solver.Value(end_vars[t.task_id]) / float(time_scale)
            schedule[t.task_id] = {
                "start": s_val,
                "end": e_val,
                "duration": round(e_val - s_val, 2),
            }
        logger.info(
            f"CP-SAT Exact Solve: Status={status_str}, Makespan={optimal_makespan:.2f}h, "
            f"SolveTime={solve_duration:.3f}s"
        )
        return RCPSPSolution(
            status=status_str,
            makespan=optimal_makespan,
            task_schedule=schedule,
            solve_time_seconds=solve_duration,
            solver_name="OR-Tools CP-SAT",
        )
    else:
        logger.warning(f"CP-SAT Solve did not find a feasible solution: {status_str}")
        return RCPSPSolution(
            status=status_str,
            makespan=float("inf"),
            solve_time_seconds=solve_duration,
            solver_name="OR-Tools CP-SAT",
        )


class BaseSchedulingPolicy(ABC):
    """
    Abstract base class for heuristic scheduling policies and dispatching rules.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def rank_tasks(self, ready_tasks: List[TaskNode], env: ProjectSchedulingEnv) -> List[TaskNode]:
        """Ranks ready tasks according to the heuristic priority rule."""
        pass

    def get_actions(self, env: ProjectSchedulingEnv) -> Dict[str, Any]:
        """
        Generates coordinated actions for all agents in the environment:
            - scheduling_agent: selects best task-agent pairing or NO_OP
            - risk_agent: NO_OP (or conservative buffer)
            - developer_agents: standard balanced execution mode (1)
            - pm_agent: balanced reward weights (1)
        """
        actions: Dict[str, Any] = {}

        # Default coordinator actions
        actions[env.pm_agent_id] = 1        # Balanced
        actions[env.risk_agent_id] = env.max_tasks  # NO_OP for risk

        for dev_id in env.dev_agent_ids:
            actions[dev_id] = 1             # Balanced mode

        # Scheduling Agent dispatch logic
        ready_tasks = env._get_ready_tasks()
        idle_dev_indices = [
            m for m in range(env.num_developers)
            if env.dev_current_task[m] is None
        ]

        if ready_tasks and idle_dev_indices:
            ranked_tasks = self.rank_tasks(ready_tasks, env)
            chosen_task = ranked_tasks[0]
            chosen_dev_idx = idle_dev_indices[0]

            task_idx = env.task_list.index(chosen_task.task_id)
            sched_action = task_idx * env.num_developers + chosen_dev_idx
            actions[env.sched_agent] = sched_action
        else:
            # NO_OP
            actions[env.sched_agent] = env.max_tasks * env.num_developers

        return actions


class SPTPolicy(BaseSchedulingPolicy):
    """
    Shortest Processing Time (SPT) dispatching rule.
    Prioritizes ready tasks with the smallest estimated duration to clear bottlenecks.
    """

    def __init__(self) -> None:
        super().__init__(name="SPT")

    def rank_tasks(self, ready_tasks: List[TaskNode], env: ProjectSchedulingEnv) -> List[TaskNode]:
        return sorted(ready_tasks, key=lambda t: t.estimated_duration)


class LPTPolicy(BaseSchedulingPolicy):
    """
    Longest Processing Time (LPT) dispatching rule.
    Prioritizes ready tasks with the largest estimated duration to initiate heavy work early.
    """

    def __init__(self) -> None:
        super().__init__(name="LPT")

    def rank_tasks(self, ready_tasks: List[TaskNode], env: ProjectSchedulingEnv) -> List[TaskNode]:
        return sorted(ready_tasks, key=lambda t: -t.estimated_duration)


class CPMPolicy(BaseSchedulingPolicy):
    """
    Critical Path Method (CPM) Priority dispatching rule.
    Prioritizes tasks on the critical path, followed by lowest slack and highest downstream work.
    """

    def __init__(self) -> None:
        super().__init__(name="CPM")

    def rank_tasks(self, ready_tasks: List[TaskNode], env: ProjectSchedulingEnv) -> List[TaskNode]:
        # Sort key:
        # 1. is_critical (True first)
        # 2. total_slack (lowest slack first)
        # 3. estimated_duration (longest first)
        return sorted(
            ready_tasks,
            key=lambda t: (not t.is_critical, t.total_slack, -t.estimated_duration),
        )


class MSFPolicy(BaseSchedulingPolicy):
    """
    Minimum Slack First (MSF) dispatching rule.
    Prioritizes tasks with the least scheduling float/slack (Late Start - Early Start).
    """

    def __init__(self) -> None:
        super().__init__(name="MSF")

    def rank_tasks(self, ready_tasks: List[TaskNode], env: ProjectSchedulingEnv) -> List[TaskNode]:
        return sorted(ready_tasks, key=lambda t: t.total_slack)


class RandomPolicy(BaseSchedulingPolicy):
    """
    Random dispatching baseline.
    Selects uniformly at random among available ready tasks.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(name="Random")
        self.rng = random.Random(seed)

    def rank_tasks(self, ready_tasks: List[TaskNode], env: ProjectSchedulingEnv) -> List[TaskNode]:
        shuffled = list(ready_tasks)
        self.rng.shuffle(shuffled)
        return shuffled
