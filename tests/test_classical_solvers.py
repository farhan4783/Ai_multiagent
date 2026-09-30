"""
Unit and integration tests for classical optimization baselines,
heuristic dispatching rules (SPT, LPT, CPM, MSF), and the OR-Tools exact solver.
"""

import pytest
import numpy as np

from project_marl.core.schemas import TaskNode
from project_marl.sim.project_env import ProjectSchedulingEnv
from project_marl.baselines.classical_solvers import (
    solve_rcpsp_exact,
    SPTPolicy,
    LPTPolicy,
    CPMPolicy,
    MSFPolicy,
    RandomPolicy,
)
from project_marl.baselines.evaluator import BenchmarkSuite, BenchmarkResult


class TestClassicalSolvers:
    """Test suite for classical solvers and heuristic dispatching rules."""

    def test_exact_rcpsp_serial_chain(self) -> None:
        """Serial dependency chain A -> B -> C on 1 worker: makespan = sum of durations."""
        tasks = [
            TaskNode(task_id="A", title="Task A", estimated_duration=2.0),
            TaskNode(task_id="B", title="Task B", estimated_duration=3.0, predecessors=["A"]),
            TaskNode(task_id="C", title="Task C", estimated_duration=4.0, predecessors=["B"]),
        ]
        sol = solve_rcpsp_exact(tasks, num_resources=1)
        assert sol.status == "OPTIMAL"
        assert pytest.approx(sol.makespan, 0.1) == 9.0

        # Precedences check
        assert sol.task_schedule["B"]["start"] >= sol.task_schedule["A"]["end"] - 1e-4
        assert sol.task_schedule["C"]["start"] >= sol.task_schedule["B"]["end"] - 1e-4

    def test_exact_rcpsp_parallel_speedup(self) -> None:
        """Two independent tasks A (4h) and B (6h) with 2 workers: makespan = max(4, 6) = 6h."""
        tasks = [
            TaskNode(task_id="A", title="Task A", estimated_duration=4.0),
            TaskNode(task_id="B", title="Task B", estimated_duration=6.0),
        ]
        sol = solve_rcpsp_exact(tasks, num_resources=2)
        assert sol.status == "OPTIMAL"
        assert pytest.approx(sol.makespan, 0.1) == 6.0

    def test_spt_dispatching_rule(self) -> None:
        """SPT should rank ready tasks by ascending estimated duration."""
        env = ProjectSchedulingEnv(num_developers=2, max_tasks=10, seed=42)
        env.reset(seed=42)

        ready_tasks = [
            TaskNode(task_id="T1", title="T1", estimated_duration=10.0),
            TaskNode(task_id="T2", title="T2", estimated_duration=2.0),
            TaskNode(task_id="T3", title="T3", estimated_duration=5.0),
        ]
        policy = SPTPolicy()
        ranked = policy.rank_tasks(ready_tasks, env)
        assert [t.task_id for t in ranked] == ["T2", "T3", "T1"]

    def test_lpt_dispatching_rule(self) -> None:
        """LPT should rank ready tasks by descending estimated duration."""
        env = ProjectSchedulingEnv(num_developers=2, max_tasks=10, seed=42)
        env.reset(seed=42)

        ready_tasks = [
            TaskNode(task_id="T1", title="T1", estimated_duration=10.0),
            TaskNode(task_id="T2", title="T2", estimated_duration=2.0),
            TaskNode(task_id="T3", title="T3", estimated_duration=5.0),
        ]
        policy = LPTPolicy()
        ranked = policy.rank_tasks(ready_tasks, env)
        assert [t.task_id for t in ranked] == ["T1", "T3", "T2"]

    def test_cpm_dispatching_rule(self) -> None:
        """CPM policy should prioritize critical path tasks first."""
        env = ProjectSchedulingEnv(num_developers=2, max_tasks=10, seed=42)
        env.reset(seed=42)

        ready_tasks = [
            TaskNode(task_id="T1", title="T1", estimated_duration=5.0, is_critical=False, total_slack=4.0),
            TaskNode(task_id="T2", title="T2", estimated_duration=5.0, is_critical=True, total_slack=0.0),
            TaskNode(task_id="T3", title="T3", estimated_duration=5.0, is_critical=False, total_slack=1.0),
        ]
        policy = CPMPolicy()
        ranked = policy.rank_tasks(ready_tasks, env)
        assert ranked[0].task_id == "T2"  # Critical task prioritized

    def test_msf_dispatching_rule(self) -> None:
        """MSF policy should prioritize tasks with minimum slack."""
        env = ProjectSchedulingEnv(num_developers=2, max_tasks=10, seed=42)
        env.reset(seed=42)

        ready_tasks = [
            TaskNode(task_id="T1", title="T1", estimated_duration=5.0, total_slack=8.0),
            TaskNode(task_id="T2", title="T2", estimated_duration=5.0, total_slack=1.5),
            TaskNode(task_id="T3", title="T3", estimated_duration=5.0, total_slack=4.0),
        ]
        policy = MSFPolicy()
        ranked = policy.rank_tasks(ready_tasks, env)
        assert [t.task_id for t in ranked] == ["T2", "T3", "T1"]

    def test_benchmark_suite_execution(self) -> None:
        """Tests BenchmarkSuite execution and report generation across multiple policies."""
        suite = BenchmarkSuite(
            num_episodes=2,
            num_developers=2,
            max_tasks=6,
            max_steps=50,
            seed=42,
        )

        policies = [SPTPolicy(), CPMPolicy()]
        results = suite.run_all(policies=policies, solve_exact=True)

        assert "SPT" in results
        assert "CPM" in results

        spt_res = results["SPT"]
        assert spt_res.num_episodes == 2
        assert spt_res.mean_makespan > 0.0
        assert spt_res.mean_opt_makespan is not None

        # Check table formatting
        table_md = suite.format_table(results)
        assert "| Policy |" in table_md
        assert "**SPT**" in table_md
        assert "**CPM**" in table_md
        assert "Static CP-SAT Exact Optimum" in table_md
