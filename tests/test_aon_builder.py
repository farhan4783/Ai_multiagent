"""
Unit tests for AONGraphBuilder and CPM solver in project_marl.graph.aon_builder.
"""

import pytest
from project_marl.core.schemas import TaskNode
from project_marl.graph.aon_builder import AONGraphBuilder


def test_cpm_linear_chain() -> None:
    builder = AONGraphBuilder()
    t1 = TaskNode(task_id="A", title="Task A", estimated_duration=5.0)
    t2 = TaskNode(task_id="B", title="Task B", estimated_duration=10.0, predecessors=["A"])
    t3 = TaskNode(task_id="C", title="Task C", estimated_duration=3.0, predecessors=["B"])

    builder.add_tasks([t1, t2, t3])
    makespan, critical_path = builder.compute_cpm()

    assert makespan == 18.0
    assert critical_path == ["A", "B", "C"]
    assert all(builder.tasks[tid].total_slack == 0.0 for tid in ["A", "B", "C"])


def test_cpm_branching_paths() -> None:
    builder = AONGraphBuilder()
    # A (4) -> B (8) -> D (2) = 14 (Critical)
    # A (4) -> C (3) -> D (2) = 9  (Slack on C = 5)
    t_a = TaskNode(task_id="A", title="A", estimated_duration=4.0)
    t_b = TaskNode(task_id="B", title="B", estimated_duration=8.0, predecessors=["A"])
    t_c = TaskNode(task_id="C", title="C", estimated_duration=3.0, predecessors=["A"])
    t_d = TaskNode(task_id="D", title="D", estimated_duration=2.0, predecessors=["B", "C"])

    builder.add_tasks([t_a, t_b, t_c, t_d])
    makespan, critical_path = builder.compute_cpm()

    assert makespan == 14.0
    assert critical_path == ["A", "B", "D"]
    assert builder.tasks["C"].total_slack == pytest.approx(5.0)
    assert not builder.tasks["C"].is_critical


def test_cycle_detection() -> None:
    builder = AONGraphBuilder()
    t1 = TaskNode(task_id="X", title="X", estimated_duration=5.0, predecessors=["Z"])
    t2 = TaskNode(task_id="Y", title="Y", estimated_duration=5.0, predecessors=["X"])
    t3 = TaskNode(task_id="Z", title="Z", estimated_duration=5.0, predecessors=["Y"])

    builder.add_tasks([t1, t2, t3])
    is_valid, cycles = builder.validate_dag()
    assert is_valid is False
    assert cycles is not None
    assert len(cycles) > 0

    with pytest.raises(ValueError, match="Cannot compute CPM on cyclic graph"):
        builder.compute_cpm()
