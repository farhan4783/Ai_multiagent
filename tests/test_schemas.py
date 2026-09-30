"""
Unit tests for core Pydantic domain models in project_marl.core.schemas.
"""

import pytest
from project_marl.core.constants import ExecutionStatus, EdgeType, AgentRole, ActionType
from project_marl.core.schemas import (
    CodebaseModuleNode,
    TaskNode,
    DependencyEdge,
    AgentState,
    ProjectGraphState,
    SchedulingAction,
)


def test_codebase_module_node_feature_vector() -> None:
    mod = CodebaseModuleNode(
        module_id="src/auth.py",
        file_path="src/auth.py",
        language="python",
        lines_of_code=250,
        cyclomatic_complexity=8.0,
        functions_count=5,
        classes_count=2,
        imports=["os", "sys", "jwt"],
        git_churn_score=1.5,
    )
    vec = mod.compute_feature_vector(dim=16)
    assert len(vec) == 16
    assert vec[0] == pytest.approx(250 / 500.0)
    assert vec[1] == pytest.approx(8.0 / 20.0)
    assert vec[2] == pytest.approx(5 / 25.0)
    assert vec[3] == pytest.approx(2 / 10.0)
    assert vec[4] == pytest.approx(3 / 20.0)
    assert vec[5] == pytest.approx(1.5)


def test_task_node_readiness_and_vector() -> None:
    task = TaskNode(
        task_id="TASK-01",
        title="Init DB",
        estimated_duration=12.0,
        predecessors=["PRE-01", "PRE-02"],
    )
    # Not ready initially
    assert not task.is_ready(completed_task_ids={"PRE-01"})
    # Ready when all predecessors complete
    assert task.is_ready(completed_task_ids={"PRE-01", "PRE-02"})

    # Check status transition
    task.status = ExecutionStatus.IN_PROGRESS
    assert not task.is_ready(completed_task_ids={"PRE-01", "PRE-02"})

    vec = task.compute_feature_vector(dim=16)
    assert len(vec) == 16
    assert vec[0] == pytest.approx(12.0 / 40.0)
    # In-progress status flag at index 3
    assert vec[3] == 1.0


def test_dependency_edge_validation() -> None:
    edge = DependencyEdge(
        source="T1",
        target="T2",
        edge_type=EdgeType.EXPLICIT_PRECEDENCE,
        weight=2.5,
        metadata={"critical": True},
    )
    assert edge.source == "T1"
    assert edge.target == "T2"
    assert edge.edge_type == EdgeType.EXPLICIT_PRECEDENCE
    assert edge.weight == 2.5
    assert edge.metadata["critical"] is True


def test_agent_state_and_scheduling_action() -> None:
    agent = AgentState(
        agent_id="agent-007",
        role=AgentRole.SCHEDULING,
        skills=["python", "fastapi"],
    )
    assert agent.is_idle is True
    assert agent.health_score == 1.0

    action = SchedulingAction(
        action_type=ActionType.DISPATCH,
        task_id="TASK-01",
        agent_id="agent-007",
        reason="Assigned based on skill match",
    )
    assert action.action_type == ActionType.DISPATCH
    assert action.task_id == "TASK-01"
