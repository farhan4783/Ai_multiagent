"""
Unit tests for ProjectGraphBuilder in project_marl.graph.builder.
"""

import pytest
import numpy as np
from project_marl.core.schemas import TaskNode, CodebaseModuleNode
from project_marl.core.constants import EdgeType, ExecutionStatus
from project_marl.graph.builder import ProjectGraphBuilder


def test_project_graph_builder_synthesis() -> None:
    builder = ProjectGraphBuilder("test-project", feature_dim=16)

    # 1. Add tasks
    tasks = [
        TaskNode(
            task_id="T1",
            title="Design Service",
            estimated_duration=6.0,
            impacted_modules=["service.py"],
        ),
        TaskNode(
            task_id="T2",
            title="Implement Client",
            estimated_duration=8.0,
            predecessors=["T1"],
            impacted_modules=["client.py"],
        ),
    ]
    builder.set_tasks(tasks)

    # 2. Add explicit modules
    mod1 = CodebaseModuleNode(
        module_id="service.py",
        file_path="service.py",
        lines_of_code=150,
        cyclomatic_complexity=4.0,
        functions_count=3,
        functions_defined=["serve"],
    )
    mod2 = CodebaseModuleNode(
        module_id="client.py",
        file_path="client.py",
        lines_of_code=80,
        cyclomatic_complexity=2.0,
        functions_count=2,
        functions_called=["serve"],
        imports=["service"],
    )
    builder.add_explicit_module(mod1)
    builder.add_explicit_module(mod2)

    state = builder.build_unified_graph()

    # Checks
    assert len(state.tasks) == 2
    assert len(state.modules) == 2
    assert state.project_makespan == 14.0
    assert state.critical_path == ["T1", "T2"]

    # Check bipartite edges
    bipartite = [e for e in state.edges if e.edge_type == EdgeType.TASK_AFFECTS_MODULE]
    assert len(bipartite) == 2
    assert any(e.source == "T1" and e.target == "service.py" for e in bipartite)
    assert any(e.source == "T2" and e.target == "client.py" for e in bipartite)

    # Check tensors
    tensors = builder.to_tensors()
    assert tensors["num_nodes"] == 4  # 2 tasks + 2 modules
    assert tensors["x"].shape == (4, 16)
    assert tensors["edge_index"].shape[0] == 2
    assert tensors["task_mask"].sum() == 2
    assert tensors["node_types"].sum() == 2  # 2 module nodes labeled 1
    assert tensors["makespan"] == 14.0
