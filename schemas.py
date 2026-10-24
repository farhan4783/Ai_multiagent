"""
Root compatibility wrapper re-exporting project_marl.core.schemas.
Maintains backward compatibility for legacy scripts and agent task executions.
"""

from project_marl.core.schemas import *  # noqa: F401, F403
from project_marl.core.schemas import (
    CodebaseModuleNode,
    TaskNode,
    DependencyEdge,
    AgentState,
    ProjectGraphState,
    SchedulingAction,
)

# Legacy task stub for developer_agent_0 / TASK-01
def execute_task_01() -> bool:
    """Define TaskNode, CodebaseModuleNode, DependencyEdge, and AgentState."""
    return True
