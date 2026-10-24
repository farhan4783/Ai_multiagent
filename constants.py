"""
Root compatibility wrapper re-exporting project_marl.core.constants.
Maintains backward compatibility for legacy scripts and agent task executions.
"""

from project_marl.core.constants import *  # noqa: F401, F403
from project_marl.core.constants import (
    ExecutionStatus,
    EdgeType,
    AgentRole,
    ActionType,
    DEFAULT_FEATURE_DIM,
    DEFAULT_CYCLOMATIC_NORMALIZER,
    DEFAULT_DURATION_NORMALIZER,
    DEFAULT_LINES_OF_CODE_NORMALIZER,
    WEIGHT_AST_CALL,
    WEIGHT_IMPORT,
    WEIGHT_CO_CHURN,
    MIN_CO_CHURN_THRESHOLD,
)

# Legacy task stub for developer_agent_0 / TASK-01
def execute_task_01() -> bool:
    """Define TaskNode, CodebaseModuleNode, DependencyEdge, and AgentState."""
    return True
