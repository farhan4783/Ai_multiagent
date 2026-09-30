"""
Core constants, enumerations, and mathematical configuration defaults
for the STGAT-MARL system.
"""

from enum import Enum, auto


class ExecutionStatus(str, Enum):
    """Execution status for tasks in the spatiotemporal project graph."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"


class EdgeType(str, Enum):
    """
    Types of edges connecting nodes in the unified heterogeneous project graph.
    Covers both explicit managerial constraints and implicit codebase coupling.
    """
    EXPLICIT_PRECEDENCE = "explicit_precedence"      # Activity-On-Node Task -> Task
    IMPLICIT_AST_CALL = "implicit_ast_call"          # Module -> Module (Function/Method invocation)
    IMPLICIT_IMPORT = "implicit_import"              # Module -> Module (Import dependency)
    IMPLICIT_CO_CHURN = "implicit_co_churn"          # Module <-> Module (Git historical co-change)
    TASK_AFFECTS_MODULE = "task_affects_module"      # Task -> Module (Task modifies module)
    MODULE_AFFECTED_BY = "module_affected_by"        # Module -> Task (Reverse edge)


class AgentRole(str, Enum):
    """Specialized roles within the cooperative MARL agent swarm."""
    SCHEDULING = "scheduling_agent"                  # Dispatches tasks, solves dynamic RCPSP
    RISK = "risk_agent"                              # Monitors STGAT attention, manages buffers
    DEVELOPER = "developer_agent"                    # Executes code generation in worktrees
    PM_COORDINATOR = "pm_coordinator"                # Macro arbitration & multi-objective balancing


class ActionType(str, Enum):
    """Discrete action primitives dispatched by MARL agents."""
    DISPATCH = "dispatch"                            # Assign task to developer agent worktree
    PREEMPT = "preempt"                              # Temporarily halt task to free critical resource
    INJECT_BUFFER = "inject_buffer"                  # Add temporal safety margin to fragile path
    TRIGGER_PREFLIGHT = "trigger_preflight"          # Early sanity tests before merge
    REASSIGN = "reassign"                            # Transfer task to higher-capacity worker


# Default normalization and feature dimension constants
DEFAULT_FEATURE_DIM = 16
DEFAULT_CYCLOMATIC_NORMALIZER = 20.0
DEFAULT_DURATION_NORMALIZER = 40.0               # in hours / story units
DEFAULT_LINES_OF_CODE_NORMALIZER = 1000.0

# Graph edge weight defaults for implicit coupling synthesis
WEIGHT_AST_CALL = 1.0
WEIGHT_IMPORT = 0.5
WEIGHT_CO_CHURN = 0.75
MIN_CO_CHURN_THRESHOLD = 0.05
