"""
Pydantic domain schemas representing the entities, graph structures,
and actions within the STGAT-MARL system.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict

from project_marl.core.constants import (
    ExecutionStatus,
    EdgeType,
    AgentRole,
    ActionType,
    DEFAULT_FEATURE_DIM,
)


class CodebaseModuleNode(BaseModel):
    """
    Structural representation of a codebase file/module extracted via AST.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    module_id: str = Field(description="Unique module identifier, typically relative path")
    file_path: str = Field(description="Full or relative path to file")
    language: str = Field(default="python", description="Programming language")
    lines_of_code: int = Field(default=0, ge=0)
    cyclomatic_complexity: float = Field(default=1.0, ge=1.0, description="McCabe cyclomatic complexity score")
    functions_count: int = Field(default=0, ge=0)
    classes_count: int = Field(default=0, ge=0)
    imports: List[str] = Field(default_factory=list, description="List of imported module names")
    functions_defined: List[str] = Field(default_factory=list, description="Function names defined in module")
    functions_called: List[str] = Field(default_factory=list, description="Functions/methods invoked in module")
    git_churn_score: float = Field(default=0.0, ge=0.0, description="Historical co-churn / commit frequency score")
    feature_vector: Optional[List[float]] = Field(default=None, description="Precomputed or normalized feature vector")

    def compute_feature_vector(self, dim: int = DEFAULT_FEATURE_DIM) -> List[float]:
        """Generate a normalized numerical feature vector for GNN ingestion."""
        vec = [0.0] * dim
        vec[0] = min(self.lines_of_code / 500.0, 5.0)
        vec[1] = min(self.cyclomatic_complexity / 20.0, 5.0)
        vec[2] = min(self.functions_count / 25.0, 5.0)
        vec[3] = min(self.classes_count / 10.0, 5.0)
        vec[4] = min(len(self.imports) / 20.0, 5.0)
        vec[5] = min(self.git_churn_score, 10.0)
        self.feature_vector = vec
        return vec


class TaskNode(BaseModel):
    """
    Representation of a project task or work breakdown item in the AON graph.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str = Field(description="Unique task identifier, e.g. TASK-101")
    title: str = Field(description="Short human-readable title")
    description: str = Field(default="", description="Detailed task requirements")
    estimated_duration: float = Field(gt=0, description="Estimated duration in hours or work units")
    elapsed_duration: float = Field(default=0.0, ge=0, description="Elapsed time spent on task")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    predecessors: List[str] = Field(default_factory=list, description="Task IDs that must complete before this task")
    successors: List[str] = Field(default_factory=list, description="Task IDs dependent on this task")
    impacted_modules: List[str] = Field(default_factory=list, description="Module IDs modified or created by this task")
    required_skills: List[str] = Field(default_factory=list, description="Required agent capabilities")
    priority: int = Field(default=1, ge=1, le=5, description="Priority level: 1 (normal) to 5 (critical)")
    historical_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # CPM (Critical Path Method) Computed Metrics
    early_start: float = Field(default=0.0, ge=0)
    early_finish: float = Field(default=0.0, ge=0)
    late_start: float = Field(default=0.0, ge=0)
    late_finish: float = Field(default=0.0, ge=0)
    total_slack: float = Field(default=0.0, description="Float/slack = Late Start - Early Start")
    is_critical: bool = Field(default=False, description="True if task lies on the critical path")
    assigned_agent_id: Optional[str] = Field(default=None)

    def is_ready(self, completed_task_ids: set[str]) -> bool:
        """Check if all predecessor constraints are fulfilled and task is pending."""
        return self.status == ExecutionStatus.PENDING and all(
            pred in completed_task_ids for pred in self.predecessors
        )

    def compute_feature_vector(self, dim: int = DEFAULT_FEATURE_DIM) -> List[float]:
        """Generate a normalized numerical feature vector for the task."""
        vec = [0.0] * dim
        vec[0] = min(self.estimated_duration / 40.0, 5.0)
        vec[1] = min(self.elapsed_duration / 40.0, 5.0)
        # Status one-hot encoding into positions 2..6
        status_map = {
            ExecutionStatus.PENDING: 2,
            ExecutionStatus.IN_PROGRESS: 3,
            ExecutionStatus.BLOCKED: 4,
            ExecutionStatus.COMPLETED: 5,
            ExecutionStatus.FAILED: 6,
        }
        idx = status_map.get(self.status, 2)
        vec[idx] = 1.0
        vec[7] = self.priority / 5.0
        vec[8] = 1.0 if self.is_critical else 0.0
        vec[9] = min(self.total_slack / 20.0, 5.0)
        vec[10] = self.historical_failure_rate
        vec[11] = min(len(self.impacted_modules) / 5.0, 3.0)
        return vec


class DependencyEdge(BaseModel):
    """
    A directed edge representing explicit managerial dependencies or implicit codebase coupling.
    """
    source: str = Field(description="Source node identifier (task_id or module_id)")
    target: str = Field(description="Target node identifier (task_id or module_id)")
    edge_type: EdgeType = Field(description="Semantic type of dependency")
    weight: float = Field(default=1.0, ge=0.0, description="Coupling or dependency intensity")
    attention_weight: Optional[float] = Field(default=None, description="Learned spatial attention weight α_ij from STGAT")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context (e.g. call count, co-churn value)")


class AgentState(BaseModel):
    """Current state of a specialized MARL worker or coordinator agent."""
    agent_id: str
    role: AgentRole
    current_task_id: Optional[str] = None
    worktree_path: Optional[str] = None
    health_score: float = Field(default=1.0, ge=0.0, le=1.0)
    is_idle: bool = True
    allocated_port: Optional[int] = None
    skills: List[str] = Field(default_factory=list)


class ProjectGraphState(BaseModel):
    """Complete snapshot of the spatiotemporal project graph at a given timestep."""
    project_id: str
    timestamp: float = 0.0
    tasks: Dict[str, TaskNode] = Field(default_factory=dict)
    modules: Dict[str, CodebaseModuleNode] = Field(default_factory=dict)
    edges: List[DependencyEdge] = Field(default_factory=list)
    critical_path: List[str] = Field(default_factory=list)
    project_makespan: float = 0.0
    total_project_slack: float = 0.0
    risk_index: float = 0.0


class SchedulingAction(BaseModel):
    """Action emitted by MARL agent swarm."""
    action_type: ActionType
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    buffer_amount: float = Field(default=0.0, ge=0.0)
    reason: str = Field(default="")
