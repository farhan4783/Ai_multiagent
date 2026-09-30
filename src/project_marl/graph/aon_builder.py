"""
Activity-On-Node (AON) graph builder and Critical Path Method (CPM) calculation engine.
Validates project schedule acyclicity, performs forward/backward passes, and computes task slacks.
"""

from typing import Dict, List, Set, Tuple, Optional
import networkx as nx

from project_marl.core.constants import EdgeType
from project_marl.core.schemas import TaskNode, DependencyEdge
from project_marl.core.logging import setup_logger

logger = setup_logger("aon_builder")


class AONGraphBuilder:
    """
    Constructs managerial Activity-On-Node DAGs and performs CPM analysis.
    """

    def __init__(self) -> None:
        self.tasks: Dict[str, TaskNode] = {}
        self.graph: nx.DiGraph = nx.DiGraph()

    def add_task(self, task: TaskNode) -> None:
        """Adds a task node to the builder."""
        self.tasks[task.task_id] = task
        self.graph.add_node(task.task_id, task=task, duration=task.estimated_duration)

    def add_tasks(self, tasks: List[TaskNode]) -> None:
        """Adds multiple task nodes."""
        for t in tasks:
            self.add_task(t)

    def validate_dag(self) -> Tuple[bool, Optional[List[List[str]]]]:
        """
        Builds graph edges based on predecessor declarations and checks for cycles.
        Returns (is_valid, cycles).
        """
        # Rebuild edges in networkx graph
        self.graph.clear_edges()
        for task_id, task in self.tasks.items():
            for pred_id in task.predecessors:
                if pred_id not in self.tasks:
                    logger.warning(f"Task {task_id} references missing predecessor {pred_id}")
                else:
                    self.graph.add_edge(pred_id, task_id)

        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            logger.error(f"Cycle detected in task dependencies: {cycles}")
            return False, cycles

        # Update successors lists on tasks
        for task_id, task in self.tasks.items():
            task.successors = sorted(list(self.graph.successors(task_id)))

        return True, None

    def compute_cpm(self) -> Tuple[float, List[str]]:
        """
        Executes Forward and Backward CPM passes to calculate Early Start/Finish,
        Late Start/Finish, Total Slack, and identifies the Critical Path.
        
        Returns:
            (project_makespan, critical_path_task_ids)
        """
        is_valid, cycles = self.validate_dag()
        if not is_valid:
            raise ValueError(f"Cannot compute CPM on cyclic graph: {cycles}")

        if not self.tasks:
            return 0.0, []

        # 1. Topological Sort for Forward Pass
        topological_order = list(nx.topological_sort(self.graph))

        # Forward Pass: Early Start (ES) and Early Finish (EF)
        for node_id in topological_order:
            task = self.tasks[node_id]
            predecessors = list(self.graph.predecessors(node_id))
            if not predecessors:
                task.early_start = 0.0
            else:
                task.early_start = max(self.tasks[p].early_finish for p in predecessors)
            task.early_finish = task.early_start + task.estimated_duration

        # Overall project makespan = max early finish of all terminal tasks
        project_makespan = max(task.early_finish for task in self.tasks.values())

        # 2. Reverse Topological Sort for Backward Pass
        reverse_topo_order = list(reversed(topological_order))

        # Backward Pass: Late Finish (LF) and Late Start (LS)
        for node_id in reverse_topo_order:
            task = self.tasks[node_id]
            successors = list(self.graph.successors(node_id))
            if not successors:
                task.late_finish = project_makespan
            else:
                task.late_finish = min(self.tasks[s].late_start for s in successors)
            task.late_start = task.late_finish - task.estimated_duration
            
            # Total Slack (Float) = Late Start - Early Start
            task.total_slack = round(task.late_start - task.early_start, 6)
            # Floating point tolerance
            task.is_critical = abs(task.total_slack) < 1e-5

        # Critical path nodes sorted by early start
        critical_nodes = [
            t.task_id for t in sorted(self.tasks.values(), key=lambda x: x.early_start)
            if t.is_critical
        ]

        logger.debug(f"CPM Completed: Makespan={project_makespan:.2f} hrs, Critical Tasks={critical_nodes}")
        return project_makespan, critical_nodes

    def build_precedence_edges(self) -> List[DependencyEdge]:
        """Returns DependencyEdge instances for all precedence links."""
        edges: List[DependencyEdge] = []
        for task_id, task in self.tasks.items():
            for pred_id in task.predecessors:
                if pred_id in self.tasks:
                    edges.append(
                        DependencyEdge(
                            source=pred_id,
                            target=task_id,
                            edge_type=EdgeType.EXPLICIT_PRECEDENCE,
                            weight=1.0,
                            metadata={"pred_duration": self.tasks[pred_id].estimated_duration},
                        )
                    )
        return edges
