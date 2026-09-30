"""
Unified Spatiotemporal Project Graph Synthesis Engine.
Fuses managerial Activity-On-Node DAGs with codebase AST structures,
git co-churn metrics, and bipartite task-module dependencies.
"""

from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
import networkx as nx

from project_marl.core.constants import (
    EdgeType,
    ExecutionStatus,
    DEFAULT_FEATURE_DIM,
)
from project_marl.core.schemas import (
    TaskNode,
    CodebaseModuleNode,
    DependencyEdge,
    ProjectGraphState,
)
from project_marl.graph.aon_builder import AONGraphBuilder
from project_marl.graph.ast_parser import CodeASTAnalyzer
from project_marl.graph.git_miner import GitChurnMiner
from project_marl.core.logging import setup_logger

logger = setup_logger("graph_builder")


class ProjectGraphBuilder:
    """
    Constructs and maintains the unified spatiotemporal project graph G_t = (V_t, E_t, X_t, W_t).
    Integrates tasks, code modules, AST coupling, git churn, and AON precedence.
    """

    def __init__(self, project_id: str = "project-alpha", feature_dim: int = DEFAULT_FEATURE_DIM) -> None:
        self.project_id = project_id
        self.feature_dim = feature_dim
        self.aon_builder = AONGraphBuilder()
        self.ast_analyzer: Optional[CodeASTAnalyzer] = None
        self.git_miner: Optional[GitChurnMiner] = None

        self.tasks: Dict[str, TaskNode] = {}
        self.modules: Dict[str, CodebaseModuleNode] = {}
        self.edges: List[DependencyEdge] = []

        # Graph mapping indices
        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: Dict[int, str] = {}
        self.combined_nx_graph: nx.DiGraph = nx.DiGraph()

    def set_tasks(self, tasks: List[TaskNode]) -> None:
        """Sets project tasks and updates AON builder."""
        self.tasks = {t.task_id: t for t in tasks}
        self.aon_builder = AONGraphBuilder()
        self.aon_builder.add_tasks(tasks)

    def analyze_codebase(self, root_dir: str, max_commits: int = 200) -> None:
        """
        Runs AST analysis and git history mining to discover code modules and relationships.
        """
        # 1. AST Analysis
        self.ast_analyzer = CodeASTAnalyzer(root_dir)
        self.ast_analyzer.analyze_directory(root_dir)
        self.modules = dict(self.ast_analyzer.modules)

        # 2. Git History Mining
        self.git_miner = GitChurnMiner(root_dir)
        if self.git_miner.is_git_repo():
            self.git_miner.mine_commit_history(max_commits=max_commits)
            # Update module churn scores
            for mod_id, mod in self.modules.items():
                mod.git_churn_score = self.git_miner.get_module_churn_score(mod_id)
                mod.compute_feature_vector(self.feature_dim)

    def add_explicit_module(self, module: CodebaseModuleNode) -> None:
        """Manually registers a module node (useful for synthetic testing or modular builds)."""
        module.compute_feature_vector(self.feature_dim)
        self.modules[module.module_id] = module

    def build_unified_graph(self) -> ProjectGraphState:
        """
        Synthesizes all layers into the unified graph representation G_t.
        """
        self.edges.clear()

        # 1. Compute CPM for tasks
        makespan, critical_path = self.aon_builder.compute_cpm()

        # 2. Add Managerial Precedence Edges (Task -> Task)
        managerial_edges = self.aon_builder.build_precedence_edges()
        self.edges.extend(managerial_edges)

        # 3. Add Code Dependencies (AST calls & imports)
        if self.ast_analyzer:
            code_edges = self.ast_analyzer.build_code_dependency_edges()
            self.edges.extend(code_edges)

        # 4. Add Git Co-Churn Edges (Module <-> Module)
        if self.git_miner and self.git_miner.is_git_repo():
            churn_edges = self.git_miner.build_co_churn_edges(
                filter_modules=set(self.modules.keys())
            )
            self.edges.extend(churn_edges)

        # 5. Add Bipartite Task <-> Module Edges
        bipartite_edges: List[DependencyEdge] = []
        for task_id, task in self.tasks.items():
            for mod_id in task.impacted_modules:
                # Find matching module or register if missing
                matched_mod = None
                for m_key in self.modules.keys():
                    if m_key == mod_id or m_key.endswith(mod_id) or mod_id.endswith(m_key):
                        matched_mod = m_key
                        break
                
                target_mod_id = matched_mod if matched_mod else mod_id
                if target_mod_id not in self.modules:
                    # Register placeholder module
                    placeholder = CodebaseModuleNode(
                        module_id=target_mod_id,
                        file_path=target_mod_id,
                        lines_of_code=100,
                        cyclomatic_complexity=2.0,
                    )
                    placeholder.compute_feature_vector(self.feature_dim)
                    self.modules[target_mod_id] = placeholder

                # Edge: Task -> Module (Task modifies module)
                bipartite_edges.append(
                    DependencyEdge(
                        source=task_id,
                        target=target_mod_id,
                        edge_type=EdgeType.TASK_AFFECTS_MODULE,
                        weight=1.0,
                        metadata={"task_priority": task.priority},
                    )
                )
                # Reverse Edge: Module -> Task
                bipartite_edges.append(
                    DependencyEdge(
                        source=target_mod_id,
                        target=task_id,
                        edge_type=EdgeType.MODULE_AFFECTED_BY,
                        weight=1.0,
                    )
                )

        self.edges.extend(bipartite_edges)

        # 6. Build Node Index Maps & NetworkX representation
        self._reindex_nodes()
        self._build_combined_nx_graph()

        # Compute total slack sum
        total_slack = sum(t.total_slack for t in self.tasks.values())

        state = ProjectGraphState(
            project_id=self.project_id,
            timestamp=0.0,
            tasks=dict(self.tasks),
            modules=dict(self.modules),
            edges=list(self.edges),
            critical_path=critical_path,
            project_makespan=makespan,
            total_project_slack=total_slack,
            risk_index=self._calculate_structural_risk_index(),
        )

        logger.info(
            f"Built Unified Graph: {len(self.tasks)} tasks, {len(self.modules)} modules, "
            f"{len(self.edges)} edges, Makespan={makespan:.1f}h"
        )
        return state

    def _reindex_nodes(self) -> None:
        """Builds contiguous 0-indexed mapping for all nodes (tasks and modules)."""
        self.node_to_idx.clear()
        self.idx_to_node.clear()

        idx = 0
        # Tasks first
        for task_id in sorted(self.tasks.keys()):
            self.node_to_idx[task_id] = idx
            self.idx_to_node[idx] = task_id
            idx += 1

        # Modules next
        for mod_id in sorted(self.modules.keys()):
            if mod_id not in self.node_to_idx:
                self.node_to_idx[mod_id] = idx
                self.idx_to_node[idx] = mod_id
                idx += 1

    def _build_combined_nx_graph(self) -> None:
        """Constructs NetworkX DiGraph combining all nodes and edges with attributes."""
        self.combined_nx_graph.clear()

        for task_id, task in self.tasks.items():
            self.combined_nx_graph.add_node(
                task_id,
                node_type="task",
                duration=task.estimated_duration,
                is_critical=task.is_critical,
                status=task.status.value,
            )

        for mod_id, mod in self.modules.items():
            self.combined_nx_graph.add_node(
                mod_id,
                node_type="module",
                loc=mod.lines_of_code,
                complexity=mod.cyclomatic_complexity,
                churn=mod.git_churn_score,
            )

        for edge in self.edges:
            self.combined_nx_graph.add_edge(
                edge.source,
                edge.target,
                edge_type=edge.edge_type.value,
                weight=edge.weight,
            )

    def to_tensors(self) -> Dict[str, Any]:
        """
        Exports the graph state into numerical tensors (NumPy arrays)
        ready for PyTorch / STGAT consumption.
        
        Returns:
            dict containing:
                - 'x': Node feature matrix (num_nodes, feature_dim)
                - 'edge_index': Edge indices (2, num_edges) as int64
                - 'edge_attr': Edge weights / attributes (num_edges, 1)
                - 'node_types': Integer node type array (0 for task, 1 for module)
                - 'task_mask': Boolean mask selecting task nodes
                - 'makespan': Float scalar project makespan
                - 'node_to_idx': Mapping dictionary
        """
        num_nodes = len(self.node_to_idx)
        x = np.zeros((num_nodes, self.feature_dim), dtype=np.float32)
        node_types = np.zeros(num_nodes, dtype=np.int64)
        task_mask = np.zeros(num_nodes, dtype=bool)

        for node_id, idx in self.node_to_idx.items():
            if node_id in self.tasks:
                task = self.tasks[node_id]
                x[idx] = task.compute_feature_vector(self.feature_dim)
                node_types[idx] = 0
                task_mask[idx] = True
            elif node_id in self.modules:
                mod = self.modules[node_id]
                x[idx] = mod.compute_feature_vector(self.feature_dim)
                node_types[idx] = 1
                task_mask[idx] = False

        # Build edge index and attribute arrays
        valid_edges = [
            e for e in self.edges
            if e.source in self.node_to_idx and e.target in self.node_to_idx
        ]

        num_edges = len(valid_edges)
        edge_index = np.zeros((2, num_edges), dtype=np.int64)
        edge_attr = np.zeros((num_edges, 1), dtype=np.float32)

        for i, edge in enumerate(valid_edges):
            edge_index[0, i] = self.node_to_idx[edge.source]
            edge_index[1, i] = self.node_to_idx[edge.target]
            edge_attr[i, 0] = edge.weight

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "node_types": node_types,
            "task_mask": task_mask,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "node_to_idx": dict(self.node_to_idx),
            "idx_to_node": dict(self.idx_to_node),
            "makespan": float(self.aon_builder.tasks and max(t.early_finish for t in self.tasks.values()) or 0.0),
        }

    def _calculate_structural_risk_index(self) -> float:
        """
        Computes a baseline structural risk index combining:
        1. Ratio of critical path duration to total duration
        2. High cyclomatic complexity modules touched by critical tasks
        """
        if not self.tasks:
            return 0.0

        total_work = sum(t.estimated_duration for t in self.tasks.values())
        crit_work = sum(t.estimated_duration for t in self.tasks.values() if t.is_critical)
        crit_ratio = crit_work / total_work if total_work > 0 else 0.0

        # Check complexity of modules touched by critical tasks
        crit_modules = set()
        for t in self.tasks.values():
            if t.is_critical:
                crit_modules.update(t.impacted_modules)

        avg_crit_complexity = 1.0
        if crit_modules:
            complexities = [
                self.modules[m].cyclomatic_complexity
                for m in crit_modules if m in self.modules
            ]
            if complexities:
                avg_crit_complexity = sum(complexities) / len(complexities)

        risk_score = float(crit_ratio * (1.0 + min(avg_crit_complexity / 10.0, 2.0)))
        return round(risk_score, 4)
