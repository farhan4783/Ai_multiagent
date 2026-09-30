"""
Procedural synthetic project DAG generator for STGAT training data.

Generates thousands of realistic project dependency graphs with:
    - Configurable topological densities (sparse chains to dense meshes).
    - Stochastic log-normal task duration perturbations.
    - Random codebase module coupling and co-churn weights.
    - Ground truth labels: actual delays, critical path membership, makespan.

These synthetic samples train the STGAT to predict cascading delays and
structural risk without requiring real-world project history.
"""

import random
import math
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from project_marl.core.schemas import TaskNode, CodebaseModuleNode, DependencyEdge
from project_marl.core.constants import (
    EdgeType,
    ExecutionStatus,
    DEFAULT_FEATURE_DIM,
)
from project_marl.graph.aon_builder import AONGraphBuilder
from project_marl.core.logging import setup_logger

logger = setup_logger("generator")


@dataclass
class ProjectSample:
    """A single synthetic project graph sample with ground truth labels."""
    # Graph tensors
    x: np.ndarray                   # [num_nodes, feature_dim]
    edge_index: np.ndarray          # [2, num_edges]
    edge_attr: np.ndarray           # [num_edges, 1]
    task_mask: np.ndarray           # [num_nodes] bool
    node_types: np.ndarray          # [num_nodes] int (0=task, 1=module)

    # Ground truth targets
    actual_delays: np.ndarray       # [num_nodes, 1] (stochastic delay offset)
    is_critical: np.ndarray         # [num_nodes, 1] float (0.0 or 1.0)
    makespan: float                 # scalar

    # Metadata
    num_tasks: int = 0
    num_modules: int = 0
    num_edges: int = 0
    critical_path: List[str] = field(default_factory=list)


class SyntheticProjectGenerator:
    """
    Procedurally generates synthetic project DAGs with stochastic delays
    for training the STGAT predictive engine.

    Args:
        min_tasks: Minimum number of tasks per project.
        max_tasks: Maximum number of tasks per project.
        min_modules: Minimum number of code modules.
        max_modules: Maximum number of code modules.
        edge_density: Probability of adding an optional dependency edge.
        delay_noise_sigma: Log-normal noise scale for duration perturbation.
        feature_dim: Node feature vector dimension.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        min_tasks: int = 5,
        max_tasks: int = 30,
        min_modules: int = 3,
        max_modules: int = 20,
        edge_density: float = 0.3,
        delay_noise_sigma: float = 0.4,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        seed: Optional[int] = None,
    ) -> None:
        self.min_tasks = min_tasks
        self.max_tasks = max_tasks
        self.min_modules = min_modules
        self.max_modules = max_modules
        self.edge_density = edge_density
        self.delay_noise_sigma = delay_noise_sigma
        self.feature_dim = feature_dim

        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def generate_one(self) -> ProjectSample:
        """Generates a single synthetic project sample with ground truth."""
        num_tasks = self.rng.randint(self.min_tasks, self.max_tasks)
        num_modules = self.rng.randint(self.min_modules, self.max_modules)

        # 1. Generate tasks with random durations and random DAG edges
        tasks = self._generate_task_dag(num_tasks)

        # 2. Generate code modules with random complexity
        modules = self._generate_modules(num_modules)

        # 3. Assign modules to tasks (bipartite links)
        task_module_links = self._link_tasks_to_modules(tasks, modules)

        # 4. Generate inter-module coupling edges
        module_edges = self._generate_module_coupling(modules)

        # 5. Compute CPM on the original DAG (gives baseline critical path)
        aon = AONGraphBuilder()
        aon.add_tasks(tasks)
        baseline_makespan, baseline_critical_path = aon.compute_cpm()

        # 6. Apply stochastic delay perturbation (log-normal noise)
        actual_delays, actual_makespan = self._apply_stochastic_delays(tasks, aon)

        # 7. Rebuild CPM with actual durations to get true critical path
        true_critical_set = set(baseline_critical_path)

        # 8. Build tensors
        return self._build_sample_tensors(
            tasks, modules, task_module_links, module_edges,
            actual_delays, true_critical_set, actual_makespan,
            baseline_critical_path,
        )

    def generate_batch(self, n: int) -> List[ProjectSample]:
        """Generates n synthetic project samples."""
        samples = []
        for i in range(n):
            try:
                sample = self.generate_one()
                samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to generate sample {i}: {e}")
                continue
        logger.info(f"Generated {len(samples)} synthetic project samples.")
        return samples

    def generate_temporal_sequence(
        self,
        base_sample: Optional[ProjectSample] = None,
        window_size: int = 5,
    ) -> Dict[str, Any]:
        """
        Generates a temporal sequence of graph snapshots simulating project evolution.

        Creates a base project, then simulates T time steps where tasks progress,
        some complete, delays accumulate, and node features evolve.

        Returns dict with:
            'x_seq': List of T feature matrices [num_nodes, feature_dim]
            'edge_index': Static edge index [2, num_edges]
            'edge_attr': Static edge attributes [num_edges, 1]
            'task_mask': Boolean mask [num_nodes]
            'targets': Ground truth dict for final time step
        """
        if base_sample is None:
            base_sample = self.generate_one()

        x_seq = []
        num_nodes = base_sample.x.shape[0]
        num_tasks = base_sample.task_mask.sum()

        # Simulate temporal evolution
        x_current = base_sample.x.copy()
        for t in range(window_size):
            # Progress simulation: randomly update elapsed time for some tasks
            progress_rate = (t + 1) / window_size
            for i in range(num_nodes):
                if base_sample.task_mask[i]:
                    # Gradually increase elapsed duration (feature index 1)
                    x_current[i, 1] = min(
                        x_current[i, 0] * progress_rate + self.np_rng.normal(0, 0.05),
                        x_current[i, 0] * 1.5,
                    )
                    # Randomly transition status for some tasks
                    if progress_rate > 0.3 and self.rng.random() < 0.3:
                        # Status one-hot indices 2-6
                        x_current[i, 2:7] = 0.0
                        x_current[i, 3] = 1.0  # in_progress
                    if progress_rate > 0.7 and self.rng.random() < 0.2:
                        x_current[i, 2:7] = 0.0
                        x_current[i, 5] = 1.0  # completed

            x_seq.append(x_current.copy())

        targets = {
            "delay": base_sample.actual_delays,
            "is_critical": base_sample.is_critical,
            "makespan": base_sample.makespan,
        }

        return {
            "x_seq": x_seq,
            "edge_index": base_sample.edge_index,
            "edge_attr": base_sample.edge_attr,
            "task_mask": base_sample.task_mask,
            "targets": targets,
        }

    def _generate_task_dag(self, num_tasks: int) -> List[TaskNode]:
        """
        Generates a random DAG of tasks with layered topology.
        Uses a layer-based approach to guarantee acyclicity.
        """
        tasks: List[TaskNode] = []

        # Assign tasks to layers
        num_layers = self.rng.randint(3, max(4, num_tasks // 3))
        layer_assignment = [0]  # First task always in layer 0
        for _ in range(1, num_tasks):
            layer_assignment.append(self.rng.randint(0, num_layers - 1))
        layer_assignment.sort()

        # Group tasks by layer
        layers: Dict[int, List[str]] = {}
        for i, layer in enumerate(layer_assignment):
            task_id = f"T-{i:03d}"
            layers.setdefault(layer, []).append(task_id)

        # Build tasks with predecessors from earlier layers
        sorted_layers = sorted(layers.keys())
        for layer_idx, layer_num in enumerate(sorted_layers):
            for task_id in layers[layer_num]:
                predecessors = []
                # Connect to tasks in earlier layers
                if layer_idx > 0:
                    prev_layer = sorted_layers[layer_idx - 1]
                    prev_tasks = layers[prev_layer]
                    # Each task connects to 1-3 predecessors from previous layer
                    num_preds = min(len(prev_tasks), self.rng.randint(1, 3))
                    predecessors = self.rng.sample(prev_tasks, num_preds)

                    # Additional random connections from older layers
                    if layer_idx > 1 and self.rng.random() < self.edge_density:
                        older_layer = sorted_layers[self.rng.randint(0, layer_idx - 2)]
                        older_tasks = layers[older_layer]
                        if older_tasks:
                            predecessors.append(self.rng.choice(older_tasks))

                duration = max(1.0, self.np_rng.lognormal(mean=2.0, sigma=0.7))
                priority = self.rng.randint(1, 5)
                fail_rate = max(0.0, min(1.0, self.np_rng.beta(1, 5)))

                task = TaskNode(
                    task_id=task_id,
                    title=f"Task {task_id}",
                    estimated_duration=round(duration, 2),
                    predecessors=sorted(list(set(predecessors))),
                    required_skills=[self.rng.choice(["python", "ml", "infra", "frontend", "qa"])],
                    priority=priority,
                    historical_failure_rate=round(fail_rate, 4),
                )
                tasks.append(task)

        return tasks

    def _generate_modules(self, num_modules: int) -> List[CodebaseModuleNode]:
        """Generates random code module nodes."""
        modules = []
        for i in range(num_modules):
            mod = CodebaseModuleNode(
                module_id=f"mod_{i:03d}.py",
                file_path=f"src/modules/mod_{i:03d}.py",
                language="python",
                lines_of_code=self.rng.randint(20, 800),
                cyclomatic_complexity=max(1.0, float(self.np_rng.lognormal(1.5, 0.8))),
                functions_count=self.rng.randint(1, 20),
                classes_count=self.rng.randint(0, 5),
                git_churn_score=max(0.0, float(self.np_rng.exponential(1.0))),
            )
            mod.compute_feature_vector(self.feature_dim)
            modules.append(mod)
        return modules

    def _link_tasks_to_modules(
        self,
        tasks: List[TaskNode],
        modules: List[CodebaseModuleNode],
    ) -> List[DependencyEdge]:
        """Creates bipartite task-to-module dependency edges."""
        edges = []
        module_ids = [m.module_id for m in modules]

        for task in tasks:
            num_affected = self.rng.randint(1, min(3, len(module_ids)))
            affected = self.rng.sample(module_ids, num_affected)
            task.impacted_modules = affected

            for mod_id in affected:
                edges.append(DependencyEdge(
                    source=task.task_id,
                    target=mod_id,
                    edge_type=EdgeType.TASK_AFFECTS_MODULE,
                    weight=1.0,
                ))
                edges.append(DependencyEdge(
                    source=mod_id,
                    target=task.task_id,
                    edge_type=EdgeType.MODULE_AFFECTED_BY,
                    weight=1.0,
                ))
        return edges

    def _generate_module_coupling(
        self,
        modules: List[CodebaseModuleNode],
    ) -> List[DependencyEdge]:
        """Generates random inter-module coupling edges (call/import/churn)."""
        edges = []
        n = len(modules)
        for i in range(n):
            for j in range(i + 1, n):
                if self.rng.random() < self.edge_density * 0.5:
                    weight = max(0.1, float(self.np_rng.exponential(0.5)))
                    edge_type = self.rng.choice([
                        EdgeType.IMPLICIT_AST_CALL,
                        EdgeType.IMPLICIT_IMPORT,
                        EdgeType.IMPLICIT_CO_CHURN,
                    ])
                    edges.append(DependencyEdge(
                        source=modules[i].module_id,
                        target=modules[j].module_id,
                        edge_type=edge_type,
                        weight=round(weight, 4),
                    ))
        return edges

    def _apply_stochastic_delays(
        self,
        tasks: List[TaskNode],
        aon: AONGraphBuilder,
    ) -> Tuple[Dict[str, float], float]:
        """
        Perturbs task durations with log-normal noise and recomputes makespan.

        Returns:
            (delay_map, actual_makespan):
                delay_map: {task_id: actual_delay_offset}
                actual_makespan: Makespan with perturbed durations.
        """
        delay_map: Dict[str, float] = {}

        # Perturb durations
        for task_id, task in aon.tasks.items():
            noise = float(self.np_rng.lognormal(0.0, self.delay_noise_sigma))
            actual_duration = task.estimated_duration * noise
            delay = actual_duration - task.estimated_duration
            delay_map[task_id] = round(delay, 4)
            task.estimated_duration = round(actual_duration, 4)

        # Recompute CPM with perturbed durations
        actual_makespan, _ = aon.compute_cpm()

        return delay_map, actual_makespan

    def _build_sample_tensors(
        self,
        tasks: List[TaskNode],
        modules: List[CodebaseModuleNode],
        bipartite_edges: List[DependencyEdge],
        module_edges: List[DependencyEdge],
        delay_map: Dict[str, float],
        critical_set: set,
        actual_makespan: float,
        critical_path: List[str],
    ) -> ProjectSample:
        """Assembles all components into tensor-based ProjectSample."""
        # Build node index mapping
        node_to_idx: Dict[str, int] = {}
        idx = 0
        for t in tasks:
            node_to_idx[t.task_id] = idx
            idx += 1
        for m in modules:
            node_to_idx[m.module_id] = idx
            idx += 1
        num_nodes = idx

        # Feature matrix
        x = np.zeros((num_nodes, self.feature_dim), dtype=np.float32)
        task_mask = np.zeros(num_nodes, dtype=bool)
        node_types = np.zeros(num_nodes, dtype=np.int64)

        for t in tasks:
            i = node_to_idx[t.task_id]
            x[i] = t.compute_feature_vector(self.feature_dim)
            task_mask[i] = True
            node_types[i] = 0

        for m in modules:
            i = node_to_idx[m.module_id]
            x[i] = m.compute_feature_vector(self.feature_dim)
            task_mask[i] = False
            node_types[i] = 1

        # Collect all edges
        all_edges: List[DependencyEdge] = []

        # Precedence edges
        for t in tasks:
            for pred_id in t.predecessors:
                if pred_id in node_to_idx:
                    all_edges.append(DependencyEdge(
                        source=pred_id,
                        target=t.task_id,
                        edge_type=EdgeType.EXPLICIT_PRECEDENCE,
                        weight=1.0,
                    ))

        all_edges.extend(bipartite_edges)
        all_edges.extend(module_edges)

        # Filter valid edges and build tensors
        valid_edges = [e for e in all_edges if e.source in node_to_idx and e.target in node_to_idx]
        num_edges = len(valid_edges)

        edge_index = np.zeros((2, num_edges), dtype=np.int64)
        edge_attr = np.zeros((num_edges, 1), dtype=np.float32)

        for i, e in enumerate(valid_edges):
            edge_index[0, i] = node_to_idx[e.source]
            edge_index[1, i] = node_to_idx[e.target]
            edge_attr[i, 0] = e.weight

        # Ground truth targets
        actual_delays = np.zeros((num_nodes, 1), dtype=np.float32)
        is_critical = np.zeros((num_nodes, 1), dtype=np.float32)

        for t in tasks:
            i = node_to_idx[t.task_id]
            actual_delays[i, 0] = delay_map.get(t.task_id, 0.0)
            if t.task_id in critical_set:
                is_critical[i, 0] = 1.0

        return ProjectSample(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            task_mask=task_mask,
            node_types=node_types,
            actual_delays=actual_delays,
            is_critical=is_critical,
            makespan=actual_makespan,
            num_tasks=len(tasks),
            num_modules=len(modules),
            num_edges=num_edges,
            critical_path=critical_path,
        )


def collate_samples_to_torch(
    samples: List[ProjectSample],
) -> List[Dict[str, torch.Tensor]]:
    """
    Converts a list of ProjectSamples to a list of torch tensor dicts.
    Each sample is independent (different graph sizes), so we don't batch
    into a single tensor — the training loop processes them individually.
    """
    torch_samples = []
    for s in samples:
        torch_samples.append({
            "x": torch.from_numpy(s.x),
            "edge_index": torch.from_numpy(s.edge_index),
            "edge_attr": torch.from_numpy(s.edge_attr),
            "task_mask": torch.from_numpy(s.task_mask),
            "targets": {
                "delay": torch.from_numpy(s.actual_delays),
                "is_critical": torch.from_numpy(s.is_critical),
                "makespan": torch.tensor([s.makespan], dtype=torch.float32),
            },
        })
    return torch_samples
