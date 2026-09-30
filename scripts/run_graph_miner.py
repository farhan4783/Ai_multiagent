"""
CLI runner and demonstration script for Phase 0 and Phase 1.
Parses codebase AST, mines git co-churn, computes CPM on tasks, and exports unified graph state.
"""

import json
import argparse
from pathlib import Path
from project_marl.core.schemas import TaskNode
from project_marl.graph.builder import ProjectGraphBuilder
from project_marl.core.logging import setup_logger

logger = setup_logger("run_graph_miner")


def create_sample_task_backlog() -> list[TaskNode]:
    """
    Creates a realistic multi-agent technical project backlog:
    Developing the STGAT-MARL Intelligent Project Management System.
    """
    return [
        TaskNode(
            task_id="TASK-01",
            title="Design Core Schemas and Pydantic Models",
            description="Define TaskNode, CodebaseModuleNode, DependencyEdge, and AgentState",
            estimated_duration=8.0,
            impacted_modules=["src/project_marl/core/schemas.py", "src/project_marl/core/constants.py"],
            required_skills=["python", "pydantic"],
            priority=5,
        ),
        TaskNode(
            task_id="TASK-02",
            title="Implement AST & Complexity Analyzer",
            description="Build CodeASTAnalyzer and McCabe ComplexityVisitor",
            estimated_duration=12.0,
            predecessors=["TASK-01"],
            impacted_modules=["src/project_marl/graph/ast_parser.py"],
            required_skills=["ast", "compiler"],
            priority=4,
        ),
        TaskNode(
            task_id="TASK-03",
            title="Implement Git Commit Co-Churn Miner",
            description="Mine git log histories to infer evolutionary file coupling",
            estimated_duration=10.0,
            predecessors=["TASK-01"],
            impacted_modules=["src/project_marl/graph/git_miner.py"],
            required_skills=["git", "python"],
            priority=3,
        ),
        TaskNode(
            task_id="TASK-04",
            title="Develop AON Graph Builder & CPM Engine",
            description="Implement forward/backward passes and critical path identification",
            estimated_duration=14.0,
            predecessors=["TASK-01"],
            impacted_modules=["src/project_marl/graph/aon_builder.py"],
            required_skills=["algorithms", "graph_theory"],
            priority=5,
        ),
        TaskNode(
            task_id="TASK-05",
            title="Synthesize Unified Spatiotemporal Graph",
            description="Fuse managerial DAG with AST call graphs and git co-churn matrices",
            estimated_duration=16.0,
            predecessors=["TASK-02", "TASK-03", "TASK-04"],
            impacted_modules=["src/project_marl/graph/builder.py"],
            required_skills=["gnn", "pytorch", "networkx"],
            priority=5,
        ),
        TaskNode(
            task_id="TASK-06",
            title="End-to-End Integration & Verification Test Suite",
            description="Build pytest suite verifying all graph operations and tensor exports",
            estimated_duration=8.0,
            predecessors=["TASK-05"],
            impacted_modules=["tests/test_graph_builder.py"],
            required_skills=["qa", "pytest"],
            priority=4,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="STGAT-MARL Phase 0 & 1 Graph Miner")
    parser.add_argument("--repo-dir", default=".", help="Root directory of the codebase to analyze")
    parser.add_argument("--output-json", default="data/graph_state.json", help="Path to save output JSON state")
    args = parser.parse_args()

    repo_path = Path(args.repo_dir).resolve()
    logger.info(f"Initiating Spatiotemporal Graph Mining on {repo_path}")

    builder = ProjectGraphBuilder(project_id="stgat-marl-orchestrator")
    tasks = create_sample_task_backlog()
    builder.set_tasks(tasks)

    # Analyze codebase AST and Git history
    builder.analyze_codebase(str(repo_path))

    # Build unified graph
    state = builder.build_unified_graph()
    tensors = builder.to_tensors()

    print("\n" + "=" * 60)
    print("STGAT-MARL SPATIOTEMPORAL GRAPH MINER SUMMARY")
    print("=" * 60)
    print(f"Project ID:               {state.project_id}")
    print(f"Total Tasks:              {len(state.tasks)}")
    print(f"Total Code Modules:       {len(state.modules)}")
    print(f"Total Dependency Edges:   {len(state.edges)}")
    print(f"Calculated Makespan:      {state.project_makespan:.1f} hours")
    print(f"Critical Path Tasks:      {' -> '.join(state.critical_path)}")
    print(f"Total Schedule Slack:     {state.total_project_slack:.1f} hours")
    print(f"Structural Risk Index:    {state.risk_index:.4f}")
    print(f"Feature Tensor X:         {tensors['x'].shape}")
    print(f"Edge Index Tensor:        {tensors['edge_index'].shape}")
    print(f"Edge Attribute Tensor:    {tensors['edge_attr'].shape}")
    print("=" * 60 + "\n")

    # Save output JSON
    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(state.model_dump_json(indent=2))
    logger.info(f"Graph state persisted to {out_file}")


if __name__ == "__main__":
    main()
