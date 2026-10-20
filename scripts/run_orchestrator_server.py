#!/usr/bin/env python3
"""
CLI Runner for the STGAT-MARL Autonomous Orchestration Server & API.

Starts the FastAPI REST and WebSocket server powering the Operator Web Dashboard.

Usage:
    python scripts/run_orchestrator_server.py --port 8000 --host 0.0.0.0
"""

from pathlib import Path
import os
import sys
import argparse
import uvicorn

# Ensure project_marl is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from project_marl.orchestrator.coordinator import (
    AutonomousProjectCoordinator,
    OrchestratorConfig,
)
from project_marl.orchestrator.api import create_app
from project_marl.core.logging import setup_logger

logger = setup_logger("run_orchestrator_server")


def main() -> None:
    parser = argparse.ArgumentParser(description="STGAT-MARL Autonomous Orchestration Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host IP")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--repo-dir", type=str, default=".", help="Base repository directory")
    parser.add_argument("--developers", type=int, default=3, help="Number of parallel developer agents")
    parser.add_argument("--stgat-checkpoint", type=str, default="checkpoints/best_stgat.pt", help="STGAT model checkpoint")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn hot reloading")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("      STGAT-MARL AUTONOMOUS PROJECT ORCHESTRATION SERVER      ")
    print("=" * 70)
    print(f"  Host:             {args.host}")
    print(f"  Port:             {args.port}")
    print(f"  Repository:       {Path(args.repo_dir).resolve()}")
    print(f"  Developer Agents: {args.developers}")
    print(f"  STGAT Checkpoint: {args.stgat_checkpoint}")
    print("=" * 70 + "\n")

    cfg = OrchestratorConfig(
        project_id="stgat-marl-system",
        repo_dir=args.repo_dir,
        num_developers=args.developers,
        stgat_checkpoint=args.stgat_checkpoint,
    )
    coordinator = AutonomousProjectCoordinator(cfg)

    # Ingest current codebase and sample project backlog
    logger.info("Ingesting repository codebase into Spatiotemporal Graph...")
    tasks = None
    try:
        sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
        from run_graph_miner import create_sample_task_backlog
        tasks = create_sample_task_backlog()
    except Exception as e:
        logger.warning("Could not load sample backlog: %s", e)

    coordinator.ingest_project(tasks=tasks, analyze_code=True)

    app = create_app(coordinator)

    logger.info("Starting Uvicorn server on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
