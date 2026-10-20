"""
FastAPI Backend Service for STGAT-MARL Operator Web Dashboard.

Provides REST and WebSocket endpoints for:
    - Real-time Spatiotemporal Graph topology & STGAT attention heatmap streaming.
    - Dynamic Gantt chart updates and Critical Path Method projections.
    - Swarm agent and physical sandboxed Git worktree monitoring.
    - PromptQL organizational wiki graph and sub-100ms RAG semantic queries.
    - Human-in-the-Loop interventions (priority overrides, task injections).
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio
import json
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from project_marl.orchestrator.coordinator import (
    AutonomousProjectCoordinator,
    OrchestratorConfig,
)
from project_marl.core.schemas import TaskNode
from project_marl.memory.rag_service import ContextQuery
from project_marl.core.logging import setup_logger

logger = setup_logger("orchestrator_api")


class InterventionRequest(BaseModel):
    """Payload for Human-in-the-Loop operator interventions."""
    intervention_type: str = Field(description="PRIORITY_OVERRIDE, INSERT_TASK, or INJECT_BUFFER")
    params: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Payload for PromptQL RAG memory retrieval."""
    query_text: str = Field(description="Natural language technical query")
    top_k: int = Field(default=3, ge=1, le=10)
    target_module: Optional[str] = None


class TelemetryConnectionManager:
    """Manages active WebSocket subscriber connections."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.debug("WebSocket client connected. Active: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.debug("WebSocket client disconnected. Active: %d", len(self.active_connections))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        payload = json.dumps(message)
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                dead.append(connection)
        for d in dead:
            self.disconnect(d)


def create_app(coordinator: Optional[AutonomousProjectCoordinator] = None) -> FastAPI:
    """FastAPI Application Factory."""
    app = FastAPI(
        title="STGAT-MARL Autonomous Project Management Engine",
        version="0.1.0",
        description="Dynamic scheduling, graph attention risk prediction, and sandboxed parallel agent execution.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Instantiate or assign coordinator
    coord = coordinator or AutonomousProjectCoordinator()
    manager = TelemetryConnectionManager()

    # Wire coordinator callbacks to WebSocket broadcast
    # Use loop-safe callback scheduling
    def on_telemetry(event: Dict[str, Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(manager.broadcast(event), loop)
        except RuntimeError:
            pass  # No running event loop in current thread

    coord.register_callback(on_telemetry)

    # -------------------------------------------------------------
    # REST Endpoints
    # -------------------------------------------------------------

    @app.get("/api/health")
    def health_check() -> Dict[str, Any]:
        """System health and operational metrics."""
        return {
            "status": "healthy",
            "project_id": coord.config.project_id,
            "step": coord.current_step,
            "active_workspaces": len(coord.active_tasks_dispatched),
            "total_tasks": len(coord.graph_builder.tasks),
            "total_modules": len(coord.graph_builder.modules),
            "vector_store_docs": coord.vector_store.size,
            "wiki_nodes": len(coord.wiki_graph.nodes),
            "uptime_seconds": time.time(),
        }

    @app.get("/api/project/state")
    def get_project_state() -> Dict[str, Any]:
        """Complete snapshot of the spatiotemporal project graph G_t."""
        return coord.get_project_state()

    @app.get("/api/project/gantt")
    def get_gantt() -> List[Dict[str, Any]]:
        """Real-time Gantt timeline reflecting CPM schedule and live progress."""
        return coord.get_gantt_data()

    @app.get("/api/project/attention")
    def get_attention_heatmap() -> Dict[str, Any]:
        """STGAT dynamic spatial attention weights α_ij and fragility scores."""
        return coord.latest_predictions

    @app.get("/api/agents/status")
    def get_agents_status() -> List[Dict[str, Any]]:
        """Live status of the agent swarm and physical sandboxed worktrees."""
        return coord.get_agent_statuses()

    @app.get("/api/memory/wiki")
    def get_wiki_graph() -> Dict[str, Any]:
        """Nodes and edges from the PromptQL organizational wiki graph."""
        nodes = [node.to_dict() for node in coord.wiki_graph.nodes.values()]
        edges = [
            {
                "source": u,
                "target": v,
                "relation": data.get("relation", "relates_to"),
                "weight": data.get("weight", 1.0),
            }
            for u, v, data in coord.wiki_graph.graph.edges(data=True)
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "interface_history": [
                {
                    "module_id": e.module_id,
                    "task_id": e.task_id,
                    "agent_id": e.agent_id,
                    "added_symbols": e.added_symbols,
                    "summary": e.summary,
                    "timestamp": e.timestamp,
                }
                for e in coord.wiki_graph.interface_history
            ],
        }

    @app.post("/api/memory/query")
    def query_memory(payload: QueryRequest) -> Dict[str, Any]:
        """Sub-100ms semantic RAG context retrieval from organizational memory."""
        query = ContextQuery(
            query_text=payload.query_text,
            top_k=payload.top_k,
            target_module=payload.target_module,
        )
        response = coord.rag_service.query(query)
        return {
            "query": response.query,
            "relevant_chunks": response.relevant_chunks,
            "coupled_modules": response.coupled_modules,
            "applicable_adrs": response.applicable_adrs,
            "api_contracts": response.api_contracts,
            "synthesized_context": response.synthesized_context,
            "retrieval_latency_ms": response.retrieval_latency_ms,
        }

    @app.post("/api/orchestrator/step")
    def trigger_step() -> Dict[str, Any]:
        """Advances one discrete orchestration dispatch step."""
        summary = coord.step()
        return summary

    @app.post("/api/orchestrator/intervention")
    def apply_intervention(req: InterventionRequest) -> Dict[str, Any]:
        """Human-in-the-loop operator intervention."""
        res = coord.apply_intervention(req.intervention_type, req.params)
        return res

    @app.post("/api/orchestrator/reset")
    def reset_orchestrator() -> Dict[str, Any]:
        """Resets the orchestrator and tears down all active worktrees."""
        coord.reset()
        return {"status": "success", "message": "Orchestrator reset completed."}

    # -------------------------------------------------------------
    # WebSocket Streaming Endpoint
    # -------------------------------------------------------------

    @app.websocket("/ws/telemetry")
    async def websocket_telemetry(websocket: WebSocket) -> None:
        """Streams real-time execution telemetry, STGAT attention updates, and logs."""
        await manager.connect(websocket)
        # Send initial state
        initial_event = {
            "event_type": "INITIAL_STATE",
            "timestamp": time.time(),
            "data": coord.get_project_state(),
        }
        await websocket.send_text(json.dumps(initial_event))
        try:
            while True:
                data = await websocket.receive_text()
                # Handle incoming operator commands over WebSocket if needed
                try:
                    cmd = json.loads(data)
                    if cmd.get("action") == "STEP":
                        step_summary = coord.step()
                        await websocket.send_text(json.dumps({
                            "event_type": "STEP_RESULT",
                            "data": step_summary,
                        }))
                except Exception as e:
                    logger.warning("Error processing ws incoming command: %s", e)
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    # -------------------------------------------------------------
    # Frontend Static File Serving (if web/dist exists)
    # -------------------------------------------------------------
    dist_path = Path(__file__).resolve().parent.parent.parent.parent / "web" / "dist"
    if dist_path.exists() and dist_path.is_dir():
        app.mount("/assets", StaticFiles(directory=str(dist_path / "assets")), name="assets")

        @app.get("/{full_path:path}")
        def serve_spa(full_path: str):
            index_file = dist_path / "index.html"
            file_candidate = dist_path / full_path
            if file_candidate.is_file():
                return FileResponse(file_candidate)
            return FileResponse(index_file)

    return app


app = create_app()
