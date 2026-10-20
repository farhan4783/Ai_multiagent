"""
Comprehensive Unit & Integration Tests for Phase 7:
Autonomous Project Coordinator and FastAPI Backend Service.

Tests:
    1. AutonomousProjectCoordinator lifecycle: project ingestion, CPM computation, STGAT forecasting.
    2. Step execution: task readiness detection, dispatch to idle agents, physical sandbox work execution.
    3. Human-in-the-Loop interventions: priority overrides, ad-hoc task injections, safety buffers.
    4. PromptQL memory integration: interface drift tracking and sub-100ms RAG retrieval.
    5. FastAPI REST endpoints: health, state, gantt, attention heatmap, agent status, and step triggers.
    6. WebSocket telemetry streaming: initial state handshake and live message transmission.
"""

from pathlib import Path
import tempfile
import json
import pytest
from fastapi.testclient import TestClient

from project_marl.orchestrator.coordinator import (
    AutonomousProjectCoordinator,
    OrchestratorConfig,
)
from project_marl.orchestrator.api import create_app
from project_marl.core.schemas import TaskNode
from project_marl.core.constants import ExecutionStatus


def sample_test_tasks() -> list[TaskNode]:
    """Generates a small test project DAG."""
    return [
        TaskNode(
            task_id="T1",
            title="Design DB Schema",
            description="Create tables and migrations",
            estimated_duration=3.0,
            impacted_modules=["src/models.py"],
            priority=4,
        ),
        TaskNode(
            task_id="T2",
            title="Implement Auth Endpoints",
            description="JWT login and registration",
            estimated_duration=4.0,
            predecessors=["T1"],
            impacted_modules=["src/auth.py"],
            priority=5,
        ),
        TaskNode(
            task_id="T3",
            title="Build Dashboard UI",
            description="React frontend components",
            estimated_duration=5.0,
            predecessors=["T1"],
            impacted_modules=["src/ui.py"],
            priority=3,
        ),
        TaskNode(
            task_id="T4",
            title="Integration Verification",
            description="Run full end-to-end tests",
            estimated_duration=2.0,
            predecessors=["T2", "T3"],
            impacted_modules=["tests/test_e2e.py"],
            priority=5,
        ),
    ]


class TestAutonomousProjectCoordinator:
    """Tests for AutonomousProjectCoordinator lifecycle and execution."""

    def test_coordinator_initialization_and_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = OrchestratorConfig(
                project_id="test-proj",
                repo_dir=tmp_dir,
                worktree_root=str(Path(tmp_dir) / ".worktrees"),
                num_developers=2,
            )
            coordinator = AutonomousProjectCoordinator(cfg)

            tasks = sample_test_tasks()
            state = coordinator.ingest_project(tasks=tasks, analyze_code=False)

            assert state.project_id == "test-proj"
            assert len(state.tasks) == 4
            # Critical path: T1 (3) + T3 (5) + T4 (2) = 10 hrs (or T1 + T2 (4) + T4 = 9)
            assert state.project_makespan >= 9.0
            assert "T1" in state.critical_path
            assert "T4" in state.critical_path

            # STGAT predictions computed
            preds = coordinator.latest_predictions
            assert "predicted_makespan_hrs" in preds
            assert "task_predictions" in preds
            assert "T1" in preds["task_predictions"]
            assert "delay_mean_hrs" in preds["task_predictions"]["T1"]
            assert "fragility_score" in preds["task_predictions"]["T1"]

    def test_coordinator_step_dispatch_and_sandbox_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = OrchestratorConfig(
                project_id="test-dispatch",
                repo_dir=tmp_dir,
                worktree_root=str(Path(tmp_dir) / ".worktrees"),
                num_developers=2,
                auto_merge_on_complete=False,
            )
            coordinator = AutonomousProjectCoordinator(cfg)
            tasks = sample_test_tasks()
            coordinator.ingest_project(tasks=tasks, analyze_code=False)

            # Step 1: T1 has no predecessors, so it should be dispatched to an idle developer
            step1 = coordinator.step()
            assert step1["step"] == 1
            assert len(step1["dispatches"]) >= 1
            assert step1["dispatches"][0]["task_id"] == "T1"
            assert coordinator.graph_builder.tasks["T1"].status == ExecutionStatus.IN_PROGRESS

            # Advance until T1 completes (duration is 3.0 + any injected buffer)
            for _ in range(6):
                coordinator.step()

            # T1 should be completed, unblocking T2 and T3
            assert coordinator.graph_builder.tasks["T1"].status == ExecutionStatus.COMPLETED

            # T2 and T3 should now be ready or dispatched
            t2 = coordinator.graph_builder.tasks["T2"]
            t3 = coordinator.graph_builder.tasks["T3"]
            assert t2.status in (ExecutionStatus.IN_PROGRESS, ExecutionStatus.COMPLETED, ExecutionStatus.PENDING)
            assert t3.status in (ExecutionStatus.IN_PROGRESS, ExecutionStatus.COMPLETED, ExecutionStatus.PENDING)

            # Clean teardown
            coordinator.reset()
            assert len(coordinator.active_tasks_dispatched) == 0

    def test_coordinator_interventions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = OrchestratorConfig(project_id="test-intervention", repo_dir=tmp_dir)
            coordinator = AutonomousProjectCoordinator(cfg)
            tasks = sample_test_tasks()
            coordinator.ingest_project(tasks=tasks, analyze_code=False)

            # 1. Priority Override
            res1 = coordinator.apply_intervention("PRIORITY_OVERRIDE", {"task_id": "T3", "priority": 5})
            assert res1["status"] == "success"
            assert coordinator.graph_builder.tasks["T3"].priority == 5

            # 2. Buffer Injection
            orig_dur = coordinator.graph_builder.tasks["T2"].estimated_duration
            res2 = coordinator.apply_intervention("INJECT_BUFFER", {"task_id": "T2", "buffer_hours": 3.0})
            assert res2["status"] == "success"
            assert coordinator.graph_builder.tasks["T2"].estimated_duration == orig_dur + 3.0

            # 3. Ad-hoc Task Injection
            res3 = coordinator.apply_intervention("INSERT_TASK", {
                "task_id": "T_HOTFIX",
                "title": "Hotfix security vulnerability",
                "duration": 2.0,
                "predecessors": ["T1"],
                "priority": 5,
            })
            assert res3["status"] == "success"
            assert "T_HOTFIX" in coordinator.graph_builder.tasks


class TestFastAPIBackendAPI:
    """Integration tests for FastAPI REST and WebSocket endpoints."""

    @pytest.fixture
    def test_client(self) -> TestClient:
        tmp_dir = tempfile.mkdtemp()
        cfg = OrchestratorConfig(
            project_id="test-api",
            repo_dir=tmp_dir,
            worktree_root=str(Path(tmp_dir) / ".worktrees"),
            num_developers=2,
            auto_merge_on_complete=False,
        )
        coordinator = AutonomousProjectCoordinator(cfg)
        coordinator.ingest_project(tasks=sample_test_tasks(), analyze_code=False)

        # Pre-seed wiki memory
        coordinator.wiki_graph.add_adr(
            adr_id="ADR-TEST",
            title="ADR-Test: Token Policy",
            status="ACCEPTED",
            decision="Tokens must use HMAC-SHA256.",
            related_modules=["src/auth.py"],
        )
        coordinator.vector_store.add_document("doc_auth_api", "Authentication uses JWT with HMAC-SHA256.")

        app = create_app(coordinator)
        return TestClient(app)

    def test_health_endpoint(self, test_client: TestClient) -> None:
        res = test_client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["project_id"] == "test-api"
        assert data["total_tasks"] == 4

    def test_project_state_endpoint(self, test_client: TestClient) -> None:
        res = test_client.get("/api/project/state")
        assert res.status_code == 200
        data = res.json()
        assert "tasks" in data
        assert "modules" in data
        assert "edges" in data
        assert "makespan" in data
        assert "critical_path" in data
        assert len(data["tasks"]) == 4

    def test_gantt_endpoint(self, test_client: TestClient) -> None:
        res = test_client.get("/api/project/gantt")
        assert res.status_code == 200
        items = res.json()
        assert isinstance(items, list)
        assert len(items) == 4
        # Sorted by start time
        assert items[0]["task_id"] == "T1"
        assert items[0]["start"] == 0.0

    def test_attention_endpoint(self, test_client: TestClient) -> None:
        res = test_client.get("/api/project/attention")
        assert res.status_code == 200
        data = res.json()
        assert "task_predictions" in data
        assert "edge_attentions" in data

    def test_agents_status_endpoint(self, test_client: TestClient) -> None:
        res = test_client.get("/api/agents/status")
        assert res.status_code == 200
        agents = res.json()
        assert len(agents) >= 4  # sched, risk, pm, + 2 devs
        roles = {a["role"] for a in agents}
        assert "scheduling_agent" in roles
        assert "developer_agent" in roles

    def test_memory_query_endpoint(self, test_client: TestClient) -> None:
        payload = {"query_text": "What token algorithm does authentication use?", "top_k": 2}
        res = test_client.post("/api/memory/query", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["retrieval_latency_ms"] < 100.0
        assert "HMAC-SHA256" in data["synthesized_context"]

    def test_step_trigger_endpoint(self, test_client: TestClient) -> None:
        res = test_client.post("/api/orchestrator/step")
        assert res.status_code == 200
        data = res.json()
        assert "step" in data
        assert data["step"] == 1
        assert len(data["dispatches"]) >= 1

    def test_intervention_endpoint(self, test_client: TestClient) -> None:
        payload = {
            "intervention_type": "PRIORITY_OVERRIDE",
            "params": {"task_id": "T1", "priority": 5},
        }
        res = test_client.post("/api/orchestrator/intervention", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"

    def test_websocket_telemetry_handshake(self, test_client: TestClient) -> None:
        with test_client.websocket_connect("/ws/telemetry") as websocket:
            init_msg = websocket.receive_text()
            data = json.loads(init_msg)
            assert data["event_type"] == "INITIAL_STATE"
            assert "data" in data
            assert data["data"]["project_id"] == "test-api"
