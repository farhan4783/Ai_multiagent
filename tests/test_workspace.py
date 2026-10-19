"""
Comprehensive Unit and Integration Test Suite for Phase 5: Sandboxed Parallel Execution Workspace.

Tests:
    1. PortAllocator atomic leasing, socket availability checks, and collision prevention.
    2. DatabaseIsolator ephemeral SQLite database branching, schema initialization, and teardown.
    3. ResourceIsolator unified port and database environment variable provisioning.
    4. GitWorktreeManager programmatic worktree creation, status, commits, merge verification, and teardown.
    5. MCPServer standardized tools (read, write, list, diff, commit, ast query) and path traversal sandboxing.
    6. MCPAgentRunner execution modes (Aggressive, Balanced, Conservative).
    7. ASTConflictResolver pre-merge static analysis and symbol collision detection.
    8. WorkspaceOrchestrator parallel agent swarm execution with zero resource collisions.
"""

from pathlib import Path
import os
import shutil
import sqlite3
import subprocess
import tempfile
import json
import pytest

from project_marl.workspace.port_allocator import PortAllocator
from project_marl.workspace.db_isolator import DatabaseIsolator
from project_marl.workspace.resource_isolator import ResourceIsolator
from project_marl.workspace.worktree_manager import GitWorktreeManager
from project_marl.workspace.conflict_resolver import ASTConflictResolver
from project_marl.workspace.mcp_server import MCPServer
from project_marl.workspace.mcp_runner import MCPAgentRunner
from project_marl.workspace.orchestrator import WorkspaceOrchestrator


def init_test_git_repo(repo_path: Path) -> None:
    """Helper to initialize a real git repository with initial commit in a temp dir."""
    repo_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=str(repo_path), check=True, capture_output=True, stdin=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.name", "Test Agent"], cwd=str(repo_path), check=True, capture_output=True, stdin=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "agent@test.local"], cwd=str(repo_path), check=True, capture_output=True, stdin=subprocess.DEVNULL)

    readme = repo_path / "README.md"
    readme.write_text("# Test Repo\n", encoding="utf-8")

    src_file = repo_path / "app.py"
    src_file.write_text(
        "def hello():\n"
        "    return 'world'\n\n"
        "def compute(x):\n"
        "    return x + 1\n",
        encoding="utf-8",
    )

    subprocess.run(["git", "add", "."], cwd=str(repo_path), check=True, capture_output=True, stdin=subprocess.DEVNULL)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(repo_path), check=True, capture_output=True, stdin=subprocess.DEVNULL)


class TestPortAllocator:
    """Tests for dynamic port allocation and system socket verification."""

    def test_port_allocator_disjoint_leases(self) -> None:
        allocator = PortAllocator(base_port=18000, max_port=18100, ports_per_agent=5)

        lease1 = allocator.allocate("agent_0", "TASK-001")
        lease2 = allocator.allocate("agent_1", "TASK-002")

        # 5 ports each
        assert len(lease1.all_ports) == 5
        assert len(lease2.all_ports) == 5

        # Completely disjoint sets
        set1 = set(lease1.all_ports)
        set2 = set(lease2.all_ports)
        assert set1.isdisjoint(set2)

        # Idempotent allocation for same agent & task
        lease1_again = allocator.allocate("agent_0", "TASK-001")
        assert lease1.primary_port == lease1_again.primary_port

        # Test environment dictionary
        env_dict = lease1.to_env_dict()
        assert "PORT" in env_dict
        assert "TEST_PORT" in env_dict
        assert "DEBUG_PORT" in env_dict

        # Release
        allocator.release("agent_0", "TASK-001")
        assert allocator.get_allocation("agent_0", "TASK-001") is None
        assert allocator.active_leases_count == 1

        allocator.release_all()
        assert allocator.active_leases_count == 0


class TestDatabaseIsolator:
    """Tests for ephemeral SQLite database provisioning."""

    def test_sqlite_branch_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            isolator = DatabaseIsolator(default_db_type="sqlite")
            init_sql = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"

            lease = isolator.provision("dev_0", "TASK-101", tmp_dir, init_sql=init_sql)
            assert lease.db_type == "sqlite"
            assert lease.db_path is not None
            assert os.path.exists(lease.db_path)
            assert lease.connection_uri.startswith("sqlite:///")

            # Connect and verify table exists
            conn = sqlite3.connect(lease.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name) VALUES ('Alice');")
            conn.commit()
            cursor.execute("SELECT name FROM users;")
            rows = cursor.fetchall()
            conn.close()
            assert rows == [("Alice",)]

            # Teardown
            isolator.teardown("dev_0", "TASK-101")
            assert not os.path.exists(lease.db_path)


class TestResourceIsolator:
    """Tests for unified resource management."""

    def test_resource_provisioning_and_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            isolator = ResourceIsolator(base_port=19000)
            res = isolator.provision("dev_1", "TASK-202", tmp_dir)

            env = res.to_env_dict()
            assert env["MARL_AGENT_ID"] == "dev_1"
            assert env["MARL_TASK_ID"] == "TASK-202"
            assert "PORT" in env
            assert "DATABASE_URL" in env

            isolator.release("dev_1", "TASK-202")


class TestGitWorktreeManager:
    """Tests for Git worktree lifecycle management."""

    def test_worktree_creation_commit_and_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_path = Path(tmp_dir) / "main_repo"
            init_test_git_repo(repo_path)

            wt_mgr = GitWorktreeManager(repo_dir=str(repo_path))
            assert wt_mgr.is_git_repo()

            # Create worktree for agent_0 on TASK-001
            wt_info = wt_mgr.create_worktree("agent_0", "TASK-001")
            assert wt_info.exists()
            assert wt_info.branch_name == "agent/agent_0/TASK-001"

            # Modify file in worktree
            wt_app = Path(wt_info.worktree_path) / "app.py"
            wt_app.write_text(
                "def hello():\n"
                "    return 'world from worktree'\n\n"
                "def compute(x):\n"
                "    return x + 1\n",
                encoding="utf-8",
            )

            # Commit inside worktree
            commit_sha = wt_mgr.commit_in_worktree("agent_0", "TASK-001", "Update hello message")
            assert commit_sha is not None

            # Verify clean merge check
            can_merge = wt_mgr.can_merge_cleanly("agent_0", "TASK-001")
            assert can_merge is True

            # Merge into main
            merged = wt_mgr.merge_worktree_into_base("agent_0", "TASK-001", target_branch="main")
            assert merged is True

            # Verify main repo now contains updated content
            main_app = repo_path / "app.py"
            assert "world from worktree" in main_app.read_text(encoding="utf-8")

            # Remove worktree
            wt_mgr.remove_worktree("agent_0", "TASK-001")
            assert not wt_info.exists()


class TestMCPServer:
    """Tests for MCP JSON-RPC 2.0 tool interface and sandboxing."""

    def test_mcp_tools_and_sandboxing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = MCPServer(worktree_path=tmp_dir, agent_id="dev_0", task_id="TASK-01")

            # 1. Write file
            w_res = server.write_file("src/module.py", "def add(a, b):\n    return a + b\n")
            assert not w_res.is_error

            # 2. Read file
            r_res = server.read_file("src/module.py")
            assert not r_res.is_error
            assert "def add(a, b):" in r_res.content

            # 3. List files
            l_res = server.list_files("src")
            assert "module.py" in l_res.content

            # 4. Query AST
            ast_res = server.query_ast("add")
            assert "Found FunctionDef 'add'" in ast_res.content

            # 5. Sandboxing path traversal security test
            # Attempting to read outside worktree must trigger error
            jailbreak = server.read_file("../../windows/system32/cmd.exe")
            assert jailbreak.is_error
            assert "Sandbox violation" in jailbreak.content

            # 6. JSON-RPC protocol
            rpc_list = server.handle_jsonrpc(json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
            }))
            list_data = json.loads(rpc_list)
            assert "result" in list_data
            assert len(list_data["result"]["tools"]) >= 8

            # 7. JSON-RPC tool call
            rpc_call = server.handle_jsonrpc(json.dumps({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "read_file", "arguments": {"path": "src/module.py"}},
            }))
            call_data = json.loads(rpc_call)
            assert "result" in call_data
            assert call_data["result"]["content"][0]["text"].startswith("def add")


class TestMCPAgentRunner:
    """Tests for MCPAgentRunner executing coding actions across execution modes."""

    def test_runner_conservative_ast_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = MCPServer(worktree_path=tmp_dir, agent_id="dev_0", task_id="TASK-01")
            runner = MCPAgentRunner(mcp_server=server, execution_mode=2)  # Conservative

            # Attempt invalid python syntax
            bad_mods = {"broken.py": "def invalid_syntax(:"}
            res = runner.apply_code_modifications(bad_mods)
            assert not res.success
            assert not res.tests_passed
            assert any("AST syntax error" in log for log in res.logs)

            # Valid syntax
            good_mods = {"good.py": "def valid(): pass\n"}
            res_good = runner.apply_code_modifications(good_mods)
            assert res_good.success
            assert "good.py" in res_good.files_modified


class TestASTConflictResolver:
    """Tests for pre-merge static analysis and symbol collision detection."""

    def test_detect_symbol_collision_between_branches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_path = Path(tmp_dir) / "repo"
            init_test_git_repo(repo_path)

            wt_mgr = GitWorktreeManager(repo_dir=str(repo_path))
            wt_a = wt_mgr.create_worktree("agent_A", "TASK-10")
            wt_b = wt_mgr.create_worktree("agent_B", "TASK-20")

            # Worktree A modifies compute() to x * 10
            app_a = Path(wt_a.worktree_path) / "app.py"
            app_a.write_text(
                "def hello():\n"
                "    return 'world'\n\n"
                "def compute(x):\n"
                "    return x * 10\n",
                encoding="utf-8",
            )

            # Worktree B modifies compute() to x - 5
            app_b = Path(wt_b.worktree_path) / "app.py"
            app_b.write_text(
                "def hello():\n"
                "    return 'world'\n\n"
                "def compute(x):\n"
                "    return x - 5\n",
                encoding="utf-8",
            )

            resolver = ASTConflictResolver(repo_dir=str(repo_path))
            report = resolver.analyze_worktrees(
                worktree_a_path=wt_a.worktree_path,
                worktree_b_path=wt_b.worktree_path,
                base_path=str(repo_path),
            )

            assert report.has_file_collision is True
            assert report.has_symbol_collision is True
            assert "app.py" in report.overlapping_files
            assert any(c["symbol_name"] == "compute" for c in report.symbol_collisions)
            assert report.conflict_risk_score > 0.5
            assert not report.is_mergeable()

            wt_mgr.teardown_all()

    def test_safe_non_overlapping_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_path = Path(tmp_dir) / "repo"
            init_test_git_repo(repo_path)

            wt_mgr = GitWorktreeManager(repo_dir=str(repo_path))
            wt_a = wt_mgr.create_worktree("agent_A", "TASK-10")
            wt_b = wt_mgr.create_worktree("agent_B", "TASK-20")

            # Worktree A modifies app.py
            (Path(wt_a.worktree_path) / "app.py").write_text("# app\n", encoding="utf-8")
            # Worktree B creates new file service.py
            (Path(wt_b.worktree_path) / "service.py").write_text("# service\n", encoding="utf-8")

            resolver = ASTConflictResolver(repo_dir=str(repo_path))
            report = resolver.analyze_worktrees(
                worktree_a_path=wt_a.worktree_path,
                worktree_b_path=wt_b.worktree_path,
                base_path=str(repo_path),
            )

            assert report.has_file_collision is False
            assert report.has_symbol_collision is False
            assert report.conflict_risk_score == 0.0
            assert report.recommendation == "SAFE_TO_MERGE"
            assert report.is_mergeable()

            wt_mgr.teardown_all()


class TestWorkspaceOrchestrator:
    """Integration test verifying parallel agent workspaces with zero collisions."""

    def test_parallel_agent_workspaces_swarm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_path = Path(tmp_dir) / "repo"
            init_test_git_repo(repo_path)

            orchestrator = WorkspaceOrchestrator(
                repo_dir=str(repo_path),
                base_port=20000,
            )

            # Provision 3 parallel agent workspaces
            ws0 = orchestrator.provision_workspace("dev_0", "TASK-001", execution_mode=1)
            ws1 = orchestrator.provision_workspace("dev_1", "TASK-002", execution_mode=1)
            ws2 = orchestrator.provision_workspace("dev_2", "TASK-003", execution_mode=1)

            # Verify distinct ports
            ports0 = set(ws0.resources.ports.all_ports)
            ports1 = set(ws1.resources.ports.all_ports)
            ports2 = set(ws2.resources.ports.all_ports)
            assert ports0.isdisjoint(ports1)
            assert ports1.isdisjoint(ports2)

            # Verify distinct worktrees and isolated DB paths
            assert ws0.worktree_path != ws1.worktree_path
            assert ws0.resources.db.connection_uri != ws1.resources.db.connection_uri

            # All 3 agents execute modifications in parallel
            res0 = ws0.runner.apply_code_modifications({"feature_a.py": "def feat_a(): return 1\n"})
            res1 = ws1.runner.apply_code_modifications({"feature_b.py": "def feat_b(): return 2\n"})
            res2 = ws2.runner.apply_code_modifications({"feature_c.py": "def feat_c(): return 3\n"})

            assert res0.success
            assert res1.success
            assert res2.success

            # Check conflicts between ws0 and ws1: completely safe
            conflict_rep = orchestrator.check_conflicts("dev_0", "TASK-001", "dev_1", "TASK-002")
            assert conflict_rep.recommendation == "SAFE_TO_MERGE"

            # Teardown all workspaces
            torn_down = orchestrator.teardown_all()
            assert torn_down == 3
