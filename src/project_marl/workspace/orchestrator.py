"""
High-Level Workspace Orchestrator for Parallel Sandboxed Agent Execution.

Coordinates Git worktrees, isolated port/DB resources, MCP tool servers,
and pre-merge conflict resolution across concurrent developer agents.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import os

from project_marl.workspace.worktree_manager import GitWorktreeManager, WorktreeInfo
from project_marl.workspace.resource_isolator import ResourceIsolator, WorkspaceResources
from project_marl.workspace.mcp_server import MCPServer
from project_marl.workspace.mcp_runner import MCPAgentRunner
from project_marl.workspace.conflict_resolver import ASTConflictResolver, ConflictReport
from project_marl.core.logging import setup_logger

logger = setup_logger("workspace_orchestrator")


@dataclass
class SandboxedWorkspace:
    """Represents a fully provisioned, isolated physical execution environment for an agent."""
    agent_id: str
    task_id: str
    worktree: WorktreeInfo
    resources: WorkspaceResources
    mcp_server: MCPServer
    runner: MCPAgentRunner

    @property
    def worktree_path(self) -> str:
        return self.worktree.worktree_path

    @property
    def branch_name(self) -> str:
        return self.worktree.branch_name

    def to_env_dict(self) -> Dict[str, str]:
        env = self.resources.to_env_dict()
        env["WORKTREE_PATH"] = self.worktree_path
        env["GIT_BRANCH"] = self.branch_name
        return env


class WorkspaceOrchestrator:
    """
    Master orchestrator managing isolated Git worktrees, dynamic port/DB allocations,
    MCP tool servers, and automated conflict analysis across swarms of parallel agents.
    """

    def __init__(
        self,
        repo_dir: Optional[str] = None,
        worktree_root: Optional[str] = None,
        base_port: int = 12000,
        max_port: int = 25000,
        default_db_type: str = "sqlite",
        postgres_base_url: Optional[str] = None,
    ) -> None:
        self.worktree_mgr = GitWorktreeManager(repo_dir=repo_dir, worktree_root=worktree_root)
        self.resource_isolator = ResourceIsolator(
            base_port=base_port,
            max_port=max_port,
            default_db_type=default_db_type,
            postgres_base_url=postgres_base_url,
        )
        self.conflict_resolver = ASTConflictResolver(repo_dir=repo_dir)

        # Key: (agent_id, task_id) -> SandboxedWorkspace
        self._workspaces: Dict[Tuple[str, str], SandboxedWorkspace] = {}

    def provision_workspace(
        self,
        agent_id: str,
        task_id: str,
        base_branch: Optional[str] = None,
        execution_mode: int = 1,
        init_sql: Optional[str] = None,
    ) -> SandboxedWorkspace:
        """
        Creates an isolated Git worktree, allocates ports and database, and initializes MCP server.
        """
        key = (agent_id, task_id)
        if key in self._workspaces:
            return self._workspaces[key]

        # 1. Create Git Worktree
        worktree_info = self.worktree_mgr.create_worktree(
            agent_id=agent_id,
            task_id=task_id,
            base_branch=base_branch,
        )

        # 2. Allocate Ports & Isolated Database
        resources = self.resource_isolator.provision(
            agent_id=agent_id,
            task_id=task_id,
            worktree_path=worktree_info.worktree_path,
            init_sql=init_sql,
        )

        # 3. Initialize Sandboxed MCP Server and Runner
        mcp_server = MCPServer(
            worktree_path=worktree_info.worktree_path,
            agent_id=agent_id,
            task_id=task_id,
        )
        runner = MCPAgentRunner(
            mcp_server=mcp_server,
            execution_mode=execution_mode,
        )

        workspace = SandboxedWorkspace(
            agent_id=agent_id,
            task_id=task_id,
            worktree=worktree_info,
            resources=resources,
            mcp_server=mcp_server,
            runner=runner,
        )
        self._workspaces[key] = workspace
        logger.info(
            "Provisioned sandboxed workspace for agent=%s task=%s at %s",
            agent_id,
            task_id,
            worktree_info.worktree_path,
        )
        return workspace

    def teardown_workspace(
        self,
        agent_id: str,
        task_id: str,
        merge: bool = False,
        target_branch: Optional[str] = None,
    ) -> bool:
        """
        Tears down the sandboxed workspace, releasing ports and database.
        Optionally merges the worktree branch into target_branch.
        """
        key = (agent_id, task_id)
        workspace = self._workspaces.pop(key, None)
        if not workspace:
            return False

        merge_success = True
        if merge:
            merge_success = self.worktree_mgr.merge_worktree_into_base(
                agent_id=agent_id,
                task_id=task_id,
                target_branch=target_branch,
            )

        # Release ports and database
        self.resource_isolator.release(agent_id, task_id)

        # Remove worktree directory and branch
        self.worktree_mgr.remove_worktree(agent_id, task_id, delete_branch=True)

        logger.info("Torn down workspace for agent=%s task=%s (merged=%s)", agent_id, task_id, merge)
        return merge_success

    def check_conflicts(
        self,
        agent_a: str,
        task_a: str,
        agent_b: str,
        task_b: str,
    ) -> ConflictReport:
        """
        Performs AST static analysis to detect code and symbol collisions between two active workspaces.
        """
        ws_a = self._workspaces.get((agent_a, task_a))
        ws_b = self._workspaces.get((agent_b, task_b))

        if not ws_a or not ws_b:
            raise ValueError("Both workspaces must be active to check conflicts.")

        return self.conflict_resolver.analyze_worktrees(
            worktree_a_path=ws_a.worktree_path,
            worktree_b_path=ws_b.worktree_path,
            branch_a_name=ws_a.branch_name,
            branch_b_name=ws_b.branch_name,
        )

    def get_workspace(self, agent_id: str, task_id: str) -> Optional[SandboxedWorkspace]:
        return self._workspaces.get((agent_id, task_id))

    def list_active_workspaces(self) -> List[SandboxedWorkspace]:
        return list(self._workspaces.values())

    def teardown_all(self) -> int:
        """Tears down all active workspaces and releases all resources."""
        count = len(self._workspaces)
        keys = list(self._workspaces.keys())
        for agent_id, task_id in keys:
            self.teardown_workspace(agent_id, task_id, merge=False)
        self.worktree_mgr.teardown_all()
        self.resource_isolator.release_all()
        return count
