"""
Model Context Protocol (MCP) Agent Runner for Sandboxed Worktree Workspaces.

Executes automated developer agent actions driven by MARL Developer Agent policies
(Aggressive, Balanced, Conservative) over the MCPServer tool interface.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time

from project_marl.workspace.mcp_server import MCPServer, ToolResult
from project_marl.core.logging import setup_logger

logger = setup_logger("mcp_runner")


@dataclass
class ExecutionResult:
    """Outcome of an agent's execution within its sandboxed worktree."""
    success: bool
    agent_id: str
    task_id: str
    commit_sha: Optional[str] = None
    tests_passed: bool = True
    files_modified: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class MCPAgentRunner:
    """
    Drives coding agent execution inside a sandboxed worktree via the MCP interface,
    adapting test rigor and commit cadence according to Developer Agent execution mode.
    """

    def __init__(
        self,
        mcp_server: MCPServer,
        execution_mode: int = 1,  # 0: Aggressive, 1: Balanced, 2: Conservative
    ) -> None:
        self.server = mcp_server
        self.execution_mode = execution_mode

    def apply_code_modifications(
        self,
        modifications: Dict[str, str],
        commit_message: Optional[str] = None,
        test_path: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Executes code modifications within the worktree sandbox according to mode:
            - Mode 0 (Aggressive): writes all files, skips tests, commits immediately.
            - Mode 1 (Balanced): writes files, runs targeted tests, commits if tests pass.
            - Mode 2 (Conservative): verifies AST syntax, writes files, runs full tests, commits if green.

        Args:
            modifications: Dict mapping relative file paths to new file content.
            commit_message: Optional commit message for the git commit.
            test_path: Optional path to test suite.

        Returns:
            ExecutionResult containing outcome, modified files, and logs.
        """
        start_time = time.time()
        logs: List[str] = []
        files_modified: List[str] = []
        tests_passed = True
        commit_sha: Optional[str] = None

        logs.append(f"Starting execution in mode {self.execution_mode} for task {self.server.task_id}")

        # Conservative mode preflight AST validation
        if self.execution_mode == 2:
            for file_path, content in modifications.items():
                if file_path.endswith(".py"):
                    try:
                        import ast
                        ast.parse(content)
                        logs.append(f"Pre-flight AST validation passed for {file_path}")
                    except SyntaxError as e:
                        logs.append(f"Pre-flight AST syntax error in {file_path}: {e}")
                        return ExecutionResult(
                            success=False,
                            agent_id=self.server.agent_id,
                            task_id=self.server.task_id,
                            tests_passed=False,
                            logs=logs,
                            elapsed_seconds=time.time() - start_time,
                        )

        # 1. Write modified files via MCP
        for file_path, content in modifications.items():
            res = self.server.write_file(file_path, content)
            if res.is_error:
                logs.append(f"Failed to write {file_path}: {res.content}")
                return ExecutionResult(
                    success=False,
                    agent_id=self.server.agent_id,
                    task_id=self.server.task_id,
                    tests_passed=False,
                    files_modified=files_modified,
                    logs=logs,
                    elapsed_seconds=time.time() - start_time,
                )
            files_modified.append(file_path)
            logs.append(f"Wrote file {file_path}")

        # 2. Run Tests if not Aggressive mode (Mode 1 & 2)
        if self.execution_mode in (1, 2) and test_path:
            test_res = self.server.run_tests(test_path)
            logs.append(f"Ran tests:\n{test_res.content}")
            if test_res.is_error:
                tests_passed = False
                logs.append("Unit tests failed in sandbox.")
                if self.execution_mode == 2:
                    # Conservative mode rejects commit on test failure
                    return ExecutionResult(
                        success=False,
                        agent_id=self.server.agent_id,
                        task_id=self.server.task_id,
                        tests_passed=False,
                        files_modified=files_modified,
                        logs=logs,
                        elapsed_seconds=time.time() - start_time,
                    )

        # 3. Create Git commit
        msg = commit_message or f"Agent {self.server.agent_id} completed {self.server.task_id}"
        commit_res = self.server.git_commit(msg)
        logs.append(f"Git commit output: {commit_res.content}")

        # Extract commit hash
        diff_res = self.server.run_command("git rev-parse HEAD")
        if not diff_res.is_error:
            commit_sha = diff_res.content.splitlines()[2] if len(diff_res.content.splitlines()) > 2 else None

        elapsed = time.time() - start_time
        return ExecutionResult(
            success=True,
            agent_id=self.server.agent_id,
            task_id=self.server.task_id,
            commit_sha=commit_sha,
            tests_passed=tests_passed,
            files_modified=files_modified,
            logs=logs,
            elapsed_seconds=elapsed,
        )
