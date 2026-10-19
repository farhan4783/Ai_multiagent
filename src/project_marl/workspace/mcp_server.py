"""
Model Context Protocol (MCP) Server for Sandboxed Worktree Workspaces.

Exposes standardized workspace tools (read_file, write_file, run_tests, git_commit,
git_diff, query_ast) compliant with MCP JSON-RPC 2.0 specifications with strict path sandboxing.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import ast
import json
import os
import subprocess

from project_marl.core.logging import setup_logger

logger = setup_logger("mcp_server")


@dataclass
class ToolResult:
    """Standardized result returned by an MCP tool invocation."""
    content: str
    is_error: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": [{"type": "text", "text": self.content}],
            "isError": self.is_error,
            "metadata": self.metadata or {},
        }


class MCPServer:
    """
    Sandboxed MCP Server providing CLI coding agents (Claude Code, Cursor CLI, custom LLMs)
    with safe, isolated filesystem, git, and testing capabilities inside their worktree.
    """

    def __init__(
        self,
        worktree_path: str,
        agent_id: str,
        task_id: str,
    ) -> None:
        self.worktree_path = Path(worktree_path).resolve()
        self.agent_id = agent_id
        self.task_id = task_id

        # Register tools
        self._tools: Dict[str, Callable[..., ToolResult]] = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_files": self.list_files,
            "run_command": self.run_command,
            "run_tests": self.run_tests,
            "git_diff": self.git_diff,
            "git_commit": self.git_commit,
            "query_ast": self.query_ast,
        }

    def _resolve_safe_path(self, rel_or_abs_path: str) -> Path:
        """
        Resolves a path and enforces sandbox containment within self.worktree_path.
        Prevents directory traversal (e.g., '../') outside the assigned worktree.
        """
        target = Path(rel_or_abs_path)
        if not target.is_absolute():
            resolved = (self.worktree_path / target).resolve()
        else:
            resolved = target.resolve()

        try:
            resolved.relative_to(self.worktree_path)
        except ValueError:
            raise PermissionError(
                f"Sandbox violation: Target path '{rel_or_abs_path}' escapes worktree root '{self.worktree_path}'"
            )

        return resolved

    def read_file(self, path: str) -> ToolResult:
        """Reads file contents from inside the worktree."""
        try:
            safe_path = self._resolve_safe_path(path)
            if not safe_path.exists():
                return ToolResult(content=f"Error: File not found at {path}", is_error=True)
            if not safe_path.is_file():
                return ToolResult(content=f"Error: {path} is not a file", is_error=True)

            text = safe_path.read_text(encoding="utf-8", errors="replace")
            return ToolResult(content=text, is_error=False)
        except Exception as e:
            return ToolResult(content=f"Error reading file: {str(e)}", is_error=True)

    def write_file(self, path: str, content: str) -> ToolResult:
        """Writes or creates a file inside the worktree."""
        try:
            safe_path = self._resolve_safe_path(path)
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            safe_path.write_text(content, encoding="utf-8")
            return ToolResult(content=f"Successfully wrote {len(content)} bytes to {path}", is_error=False)
        except Exception as e:
            return ToolResult(content=f"Error writing file: {str(e)}", is_error=True)

    def list_files(self, directory: str = ".") -> ToolResult:
        """Lists files and directories inside the worktree path."""
        try:
            safe_path = self._resolve_safe_path(directory)
            if not safe_path.exists():
                return ToolResult(content=f"Directory {directory} does not exist", is_error=True)

            items = []
            for entry in sorted(os.listdir(safe_path)):
                if entry.startswith(".git"):
                    continue
                p = safe_path / entry
                kind = "DIR" if p.is_dir() else "FILE"
                items.append(f"[{kind}] {entry}")

            return ToolResult(content="\n".join(items) if items else "(empty directory)")
        except Exception as e:
            return ToolResult(content=f"Error listing files: {str(e)}", is_error=True)

    def run_command(self, command: str, timeout: int = 60) -> ToolResult:
        """Executes a command strictly within the worktree directory."""
        try:
            res = subprocess.run(
                command,
                shell=True,
                cwd=str(self.worktree_path),
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
            )
            out = f"Return Code: {res.returncode}\n--- STDOUT ---\n{res.stdout}\n--- STDERR ---\n{res.stderr}"
            return ToolResult(content=out, is_error=(res.returncode != 0))
        except subprocess.TimeoutExpired:
            return ToolResult(content=f"Command timed out after {timeout} seconds", is_error=True)
        except Exception as e:
            return ToolResult(content=f"Execution error: {str(e)}", is_error=True)

    def run_tests(self, test_path: str = "tests") -> ToolResult:
        """Executes pytest inside the worktree."""
        safe_path = self._resolve_safe_path(test_path)
        cmd = f'python -m pytest "{safe_path}" -v'
        return self.run_command(cmd, timeout=120)

    def git_diff(self) -> ToolResult:
        """Returns the current git diff of changes made inside the worktree."""
        return self.run_command("git diff HEAD", timeout=30)

    def git_commit(self, message: str) -> ToolResult:
        """Stages all changes and creates a git commit."""
        self.run_command("git add -A")
        return self.run_command(f'git commit -m "{message}"')

    def query_ast(self, symbol_name: str) -> ToolResult:
        """Inspects Python source files in the worktree for matching functions or classes."""
        matches = []
        for root, _, files in os.walk(self.worktree_path):
            if ".git" in root or "__pycache__" in root:
                continue
            for f in files:
                if f.endswith(".py"):
                    full_p = Path(root) / f
                    try:
                        tree = ast.parse(full_p.read_text(encoding="utf-8", errors="replace"))
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                if node.name == symbol_name:
                                    rel_p = full_p.relative_to(self.worktree_path)
                                    matches.append(
                                        f"Found {type(node).__name__} '{node.name}' in {rel_p} at line {node.lineno}"
                                    )
                    except Exception:
                        pass

        if matches:
            return ToolResult(content="\n".join(matches))
        return ToolResult(content=f"Symbol '{symbol_name}' not found in Python files.")

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns MCP tool definitions conforming to MCP tools/list."""
        return [
            {
                "name": "read_file",
                "description": "Reads file contents safely from inside the worktree sandbox.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": "Writes content to a file inside the worktree sandbox.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "list_files",
                "description": "Lists files and directories in the worktree.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"directory": {"type": "string", "default": "."}},
                },
            },
            {
                "name": "run_command",
                "description": "Runs a command strictly inside the worktree.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
            {
                "name": "run_tests",
                "description": "Runs pytest test suites inside the worktree.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"test_path": {"type": "string", "default": "tests"}},
                },
            },
            {
                "name": "git_diff",
                "description": "Returns current git diff inside the worktree.",
                "inputSchema": {"type": "object"},
            },
            {
                "name": "git_commit",
                "description": "Commits all changes inside the worktree.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            },
            {
                "name": "query_ast",
                "description": "Searches for Python AST symbols across the worktree.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"symbol_name": {"type": "string"}},
                    "required": ["symbol_name"],
                },
            },
        ]

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Dispatches an MCP tool call by name."""
        args = arguments or {}
        if name not in self._tools:
            return ToolResult(content=f"Unknown tool: {name}", is_error=True)
        return self._tools[name](**args)

    def handle_jsonrpc(self, request_str: str) -> str:
        """
        Processes a raw JSON-RPC 2.0 message according to the Model Context Protocol specification.
        """
        try:
            req = json.loads(request_str)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if method == "tools/list":
                res = {"tools": self.list_tools()}
                return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": res})

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                result = self.call_tool(tool_name, tool_args)
                return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result.to_dict()})

            elif method == "initialize":
                res = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": f"stgat-marl-sandbox-{self.agent_id}",
                        "version": "1.0.0",
                    },
                }
                return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": res})

            else:
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                })

        except Exception as e:
            return json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
            })
