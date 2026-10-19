"""
Sandboxed Parallel Execution Workspace for Multi-Agent Coding Swarms.

Provides Git worktrees, dynamic port and ephemeral DB isolation, MCP tool servers,
and AST pre-merge conflict resolution.
"""

from project_marl.workspace.port_allocator import PortAllocator, PortAllocation
from project_marl.workspace.db_isolator import DatabaseIsolator, DBLease
from project_marl.workspace.resource_isolator import ResourceIsolator, WorkspaceResources
from project_marl.workspace.worktree_manager import GitWorktreeManager, WorktreeInfo
from project_marl.workspace.conflict_resolver import (
    ASTConflictResolver,
    ConflictReport,
    SymbolModification,
)
from project_marl.workspace.mcp_server import MCPServer, ToolResult
from project_marl.workspace.mcp_runner import MCPAgentRunner, ExecutionResult
from project_marl.workspace.orchestrator import (
    WorkspaceOrchestrator,
    SandboxedWorkspace,
)

__all__ = [
    "PortAllocator",
    "PortAllocation",
    "DatabaseIsolator",
    "DBLease",
    "ResourceIsolator",
    "WorkspaceResources",
    "GitWorktreeManager",
    "WorktreeInfo",
    "ASTConflictResolver",
    "ConflictReport",
    "SymbolModification",
    "MCPServer",
    "ToolResult",
    "MCPAgentRunner",
    "ExecutionResult",
    "WorkspaceOrchestrator",
    "SandboxedWorkspace",
]
