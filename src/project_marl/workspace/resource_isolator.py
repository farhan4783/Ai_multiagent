"""
Unified Resource Isolator coordinating port allocation and ephemeral databases.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from project_marl.workspace.port_allocator import PortAllocator, PortAllocation
from project_marl.workspace.db_isolator import DatabaseIsolator, DBLease
from project_marl.core.logging import setup_logger

logger = setup_logger("resource_isolator")


@dataclass
class WorkspaceResources:
    """Unified container for all isolated system resources reserved for a worktree."""
    agent_id: str
    task_id: str
    ports: PortAllocation
    db: DBLease

    def to_env_dict(self) -> Dict[str, str]:
        """Combines all isolated port and database configurations into environment variables."""
        env = {
            "MARL_AGENT_ID": self.agent_id,
            "MARL_TASK_ID": self.task_id,
        }
        env.update(self.ports.to_env_dict())
        env.update(self.db.to_env_dict())
        return env


class ResourceIsolator:
    """
    High-level orchestrator ensuring complete hardware & state isolation across parallel worktrees:
    port allocation, dynamic SQLite/Postgres instances, and environment isolation.
    """

    def __init__(
        self,
        base_port: int = 12000,
        max_port: int = 25000,
        default_db_type: str = "sqlite",
        postgres_base_url: Optional[str] = None,
    ) -> None:
        self.port_allocator = PortAllocator(base_port=base_port, max_port=max_port)
        self.db_isolator = DatabaseIsolator(
            default_db_type=default_db_type,
            postgres_base_url=postgres_base_url,
        )
        self._resources: Dict[Tuple[str, str], WorkspaceResources] = {}

    def provision(
        self,
        agent_id: str,
        task_id: str,
        worktree_path: str,
        init_sql: Optional[str] = None,
    ) -> WorkspaceResources:
        """
        Atomically provisions both isolated ports and ephemeral database for an agent.
        """
        key = (agent_id, task_id)
        if key in self._resources:
            return self._resources[key]

        ports = self.port_allocator.allocate(agent_id, task_id)
        db = self.db_isolator.provision(agent_id, task_id, worktree_path, init_sql=init_sql)

        resources = WorkspaceResources(
            agent_id=agent_id,
            task_id=task_id,
            ports=ports,
            db=db,
        )
        self._resources[key] = resources
        return resources

    def release(self, agent_id: str, task_id: str) -> None:
        """Releases ports and database lease."""
        key = (agent_id, task_id)
        self._resources.pop(key, None)
        self.port_allocator.release(agent_id, task_id)
        self.db_isolator.teardown(agent_id, task_id)

    def release_all(self) -> None:
        """Tears down all allocated resources."""
        self._resources.clear()
        self.port_allocator.release_all()
        self.db_isolator.teardown_all()
