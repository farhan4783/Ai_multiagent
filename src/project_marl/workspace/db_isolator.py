"""
Ephemeral Database Isolator for Sandboxed Worktree Workspaces.

Provisions isolated database instances (lightweight SQLite branches or isolated
PostgreSQL schema spaces) per worktree to prevent shared-state corruption.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import os
import shutil
import sqlite3
import time

from project_marl.core.logging import setup_logger

logger = setup_logger("db_isolator")


@dataclass
class DBLease:
    """Represents an isolated database lease for an agent worktree."""
    agent_id: str
    task_id: str
    db_type: str  # 'sqlite' or 'postgres'
    connection_uri: str
    db_path: Optional[str] = None
    schema_name: Optional[str] = None
    allocated_at: float = field(default_factory=time.time)

    def to_env_dict(self) -> Dict[str, str]:
        return {
            "DATABASE_URL": self.connection_uri,
            "DB_TYPE": self.db_type,
            "DB_SCHEMA": self.schema_name or "",
        }


class DatabaseIsolator:
    """
    Manages isolated database branches for concurrent developer agents.
    Default backend: lightweight SQLite instances created within the worktree sandbox.
    """

    def __init__(
        self,
        default_db_type: str = "sqlite",
        postgres_base_url: Optional[str] = None,
    ) -> None:
        self.default_db_type = default_db_type
        self.postgres_base_url = postgres_base_url
        self._leases: Dict[Tuple[str, str], DBLease] = {}

    def provision(
        self,
        agent_id: str,
        task_id: str,
        worktree_path: str,
        init_sql: Optional[str] = None,
    ) -> DBLease:
        """
        Provisions an isolated database for the agent worktree.

        Args:
            agent_id: Identifier of the developer agent.
            task_id: Identifier of the assigned task.
            worktree_path: Absolute path to the sandboxed worktree.
            init_sql: Optional initial schema SQL script to execute.

        Returns:
            DBLease containing the connection URI.
        """
        key = (agent_id, task_id)
        if key in self._leases:
            return self._leases[key]

        if self.default_db_type == "sqlite":
            db_dir = os.path.join(worktree_path, ".db")
            os.makedirs(db_dir, exist_ok=True)
            db_file = os.path.join(db_dir, f"{agent_id}_{task_id}.sqlite")

            # Initialize SQLite database
            conn = sqlite3.connect(db_file)
            if init_sql:
                conn.executescript(init_sql)
            conn.commit()
            conn.close()

            # Normalized URI (forward slashes for SQLite URI)
            normalized_path = db_file.replace("\\", "/")
            uri = f"sqlite:///{normalized_path}"

            lease = DBLease(
                agent_id=agent_id,
                task_id=task_id,
                db_type="sqlite",
                connection_uri=uri,
                db_path=db_file,
            )
            self._leases[key] = lease
            logger.info("Provisioned SQLite database at %s for agent=%s task=%s", db_file, agent_id, task_id)
            return lease

        elif self.default_db_type == "postgres":
            # Sanitize schema name: only alphanumeric and underscores
            safe_agent = agent_id.replace("-", "_")
            safe_task = task_id.replace("-", "_")
            schema_name = f"schema_{safe_agent}_{safe_task}".lower()

            base_url = self.postgres_base_url or "postgresql://postgres:postgres@localhost:5432/marl_db"
            separator = "&" if "?" in base_url else "?"
            uri = f"{base_url}{separator}options=-csearch_path%3D{schema_name}"

            lease = DBLease(
                agent_id=agent_id,
                task_id=task_id,
                db_type="postgres",
                connection_uri=uri,
                schema_name=schema_name,
            )
            self._leases[key] = lease
            logger.info("Provisioned Postgres schema %s for agent=%s task=%s", schema_name, agent_id, task_id)
            return lease

        else:
            raise ValueError(f"Unsupported database type: {self.default_db_type}")

    def teardown(self, agent_id: str, task_id: str) -> Optional[DBLease]:
        """
        Tears down the isolated database and cleans up stored files or schema.
        """
        key = (agent_id, task_id)
        lease = self._leases.pop(key, None)
        if lease is None:
            return None

        if lease.db_type == "sqlite" and lease.db_path and os.path.exists(lease.db_path):
            try:
                os.remove(lease.db_path)
                db_parent = os.path.dirname(lease.db_path)
                if os.path.exists(db_parent) and not os.listdir(db_parent):
                    os.rmdir(db_parent)
                logger.info("Removed isolated SQLite database at %s", lease.db_path)
            except OSError as e:
                logger.warning("Could not remove sqlite file %s: %s", lease.db_path, e)

        return lease

    def teardown_all(self) -> int:
        """Tears down all active database leases."""
        keys = list(self._leases.keys())
        for agent_id, task_id in keys:
            self.teardown(agent_id, task_id)
        return len(keys)
