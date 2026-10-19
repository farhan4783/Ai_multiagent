"""
Dynamic Atomic Port Lease Allocator for Parallel Sandboxed Workspaces.

Guarantees collision-free localhost port blocks per agent worktree,
validating system socket availability and managing lease lifecycles.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import socket
import threading
import time

from project_marl.core.logging import setup_logger

logger = setup_logger("port_allocator")


@dataclass
class PortAllocation:
    """Represents a leased block of ports for a specific agent and task."""
    agent_id: str
    task_id: str
    primary_port: int
    test_port: int
    debug_port: int
    websocket_port: int
    metrics_port: int
    allocated_at: float = field(default_factory=time.time)

    @property
    def all_ports(self) -> List[int]:
        return [
            self.primary_port,
            self.test_port,
            self.debug_port,
            self.websocket_port,
            self.metrics_port,
        ]

    def to_env_dict(self) -> Dict[str, str]:
        """Exports ports as environment variable dictionary."""
        return {
            "PORT": str(self.primary_port),
            "TEST_PORT": str(self.test_port),
            "DEBUG_PORT": str(self.debug_port),
            "WS_PORT": str(self.websocket_port),
            "METRICS_PORT": str(self.metrics_port),
        }


class PortAllocator:
    """
    Thread-safe dynamic port allocator that reserves non-overlapping blocks of ports
    for parallel developer agents, ensuring zero binding collisions.
    """

    def __init__(
        self,
        base_port: int = 12000,
        max_port: int = 25000,
        ports_per_agent: int = 5,
    ) -> None:
        self.base_port = base_port
        self.max_port = max_port
        self.ports_per_agent = ports_per_agent

        self._lock = threading.Lock()
        # Key: (agent_id, task_id) -> PortAllocation
        self._allocations: Dict[Tuple[str, str], PortAllocation] = {}
        # Set of currently leased ports
        self._leased_ports: Set[int] = set()
        self._next_port = base_port

    @staticmethod
    def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
        """
        Tests if a TCP port is currently open and bindable on the host.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return True
            except (socket.error, OSError):
                return False

    def allocate(self, agent_id: str, task_id: str) -> PortAllocation:
        """
        Atomically leases a block of contiguous or verified-available ports for an agent task.

        Args:
            agent_id: Identifier of the developer agent.
            task_id: Identifier of the assigned task.

        Returns:
            PortAllocation instance containing reserved ports.
        """
        key = (agent_id, task_id)
        with self._lock:
            if key in self._allocations:
                return self._allocations[key]

            allocated_ports: List[int] = []
            candidate = self._next_port

            while len(allocated_ports) < self.ports_per_agent:
                if candidate > self.max_port:
                    candidate = self.base_port  # Wrap around

                if candidate not in self._leased_ports and self.is_port_available(candidate):
                    allocated_ports.append(candidate)
                    self._leased_ports.add(candidate)

                candidate += 1
                if candidate == self._next_port and len(allocated_ports) < self.ports_per_agent:
                    raise RuntimeError("Port exhaustion: No free ports available in prescribed range.")

            self._next_port = candidate

            allocation = PortAllocation(
                agent_id=agent_id,
                task_id=task_id,
                primary_port=allocated_ports[0],
                test_port=allocated_ports[1],
                debug_port=allocated_ports[2],
                websocket_port=allocated_ports[3],
                metrics_port=allocated_ports[4],
            )
            self._allocations[key] = allocation
            logger.info(
                "Allocated ports %s to agent=%s task=%s",
                allocated_ports,
                agent_id,
                task_id,
            )
            return allocation

    def release(self, agent_id: str, task_id: str) -> Optional[PortAllocation]:
        """
        Releases port allocation for the specified agent and task.
        """
        key = (agent_id, task_id)
        with self._lock:
            allocation = self._allocations.pop(key, None)
            if allocation is not None:
                for p in allocation.all_ports:
                    self._leased_ports.discard(p)
                logger.info(
                    "Released ports %s for agent=%s task=%s",
                    allocation.all_ports,
                    agent_id,
                    task_id,
                )
            return allocation

    def get_allocation(self, agent_id: str, task_id: str) -> Optional[PortAllocation]:
        """Retrieves existing allocation if present."""
        with self._lock:
            return self._allocations.get((agent_id, task_id))

    def release_all(self) -> int:
        """Releases all active port allocations."""
        with self._lock:
            count = len(self._allocations)
            self._allocations.clear()
            self._leased_ports.clear()
            self._next_port = self.base_port
            return count

    @property
    def active_leases_count(self) -> int:
        with self._lock:
            return len(self._allocations)
