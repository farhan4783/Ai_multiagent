"""
Dynamic Self-Updating Organizational Wiki Graph for PromptQL-style Shared Context.

Maintains a semantic topological graph connecting concepts, modules, APIs,
Architectural Decision Records (ADRs), requirements, and parallel agent tasks.
Detects interface drift and generates automated summaries upon API alterations.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import networkx as nx
import json
import re
import time

from project_marl.core.logging import setup_logger

logger = setup_logger("wiki_graph")


@dataclass
class WikiNode:
    """Represents an entity in the organizational memory graph."""
    node_id: str
    node_type: str  # 'concept', 'module', 'api', 'adr', 'requirement', 'task'
    title: str
    description: str = ""
    content: str = ""
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "version": self.version,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WikiNode":
        return cls(**data)


@dataclass
class InterfaceDriftEvent:
    """Recorded event when a parallel worktree alters a module's public interface."""
    module_id: str
    task_id: str
    agent_id: str
    added_symbols: List[str]
    removed_symbols: List[str]
    modified_symbols: List[str]
    summary: str
    timestamp: float = field(default_factory=time.time)


class WikiGraph:
    """
    Self-Updating Organizational Wiki Graph connecting code modules, APIs,
    architectural requirements, and project tasks into a navigable semantic network.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, WikiNode] = {}
        self.interface_history: List[InterfaceDriftEvent] = []

    def add_node(self, node: WikiNode) -> None:
        """Adds or updates a node in the graph."""
        self.nodes[node.node_id] = node
        self.graph.add_node(
            node.node_id,
            node_type=node.node_type,
            title=node.title,
            version=node.version,
        )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Adds a directed relation between two entities in the wiki graph.
        Relations: 'implements', 'depends_on', 'calls', 'documented_in', 'alters_interface', 'modifies'
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            raise KeyError(f"Both nodes must exist before linking: {source_id} -> {target_id}")

        self.graph.add_edge(
            source_id,
            target_id,
            relation=relation,
            weight=weight,
            metadata=metadata or {},
        )

    def get_node(self, node_id: str) -> Optional[WikiNode]:
        return self.nodes.get(node_id)

    def add_concept(
        self,
        concept_id: str,
        title: str,
        description: str,
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WikiNode:
        node = WikiNode(
            node_id=concept_id,
            node_type="concept",
            title=title,
            description=description,
            content=content,
            metadata=metadata or {},
        )
        self.add_node(node)
        return node

    def add_module(
        self,
        module_id: str,
        name: str,
        path: str = "",
        exported_symbols: Optional[List[str]] = None,
    ) -> WikiNode:
        meta = {"path": path, "exported_symbols": exported_symbols or []}
        node = WikiNode(
            node_id=module_id,
            node_type="module",
            title=name,
            description=f"Codebase module {name} at {path}",
            metadata=meta,
        )
        self.add_node(node)
        return node

    def add_api(
        self,
        api_id: str,
        signature: str,
        module_id: str,
        docstring: str = "",
    ) -> WikiNode:
        node = WikiNode(
            node_id=api_id,
            node_type="api",
            title=signature,
            description=docstring,
            metadata={"module_id": module_id, "signature": signature},
        )
        self.add_node(node)
        if module_id in self.nodes:
            self.add_edge(module_id, api_id, relation="defines")
        return node

    def add_adr(
        self,
        adr_id: str,
        title: str,
        status: str = "ACCEPTED",
        decision: str = "",
        related_modules: Optional[List[str]] = None,
    ) -> WikiNode:
        meta = {"status": status, "related_modules": related_modules or []}
        node = WikiNode(
            node_id=adr_id,
            node_type="adr",
            title=title,
            description=decision,
            content=decision,
            metadata=meta,
        )
        self.add_node(node)
        for mod in (related_modules or []):
            if mod in self.nodes:
                self.add_edge(adr_id, mod, relation="governs")
        return node

    def add_task(
        self,
        task_id: str,
        title: str,
        impacted_modules: Optional[List[str]] = None,
    ) -> WikiNode:
        meta = {"impacted_modules": impacted_modules or []}
        node = WikiNode(
            node_id=task_id,
            node_type="task",
            title=title,
            metadata=meta,
        )
        self.add_node(node)
        for mod in (impacted_modules or []):
            if mod in self.nodes:
                self.add_edge(task_id, mod, relation="modifies")
        return node

    def ingest_markdown_doc(
        self,
        doc_path: str,
        text: str,
        doc_type: str = "adr",
        related_modules: Optional[List[str]] = None,
    ) -> WikiNode:
        """
        Parses Markdown documentation or ADR into structured wiki node with metadata.
        """
        lines = text.strip().splitlines()
        title = lines[0].replace("#", "").strip() if lines else doc_path

        # Extract sections
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        doc_id = re.sub(r"[^a-zA-Z0-9_-]", "_", doc_path).lower()
        node = WikiNode(
            node_id=doc_id,
            node_type=doc_type,
            title=title,
            description=body[:200] + ("..." if len(body) > 200 else ""),
            content=text,
            metadata={"path": doc_path, "doc_type": doc_type},
        )
        self.add_node(node)

        for mod in (related_modules or []):
            if mod in self.nodes:
                self.add_edge(doc_id, mod, relation="documents")

        logger.info("Ingested Markdown document %s (%s)", doc_path, doc_type)
        return node

    def update_module_interface(
        self,
        module_id: str,
        new_exported_symbols: List[str],
        agent_id: str,
        task_id: str,
    ) -> Optional[InterfaceDriftEvent]:
        """
        Detects changes in a module's public interface, creates an automated summary,
        records an interface drift event, and links the modifying task.
        """
        mod_node = self.nodes.get(module_id)
        if not mod_node:
            return None

        old_symbols = set(mod_node.metadata.get("exported_symbols", []))
        new_symbols = set(new_exported_symbols)

        added = sorted(list(new_symbols - old_symbols))
        removed = sorted(list(old_symbols - new_symbols))

        if not added and not removed:
            return None  # No interface drift

        summary_parts = []
        if added:
            summary_parts.append(f"Added symbols: {', '.join(added)}")
        if removed:
            summary_parts.append(f"Removed symbols: {', '.join(removed)}")
        summary = "; ".join(summary_parts)

        # Update module node
        mod_node.version += 1
        mod_node.metadata["exported_symbols"] = new_exported_symbols
        mod_node.updated_at = time.time()

        drift_event = InterfaceDriftEvent(
            module_id=module_id,
            task_id=task_id,
            agent_id=agent_id,
            added_symbols=added,
            removed_symbols=removed,
            modified_symbols=[],
            summary=summary,
        )
        self.interface_history.append(drift_event)

        # Connect task with alters_interface edge
        if task_id in self.nodes:
            self.add_edge(
                task_id,
                module_id,
                relation="alters_interface",
                metadata={"drift_summary": summary, "agent_id": agent_id},
            )

        # Flag dependent modules/tasks of interface changes
        for predecessor in self.graph.predecessors(module_id):
            edge_data = self.graph.get_edge_data(predecessor, module_id)
            if edge_data and edge_data.get("relation") == "depends_on":
                self.nodes[predecessor].metadata["interface_drift_alert"] = True

        logger.info("Interface drift detected in %s by %s: %s", module_id, task_id, summary)
        return drift_event

    def get_dependencies(self, node_id: str, depth: int = 1) -> Set[str]:
        """Returns nodes that node_id depends on (successors)."""
        if node_id not in self.graph:
            return set()
        visited = set()
        current = {node_id}
        for _ in range(depth):
            next_level = set()
            for n in current:
                for succ in self.graph.successors(n):
                    if succ not in visited and succ != node_id:
                        visited.add(succ)
                        next_level.add(succ)
            current = next_level
        return visited

    def get_dependents(self, node_id: str, depth: int = 1) -> Set[str]:
        """Returns nodes that depend on node_id (predecessors)."""
        if node_id not in self.graph:
            return set()
        visited = set()
        current = {node_id}
        for _ in range(depth):
            next_level = set()
            for n in current:
                for pred in self.graph.predecessors(n):
                    if pred not in visited and pred != node_id:
                        visited.add(pred)
                        next_level.add(pred)
            current = next_level
        return visited

    def query_subgraph_context(self, node_ids: List[str]) -> Dict[str, Any]:
        """Gathers connected concepts, ADRs, APIs, and requirements for a set of nodes."""
        related_node_ids: Set[str] = set(node_ids)
        for n_id in node_ids:
            if n_id in self.graph:
                related_node_ids.update(self.graph.successors(n_id))
                related_node_ids.update(self.graph.predecessors(n_id))

        concepts = []
        adrs = []
        apis = []
        modules = []

        for nid in related_node_ids:
            node = self.nodes.get(nid)
            if not node:
                continue
            if node.node_type == "concept":
                concepts.append({"id": node.node_id, "title": node.title, "desc": node.description})
            elif node.node_type == "adr":
                adrs.append({"id": node.node_id, "title": node.title, "decision": node.description})
            elif node.node_type == "api":
                apis.append({"id": node.node_id, "signature": node.title, "doc": node.description})
            elif node.node_type == "module":
                modules.append({"id": node.node_id, "name": node.title, "exports": node.metadata.get("exported_symbols", [])})

        return {
            "concepts": concepts,
            "adrs": adrs,
            "apis": apis,
            "modules": modules,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes wiki graph state."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "relation": data.get("relation", ""),
                    "weight": data.get("weight", 1.0),
                    "metadata": data.get("metadata", {}),
                }
                for u, v, data in self.graph.edges(data=True)
            ],
            "interface_history": [
                {
                    "module_id": e.module_id,
                    "task_id": e.task_id,
                    "agent_id": e.agent_id,
                    "added": e.added_symbols,
                    "removed": e.removed_symbols,
                    "summary": e.summary,
                    "timestamp": e.timestamp,
                }
                for e in self.interface_history
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WikiGraph":
        """Deserializes wiki graph state."""
        wiki = cls()
        for nd in data.get("nodes", []):
            wiki.add_node(WikiNode.from_dict(nd))
        for ed in data.get("edges", []):
            wiki.add_edge(
                ed["source"],
                ed["target"],
                relation=ed["relation"],
                weight=ed.get("weight", 1.0),
                metadata=ed.get("metadata", {}),
            )
        return wiki
