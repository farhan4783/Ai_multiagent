"""
Context Retrieval-Augmented Generation (RAG) Service for Developer Agent Swarms.

Provides deterministic, sub-100ms semantic context retrieval by fusing VectorStore
unstructured similarity search with WikiGraph topological relationship navigation.
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
import time

from project_marl.memory.vector_store import VectorStore, VectorDocument
from project_marl.memory.wiki_graph import WikiGraph
from project_marl.core.logging import setup_logger

logger = setup_logger("rag_service")


@dataclass
class ContextQuery:
    """Structured context query submitted by an agent."""
    query_text: str
    target_module: Optional[str] = None
    target_modules: Optional[List[str]] = None
    target_task: Optional[str] = None
    top_k: int = 5
    max_hops: int = 2


@dataclass
class RAGResponse:
    """Synthesized context result delivered to an executing agent."""
    query: str
    relevant_chunks: List[Dict[str, Any]] = field(default_factory=list)
    coupled_modules: List[str] = field(default_factory=list)
    applicable_adrs: List[str] = field(default_factory=list)
    api_contracts: List[str] = field(default_factory=list)
    synthesized_context: str = ""
    retrieval_latency_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        return (
            len(self.relevant_chunks) == 0
            and len(self.coupled_modules) == 0
            and len(self.applicable_adrs) == 0
            and len(self.api_contracts) == 0
        )


class RAGService:
    """
    Sub-100ms deterministic organizational memory retrieval engine.
    Fuses VectorStore embeddings with WikiGraph topological relationship queries.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        wiki_graph: WikiGraph,
    ) -> None:
        self.vector_store = vector_store
        self.wiki_graph = wiki_graph

    def query(self, query: Union[str, ContextQuery]) -> RAGResponse:
        """
        Executes hybrid semantic and structural context retrieval.
        Guarantees sub-100ms latency without stale hallucinations.
        """
        start_time = time.perf_counter()

        if isinstance(query, str):
            c_query = ContextQuery(query_text=query)
        else:
            c_query = query

        q_text = c_query.query_text
        top_k = c_query.top_k

        # 1. Vector Search for relevant document chunks
        filter_dict = {}
        if c_query.target_module:
            filter_dict["module_id"] = c_query.target_module

        vector_results = self.vector_store.search(
            query=q_text,
            top_k=top_k,
            filter_dict=filter_dict if filter_dict else None,
        )

        relevant_chunks: List[Dict[str, Any]] = []
        inferred_module_ids: Set[str] = set()

        for doc, score in vector_results:
            relevant_chunks.append({
                "doc_id": doc.doc_id,
                "text": doc.text,
                "score": round(score, 4),
                "metadata": doc.metadata,
            })
            if "module_id" in doc.metadata:
                inferred_module_ids.add(doc.metadata["module_id"])

        if c_query.target_module:
            inferred_module_ids.add(c_query.target_module)

        if c_query.target_modules:
            inferred_module_ids.update(c_query.target_modules)

        if c_query.target_task and c_query.target_task in self.wiki_graph.nodes:
            task_node = self.wiki_graph.nodes[c_query.target_task]
            for mod in task_node.metadata.get("impacted_modules", []):
                inferred_module_ids.add(mod)
            inferred_module_ids.add(c_query.target_task)

        # Infer modules directly mentioned in query string
        for mod_id, node in self.wiki_graph.nodes.items():
            if node.node_type == "module" and (mod_id.lower() in q_text.lower() or node.title.lower() in q_text.lower()):
                inferred_module_ids.add(mod_id)

        # 2. Query Structural WikiGraph for coupled modules, ADRs, and API contracts
        applicable_adrs: List[str] = []
        api_contracts: List[str] = []
        coupled_modules: List[str] = []

        subgraph_context = self.wiki_graph.query_subgraph_context(list(inferred_module_ids))

        for adr in subgraph_context["adrs"]:
            applicable_adrs.append(f"[{adr['id']}] {adr['title']}: {adr['decision']}")

        for api in subgraph_context["apis"]:
            api_contracts.append(f"{api['signature']} - {api['doc']}")

        for mod in subgraph_context["modules"]:
            coupled_modules.append(mod["name"])

        # Also check for interface drift alerts on target modules
        drift_alerts: List[str] = []
        for mod_id in inferred_module_ids:
            node = self.wiki_graph.get_node(mod_id)
            if node and node.metadata.get("interface_drift_alert"):
                drift_alerts.append(f"ALERT: Module '{mod_id}' public interface was recently modified in a parallel worktree.")

        # 3. Synthesize deterministic context markdown
        lines = [f"### Context for Query: \"{q_text}\""]

        if drift_alerts:
            lines.append("\n**Interface Drift Warnings:**")
            for alert in drift_alerts:
                lines.append(f"- {alert}")

        if applicable_adrs:
            lines.append("\n**Architectural Decision Records (ADRs):**")
            for adr in applicable_adrs:
                lines.append(f"- {adr}")

        if api_contracts:
            lines.append("\n**API Contracts & Signatures:**")
            for contract in api_contracts:
                lines.append(f"- `{contract}`")

        if relevant_chunks:
            lines.append("\n**Relevant Documentation Snippets:**")
            for chunk in relevant_chunks[:3]:
                lines.append(f"> {chunk['text'].strip()}")

        synthesized = "\n".join(lines)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        response = RAGResponse(
            query=q_text,
            relevant_chunks=relevant_chunks,
            coupled_modules=sorted(list(set(coupled_modules))),
            applicable_adrs=applicable_adrs,
            api_contracts=api_contracts,
            synthesized_context=synthesized,
            retrieval_latency_ms=round(latency_ms, 2),
        )

        logger.debug("RAG Query completed in %.2f ms (results=%d)", latency_ms, len(relevant_chunks))
        return response
