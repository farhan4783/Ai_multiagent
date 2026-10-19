"""
Semantic State Builder & Context Injector for STGAT-MARL Integration.

Extracts semantic context vectors from organizational memory to enrich the STGAT
node feature matrix X_t and injects fresh, zero-stale context into developer agent prompts.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from project_marl.memory.vector_store import VectorStore
from project_marl.memory.wiki_graph import WikiGraph
from project_marl.memory.rag_service import RAGService, ContextQuery
from project_marl.core.schemas import TaskNode
from project_marl.core.logging import setup_logger

logger = setup_logger("context_injector")


class SemanticContextInjector:
    """
    Enriches STGAT task node features with organizational memory vectors
    and provides Developer Agents with immediate, deterministic context.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        wiki_graph: WikiGraph,
        rag_service: Optional[RAGService] = None,
        semantic_dim: int = 16,
    ) -> None:
        self.vector_store = vector_store
        self.wiki_graph = wiki_graph
        self.rag_service = rag_service or RAGService(vector_store, wiki_graph)
        self.semantic_dim = semantic_dim

    def build_task_semantic_vector(
        self,
        task_id: str,
        task_title: str,
        impacted_modules: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Builds a fixed-dimensional semantic vector e_i^semantic in R^d encapsulating
        the requirements, ADR constraints, and module coupling contracts of a task.
        """
        # Formulate query from title and impacted modules
        mods = impacted_modules or []
        query_text = f"{task_title} {' '.join(mods)}"

        response = self.rag_service.query(ContextQuery(query_text=query_text, top_k=3))

        # Combine text content from retrieved ADRs, APIs, and chunks
        context_corpus = [task_title]
        for adr in response.applicable_adrs:
            context_corpus.append(adr)
        for contract in response.api_contracts:
            context_corpus.append(contract)
        for chunk in response.relevant_chunks:
            context_corpus.append(chunk["text"])

        joint_text = " ".join(context_corpus)

        # Project using vector_store embedding engine and slice/project to semantic_dim
        raw_emb = self.vector_store.engine.embed_text(joint_text)

        if len(raw_emb) == self.semantic_dim:
            return raw_emb
        elif len(raw_emb) > self.semantic_dim:
            # Deterministic projection/folding
            folded = np.zeros(self.semantic_dim, dtype=np.float32)
            for i, val in enumerate(raw_emb):
                folded[i % self.semantic_dim] += val
            norm = np.linalg.norm(folded)
            return folded / max(norm, 1e-8)
        else:
            padded = np.zeros(self.semantic_dim, dtype=np.float32)
            padded[:len(raw_emb)] = raw_emb
            return padded

    def enrich_task_node_features(
        self,
        task: TaskNode,
        base_features: np.ndarray,
    ) -> np.ndarray:
        """
        Appends the semantic vector e_i^semantic to the task node feature vector:
        x_i = [base_features, e_i^semantic]
        """
        sem_vec = self.build_task_semantic_vector(
            task_id=task.task_id,
            task_title=task.title,
            impacted_modules=task.impacted_modules,
        )
        return np.concatenate([base_features, sem_vec]).astype(np.float32)

    def generate_agent_prompt_context(
        self,
        agent_id: str,
        task: TaskNode,
    ) -> str:
        """
        Assembles fresh, structured context prompt for the developer agent
        executing inside a sandboxed worktree.
        """
        query_text = f"Implementation specification for {task.title} modifying {', '.join(task.impacted_modules)}"
        rag_resp = self.rag_service.query(ContextQuery(
            query_text=query_text,
            target_task=task.task_id,
            target_modules=task.impacted_modules,
            top_k=3,
        ))

        prompt_lines = [
            f"# Developer Agent Instructions: {agent_id}",
            f"**Assigned Task:** [{task.task_id}] {task.title}",
            f"**Estimated Duration:** {task.estimated_duration:.1f} hours",
            f"**Target Impacted Modules:** {', '.join(task.impacted_modules) if task.impacted_modules else 'None'}",
            "",
            "## Shared Organizational Memory & Architecture Context",
            rag_resp.synthesized_context,
            "",
            "## Execution Directives:",
            "1. Adhere strictly to the API contracts and ADR decisions documented above.",
            "2. Execute in your isolated sandboxed worktree with zero branch pollution.",
            "3. If public interfaces are modified, verify backward compatibility for coupled modules.",
        ]
        return "\n".join(prompt_lines)
