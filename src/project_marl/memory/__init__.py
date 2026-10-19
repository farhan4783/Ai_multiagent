"""
PromptQL-Inspired Shared Context Wiki and Organizational Memory Layer.

Provides continuous documentation ingestion, topological semantic wiki graph,
sub-100ms RAG context retrieval, and STGAT semantic vector injection.
"""

from project_marl.memory.vector_store import (
    VectorDocument,
    EmbeddingEngine,
    VectorStore,
)
from project_marl.memory.wiki_graph import (
    WikiNode,
    WikiGraph,
    InterfaceDriftEvent,
)
from project_marl.memory.rag_service import (
    ContextQuery,
    RAGResponse,
    RAGService,
)
from project_marl.memory.context_injector import SemanticContextInjector

# Architectural deliverable aliases
wiki_engine = WikiGraph
wiki_graph = WikiGraph
vector_store = VectorStore
context_injector = SemanticContextInjector

__all__ = [
    "VectorDocument",
    "EmbeddingEngine",
    "VectorStore",
    "WikiNode",
    "WikiGraph",
    "InterfaceDriftEvent",
    "ContextQuery",
    "RAGResponse",
    "RAGService",
    "SemanticContextInjector",
    "wiki_engine",
    "wiki_graph",
    "vector_store",
    "context_injector",
]
