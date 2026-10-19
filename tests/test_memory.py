"""
Comprehensive Unit and Integration Test Suite for Phase 6: Shared Context Wiki & Organizational Memory.

Tests:
    1. VectorStore semantic embedding generation, cosine similarity search, and metadata filtering.
    2. VectorStore serialization and deserialization.
    3. WikiGraph topological entity representation and dependency traversal.
    4. Continuous Markdown and ADR ingestion pipeline.
    5. Self-updating interface drift detection, automated summary generation, and drift alerts.
    6. RAGService hybrid retrieval, sub-100ms latency verification, and context synthesis.
    7. SemanticContextInjector STGAT feature vector enrichment and prompt assembly.
"""

from pathlib import Path
import os
import tempfile
import time
import pytest
import numpy as np

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
from project_marl.core.schemas import TaskNode


class TestVectorStore:
    """Tests for VectorStore and EmbeddingEngine."""

    def test_embedding_engine_normalization(self) -> None:
        engine = EmbeddingEngine(dim=64)
        emb = engine.embed_text("Authentication with OAuth and JWT tokens")
        assert emb.shape == (64,)
        # Must be unit norm
        norm = np.linalg.norm(emb)
        assert pytest.approx(norm, abs=1e-5) == 1.0

    def test_vector_search_and_similarity_ranking(self) -> None:
        store = VectorStore(dim=64)

        doc1 = store.add_document(
            doc_id="doc_auth",
            text="OAuth2 and JWT token authentication protocol for user sessions.",
            metadata={"module_id": "auth_service", "doc_type": "security"},
        )
        doc2 = store.add_document(
            doc_id="doc_db",
            text="PostgreSQL index tuning and connection pooling for high-throughput queries.",
            metadata={"module_id": "db_layer", "doc_type": "infrastructure"},
        )
        doc3 = store.add_document(
            doc_id="doc_billing",
            text="Stripe payment gateway integration and invoice calculation.",
            metadata={"module_id": "billing", "doc_type": "finance"},
        )

        assert store.size == 3

        # Query for auth should rank doc1 highest
        results = store.search("How do user sessions authenticate with JWT?", top_k=2)
        assert len(results) == 2
        top_doc, top_score = results[0]
        assert top_doc.doc_id == "doc_auth"
        assert top_score > results[1][1]

        # Metadata filtering
        filtered_results = store.search("performance", filter_dict={"module_id": "db_layer"})
        assert len(filtered_results) == 1
        assert filtered_results[0][0].doc_id == "doc_db"

    def test_vector_store_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "test_vstore.json")

            store = VectorStore(dim=64)
            store.add_document("d1", "Microservice architecture ADR", {"category": "architecture"})
            store.add_document("d2", "Redis distributed caching", {"category": "caching"})
            store.save(file_path)

            loaded_store = VectorStore.load(file_path)
            assert loaded_store.size == 2
            assert loaded_store.dim == 64
            assert loaded_store.get_document("d1") is not None

            # Test search on loaded store
            res = loaded_store.search("Redis cache", top_k=1)
            assert res[0][0].doc_id == "d2"


class TestWikiGraph:
    """Tests for WikiGraph semantic network and interface drift detection."""

    def test_wiki_graph_traversal(self) -> None:
        wiki = WikiGraph()

        # Add concepts and modules
        wiki.add_concept("concept_auth", "OAuth2 Flow", "Standard authorization protocol")
        wiki.add_module("mod_auth", "AuthModule", "src/auth.py", ["login", "verify_token"])
        wiki.add_api("api_login", "def login(user, pwd)", "mod_auth", "User authentication endpoint")
        wiki.add_adr("adr_001", "ADR-001: Token-based Auth", "ACCEPTED", "Use RS256 signed JWTs", ["mod_auth"])

        # Add relations
        wiki.add_edge("mod_auth", "concept_auth", relation="implements")

        # Verify dependencies
        deps = wiki.get_dependencies("mod_auth")
        assert "concept_auth" in deps or "api_login" in deps

        dependents = wiki.get_dependents("mod_auth")
        assert "adr_001" in dependents

        # Subgraph query
        subgraph = wiki.query_subgraph_context(["mod_auth"])
        assert any(a["id"] == "adr_001" for a in subgraph["adrs"])
        assert any(m["id"] == "mod_auth" for m in subgraph["modules"])

    def test_markdown_and_adr_ingestion(self) -> None:
        wiki = WikiGraph()
        wiki.add_module("payment_worker", "Payment Worker")

        md_content = """# ADR-042: Idempotent Payment Processing
## Status
ACCEPTED
## Context and Decision
All payment webhooks must carry idempotency keys stored in Redis for 24 hours.
"""
        node = wiki.ingest_markdown_doc("docs/adr_042.md", md_content, doc_type="adr", related_modules=["payment_worker"])
        assert node.title == "ADR-042: Idempotent Payment Processing"
        assert wiki.graph.has_edge("docs_adr_042_md", "payment_worker")

    def test_interface_drift_detection_and_alerts(self) -> None:
        wiki = WikiGraph()
        wiki.add_module("user_service", "UserService", "src/user.py", ["get_user", "update_user"])
        wiki.add_module("order_service", "OrderService", "src/order.py", ["place_order"])
        wiki.add_task("TASK-104", "Refactor User Service")

        # order_service depends on user_service
        wiki.add_edge("order_service", "user_service", relation="depends_on")

        # Parallel worktree modifies user_service interface: removes update_user, adds update_profile
        drift = wiki.update_module_interface(
            module_id="user_service",
            new_exported_symbols=["get_user", "update_profile"],
            agent_id="agent-0",
            task_id="TASK-104",
        )

        assert drift is not None
        assert "update_user" in drift.removed_symbols
        assert "update_profile" in drift.added_symbols

        # Alters interface edge created
        assert wiki.graph.has_edge("TASK-104", "user_service")
        edge_data = wiki.graph.get_edge_data("TASK-104", "user_service")
        assert edge_data["relation"] == "alters_interface"

        # Dependent order_service receives interface drift alert
        assert wiki.nodes["order_service"].metadata.get("interface_drift_alert") is True


class TestRAGService:
    """Tests for sub-100ms RAG retrieval and context synthesis."""

    def test_rag_query_latency_and_context_synthesis(self) -> None:
        store = VectorStore(dim=64)
        wiki = WikiGraph()

        # Ingest docs and graph
        wiki.add_module("crypto_lib", "CryptoLib", "src/crypto.py", ["encrypt", "decrypt"])
        wiki.add_adr("adr_sec", "ADR-Sec: AES-GCM", "ACCEPTED", "All data at rest must use AES-256-GCM", ["crypto_lib"])
        wiki.add_api("api_enc", "encrypt(data: bytes) -> bytes", "crypto_lib", "Encrypts payload using AES-256-GCM")

        store.add_document("doc_crypto", "CryptoLib provides cryptographic primitives including AES-256-GCM encryption.", {"module_id": "crypto_lib"})

        rag = RAGService(vector_store=store, wiki_graph=wiki)

        # Benchmark query execution time
        start_t = time.perf_counter()
        response = rag.query(ContextQuery(query_text="What encryption cipher is used by CryptoLib?"))
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        # Sub-100ms acceptance criteria verification
        assert latency_ms < 100.0, f"RAG query latency {latency_ms:.2f}ms exceeded 100ms threshold!"
        assert response.retrieval_latency_ms < 100.0
        assert not response.is_empty

        # Synthesized content checks
        assert "AES-256-GCM" in response.synthesized_context
        assert any("ADR-Sec" in adr for adr in response.applicable_adrs)
        assert any("encrypt" in contract for contract in response.api_contracts)


class TestSemanticContextInjector:
    """Tests for STGAT feature vector enrichment and prompt assembly."""

    def test_semantic_vector_injection(self) -> None:
        store = VectorStore(dim=64)
        wiki = WikiGraph()
        wiki.add_module("auth", "AuthModule")
        wiki.add_adr("adr_01", "JWT Specification", "ACCEPTED", "Tokens must expire after 15 minutes", ["auth"])

        injector = SemanticContextInjector(vector_store=store, wiki_graph=wiki, semantic_dim=16)

        sem_vec = injector.build_task_semantic_vector(
            task_id="TASK-01",
            task_title="Implement JWT token expiration",
            impacted_modules=["auth"],
        )
        assert sem_vec.shape == (16,)
        assert not np.isnan(sem_vec).any()

        # Augment base features
        task = TaskNode(
            task_id="TASK-01",
            title="Implement JWT token expiration",
            estimated_duration=4.0,
            impacted_modules=["auth"],
        )
        base_features = np.array([4.0, 0.0, 1.5, 0.0, 1.0], dtype=np.float32)
        enriched = injector.enrich_task_node_features(task, base_features)

        assert enriched.shape == (5 + 16,)

    def test_agent_prompt_assembly(self) -> None:
        store = VectorStore(dim=64)
        wiki = WikiGraph()
        wiki.add_module("billing", "BillingService")
        wiki.add_adr("adr_stripe", "Stripe API v2024", "ACCEPTED", "Handle idempotent webhooks", ["billing"])

        injector = SemanticContextInjector(vector_store=store, wiki_graph=wiki, semantic_dim=16)
        task = TaskNode(
            task_id="TASK-55",
            title="Implement Stripe Webhook Handler",
            estimated_duration=6.0,
            impacted_modules=["billing"],
        )

        prompt = injector.generate_agent_prompt_context("developer_agent_0", task)
        assert "developer_agent_0" in prompt
        assert "TASK-55" in prompt
        assert "Stripe API v2024" in prompt
        assert "Execution Directives" in prompt
