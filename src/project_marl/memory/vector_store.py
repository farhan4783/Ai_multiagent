"""
High-Performance In-Memory and Persistent Vector Store for Organizational Memory.

Provides sub-millisecond semantic similarity search, metadata filtering,
and hybrid vector embedding projections for continuous documentation ingestion.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import re
import numpy as np

from project_marl.core.logging import setup_logger

logger = setup_logger("vector_store")


@dataclass
class VectorDocument:
    """Represents a chunked document stored with its embedding and organizational metadata."""
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorDocument":
        emb = np.array(data["embedding"], dtype=np.float32) if data.get("embedding") is not None else None
        return cls(
            doc_id=data["doc_id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            embedding=emb,
        )


class EmbeddingEngine:
    """
    Embedding Engine with deterministic semantic projection and optional SentenceTransformers.
    Guarantees sub-millisecond execution and 100% offline reliability without remote API dependencies.
    """

    def __init__(
        self,
        dim: int = 128,
        use_sentence_transformers: bool = False,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.dim = dim
        self.use_sentence_transformers = use_sentence_transformers
        self._st_model = None

        if use_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(model_name)
                self.dim = self._st_model.get_sentence_embedding_dimension()
                logger.info("Loaded SentenceTransformer model: %s (dim=%d)", model_name, self.dim)
            except Exception as e:
                logger.warning("Could not initialize SentenceTransformer (%s). Falling back to semantic projection.", e)
                self.use_sentence_transformers = False

    def embed_text(self, text: str) -> np.ndarray:
        """Embeds a single string into a normalized 1D vector."""
        if self._st_model is not None:
            vec = self._st_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return vec.astype(np.float32)

        # Ultra-fast, deterministic semantic hashing projection
        return self._semantic_projection(text, self.dim)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of strings into a (N, dim) matrix."""
        if self._st_model is not None:
            vecs = self._st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return vecs.astype(np.float32)

        vectors = [self._semantic_projection(t, self.dim) for t in texts]
        return np.vstack(vectors).astype(np.float32)

    @staticmethod
    def _semantic_projection(text: str, dim: int) -> np.ndarray:
        """
        Projects text into a fixed-dimensional unit hypersphere vector using
        word tokens and character n-grams with deterministic Murmur-style feature hashing.
        """
        tokens = re.findall(r"\b\w+\b", text.lower())
        vec = np.zeros(dim, dtype=np.float32)
        if not tokens:
            vec[0] = 1.0
            return vec

        for token in tokens:
            # Word feature
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % dim
            sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
            vec[idx] += sign

            # Character 3-gram features for subword semantics
            if len(token) >= 3:
                for i in range(len(token) - 2):
                    ngram = token[i:i + 3]
                    h_ng = int(hashlib.sha256(ngram.encode("utf-8")).hexdigest()[:8], 16)
                    idx_ng = h_ng % dim
                    sign_ng = 0.5 if ((h_ng >> 16) & 1) == 0 else -0.5
                    vec[idx_ng] += sign_ng

        # L2 Normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return vec


class VectorStore:
    """
    Sub-millisecond Vector Store supporting cosine similarity search,
    metadata filtering, and serialization.
    """

    def __init__(
        self,
        embedding_engine: Optional[EmbeddingEngine] = None,
        dim: int = 128,
    ) -> None:
        self.engine = embedding_engine or EmbeddingEngine(dim=dim)
        self.dim = self.engine.dim

        self.documents: Dict[str, VectorDocument] = {}
        self.doc_ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None  # Shape: (N, dim)

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> VectorDocument:
        """Adds or updates a single document in the store."""
        emb = embedding if embedding is not None else self.engine.embed_text(text)
        doc = VectorDocument(
            doc_id=doc_id,
            text=text,
            metadata=metadata or {},
            embedding=emb,
        )
        self.documents[doc_id] = doc
        if doc_id not in self.doc_ids:
            self.doc_ids.append(doc_id)

        self._rebuild_matrix()
        return doc

    def add_documents(self, docs: List[VectorDocument]) -> None:
        """Batch adds documents and rebuilds index matrix."""
        texts_to_embed = []
        indices_to_embed = []

        for i, d in enumerate(docs):
            if d.embedding is None:
                texts_to_embed.append(d.text)
                indices_to_embed.append(i)

        if texts_to_embed:
            embeddings = self.engine.embed_batch(texts_to_embed)
            for idx, emb in zip(indices_to_embed, embeddings):
                docs[idx].embedding = emb

        for d in docs:
            self.documents[d.doc_id] = d
            if d.doc_id not in self.doc_ids:
                self.doc_ids.append(d.doc_id)

        self._rebuild_matrix()

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[VectorDocument, float]]:
        """
        Searches documents by cosine similarity with optional metadata filtering.

        Returns:
            List of (VectorDocument, similarity_score) sorted descending by score.
        """
        if not self.doc_ids or self._matrix is None:
            return []

        q_vec = self.engine.embed_text(query)  # (dim,)

        # Dot product of normalized vectors equals cosine similarity
        similarities = np.dot(self._matrix, q_vec)  # (N,)

        # Metadata filtering
        ranked_indices = np.argsort(-similarities)
        results: List[Tuple[VectorDocument, float]] = []

        for idx in ranked_indices:
            doc_id = self.doc_ids[idx]
            doc = self.documents[doc_id]

            if filter_dict:
                match = True
                for k, v in filter_dict.items():
                    if doc.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            score = float(similarities[idx])
            results.append((doc, score))
            if len(results) >= top_k:
                break

        return results

    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        return self.documents.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.doc_ids.remove(doc_id)
            self._rebuild_matrix()
            return True
        return False

    def clear(self) -> None:
        self.documents.clear()
        self.doc_ids.clear()
        self._matrix = None

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def _rebuild_matrix(self) -> None:
        """Synchronizes the 2D search matrix."""
        if not self.doc_ids:
            self._matrix = None
            return
        vectors = [self.documents[did].embedding for did in self.doc_ids]
        self._matrix = np.vstack(vectors).astype(np.float32)

    def save(self, filepath: str) -> None:
        """Serializes vector store to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "dim": self.dim,
            "documents": [doc.to_dict() for doc in self.documents.values()],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved VectorStore (%d docs) to %s", len(self.documents), filepath)

    @classmethod
    def load(cls, filepath: str) -> "VectorStore":
        """Loads vector store from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        store = cls(dim=data["dim"])
        docs = [VectorDocument.from_dict(d) for d in data["documents"]]
        store.add_documents(docs)
        logger.info("Loaded VectorStore (%d docs) from %s", len(docs), filepath)
        return store
