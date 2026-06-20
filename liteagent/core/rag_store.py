"""RAG Store for LiteAgent framework.

Provides a vector store for indexing code examples (design-document +
implementation pairs) and retrieving similar examples during code
generation.

Supports two backends:
  - ChromaDB (persistent, requires `chromadb` package)
  - InMemory (ephemeral, for testing, always available)

Layer: Core infrastructure.
"""

import logging
import os
import uuid
from typing import Optional

_log = logging.getLogger(__name__)

DEFAULT_COLLECTION = "code_examples"


def create_rag_store(
    persist_dir: str = "./chroma_data",
    api_key: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    collection_name: str = DEFAULT_COLLECTION,
) -> "BaseRAGStore":
    """Factory: create the appropriate RAG store backend.

    Tries ChromaDB first; falls back to InMemory if chromadb is not installed.

    Args:
        persist_dir: Directory for persistent storage (ChromaDB only).
        api_key: OpenAI API key for embeddings. Falls back to env var.
        embedding_model: OpenAI embedding model name.
        collection_name: Collection/table name.

    Returns:
        A BaseRAGStore implementation.
    """
    try:
        import chromadb
        return ChromaRAGStore(
            persist_dir=persist_dir,
            api_key=api_key,
            embedding_model=embedding_model,
            collection_name=collection_name,
        )
    except ImportError:
        _log.warning("chromadb_not_installed", extra={
            "layer": "core",
            "detail": "ChromaDB not available; using in-memory RAG store.",
        })
        return InMemoryRAGStore(collection_name=collection_name)


class BaseRAGStore:
    """Abstract interface for RAG vector stores."""

    def add_example(
        self,
        design_content: str,
        code_content: str,
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Index a design-doc + implementation code pair."""
        raise NotImplementedError

    def query_similar(self, query_text: str, n_results: int = 5) -> list[dict]:
        """Retrieve the most similar examples for a query."""
        raise NotImplementedError

    def count(self) -> int:
        """Return the number of indexed examples."""
        raise NotImplementedError

    def clear(self) -> None:
        """Remove all entries from the store."""
        raise NotImplementedError

    def list_ids(self) -> list[str]:
        """List all document IDs in the store."""
        raise NotImplementedError

    @property
    def collection_name(self) -> str:
        """The collection name."""
        raise NotImplementedError


class InMemoryRAGStore(BaseRAGStore):
    """In-memory RAG store for testing and standalone usage.

    Uses simple TF-IDF + cosine similarity (via scikit-learn / numpy)
    when available, otherwise falls back to substring matching.
    """

    def __init__(self, collection_name: str = DEFAULT_COLLECTION):
        """Initialize the in-memory store.

        Args:
            collection_name: Collection name (for interface compatibility).
        """
        self._collection_name = collection_name
        self._docs: dict[str, dict] = {}

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def add_example(
        self,
        design_content: str,
        code_content: str,
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        combined = f"DESIGN DOCUMENT:\n{design_content}\n\nIMPLEMENTATION CODE:\n{code_content}"
        meta = dict(metadata or {})
        meta["has_code"] = bool(code_content.strip())

        doc_id = doc_id or str(uuid.uuid4())

        self._docs[doc_id] = {
            "document": combined,
            "metadata": meta,
            "design": design_content,
            "code": code_content,
        }
        return doc_id

    def query_similar(self, query_text: str, n_results: int = 5) -> list[dict]:
        if not self._docs:
            return []

        results = self._rank_by_similarity(query_text)
        return results[:n_results]

    def _rank_by_similarity(self, query_text: str) -> list[dict]:
        """Rank documents by similarity to query.

        Tries to use sklearn TfidfVectorizer; falls back to simple
        word-overlap scoring.
        """
        query_lower = query_text.lower()
        scored = []

        for doc_id, entry in self._docs.items():
            doc_text = entry["document"].lower()

            try:
                import numpy as np
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity

                if not hasattr(self, "_tfidf"):
                    self._tfidf_list: list[str] = []
                    self._tfidf_ids: list[str] = []

                if doc_id not in self._tfidf_ids:
                    self._tfidf_ids.append(doc_id)
                    self._tfidf_list.append(doc_text)

                if not self._tfidf_list:
                    continue

                vectorizer = TfidfVectorizer()
                all_texts = self._tfidf_list + [query_lower]
                tfidf_matrix = vectorizer.fit_transform(all_texts)
                similarity = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]

                idx = self._tfidf_ids.index(doc_id)
                score = float(similarity[idx])
            except Exception:
                query_words = set(query_lower.split())
                doc_words = set(doc_text.split())
                overlap = len(query_words & doc_words)
                total = len(query_words | doc_words)
                score = overlap / max(total, 1)

            scored.append({
                "id": doc_id,
                "document": entry["document"],
                "metadata": entry["metadata"],
                "distance": 1.0 - score,
            })

        scored.sort(key=lambda x: x["distance"])
        return scored

    def count(self) -> int:
        return len(self._docs)

    def clear(self) -> None:
        self._docs.clear()
        if hasattr(self, "_tfidf_ids"):
            self._tfidf_ids.clear()
            self._tfidf_list.clear()

    def list_ids(self) -> list[str]:
        return list(self._docs.keys())


class ChromaRAGStore(BaseRAGStore):
    """ChromaDB-backed persistent vector store.

    Uses OpenAI embeddings for semantic similarity search.
    Requires the `chromadb` package.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_data",
        api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = DEFAULT_COLLECTION,
    ):
        """Initialize the ChromaDB-backed RAG store.

        Args:
            persist_dir: Directory for persistent ChromaDB storage.
            api_key: OpenAI API key for embeddings.
            embedding_model: OpenAI embedding model name.
            collection_name: ChromaDB collection name.
        """
        import chromadb
        from chromadb.config import Settings

        self._persist_dir = persist_dir
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._embedding_model = embedding_model
        self._collection_name = collection_name

        os.makedirs(persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self._openai = None
        if self._api_key:
            from openai import OpenAI
            self._openai = OpenAI(api_key=self._api_key)
        self._embed_fn = self._embed_remote if self._openai else self._embed_dummy

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Code examples: design doc + implementation pairs"},
        )

    def _embed_remote(self, text: str) -> list[float]:
        resp = self._openai.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        return resp.data[0].embedding

    @staticmethod
    def _embed_dummy(text: str) -> list[float]:
        return [0.0] * 128

    def add_example(
        self,
        design_content: str,
        code_content: str,
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        combined = f"DESIGN DOCUMENT:\n{design_content}\n\nIMPLEMENTATION CODE:\n{code_content}"
        meta = dict(metadata or {})
        meta["has_code"] = bool(code_content.strip())

        doc_id = doc_id or str(uuid.uuid4())

        embedding = self._embed_fn(combined)

        existing = self._collection.get(ids=[doc_id])
        if existing and existing["ids"]:
            self._collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[meta],
                documents=[combined],
            )
            _log.debug("rag_update", extra={"layer": "core", "doc_id": doc_id})
        else:
            self._collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[meta],
                documents=[combined],
            )
            _log.debug("rag_add", extra={"layer": "core", "doc_id": doc_id})

        return doc_id

    def query_similar(self, query_text: str, n_results: int = 5) -> list[dict]:
        if self._collection.count() == 0:
            return []

        embedding = self._embed_fn(query_text)

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, self._collection.count()),
        )

        output = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            output.append({
                "id": doc_id,
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else 0.0,
            })

        return output

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"description": "Code examples: design doc + implementation pairs"},
        )

    def list_ids(self) -> list[str]:
        result = self._collection.get()
        return result.get("ids", [])

    @property
    def collection_name(self) -> str:
        return self._collection_name
