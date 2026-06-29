"""RAG Store for LiteAgent framework.

Provides a vector store for indexing code examples (design-document +
implementation pairs) and retrieving similar examples during code
generation.

Supports two backends:
  - ChromaDB with local embeddings (default, requires `chromadb` package)
  - InMemory with JSON file persistence (fallback)

Layer: Core infrastructure.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

DEFAULT_COLLECTION = "code_examples"


def create_rag_store(
    persist_dir: str = "./chroma_data",
    api_key: str = "",
    embedding_model: str = "all-MiniLM-L6-v2",
    embedding_provider: str = "local",
    collection_name: str = DEFAULT_COLLECTION,
    base_url: str = "",
) -> "BaseRAGStore":
    """Factory: create the appropriate RAG store backend.

    Tries ChromaRAGStore with local embeddings first; falls back
    to InMemoryRAGStore if chromadb is not installed.

    Args:
        persist_dir: Directory for persistent ChromaDB storage.
        api_key: API key for OpenAI embeddings (only when embedding_provider="openai").
        embedding_model: Model name. For local: all-MiniLM-L6-v2 (ONNX). For openai: text-embedding-3-small.
        embedding_provider: "local" (default) or "openai".
        collection_name: Collection name.
        base_url: Custom base URL for OpenAI-compatible embeddings API.

    Returns:
        A BaseRAGStore implementation.
    """
    try:
        return ChromaRAGStore(
            persist_dir=persist_dir,
            api_key=api_key,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            collection_name=collection_name,
            base_url=base_url,
        )
    except ImportError:
        _log.warning("chromadb_not_available", extra={
            "layer": "core",
            "detail": "ChromaDB not available; using in-memory RAG store with JSON persistence.",
        })
        persist_path = os.path.join(persist_dir, f"{collection_name}.json")
        return InMemoryRAGStore(collection_name=collection_name, persist_path=persist_path)
    except Exception as e:
        _log.warning("chromadb_init_failed", extra={
            "layer": "core",
            "detail": str(e),
        })
        persist_path = os.path.join(persist_dir, f"{collection_name}.json")
        return InMemoryRAGStore(collection_name=collection_name, persist_path=persist_path)


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
    """In-memory RAG store with JSON file persistence.

    Uses simple TF-IDF + cosine similarity (via scikit-learn / numpy)
    when available, otherwise falls back to substring matching.

    Documents auto-load from disk on init and auto-save on every add.
    """

    def __init__(self, collection_name: str = DEFAULT_COLLECTION, persist_path: str = ""):
        """Initialize the in-memory store.

        Args:
            collection_name: Collection name (for interface compatibility).
            persist_path: Path to the JSON file for persistence.
                If the file exists, documents are loaded on init.
        """
        self._collection_name = collection_name
        self._persist_path = persist_path
        self._docs: dict[str, dict] = {}

        if self._persist_path:
            self._load()

    def _load(self) -> None:
        """Load documents from the persistent JSON file."""
        path = Path(self._persist_path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._docs = data.get("docs", {})
            _log.info("rag_load", extra={
                "layer": "core",
                "path": self._persist_path,
                "count": len(self._docs),
            })
        except Exception as e:
            _log.warning("rag_load_failed", extra={
                "layer": "core",
                "path": self._persist_path,
                "error": str(e),
            })

    def _save(self) -> None:
        """Save documents to the persistent JSON file."""
        if not self._persist_path:
            return
        try:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {"docs": self._docs, "collection": self._collection_name}
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            _log.debug("rag_save", extra={
                "layer": "core",
                "path": self._persist_path,
                "count": len(self._docs),
            })
        except Exception as e:
            _log.warning("rag_save_failed", extra={
                "layer": "core",
                "path": self._persist_path,
                "error": str(e),
            })

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
        self._save()
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
        if self._persist_path:
            pp = Path(self._persist_path)
            if pp.exists():
                pp.unlink()

    def list_ids(self) -> list[str]:
        return list(self._docs.keys())


class ChromaRAGStore(BaseRAGStore):
    """ChromaDB-backed persistent vector store.

    Uses ChromaDB's built-in embedding functions:
      - Local: SentenceTransformer (all-MiniLM-L6-v2), no API key needed.
      - OpenAI: text-embedding-3-small, requires OPENAI_API_KEY.

    Embedding is delegated to ChromaDB internally — add/query only
    pass documents and query texts, not raw vectors.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_data",
        api_key: str = "",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "local",
        collection_name: str = DEFAULT_COLLECTION,
        base_url: str = "",
    ):
        """Initialize the ChromaDB-backed RAG store.

        Args:
            persist_dir: Directory for persistent ChromaDB storage.
            api_key: API key for OpenAI embeddings.
            embedding_model: Model name for the embedding function.
            embedding_provider: "local" or "openai".
            collection_name: ChromaDB collection name.
            base_url: Custom base URL for OpenAI embeddings API.
        """
        import chromadb
        from chromadb.config import Settings

        self._collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model
        self._api_key = api_key
        self._base_url = base_url

        os.makedirs(persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self._ef = self._create_embedding_function()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
            metadata={"description": "Code examples: design doc + implementation pairs"},
        )
        _log.info("chromadb_init", extra={
            "layer": "core",
            "provider": embedding_provider,
            "model": embedding_model,
            "path": persist_dir,
        })

    def _create_embedding_function(self):
        """Create the appropriate ChromaDB embedding function."""
        if self._embedding_provider == "openai" and self._api_key:
            api_key = self._api_key or os.environ.get("OPENAI_API_KEY", "")
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            kwargs = {"api_key": api_key, "model_name": self._embedding_model}
            if self._base_url:
                kwargs["api_base"] = self._base_url
            return OpenAIEmbeddingFunction(**kwargs)
        else:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
            _log.info("chromadb_local_embedding", extra={
                "layer": "core",
                "model": "all-MiniLM-L6-v2 (ONNX)",
                "detail": "Using local ONNX embeddings, no API key needed.",
            })
            return ONNXMiniLM_L6_V2()

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

        existing = self._collection.get(ids=[doc_id])
        if existing and existing["ids"]:
            self._collection.update(
                ids=[doc_id],
                documents=[combined],
                metadatas=[meta],
            )
            _log.debug("rag_update", extra={"layer": "core", "doc_id": doc_id})
        else:
            self._collection.add(
                ids=[doc_id],
                documents=[combined],
                metadatas=[meta],
            )
            _log.debug("rag_add", extra={"layer": "core", "doc_id": doc_id})

        return doc_id

    def query_similar(self, query_text: str, n_results: int = 5) -> list[dict]:
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query_text],
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
            embedding_function=self._ef,
            metadata={"description": "Code examples: design doc + implementation pairs"},
        )

    def list_ids(self) -> list[str]:
        result = self._collection.get()
        return result.get("ids", [])

    @property
    def collection_name(self) -> str:
        return self._collection_name
