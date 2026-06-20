"""Code retriever skill implementation.

Queries the RAG vector database to find similar code examples based
on a design document or requirement query.

Layer: Skill layer (first layer).
"""

import json
from typing import Optional

from liteagent.core.base_skill import BaseSkill


class CodeRetrieverImpl(BaseSkill):
    """Query the RAG store for similar code examples."""

    def __init__(self, contract_path: Optional[str] = None, rag_store=None):
        """Initialize with an optional external RAGStore reference.

        Args:
            contract_path: Path to the .md contract file.
            rag_store: A RAGStore instance. If None, will be injected later.
        """
        super().__init__(contract_path)
        self._rag_store = rag_store

    def set_rag_store(self, rag_store) -> None:
        """Inject a RAGStore instance.

        Args:
            rag_store: A RAGStore instance for querying.
        """
        self._rag_store = rag_store

    def execute(self, query_text: str, n_results: int = 5) -> dict:
        """Query RAG for similar code examples.

        Args:
            query_text: The search query (design document or requirement).
            n_results: Number of results to return (1-10).

        Returns:
            dict with 'status' and 'results' list or 'error'.
        """
        try:
            if self._rag_store is None:
                return {"status": "error", "error": "RAG store not initialized. Call set_rag_store() first."}

            n_results = max(min(n_results, 10), 1)

            if len(query_text) > 50000:
                return {"status": "error", "error": f"Query too long ({len(query_text)} chars). Max: 50000."}

            total = self._rag_store.count()
            if total == 0:
                return {
                    "status": "ok",
                    "results": [],
                    "total_docs": 0,
                    "note": "RAG store is empty. Index some examples first.",
                }

            results = self._rag_store.query_similar(query_text, n_results=n_results)

            return {
                "status": "ok",
                "results": results,
                "total_docs": total,
                "retrieved": len(results),
            }

        except Exception as e:
            return {"status": "error", "error": f"Failed to retrieve examples: {e}"}
