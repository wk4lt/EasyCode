"""Code embedder skill implementation.

Embeds design document + implementation code pairs and stores them in the
RAG vector database (ChromaDB).

Layer: Skill layer (first layer).
"""

import json
from typing import Optional

from liteagent.core.base_skill import BaseSkill


class CodeEmbedderImpl(BaseSkill):
    """Embed code examples and store in the RAG vector store."""

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
            rag_store: A RAGStore instance for embedding and storage.
        """
        self._rag_store = rag_store

    def execute(
        self,
        design_content: str,
        code_content: str,
        metadata_json: str = "{}",
        doc_id: str = "",
    ) -> dict:
        """Embed a design+code pair and store in RAG.

        Args:
            design_content: The design document content.
            code_content: The implementation code content.
            metadata_json: JSON string with extra metadata.
            doc_id: Optional explicit document ID.

        Returns:
            dict with 'status' and 'doc_id' or 'error'.
        """
        try:
            if self._rag_store is None:
                return {"status": "error", "error": "RAG store not initialized. Call set_rag_store() first."}

            if len(design_content) > 50000:
                return {"status": "error", "error": f"Design content too long ({len(design_content)} chars). Max: 50000."}

            if len(code_content) > 100000:
                return {"status": "error", "error": f"Code content too long ({len(code_content)} chars). Max: 100000."}

            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
            except json.JSONDecodeError:
                metadata = {}

            store_id = self._rag_store.add_example(
                design_content=design_content,
                code_content=code_content,
                metadata=metadata,
                doc_id=doc_id if doc_id else None,
            )

            return {
                "status": "ok",
                "doc_id": store_id,
                "total_docs": self._rag_store.count(),
            }

        except Exception as e:
            return {"status": "error", "error": f"Failed to embed code example: {e}"}
