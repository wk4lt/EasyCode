"""Tests for rag_store.py — RAG store backends."""

from unittest.mock import MagicMock, patch

import pytest

from liteagent.core.rag_store import InMemoryRAGStore, create_rag_store


class TestInMemoryRAGStore:
    @pytest.fixture
    def store(self):
        return InMemoryRAGStore()

    def test_add_example_and_count(self, store):
        doc_id = store.add_example(
            design_content="# Design\nA test module.",
            code_content="def hello():\n    return 'world'",
            metadata={"source": "test"},
        )
        assert doc_id
        assert store.count() == 1

    def test_add_example_idempotent(self, store):
        store.add_example("design", "code", doc_id="fixed-1")
        assert store.count() == 1
        store.add_example("design-updated", "code-updated", doc_id="fixed-1")
        assert store.count() == 1

    def test_query_similar_on_empty_store(self, store):
        results = store.query_similar("query")
        assert results == []

    def test_query_similar_returns_results(self, store):
        store.add_example("Python web scraping with requests", "import requests\n...")
        store.add_example("Java database connection with JDBC", "import java.sql\n...")

        results = store.query_similar("How to do web scraping in Python", n_results=2)
        assert len(results) >= 1
        for r in results:
            assert "id" in r
            assert "document" in r
            assert "metadata" in r
            assert "distance" in r

    def test_query_ranks_relevant_higher(self, store):
        store.add_example("Python function to add numbers", "def add(a, b): return a+b")
        store.add_example("Java class for database", "public class DB {}")

        results = store.query_similar("python math addition", n_results=2)
        assert len(results) >= 1
        first_doc = results[0]["document"]
        assert "add" in first_doc.lower()

    def test_list_ids(self, store):
        id1 = store.add_example("d1", "c1")
        id2 = store.add_example("d2", "c2")
        ids = store.list_ids()
        assert len(ids) == 2
        assert id1 in ids
        assert id2 in ids

    def test_clear(self, store):
        store.add_example("d1", "c1")
        assert store.count() == 1
        store.clear()
        assert store.count() == 0

    def test_collection_name(self, store):
        assert store.collection_name == "code_examples"

    def test_collection_name_custom(self):
        store = InMemoryRAGStore(collection_name="my_coll")
        assert store.collection_name == "my_coll"


class TestCreateRAGStore:
    def test_factory_creates_inmemory_when_chromadb_missing(self):
        import builtins as _bi
        orig = _bi.__import__

        def _fail_import(name, *a, **kw):
            if name == "chromadb":
                raise ImportError("No chromadb")
            return orig(name, *a, **kw)

        with patch("builtins.__import__", _fail_import):
            store = create_rag_store()
        assert isinstance(store, InMemoryRAGStore)

    def test_factory_creates_chroma_when_available(self):
        store = create_rag_store(api_key="sk-test")
        from liteagent.core.rag_store import ChromaRAGStore, InMemoryRAGStore
        assert isinstance(store, (ChromaRAGStore, InMemoryRAGStore))
