"""Tests for rag_store.py — RAG store backends."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from liteagent.core.rag_store import InMemoryRAGStore, ChromaRAGStore, create_rag_store


class TestInMemoryRAGStore:
    @pytest.fixture
    def store(self):
        return InMemoryRAGStore()

    @pytest.fixture
    def tmp_persist_path(self, tmp_path):
        return str(tmp_path / "test_store.json")

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

    def test_persist_save_and_load(self, tmp_persist_path):
        store = InMemoryRAGStore(persist_path=tmp_persist_path)
        store.add_example("Design A", "Code A", doc_id="id-1")
        store.add_example("Design B", "Code B", doc_id="id-2")
        assert store.count() == 2
        assert Path(tmp_persist_path).exists()

        store2 = InMemoryRAGStore(persist_path=tmp_persist_path)
        assert store2.count() == 2
        assert "id-1" in store2.list_ids()
        assert "id-2" in store2.list_ids()
        assert store2._docs["id-1"]["design"] == "Design A"

    def test_persist_clear_removes_file(self, tmp_persist_path):
        store = InMemoryRAGStore(persist_path=tmp_persist_path)
        store.add_example("Design", "Code")
        assert Path(tmp_persist_path).exists()
        store.clear()
        assert store.count() == 0
        assert not Path(tmp_persist_path).exists()

    def test_persist_loads_empty_when_no_file(self, tmp_persist_path):
        store = InMemoryRAGStore(persist_path=tmp_persist_path)
        assert store.count() == 0

    def test_no_persist_when_path_empty(self):
        store = InMemoryRAGStore()
        store.add_example("Design", "Code")
        assert store.count() == 1


class TestChromaRAGStore:
    @pytest.fixture
    def chroma_store(self, tmp_path):
        store = ChromaRAGStore(
            persist_dir=str(tmp_path / "chroma_test"),
            collection_name="test_coll",
            embedding_provider="local",
            embedding_model="all-MiniLM-L6-v2",
        )
        yield store
        store.clear()

    def test_add_and_query(self, chroma_store):
        chroma_store.add_example(
            design_content="Python web scraping with requests library",
            code_content="import requests\nresponse = requests.get(url)",
            doc_id="scrape-1",
        )
        chroma_store.add_example(
            design_content="Java database connection using JDBC",
            code_content="import java.sql.Connection;\nDriverManager.getConnection(url)",
            doc_id="db-1",
        )
        assert chroma_store.count() == 2

        results = chroma_store.query_similar("How to scrape websites", n_results=2)
        assert len(results) >= 1
        first = results[0]
        assert "id" in first
        assert "document" in first
        assert "metadata" in first
        assert "distance" in first

    def test_query_empty_store(self, chroma_store):
        assert chroma_store.query_similar("anything") == []

    def test_list_ids(self, chroma_store):
        chroma_store.add_example("d1", "c1", doc_id="id-1")
        chroma_store.add_example("d2", "c2", doc_id="id-2")
        ids = chroma_store.list_ids()
        assert sorted(ids) == ["id-1", "id-2"]

    def test_clear(self, chroma_store):
        chroma_store.add_example("d1", "c1")
        assert chroma_store.count() == 1
        chroma_store.clear()
        assert chroma_store.count() == 0

    def test_add_example_idempotent(self, chroma_store):
        chroma_store.add_example("d1", "c1", doc_id="fixed")
        assert chroma_store.count() == 1
        chroma_store.add_example("d1-updated", "c1-updated", doc_id="fixed")
        assert chroma_store.count() == 1

    def test_collection_name(self, chroma_store):
        assert chroma_store.collection_name == "test_coll"


class TestCreateRAGStore:
    def test_factory_creates_chromadb_when_available(self, tmp_path):
        store = create_rag_store(
            persist_dir=str(tmp_path / "rag"),
            collection_name="factory_test",
            embedding_provider="local",
        )
        assert isinstance(store, ChromaRAGStore)
        store.clear()

    def test_factory_fallback_to_inmemory(self):
        def _fail_import(name, *a, **kw):
            if name == "chromadb":
                raise ImportError("No chromadb")
            return __builtins__["__import__"](name, *a, **kw)

        with patch("builtins.__import__", side_effect=_fail_import):
            store = create_rag_store()
        assert isinstance(store, InMemoryRAGStore)
