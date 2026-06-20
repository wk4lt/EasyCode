"""Tests for codegen skills — file_reader, code_embedder, code_retriever,
code_generator, code_tester."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


CONTRACT_DIR = Path(__file__).parent.parent / "examples" / "domain_skills" / "codegen_tools"


def _skill_path(name):
    return str(CONTRACT_DIR / f"SKILL_{name}.md")


class TestFileReader:
    @pytest.fixture
    def reader(self):
        from examples.domain_skills.codegen_tools.file_reader_impl import FileReaderImpl
        return FileReaderImpl(contract_path=_skill_path("file_reader"))

    def test_reads_existing_file(self, reader):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello')")
            path = f.name

        result = reader.execute(file_path=path)
        assert result["status"] == "ok"
        assert "print('hello')" in result["content"]
        assert result["file_name"] == Path(path).name

    def test_file_not_found(self, reader):
        result = reader.execute(file_path="/nonexistent/file.py")
        assert result["status"] == "error"
        assert "not found" in result["error"]

    def test_unsupported_extension(self, reader):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bin", delete=False) as f:
            f.write("data")
            path = f.name

        result = reader.execute(file_path=path)
        assert result["status"] == "error"
        assert "Unsupported file type" in result["error"]


class TestCodeEmbedder:
    @pytest.fixture
    def rag_store(self):
        return MagicMock()

    @pytest.fixture
    def embedder(self, rag_store):
        from examples.domain_skills.codegen_tools.code_embedder_impl import CodeEmbedderImpl
        impl = CodeEmbedderImpl(contract_path=_skill_path("code_embedder"))
        impl.set_rag_store(rag_store)
        return impl

    def test_embeds_and_stores(self, embedder, rag_store):
        rag_store.add_example.return_value = "doc-123"
        rag_store.count.return_value = 1

        result = embedder.execute(
            design_content="# Design",
            code_content="def foo(): pass",
        )
        assert result["status"] == "ok"
        assert result["doc_id"] == "doc-123"
        assert result["total_docs"] == 1
        rag_store.add_example.assert_called_once()

    def test_rejects_oversized_design(self, embedder):
        result = embedder.execute(
            design_content="x" * 50001,
            code_content="code",
        )
        assert result["status"] == "error"
        assert "too long" in result["error"]

    def test_no_rag_store(self):
        from examples.domain_skills.codegen_tools.code_embedder_impl import CodeEmbedderImpl
        impl = CodeEmbedderImpl(contract_path=_skill_path("code_embedder"))
        result = impl.execute(design_content="d", code_content="c")
        assert result["status"] == "error"
        assert "not initialized" in result["error"]


class TestCodeRetriever:
    @pytest.fixture
    def rag_store(self):
        store = MagicMock()
        store.count.return_value = 2
        store.query_similar.return_value = [
            {"id": "1", "document": "design+code1", "metadata": {}, "distance": 0.1},
            {"id": "2", "document": "design+code2", "metadata": {}, "distance": 0.3},
        ]
        return store

    @pytest.fixture
    def retriever(self, rag_store):
        from examples.domain_skills.codegen_tools.code_retriever_impl import CodeRetrieverImpl
        impl = CodeRetrieverImpl(contract_path=_skill_path("code_retriever"))
        impl.set_rag_store(rag_store)
        return impl

    def test_retrieves_results(self, retriever, rag_store):
        result = retriever.execute(query_text="how to parse JSON")
        assert result["status"] == "ok"
        assert result["total_docs"] == 2
        assert result["retrieved"] == 2
        assert len(result["results"]) == 2

    def test_empty_store(self):
        store = MagicMock()
        store.count.return_value = 0

        from examples.domain_skills.codegen_tools.code_retriever_impl import CodeRetrieverImpl
        impl = CodeRetrieverImpl(contract_path=_skill_path("code_retriever"))
        impl.set_rag_store(store)
        result = impl.execute(query_text="test")
        assert result["status"] == "ok"
        assert result["results"] == []
        assert "empty" in result.get("note", "")

    def test_clamps_n_results(self, retriever):
        result = retriever.execute(query_text="test", n_results=20)
        assert result["retrieved"] <= 10


class TestCodeGenerator:
    @pytest.fixture
    def mock_llm(self):
        from liteagent.core.llm_interface import ChatResponse

        llm = MagicMock()
        llm.chat_completion.return_value = ChatResponse(
            content="def generate():\n    return 'code'",
            token_usage={"total_tokens": 50},
        )
        return llm

    @pytest.fixture
    def generator(self, mock_llm):
        from examples.domain_skills.codegen_tools.code_generator_impl import CodeGeneratorImpl
        impl = CodeGeneratorImpl(contract_path=_skill_path("code_generator"))
        impl.set_llm(mock_llm)
        return impl

    def test_generates_code(self, generator, mock_llm):
        result = generator.execute(
            design_content="# Design: echo tool",
            test_content="def test_echo(): pass",
        )
        assert result["status"] == "ok"
        assert "def generate" in result["generated_code"]
        assert result["token_usage"]["total_tokens"] == 50

    def test_generates_fix_attempt(self, generator, mock_llm):
        result = generator.execute(
            design_content="# Design",
            test_content="tests",
            previous_attempt="def broken(): pass",
            test_error="NameError: name 'x' is not defined",
        )
        assert result["status"] == "ok"
        mock_llm.chat_completion.assert_called_once()

    def test_no_llm(self):
        from examples.domain_skills.codegen_tools.code_generator_impl import CodeGeneratorImpl
        impl = CodeGeneratorImpl(contract_path=_skill_path("code_generator"))
        result = impl.execute(design_content="d", test_content="t")
        assert result["status"] == "error"

    def test_strips_code_fences(self, generator, mock_llm):
        mock_llm.chat_completion.return_value = type('obj', (object,), {
            'content': '```python\ndef hello():\n    return "hi"\n```',
            'token_usage': {'total_tokens': 10},
        })()
        result = generator.execute(design_content="d", test_content="t")
        assert 'def hello():' in result["generated_code"]
        assert '```' not in result["generated_code"]


class TestCodeTester:
    @pytest.fixture
    def tester(self):
        from examples.domain_skills.codegen_tools.code_tester_impl import CodeTesterImpl
        return CodeTesterImpl(contract_path=_skill_path("code_tester"))

    def test_missing_impl_file(self, tester):
        result = tester.execute(
            impl_file_path="/nonexistent/impl.py",
            test_file_path="/nonexistent/test.py",
        )
        assert result["status"] == "error"

    def test_runs_passing_tests(self, tester):
        with tempfile.TemporaryDirectory() as tmpdir:
            impl_path = Path(tmpdir) / "math_utils.py"
            impl_path.write_text("def add(a, b):\n    return a + b\n")

            test_path = Path(tmpdir) / "test_math_utils.py"
            test_path.write_text(
                "import sys\nsys.path.insert(0, '.')\n"
                "from math_utils import add\n\n"
                "def test_add():\n    assert add(2, 3) == 5\n"
            )

            result = tester.execute(
                impl_file_path=str(impl_path),
                test_file_path=str(test_path),
            )
            assert result["status"] == "ok"
            assert result["passed"] is True

    def test_runs_failing_tests(self, tester):
        with tempfile.TemporaryDirectory() as tmpdir:
            impl_path = Path(tmpdir) / "broken.py"
            impl_path.write_text("def broken():\n    return 1\n")

            test_path = Path(tmpdir) / "test_broken.py"
            test_path.write_text(
                "import sys\nsys.path.insert(0, '.')\n"
                "from broken import broken\n\n"
                "def test_broken():\n    assert broken() == 2\n"
            )

            result = tester.execute(
                impl_file_path=str(impl_path),
                test_file_path=str(test_path),
            )
            assert result["status"] == "ok"
            assert result["passed"] is False
