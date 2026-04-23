"""
RetailMind Unit Tests
=====================
Tests core logic WITHOUT making real API calls.
Uses mocking for Gemini, Groq, ChromaDB, and Tavily.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile, os


# ─────────────────────────────────────────────────────
# RAG Pipeline Tests
# ─────────────────────────────────────────────────────
class TestDocumentIngester:

    @patch("rag_service.rag_pipeline.genai")
    @patch("rag_service.rag_pipeline.chromadb.PersistentClient")
    def test_ingest_text_returns_correct_shape(self, mock_chroma, mock_genai):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_genai.embed_content.return_value = {"embedding": [0.1] * 768}

        from rag_service.rag_pipeline import RetailVectorStore, DocumentIngester
        vs = RetailVectorStore()
        ingester = DocumentIngester(vs)

        result = ingester.ingest_text("Sample retail text for testing.", "test_doc.txt")

        assert "doc_id"  in result
        assert "chunks"  in result
        assert result["source"] == "test_doc.txt"
        assert result["chunks"] >= 1

    @patch("rag_service.rag_pipeline.genai")
    @patch("rag_service.rag_pipeline.chromadb.PersistentClient")
    def test_ingest_pdf_raises_on_missing_file(self, mock_chroma, mock_genai):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        from rag_service.rag_pipeline import RetailVectorStore, DocumentIngester
        vs = RetailVectorStore()
        ingester = DocumentIngester(vs)

        with pytest.raises(FileNotFoundError):
            ingester.ingest_pdf("/nonexistent/path/file.pdf")

    @patch("rag_service.rag_pipeline.genai")
    @patch("rag_service.rag_pipeline.chromadb.PersistentClient")
    def test_rag_engine_returns_no_docs_message(self, mock_chroma, mock_genai):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.query.return_value = {
            "documents": [[]], "metadatas": [[]], "distances": [[]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_genai.embed_content.return_value = {"embedding": [0.1] * 768}

        from rag_service.rag_pipeline import RetailVectorStore, RAGEngine
        vs = RetailVectorStore()
        engine = RAGEngine(vs)

        result = engine.answer("What is the revenue?")
        assert "No relevant documents" in result["answer"]
        assert result["chunks_used"] == 0

    @patch("rag_service.rag_pipeline.genai")
    @patch("rag_service.rag_pipeline.chromadb.PersistentClient")
    def test_rag_engine_returns_answer_with_chunks(self, mock_chroma, mock_genai):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 3
        mock_collection.query.return_value = {
            "documents": [["Retail revenue grew 15% in 2024."]],
            "metadatas": [[{"source": "report.pdf", "chunk_index": 0, "doc_id": "abc"}]],
            "distances": [[0.1]],
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_response = MagicMock()
        mock_response.text = "Retail revenue grew by 15% in 2024 according to the report."
        mock_genai.embed_content.return_value = {"embedding": [0.1] * 768}
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response

        from rag_service.rag_pipeline import RetailVectorStore, RAGEngine
        vs = RetailVectorStore()
        engine = RAGEngine(vs)

        result = engine.answer("What was retail revenue growth?")
        assert result["chunks_used"] == 1
        assert "report.pdf" in result["sources"]
        assert len(result["answer"]) > 0


# ─────────────────────────────────────────────────────
# Agent Service Tests
# ─────────────────────────────────────────────────────
class TestAgentKnowledgeRepo:

    def test_save_to_knowledge_repo_creates_file(self):
        from agent_service.retail_agent import save_to_knowledge_repo, KNOWLEDGE_DIR
        content = "Test retail intelligence report content."
        query   = "test retail trends"

        filepath = save_to_knowledge_repo(content, query)

        assert filepath.exists()
        text = filepath.read_text(encoding="utf-8")
        assert "RetailMind Intelligence Report" in text
        assert content in text
        assert query in text

        filepath.unlink()   # cleanup

    def test_knowledge_dir_exists(self):
        from agent_service.retail_agent import KNOWLEDGE_DIR
        assert KNOWLEDGE_DIR.exists()
        assert KNOWLEDGE_DIR.is_dir()

    def test_save_sanitises_query_in_filename(self):
        from agent_service.retail_agent import save_to_knowledge_repo
        filepath = save_to_knowledge_repo("content", "Query with SPACES & symbols!")
        assert " " not in filepath.name
        assert "&" not in filepath.name
        filepath.unlink()

    @patch("agent_service.retail_agent._ddg_run")
    def test_search_tool_appends_retail_context(self, mock_ddg):
        """DuckDuckGo tool should append 'retail industry' if not present."""
        mock_ddg.run.return_value = "Sample search results about e-commerce trends."
        from agent_service.retail_agent import search_tool
        result = search_tool("e-commerce trends 2024")
        # Should have called with retail context appended
        call_arg = mock_ddg.run.call_args[0][0]
        assert "retail" in call_arg.lower()
        assert len(result) > 0

    @patch("agent_service.retail_agent._ddg_run")
    def test_search_tool_handles_failure_gracefully(self, mock_ddg):
        """DuckDuckGo tool should return error string on failure, not raise."""
        mock_ddg.run.side_effect = Exception("Rate limited")
        from agent_service.retail_agent import search_tool
        result = search_tool("some query")
        assert "Search failed" in result or isinstance(result, str)


# ─────────────────────────────────────────────────────
# API Layer Tests
# ─────────────────────────────────────────────────────
class TestAPIEndpoints:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        with patch("main.ingest_document"), patch("main.ask_question"), \
             patch("main.run_retail_research_agent"):
            from main import app
            return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "gemini-2.5-flash" in data["rag_model"]
        assert "groq" in data["agent_model"]

    def test_ask_empty_question_returns_400(self, client):
        with patch("main.ask_question"):
            resp = client.post("/api/ask", json={"question": "  "})
            assert resp.status_code == 400

    def test_ingest_non_pdf_returns_400(self, client):
        resp = client.post(
            "/api/ingest",
            files={"file": ("test.docx", b"fake content", "application/vnd.openxmlformats")}
        )
        assert resp.status_code == 400

    def test_knowledge_list_returns_list(self, client):
        resp = client.get("/api/knowledge")
        assert resp.status_code == 200
        assert "reports" in resp.json()