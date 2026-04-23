"""
RetailMind RAG Service
======================
Embeddings : HuggingFace sentence-transformers/all-MiniLM-L6-v2 (local, free)
Q&A        : Gemini 2.5 Flash (API)
Vector store: ChromaDB (local dev) / Pinecone (production)

Embedding model details:
  - 384-dimensional dense vectors
  - Trained on 1B+ sentence pairs
  - ~80 MB download, runs on CPU
  - No API key or internet call needed at inference time
"""

import os
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retailmind.rag")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
env_path = Path(__file__).resolve().parent.parent / ".env"

# 2. Check if the file actually exists before loading
if not env_path.exists():
    print(f"CRITICAL ERROR: No .env file found at {env_path}")
else:
    load_dotenv(dotenv_path=env_path)
    print(".env file loaded successfully.")

# 3. Use .get() to prevent the KeyError crash
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY not found in environment variables.")
else:
    # ADD THIS LINE:
    genai.configure(api_key=api_key)



HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local HuggingFace model
EMBED_DIM      = 384                           # MiniLM-L6-v2 output dimension
CHAT_MODEL     = "gemini-2.5-flash"            # Gemini 2.5 Flash for Q&A only
CHUNK_SIZE     = 256                           # MiniLM has 256-token context window
CHUNK_OVERLAP  = 40                            # ~15% overlap
TOP_K          = 5                             # top chunks to retrieve
COLLECTION     = "retailmind_docs"


# ──────────────────────────────────────────────
# HuggingFace Embedding Function (ChromaDB-compatible)
# ──────────────────────────────────────────────
class MiniLMEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    Wraps sentence-transformers/all-MiniLM-L6-v2 for ChromaDB.
    Model is downloaded once (~80 MB) and cached locally in ~/.cache/huggingface.
    All inference runs on CPU — no GPU or API key required.
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {HF_EMBED_MODEL}")
        self._model = SentenceTransformer(HF_EMBED_MODEL)
        logger.info(f"Embedding model ready (dim={EMBED_DIM})")

    def __call__(self, input: list[str]) -> list[list[float]]:
        """
        Encode a batch of texts into 384-dim vectors.
        sentence-transformers handles batching, padding and truncation internally.
        normalize_embeddings=True ensures cosine similarity works correctly.
        """
        vectors = self._model.encode(
            input,
            batch_size=32,
            normalize_embeddings=True,   # unit-length vectors → cosine = dot product
            show_progress_bar=False,
        )
        return vectors.tolist()


# ──────────────────────────────────────────────
# Singleton embedding model (loaded once per process)
# ──────────────────────────────────────────────
_embed_fn: MiniLMEmbeddingFunction | None = None

def get_embed_fn() -> MiniLMEmbeddingFunction:
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = MiniLMEmbeddingFunction()
    return _embed_fn


# ──────────────────────────────────────────────
# Vector Store
# ──────────────────────────────────────────────
class RetailVectorStore:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client   = chromadb.PersistentClient(path=persist_dir)
        self.embed_fn = get_embed_fn()           # MiniLM-L6-v2, local, free
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},   # cosine on normalized MiniLM vectors
        )
        logger.info(f"Vector store ready — {self.collection.count()} chunks indexed")

    def upsert_chunks(self, chunks: list[str], doc_id: str, source: str):
        """Add or update document chunks in the vector store."""
        ids       = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "chunk_index": i, "doc_id": doc_id}
                     for i in range(len(chunks))]
        self.collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
        logger.info(f"Upserted {len(chunks)} chunks from '{source}'")

    def query(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """Embed question with MiniLM and retrieve top-k similar chunks."""
        # Use the same MiniLM model that was used for document embedding
        query_vector = self.embed_fn([question])[0]

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text":        doc,
                "source":      meta["source"],
                "chunk_index": meta["chunk_index"],
                "relevance":   round(1 - dist, 4),   # cosine similarity score (0–1)
            })
        return chunks


# ──────────────────────────────────────────────
# Document Ingestion
# ──────────────────────────────────────────────
class DocumentIngester:
    def __init__(self, vector_store: RetailVectorStore):
        self.vs = vector_store
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def ingest_pdf(self, pdf_path: str) -> dict:
        """Extract text from PDF, chunk it, embed and store."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Ingesting PDF: {path.name}")

        # Extract text from all pages
        reader   = PdfReader(pdf_path)
        raw_text = "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        )
        raw_text = raw_text.strip()

        if not raw_text:
            raise ValueError(f"No extractable text in {path.name}")

        # Chunk
        chunks = self.splitter.split_text(raw_text)

        # Stable doc_id from filename hash
        doc_id = hashlib.md5(path.name.encode()).hexdigest()[:12]

        # Store
        self.vs.upsert_chunks(chunks, doc_id=doc_id, source=path.name)

        return {
            "doc_id":      doc_id,
            "source":      path.name,
            "pages":       len(reader.pages),
            "chunks":      len(chunks),
            "ingested_at": datetime.utcnow().isoformat(),
        }

    def ingest_text(self, text: str, source_name: str) -> dict:
        """Ingest plain text (e.g. agent knowledge-repo summaries)."""
        chunks = self.splitter.split_text(text)
        doc_id = hashlib.md5(source_name.encode()).hexdigest()[:12]
        self.vs.upsert_chunks(chunks, doc_id=doc_id, source=source_name)
        return {"doc_id": doc_id, "source": source_name, "chunks": len(chunks)}


# ──────────────────────────────────────────────
# RAG Q&A Engine — Gemini 2.5 Flash
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are RetailMind, an expert retail intelligence assistant.
Answer questions STRICTLY based on the provided context chunks.
If the answer is not in the context, say: "This information is not available in the uploaded documents."
Always cite the source document name when referencing specific facts.
Be concise, structured, and professional."""


class RAGEngine:
    def __init__(self, vector_store: RetailVectorStore):
        self.vs    = vector_store
        self.model = genai.GenerativeModel(
            model_name=CHAT_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )

    def answer(self, question: str, top_k: int = TOP_K) -> dict:
        """Retrieve context and generate a grounded answer."""
        # 1. Retrieve relevant chunks
        chunks = self.vs.query(question, top_k=top_k)

        if not chunks:
            return {
                "answer":  "No relevant documents found. Please upload retail documents first.",
                "sources": [],
                "chunks_used": 0,
            }

        # 2. Build context block
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['source']} | relevance: {chunk['relevance']}]\n{chunk['text']}"
            )
        context_str = "\n\n---\n\n".join(context_parts)

        # 3. Build prompt
        prompt = f"""Context from retail documents:
{context_str}

Question: {question}

Answer:"""

        # 4. Generate with Gemini 2.5 Flash
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,       # low temp for factual Q&A
                max_output_tokens=1024,
            ),
        )

        # 5. Deduplicate sources
        sources = list({c["source"] for c in chunks})

        return {
            "answer":      response.text,
            "sources":     sources,
            "chunks_used": len(chunks),
            "model":       CHAT_MODEL,
        }


# ──────────────────────────────────────────────
# Public API (used by the FastAPI / Express layer)
# ──────────────────────────────────────────────
_vs      = None
_ingester = None
_engine  = None

def get_rag_components():
    global _vs, _ingester, _engine
    if _vs is None:
        _vs      = RetailVectorStore()
        _ingester = DocumentIngester(_vs)
        _engine  = RAGEngine(_vs)
    return _vs, _ingester, _engine


def ingest_document(file_path: str) -> dict:
    _, ingester, _ = get_rag_components()
    return ingester.ingest_pdf(file_path)


def ask_question(question: str) -> dict:
    _, _, engine = get_rag_components()
    return engine.answer(question)


def ingest_agent_summary(text: str, source_name: str) -> dict:
    """Bridge: agent knowledge-repo summaries flow into the RAG index."""
    _, ingester, _ = get_rag_components()
    return ingester.ingest_text(text, source_name)


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    vs, ingester, engine = get_rag_components()

    # Test with a dummy text document
    sample = """
    Q3 2024 Retail Market Report
    ============================
    E-commerce growth reached 18% YoY in Q3 2024.
    Grocery delivery saw the highest adoption, up 34%.
    Luxury fashion retained margins at 62% despite inflation.
    Key trend: AI-powered personalisation increased conversion by 22%.
    """
    result = ingester.ingest_text(sample, "q3_retail_report.txt")
    print("Ingested:", result)

    answer = engine.answer("What was e-commerce growth in Q3 2024?")
    print("\nAnswer:", answer["answer"])
    print("Sources:", answer["sources"])