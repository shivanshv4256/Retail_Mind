"""
RetailMind FastAPI Backend
===========================
Exposes unified REST endpoints for:
  POST /api/ingest          → Upload & index a PDF via Gemini RAG
  POST /api/ask             → Q&A against indexed documents
  POST /api/research        → Trigger CrewAI + Groq 70B agent
  GET  /api/knowledge       → List saved agent reports
  GET  /api/health          → Health check
"""

import os
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Internal services
from rag_service.rag_pipeline   import ingest_document, ask_question
from agent_service.retail_agent import run_retail_research_agent, KNOWLEDGE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retailmind.api")

app = FastAPI(
    title="RetailMind API",
    description="Retail document intelligence — RAG (Gemini 2.5 Flash) + Agent (Groq 70B)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# 3. Serve your index.html at the root URL
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    top_k:    int = 5

class QuestionResponse(BaseModel):
    answer:      str
    sources:     list[str]
    chunks_used: int
    model:       str

class ResearchRequest(BaseModel):
    query: str

class ResearchResponse(BaseModel):
    report:    str
    filepath:  str
    query:     str
    model:     str
    timestamp: str


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status":     "ok",
        "rag_model":  "gemini-2.5-flash",
        "agent_model": "groq/llama-3.3-70b-versatile",
    }


@app.post("/api/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload a PDF and index it into the Gemini-powered RAG pipeline."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    save_path = UPLOAD_DIR / file.filename
    save_path.write_bytes(await file.read())

    try:
        result = ingest_document(str(save_path))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(500, f"Ingestion error: {e}")

    return {
        "message": f"Successfully indexed '{file.filename}'",
        **result,
    }


@app.post("/api/ask", response_model=QuestionResponse)
def ask(req: QuestionRequest):
    """
    Ask a question — answered strictly from indexed retail documents.
    Powered by Gemini 2.5 Flash with RAG context injection.
    """
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    try:
        result = ask_question(req.question)
    except Exception as e:
        logger.error(f"Q&A failed: {e}")
        raise HTTPException(500, f"Q&A error: {e}")

    return QuestionResponse(**result)


@app.post("/api/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    """
    Trigger the autonomous CrewAI research agent.
    3 agents (Researcher → Analyst → Writer) powered by Groq LLaMA 3.3 70B.
    Report auto-saved to knowledge_repo/ and ingested into RAG.
    """
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    try:
        result = run_retail_research_agent(req.query)
    except Exception as e:
        logger.error(f"Agent run failed: {e}")
        raise HTTPException(500, f"Agent error: {e}")

    return ResearchResponse(
        report=result["report"],
        filepath=str(result["filepath"]),
        query=result["query"],
        model=result["model"],
        timestamp=result["timestamp"],
    )


@app.get("/api/knowledge")
def list_knowledge():
    """List all saved intelligence reports in the knowledge repository."""
    files = sorted(KNOWLEDGE_DIR.glob("*.txt"), reverse=True)
    return {
        "count":   len(files),
        "reports": [
            {
                "filename": f.name,
                "size_kb":  round(f.stat().st_size / 1024, 1),
                "modified": f.stat().st_mtime,
            }
            for f in files
        ],
    }


@app.get("/api/knowledge/{filename}")
def get_report(filename: str):
    """Retrieve a specific saved report."""
    filepath = KNOWLEDGE_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, "Report not found")
    return {"filename": filename, "content": filepath.read_text(encoding="utf-8")}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)