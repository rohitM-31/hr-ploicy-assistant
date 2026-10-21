"""
main.py - FastAPI backend
Run:  uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import PolicyRAG

app = FastAPI(title="HR Policy Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = PolicyRAG()  # load + embed policies once at startup


class AskRequest(BaseModel):
    question: str


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": rag.mode, "policies": rag.policies, "sections": len(rag.chunks)}


@app.post("/api/ask")
def ask(req: AskRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is empty")
    try:
        return rag.ask(q)
    except Exception as e:  # e.g. Gemini quota / network error
        raise HTTPException(status_code=502, detail=f"AI service error: {e}")


@app.post("/api/reload")
def reload_policies():
    """Call after editing files in /policies - no restart needed."""
    global rag
    rag = PolicyRAG()
    return {"status": "reloaded", "sections": len(rag.chunks)}
