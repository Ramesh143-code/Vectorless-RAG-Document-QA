"""
server.py
─────────────────────────────────────────────────────────────────────────────
Vectorless RAG — FastAPI Web Server
Place in ROOT of project (same folder as main.py)

Run:
    uvicorn server:app --reload --port 8000
Then open: http://localhost:8000
─────────────────────────────────────────────────────────────────────────────
"""

import os
import uuid
import math
import re
from collections import defaultdict
from typing import List, Dict, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

# ── API Key & Client ──────────────────────────────────────────────────────────
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not found in .env file! Add: GROQ_API_KEY=gsk_...")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Vectorless RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend/index.html ─────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
def serve_ui():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Server running. Put index.html inside a 'frontend' folder."}

# ── In-memory stores ──────────────────────────────────────────────────────────
documents: Dict[str, dict] = {}
bm25_index: Optional[dict] = None

# ═════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ═════════════════════════════════════════════════════════════════════════════
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80

def chunk_text(text: str, doc_id: str, filename: str) -> List[dict]:
    words = text.split()
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, max(1, len(words) - CHUNK_OVERLAP), step):
        chunk_words = words[i : i + CHUNK_SIZE]
        if not chunk_words:
            break
        chunks.append({
            "id":          f"{doc_id}_chunk_{len(chunks)}",
            "doc_id":      doc_id,
            "filename":    filename,
            "text":        " ".join(chunk_words),
            "chunk_index": len(chunks),
        })
    return chunks

# ═════════════════════════════════════════════════════════════════════════════
# BM25 — pure Python, no external library
# ═════════════════════════════════════════════════════════════════════════════
def tokenize(text: str) -> List[str]:
    return re.findall(r'\b[a-z0-9]+\b', text.lower())

def build_bm25(all_chunks: List[dict]) -> dict:
    k1, b = 1.5, 0.75
    N = len(all_chunks)
    df: Dict[str, int] = defaultdict(int)
    doc_tfs, doc_lens = [], []

    for chunk in all_chunks:
        tokens = tokenize(chunk["text"])
        doc_lens.append(len(tokens))
        tf: Dict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        doc_tfs.append(dict(tf))
        for t in set(tokens):
            df[t] += 1

    avg_dl = sum(doc_lens) / max(N, 1)
    idf = {
        t: math.log((N - f + 0.5) / (f + 0.5) + 1)
        for t, f in df.items()
    }
    return {
        "chunks": all_chunks,
        "doc_tfs": doc_tfs,
        "doc_lens": doc_lens,
        "avg_dl": avg_dl,
        "idf": idf,
        "k1": k1,
        "b": b,
    }

def bm25_search(index: dict, query: str, top_k: int = 5) -> List[dict]:
    tokens = tokenize(query)
    k1, b, avg_dl = index["k1"], index["b"], index["avg_dl"]
    scores = []
    for i, (tf, dl) in enumerate(zip(index["doc_tfs"], index["doc_lens"])):
        score = 0.0
        for t in tokens:
            if t not in index["idf"]:
                continue
            f = tf.get(t, 0)
            score += index["idf"][t] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avg_dl))
        scores.append((score, i))
    scores.sort(reverse=True)
    results = []
    for score, idx in scores[:top_k]:
        if score > 0:
            c = index["chunks"][idx].copy()
            c["bm25_score"] = round(score, 4)
            results.append(c)
    return results

# ═════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "docs_loaded":   len(documents),
        "index_built":   bm25_index is not None,
        "groq_key_set":  bool(api_key),
        "model":         "llama-3.1-8b-instant",
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global bm25_index

    name = file.filename.lower()
    if not name.endswith((".pdf", ".txt", ".md")):
        raise HTTPException(400, "Only PDF, TXT, and MD files are supported.")

    raw = await file.read()

    if name.endswith(".pdf"):
        try:
            pdf  = fitz.open(stream=raw, filetype="pdf")
            text = "\n".join(page.get_text() for page in pdf)
            pdf.close()
        except Exception as e:
            raise HTTPException(500, f"PDF parse error: {e}")
    else:
        text = raw.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(400, "Could not extract any text from this file.")

    doc_id = str(uuid.uuid4())[:8]
    chunks = chunk_text(text, doc_id, file.filename)

    documents[doc_id] = {
        "doc_id":      doc_id,
        "filename":    file.filename,
        "chunks":      chunks,
        "char_count":  len(text),
        "chunk_count": len(chunks),
    }

    bm25_index = None  # invalidate index on new upload

    return {
        "doc_id":      doc_id,
        "filename":    file.filename,
        "chunk_count": len(chunks),
        "char_count":  len(text),
        "status":      "parsed",
    }


@app.post("/index")
def build_index():
    global bm25_index
    if not documents:
        raise HTTPException(400, "No documents uploaded yet.")

    all_chunks = [c for doc in documents.values() for c in doc["chunks"]]
    bm25_index = build_bm25(all_chunks)

    return {
        "status":       "indexed",
        "total_docs":   len(documents),
        "total_chunks": len(all_chunks),
    }


class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    model: str = "llama-3.1-8b-instant"   # Groq free model


@app.post("/ask")
def ask(req: AskRequest):
    if bm25_index is None:
        raise HTTPException(400, "Index not built yet. Click 'Build Index' first.")
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")

    top_chunks = bm25_search(bm25_index, req.query, top_k=req.top_k)
    if not top_chunks:
        return {
            "answer":    "No relevant content found for your question.",
            "citations": [],
            "chunks":    [],
        }

    context = "\n\n---\n\n".join(
        f"[Source: {c['filename']} | Chunk {c['chunk_index']}]\n{c['text']}"
        for c in top_chunks
    )

    system_prompt = (
        "You are a precise document Q&A assistant using Retrieval-Augmented Generation (RAG).\n"
        "Answer the user's question using ONLY the document excerpts provided below.\n"
        "Always cite the source filename when referencing information.\n"
        "If the answer is not present in the context, clearly say so.\n\n"
        f"CONTEXT:\n{context}"
    )

    try:
        response = client.chat.completions.create(
            model=req.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": req.query},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(500, f"Groq API error: {e}")

    seen, citations = set(), []
    for c in top_chunks:
        if c["filename"] not in seen:
            seen.add(c["filename"])
            citations.append({
                "filename":    c["filename"],
                "doc_id":      c["doc_id"],
                "chunk_index": c["chunk_index"],
            })

    return {
        "answer":    answer,
        "citations": citations,
        "chunks": [
            {
                "label":   f"{c['filename']} › chunk_{c['chunk_index']}",
                "score":   c["bm25_score"],
                "preview": c["text"][:120] + "...",
            }
            for c in top_chunks
        ],
    }


@app.get("/documents")
def list_documents():
    return [
        {
            "doc_id":      d["doc_id"],
            "filename":    d["filename"],
            "chunk_count": d["chunk_count"],
            "char_count":  d["char_count"],
        }
        for d in documents.values()
    ]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    global bm25_index
    if doc_id not in documents:
        raise HTTPException(404, "Document not found.")
    del documents[doc_id]
    bm25_index = None
    return {"status": "deleted", "doc_id": doc_id}
