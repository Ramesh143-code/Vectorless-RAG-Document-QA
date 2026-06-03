"""
server.py
─────────────────────────────────────────────────────────────────────────────
Vectorless RAG — FastAPI Web Server with OCR Support
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
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

# Import from src package (using the __init__.py exports)
from src import PDFParser, create_parser, __version__

load_dotenv()

# ── API Key & Client ──────────────────────────────────────────────────────────
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not found in .env file! Add: GROQ_API_KEY=gsk_...")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# ── Initialize PDF Parser with OCR Settings ───────────────────────────────────
# Use the convenience function from __init__.py
pdf_parser = create_parser(
    ocr_quality="BALANCED",      # FAST, BALANCED, HIGH_QUALITY, VERY_HIGH, MAXIMUM
    ocr_language="eng",           # English (change to "eng+fra" for French, etc.)
    parallel_processing=True,     # Faster for multi-page PDFs
    max_workers=4                 # Number of parallel workers
)

print(f"✅ Vectorless RAG v{__version__} with OCR support initialized")

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vectorless RAG API", 
    version=__version__,
    description="PDF Q&A with OCR support for scanned documents"
)

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
    """Split text into overlapping chunks for better retrieval"""
    if not text or not text.strip():
        return []
    
    words = text.split()
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    
    if len(words) == 0:
        return []
    
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
    """Simple tokenizer for BM25"""
    return re.findall(r'\b[a-z0-9]+\b', text.lower())

def build_bm25(all_chunks: List[dict]) -> dict:
    """Build BM25 index from chunks"""
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
    """Search using BM25 algorithm"""
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
        "ocr_enabled":   True,
        "ocr_quality":   "BALANCED",
        "version":       __version__,
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and parse a PDF file (supports both text and scanned PDFs)"""
    global bm25_index

    name = file.filename.lower()
    if not name.endswith((".pdf", ".txt", ".md")):
        raise HTTPException(400, "Only PDF, TXT, and MD files are supported.")

    # Create temp directory if it doesn't exist
    temp_dir = Path("/tmp") if os.name != 'nt' else Path(os.environ.get('TEMP', '.'))
    temp_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"
    temp_path.parent.mkdir(exist_ok=True)
    
    try:
        # Save file
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Parse using OCR-enabled parser
        if name.endswith(".pdf"):
            try:
                # Use our enhanced PDF parser with OCR
                parsed_doc = pdf_parser.parse(temp_path)
                text = parsed_doc.get_all_text()
                
                # Get OCR status
                is_scanned = parsed_doc.metadata.is_scanned
                ocr_quality = parsed_doc.metadata.ocr_quality if is_scanned else "N/A"
                pages = parsed_doc.metadata.page_count
                
                print(f"✅ PDF processed: {file.filename} (Scanned: {is_scanned}, Pages: {pages}, OCR Quality: {ocr_quality})")
                
            except Exception as e:
                print(f"❌ PDF parse error: {e}")
                raise HTTPException(500, f"PDF parse error with OCR: {str(e)}")
        else:
            # For text files
            text = content.decode("utf-8", errors="ignore")
            is_scanned = False
            ocr_quality = "N/A"

        if not text or not text.strip():
            raise HTTPException(400, "Could not extract any text from this file.")

        # Create document chunks
        doc_id = str(uuid.uuid4())[:8]
        chunks = chunk_text(text, doc_id, file.filename)

        if not chunks:
            raise HTTPException(400, "Text extracted but no content could be chunked.")

        # Store document
        documents[doc_id] = {
            "doc_id":      doc_id,
            "filename":    file.filename,
            "chunks":      chunks,
            "char_count":  len(text),
            "chunk_count": len(chunks),
            "is_scanned":  is_scanned,
            "ocr_quality": ocr_quality,
            "pages":       pages if name.endswith(".pdf") else 0,
        }

        # Invalidate index on new upload
        bm25_index = None

        return {
            "doc_id":      doc_id,
            "filename":    file.filename,
            "chunk_count": len(chunks),
            "char_count":  len(text),
            "is_scanned":  is_scanned,
            "ocr_quality": ocr_quality,
            "pages":       pages if name.endswith(".pdf") else 0,
            "status":      "parsed",
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(500, f"Upload processing error: {str(e)}")
    finally:
        # Clean up temp file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass


@app.post("/index")
def build_index():
    """Build BM25 index from all uploaded documents"""
    global bm25_index
    if not documents:
        raise HTTPException(400, "No documents uploaded yet. Upload a PDF first.")

    all_chunks = [c for doc in documents.values() for c in doc["chunks"]]
    
    if not all_chunks:
        raise HTTPException(400, "No chunks found to index. Try re-uploading your document.")
    
    bm25_index = build_bm25(all_chunks)

    # Count scanned vs text documents
    scanned_count = sum(1 for d in documents.values() if d.get("is_scanned", False))
    text_count = len(documents) - scanned_count

    return {
        "status":       "indexed",
        "total_docs":   len(documents),
        "total_chunks": len(all_chunks),
        "scanned_docs": scanned_count,
        "text_docs":    text_count,
    }


class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    model: str = "llama-3.1-8b-instant"


@app.post("/ask")
def ask(req: AskRequest):
    """Ask a question based on indexed documents"""
    if bm25_index is None:
        raise HTTPException(400, "Index not built yet. Click 'Build Index' first.")
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")

    top_chunks = bm25_search(bm25_index, req.query, top_k=req.top_k)
    if not top_chunks:
        return {
            "answer":    "No relevant content found for your question. Try uploading different documents or rephrasing your question.",
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
                "is_scanned":  documents.get(c["doc_id"], {}).get("is_scanned", False),
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
    """List all uploaded documents with metadata"""
    return [
        {
            "doc_id":      d["doc_id"],
            "filename":    d["filename"],
            "chunk_count": d["chunk_count"],
            "char_count":  d["char_count"],
            "is_scanned":  d.get("is_scanned", False),
            "ocr_quality": d.get("ocr_quality", "N/A"),
            "pages":       d.get("pages", 0),
        }
        for d in documents.values()
    ]


@app.get("/documents/{doc_id}")
def get_document_details(doc_id: str):
    """Get detailed information about a specific document"""
    if doc_id not in documents:
        raise HTTPException(404, "Document not found.")
    
    doc = documents[doc_id]
    return {
        "doc_id":      doc["doc_id"],
        "filename":    doc["filename"],
        "chunk_count": doc["chunk_count"],
        "char_count":  doc["char_count"],
        "is_scanned":  doc.get("is_scanned", False),
        "ocr_quality": doc.get("ocr_quality", "N/A"),
        "pages":       doc.get("pages", 0),
        "chunks_preview": [
            {
                "index": c["chunk_index"],
                "preview": c["text"][:150] + "...",
                "length": len(c["text"])
            }
            for c in doc["chunks"][:3]  # Show first 3 chunks
        ]
    }


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document from the store"""
    global bm25_index
    if doc_id not in documents:
        raise HTTPException(404, "Document not found.")
    
    filename = documents[doc_id]["filename"]
    del documents[doc_id]
    bm25_index = None  # Invalidate index
    
    return {"status": "deleted", "doc_id": doc_id, "filename": filename}


@app.delete("/documents")
def delete_all_documents():
    """Delete all documents"""
    global documents, bm25_index
    count = len(documents)
    documents = {}
    bm25_index = None
    return {"status": "deleted", "count": count}


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 Vectorless RAG Server with OCR Support")
    print("=" * 60)
    print(f"📚 Version: {__version__}")
    print(f"📁 Data directory: {Path('data').absolute()}")
    print(f"🌐 Frontend directory: {FRONTEND_DIR.absolute() if FRONTEND_DIR.exists() else 'Not found'}")
    print(f"🔍 OCR: Enabled (Quality: BALANCED, Language: eng)")
    print("=" * 60)
    print("🌎 Open http://localhost:8000 in your browser")
    print("💡 Press CTRL+C to stop the server")
    print("=" * 60)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )