# Vectorless RAG — Local Setup Guide

No vector database. No embeddings. Pure BM25 retrieval + OpenAI GPT answers.

---

## Project Structure

```
vectorless-rag/
├── backend/
│   ├── server.py          ← FastAPI app (upload, index, ask)
│   ├── requirements.txt   ← Python dependencies
│   └── .env               ← Your OpenAI key goes here
└── frontend/
    └── index.html         ← Open this in browser
```

---

## 1. Backend Setup

### Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Add your OpenAI key
Edit `backend/.env`:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

### Run the server
```bash
uvicorn server:app --reload --port 8000
```

Server runs at: http://localhost:8000

---

## 2. Frontend

Just open `frontend/index.html` in your browser — no build step needed.

> **Note:** If CORS issues arise, serve it with:
> ```bash
> cd frontend
> python -m http.server 5500
> ```
> Then open http://localhost:5500

---

## How It Works

| Step | What Happens |
|------|-------------|
| Upload | PDF parsed via PyMuPDF → split into 400-word chunks with 80-word overlap |
| Build Index | BM25 index built from all chunks (TF-IDF without vectors) |
| Ask | Query scored against all chunks → top-5 chunks sent to GPT as context |
| Answer | GPT-4o-mini answers using only the retrieved context + cites source |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a PDF/TXT/MD file |
| POST | `/index` | Build BM25 index from uploaded docs |
| POST | `/ask` | `{query, top_k, model}` → answer + citations |
| GET | `/documents` | List all uploaded documents |
| DELETE | `/documents/{doc_id}` | Remove a document |

---

## Tips

- Supports `.pdf`, `.txt`, `.md` files
- Upload multiple PDFs before building the index
- Re-upload and re-build after adding new documents
- Change `model` in the ask request to use `gpt-4o` for better answers
- BM25 scores shown in UI — higher = more relevant chunk
