<div align="center">

```
██╗   ██╗███████╗ ██████╗████████╗ ██████╗ ██████╗ ██╗     ███████╗███████╗███████╗
██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██║     ██╔════╝██╔════╝██╔════╝
██║   ██║█████╗  ██║        ██║   ██║   ██║██████╔╝██║     █████╗  ███████╗███████╗
╚██╗ ██╔╝██╔══╝  ██║        ██║   ██║   ██║██╔══██╗██║     ██╔══╝  ╚════██║╚════██║
 ╚████╔╝ ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║███████╗███████╗███████║███████║
  ╚═══╝  ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚══════╝
                                    R A G
```

### **No Vectors. No Embeddings. No Nonsense.**
#### BM25-powered Retrieval-Augmented Generation on Structured PDFs

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Groq-Ultra--Fast_LLM-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![BM25](https://img.shields.io/badge/Retrieval-BM25_+_TF--IDF-6366F1?style=for-the-badge&logo=searchengin&logoColor=white)](#)
[![OCR](https://img.shields.io/badge/OCR-Scanned_PDFs-22C55E?style=for-the-badge&logo=files&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/yourusername/vectorless-rag?style=for-the-badge&color=gold&logo=github)](https://github.com/yourusername/vectorless-rag/stargazers)

<br/>

> **The question everyone gets wrong:** *"Why would you build RAG without vectors?"*
>
> **The answer:** Because for structured PDFs — reports, manuals, research papers, forms —
> BM25 + document structure beats dense retrieval. Faster. Cheaper. More accurate. No GPU needed.

<br/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=24&pause=1500&color=22C55E&center=true&vCenter=true&width=950&lines=Vectorless+RAG+for+Structured+Documents;PDF+Parsing+%7C+OCR+%7C+BM25+Retrieval;No+Embeddings.+No+Vector+DB.;Just+Relevant+Answers." />

</div>

---

## 📸 Demo

<div align="">

| Upload & Index | Ask Questions | See Sources |
|:-:|:-:|:-:|
| Drop any PDF — scanned or digital | Natural language Q&A with full history | Every answer cites exact page + section |

</div>

> **Live on [localhost:8000](http://localhost:8000)** — Drop a PDF, hit Index, start asking. That's it.
> **Live on [localhost:8000](http://127.0.0.1:8000)** — Drop a PDF, hit Index, start asking. That's it.

---

## ⚡ Why Vectorless?

<div align="">

| | **Vectorless RAG** (this project) | **Traditional RAG** |
|---|:---:|:---:|
| 💾 Vector Database | ❌ Not needed | ✅ Required (Pinecone, Chroma, etc.) |
| 🧠 Embedding Model | ❌ Not needed | ✅ Required (OpenAI, sentence-transformers) |
| 💰 Cost per query | ~$0.0001 | ~$0.002+ |
| ⚡ Index build time | **< 2 seconds** | 10–60 seconds |
| 🎯 Structured PDF accuracy | **Higher** | Lower (misses structure) |
| 🖥️ GPU required | ❌ None | ✅ Often yes |
| 📦 Setup complexity | **Minimal** | High |

</div>

**The insight:** Structured PDFs (reports, manuals, papers) have consistent vocabulary. The words in your question appear verbatim in the relevant section. BM25 handles this perfectly — no learned representations needed.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VECTORLESS RAG PIPELINE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   📄 PDF Input                                                      │
│      │                                                              │
│      ▼                                                              │
│   ┌──────────────────────────────────────────────┐                 │
│   │           pdf_parser.py                      │                 │
│   │   pdfplumber + PyMuPDF + OCR (Tesseract)     │                 │
│   │   → Text blocks  → Tables  → Headings        │                 │
│   │   → Font sizes   → TOC     → Metadata        │                 │
│   └──────────────────────┬───────────────────────┘                 │
│                          │ ParsedDocument                          │
│                          ▼                                         │
│   ┌──────────────────────────────────────────────┐                 │
│   │           chunker.py                         │                 │
│   │   Structure-Aware Chunking                   │                 │
│   │   → Heading-bounded sections                 │                 │
│   │   → Table-isolated chunks (never split)      │                 │
│   │   → Paragraph merging                        │                 │
│   │   → Overflow split with sentence overlap     │                 │
│   └──────────────────────┬───────────────────────┘                 │
│                          │ list[Chunk]                             │
│                          ▼                                         │
│   ┌──────────────────────────────────────────────┐                 │
│   │           retriever.py                       │                 │
│   │   Hybrid BM25 + TF-IDF                       │                 │
│   │   score = 0.65×BM25 + 0.35×TFIDF + boost    │                 │
│   │   → Heading trail boost (+0.15)              │                 │
│   │   → Table type boost (+0.10)                 │                 │
│   └──────────────────────┬───────────────────────┘                 │
│                          │ Top-K RetrievalResults                  │
│                          ▼                                         │
│   ┌──────────────────────────────────────────────┐                 │
│   │           rag_pipeline.py                    │                 │
│   │   Context builder + Groq LLM                 │                 │
│   │   → Structured prompt with source metadata   │                 │
│   │   → Multi-turn conversation history          │                 │
│   │   → Citation extraction                      │                 │
│   └──────────────────────┬───────────────────────┘                 │
│                          │ RAGResponse                             │
│                          ▼                                         │
│   ┌──────────────────────────────────────────────┐                 │
│   │         server.py + ui/index.html            │                 │
│   │   FastAPI REST API + Web Interface           │                 │
│   │   → /upload  /index  /ask  /stats            │                 │
│   └──────────────────────────────────────────────┘                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🔍 Retrieval Engine
- **BM25 Okapi** — gold standard keyword search, beats TF-IDF on short queries
- **TF-IDF hybrid** — bigram-aware term weighting with sublinear scaling
- **Structural boost** — chunks under matching headings score higher
- **Table boost** — structured data chunks are prioritised for data questions
- **Multi-query fusion** — optional query rephrasing for broader recall

### 📄 PDF Processing
- **Digital PDFs** — pdfplumber extracts text with full layout awareness
- **Scanned PDFs** — OCR pipeline via Tesseract for image-based documents
- **Table extraction** — full table parsing into structured rows/columns
- **Heading detection** — font-size-based H1/H2/H3 detection (no hardcoding)
- **Multi-document** — query across multiple PDFs simultaneously
- **TOC parsing** — bookmark-based outline extraction via PyMuPDF

### 💬 Q&A Interface
- **Multi-turn conversation** — full history context across questions
- **Source citations** — every answer links to exact file, page, and section
- **Chunk transparency** — see exactly which chunks were retrieved and their scores
- **Real-time UI** — clean web interface with drag-and-drop upload

### ⚡ Performance
- **Groq inference** — among the fastest LLM APIs available (hundreds of tokens/sec)
- **Sub-second retrieval** — BM25 index lookup is near-instantaneous
- **No GPU required** — runs entirely on CPU
- **Lightweight** — no vector DB, no embedding model, no cloud dependencies

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/vectorless-rag.git
cd vectorless-rag
```

### 2. Create environment & install

```bash
# Using uv (recommended — 10x faster than pip)
uv venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

uv pip install -r requirements.txt
```

```bash
# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download NLTK data (once)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

### 4. Set up `.env`

```env
# ── LLM ──────────────────────────────────────────────────────
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=llama-3.1-8b-instant        # or mixtral-8x7b-32768, llama3-70b-8192

# ── Project Paths ─────────────────────────────────────────────
PDF_INPUT_DIR=data/
OUTPUT_DIR=outputs/

# ── RAG Settings ──────────────────────────────────────────────
TOP_K_RESULTS=5
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

> Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the server

```bash
uvicorn server:app --reload --port 8000
```

### 6. Open the UI

```
http://localhost:8000/ui
```

**That's it.** Drop a PDF → click Index → start asking questions.

---

## 📁 Project Structure

```
vectorless-rag/
│
├── src/
│   ├── __init__.py          # Package exports
│   ├── pdf_parser.py        # PDF → ParsedDocument (text, tables, headings)
│   ├── chunker.py           # ParsedDocument → list[Chunk] (structure-aware)
│   ├── retriever.py         # BM25 + TF-IDF hybrid search index
│   └── rag_pipeline.py      # End-to-end orchestration + Groq integration
│
├── ui/
│   └── index.html           # Web interface (upload, chat, citations)
│
├── data/                    # Drop your PDFs here
├── outputs/                 # Logs and processed outputs
├── notebooks/               # Experiments and exploration
│
├── server.py                # FastAPI REST API
├── main.py                  # CLI entry point
├── requirements.txt         # Dependencies
└── .env                     # API keys and config
```

---

## 🔌 API Reference

### `POST /upload`
Upload a PDF file.
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"
```
```json
{ "status": "uploaded", "filename": "your_document.pdf", "size_kb": 245.3 }
```

### `POST /index`
Build the BM25 + TF-IDF index from all uploaded PDFs.
```bash
curl -X POST http://localhost:8000/index
```
```json
{ "status": "ready", "documents": 2, "total_chunks": 87, "latency_ms": 1243.5 }
```

### `POST /ask`
Ask a question.
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main risk factors?", "top_k": 5}'
```
```json
{
  "query": "What are the main risk factors?",
  "answer": "According to the document, the main risk factors include...",
  "citations": [
    { "source": "report.pdf", "page_start": 4, "label": "[report.pdf, p.4]" }
  ],
  "chunks": [
    { "rank": 1, "final_score": 0.921, "chunk_type": "heading_section", "page": 4 }
  ],
  "latency_ms": 843.2
}
```

### `GET /documents`
List all uploaded documents.

### `DELETE /documents/{filename}`
Remove a document.

### `GET /stats`
Index statistics.

### `POST /clear-history`
Reset conversation history.

---

## 🧠 How Chunking Works

Unlike naive fixed-size chunking, this project chunks along the document's **own structure**:

```
Document
├── [H1] Introduction              → Heading chunk
│   ├── paragraph 1 + 2 merged    → Paragraph chunk (under H1 context)
│   └── [TABLE] Results table     → Table chunk (never split, isolated)
├── [H2] Methodology               → Heading chunk
│   ├── paragraph 1               → Paragraph chunk (under H1 > H2 context)
│   └── paragraph 2 + 3 merged    → Paragraph chunk
└── [H1] Conclusion                → Heading chunk
```

Every chunk carries a **heading breadcrumb trail** — e.g. `"Introduction > 1.2 Methods"` — which is used both for retrieval boosting and citation display.

---

## 📊 Retrieval Scoring

```
final_score = (0.65 × BM25_normalized) + (0.35 × TF-IDF_cosine) + structural_boost

structural_boost:
  + 0.15 × (query_terms ∩ heading_trail_terms) / 3    # heading relevance
  + 0.10                                               # if chunk is a TABLE
```

BM25 scores are **min-max normalized** per query before fusion. TF-IDF uses **sublinear term frequency** with **bigram tokenization** for phrase matching.

---

## 🗺️ Roadmap

- [x] BM25 + TF-IDF hybrid retrieval
- [x] Structure-aware chunking (headings, tables, paragraphs)
- [x] OCR support for scanned PDFs
- [x] Multi-document indexing
- [x] Multi-turn conversation history
- [x] Web UI with drag-and-drop upload
- [x] FastAPI REST server with full CORS support
- [ ] Query decomposition for complex multi-hop questions
- [ ] Reranking layer (cross-encoder)
- [ ] Metadata filters (date range, document source)
- [ ] Export conversation as PDF/Markdown
- [ ] Docker containerization
- [ ] Batch document processing API

---

## 🤝 Contributing

Contributions are welcome! Here's how:

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
# Open a Pull Request
```

Please keep PRs focused and include a short description of what changed and why.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Vectorless-RAG-Document-QA**

Designed and developed by **P Ramesh**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/p-ramesh-477482304)
[![GitHub](https://img.shields.io/badge/GitHub-Ramesh143--code-black?logo=github)](https://github.com/Ramesh143-code)

</div>
