<div align="center">

```
██╗   ██╗███████╗ ██████╗████████╗ ██████╗ ██████╗ ██╗     ███████╗███████╗
██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██║     ██╔════╝██╔════╝
██║   ██║█████╗  ██║        ██║   ██║   ██║██████╔╝██║     █████╗  ███████╗███████╗
╚██╗ ██╔╝██╔══╝  ██║        ██║   ██║   ██║██╔══██╗██║     ██╔══╝  ╚════██║╚════██║
 ╚████╔╝ ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║███████╗███████╗██████╔╝███████╝
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

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=24&pause=1500&color=22C55E&center=true&vCenter=true&width=950&lines=Vectorless+RAG+for+Structured+Documents;PDF+Parsing+%26+Chunking;BM25+%2B+TF-IDF+Retrieval;Lightning+Fast+Q%26A" alt="animated title">

</div>

---

## 📸 Demo & Screenshots

<div align="center">

| Upload & Index | Ask Questions | See Sources |
|:-:|:-:|:-:|
| Drop any PDF — scanned or digital | Natural language Q&A with full history | Every answer cites exact page + section |

</div>

### Step-by-Step Usage Guide

#### **Step 1: Upload & Prepare Documents**
Upload your PDF documents using the intuitive drag-and-drop interface. The system automatically handles both digital and scanned PDFs through OCR.

```
[Vectorless RAG UI - Upload Screen]
├── Left Panel: Document Management
│   ├── Drop PDFs here (or click to browse)
│   ├── List of uploaded documents with status indicators
│   └── File size and chunk count display
└── Right Panel: Main Chat Area
    ├── Status: "Drop PDFs here or click to browse"
    └── Ready for document upload
```

#### **Step 2: Index Your Documents**
Click the **"Index Ready"** button to process all uploaded PDFs. The system performs:
- Structure extraction (headings, tables, sections)
- BM25 + TF-IDF index building
- Heading trail context creation for citations

**Result:** `Index built – 1 doc, 24 chunks ready.`

#### **Step 3: Ask Natural Language Questions**
Type your questions in plain English. The retrieval engine handles context from document structure automatically.

**Example Query:** *"What is problem statement of the project"*

**Retrieved Response:**
```
According to [Source: project report final XAI.pdf | Chunk 4], 
the problem statement of the project is:

"Although machine learning techniques have been widely used for predicting 
heart disease risk, many existing models operate as black-box systems that 
do not provide clear explanations for their predictions. In healthcare 
applications, lack of transparency can reduce trust among medical professionals 
and limit the practical adoption of such systems. Therefore, there is a need 
for a predictive system that not only estimates the risk of heart disease 
accurately but also explains the reasoning behind each prediction."

Retrieved Chunks (with BM25 scores):
├── project report final XAI.pdf · chunk_2     BM25: 5.5601
├── project report final XAI.pdf · chunk_4     BM25: 4.6484
├── project report final XAI.pdf · chunk_13    BM25: 4.6202
├── project report final XAI.pdf · chunk_14    BM25: 3.1692
└── project report final XAI.pdf · chunk_9     BM25: 1.9571
```

#### **Step 4: View Evaluation Metrics**
Click **"Evaluation Metrics"** to see real-time performance analytics:

```
┌─────────────────────────────────────────┐
│         EVALUATION DASHBOARD             │
├─────────────────────────────────────────┤
│  84%  Average Score                     │
│   1   Query Count                       │
│ 2956ms Average Latency                  │
│                                         │
│  QUERY: "what is problem statement..." │
│  Score: 84%                             │
│  ├─ Faithfulness:        98%  ████████ │
│  ├─ Answer Relevance:    99%  ████████ │
│  ├─ Context Precision:   99%  ████████ │
│  ├─ Context Recall:      61%  █████    │
│  ├─ Chunk Diversity:     95%  ████████ │
│  └─ Latency Score:       56%  ████     │
│  Rating: ⭐⭐⭐⭐☆                       │
└─────────────────────────────────────────┘

QUERY HISTORY:
├── "what is problem statement..." (84%)
└── "how this project is useful"   (83%)
```

---

## 📊 Key Performance Indicators

<div align="">

### Real-World Metrics (from screenshots):
| Metric | Value | Status |
|--------|-------|--------|
| **Average Score** | 84% | ✅ Excellent |
| **Faithfulness** | 98% | ✅ High accuracy |
| **Answer Relevance** | 99% | ✅ Highly relevant |
| **Context Precision** | 99% | ✅ Exact matches |
| **Context Recall** | 61% | ⚠️ Room for improvement |
| **Chunk Diversity** | 95% | ✅ Well-distributed |
| **Average Latency** | 2956ms | ⚠️ Can be optimized |
| **Query Count** | 1-2 | 📊 Growing |

</div>

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
┌─────────────────────────────────────────────────────────────────┐
│                        VECTORLESS RAG PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
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
│   ┌─────────────────────────────────────────────���┐                 │
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
└─────────────────────────────────────────────────────────────────┘
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
- **Evaluation metrics** — track faithfulness, relevance, precision, recall in real-time

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

## 🚀 Planned Improvements & Enhancements

### Phase 1: Performance Optimization (High Priority)
- **Latency Reduction** — Target: < 1500ms average response time
  - Implement query caching layer
  - Optimize TF-IDF computation with sparse matrices
  - Add background indexing for large PDFs
  
- **Context Recall Enhancement** — Target: > 80% recall
  - Implement hybrid query expansion using synonyms
  - Add semantic keyword extraction pre-processing
  - Multi-pass retrieval strategy for complex queries

### Phase 2: Advanced Features (Medium Priority)
- [ ] **Query Decomposition** — Break multi-hop questions into retrievable sub-queries
- [ ] **Reranking Layer** — Cross-encoder reranking for top-5 results
- [ ] **Metadata Filters** — Date range, document source, section type filtering
- [ ] **Export Capabilities** — Save conversations as PDF/Markdown with proper citations

### Phase 3: Production Readiness (Medium Priority)
- [ ] **Docker Containerization** — One-command deployment
- [ ] **Batch Processing API** — `/batch/ask` for bulk questions
- [ ] **Caching Strategy** — Redis integration for query result caching
- [ ] **Rate Limiting** — API throttling and quota management

### Phase 4: Enterprise Features (Lower Priority)
- [ ] **Document Versioning** — Track PDF updates and maintain version history
- [ ] **User Feedback Loop** — Rating system for answer quality
- [ ] **Analytics Dashboard** — Detailed query analytics and insights
- [ ] **Multi-language Support** — Question answering in multiple languages

---

## 🗺️ Roadmap

- [x] BM25 + TF-IDF hybrid retrieval
- [x] Structure-aware chunking (headings, tables, paragraphs)
- [x] OCR support for scanned PDFs
- [x] Multi-document indexing
- [x] Multi-turn conversation history
- [x] Web UI with drag-and-drop upload
- [x] FastAPI REST server with full CORS support
- [x] Real-time evaluation metrics dashboard
- [ ] Query decomposition for complex multi-hop questions
- [ ] Reranking layer (cross-encoder)
- [ ] Metadata filters (date range, document source)
- [ ] Export conversation as PDF/Markdown
- [ ] Docker containerization
- [ ] Batch document processing API
- [ ] Latency optimization (target < 1.5s)
- [ ] Context recall enhancement (target > 80%)

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
