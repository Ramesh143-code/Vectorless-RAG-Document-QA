"""
rag_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Vectorless RAG — End-to-End Pipeline (OpenAI version)

Ties together all components:
    PDF Parser → Chunker → Retriever → OpenAI GPT (Generator)

Flow for a single query:
    1. (Once) Parse all PDFs in data/ into ParsedDocuments
    2. (Once) Chunk all documents → flat list of Chunks
    3. (Once) Index chunks in BM25Retriever
    4. (Per query) Retrieve top-K relevant chunks
    5. (Per query) Build a context-aware prompt with heading breadcrumbs
    6. (Per query) Send to OpenAI → get answer
    7. Return RAGResponse with answer + source citations

Supports:
    - Single-turn Q&A
    - Multi-turn conversation (maintains message history)
    - Multi-query fusion (rephrase + retrieve for better recall)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

from chunker import Chunk, StructuredChunker
from pdf_parser import PDFParser
from retriever import BM25Retriever, RetrievalResult, print_retrieval_results

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PDF_INPUT_DIR  = os.getenv("PDF_INPUT_DIR", "data/")
TOP_K          = int(os.getenv("TOP_K_RESULTS", 5))
MODEL          = os.getenv("OPENAI_MODEL", "gpt-4o-mini")   # override in .env
MAX_TOKENS     = 1024

# ─── Prompt Templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise document assistant for structured PDF documents.

Your job:
- Answer questions ONLY using the provided context chunks.
- Each chunk includes its source file, page number, section heading trail, and content.
- Always cite your sources using the format: [FileName, p.N] or [FileName, p.N-M].
- If the context does not contain enough information to answer, say so clearly.
- Do not fabricate information or use outside knowledge.
- If a table is provided in context, interpret it accurately.
- Be concise and structured in your answers."""

CONTEXT_TEMPLATE = """--- CONTEXT CHUNK {rank} ---
Source : {source}
Pages  : {pages}
Section: {section}
Type   : {chunk_type}

{text}
"""

QUERY_TEMPLATE = """Based on the context chunks above, answer the following question:

Question: {query}

Instructions:
- Answer directly and concisely.
- Cite sources as [FileName, p.N] at the end of relevant statements.
- If the answer spans multiple chunks, synthesize clearly.
- If the context is insufficient, say: "The provided documents do not contain enough information to answer this question."
"""


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class Citation:
    """A single source citation in an answer."""
    source_file : str
    page_start  : int
    page_end    : int
    chunk_id    : str
    heading     : str

    def __str__(self) -> str:
        pages = (f"p.{self.page_start}-{self.page_end}"
                 if self.page_start != self.page_end
                 else f"p.{self.page_start}")
        return f"[{self.source_file}, {pages}]"


@dataclass
class RAGResponse:
    """
    Full response from the RAG pipeline.

    Attributes:
        query       : The original question asked.
        answer      : GPT's generated answer.
        citations   : List of Citation objects for sources used.
        chunks_used : The actual Chunk objects retrieved.
        latency_ms  : End-to-end latency in milliseconds.
        model       : Model used for generation.
    """
    query      : str
    answer     : str
    citations  : list[Citation]
    chunks_used: list[RetrievalResult]
    latency_ms : float
    model      : str = MODEL

    def __str__(self) -> str:
        lines = [
            f"\n{'═'*60}",
            f"  Q: {self.query}",
            f"{'─'*60}",
            f"  A: {self.answer}",
            f"{'─'*60}",
            f"  Sources used ({len(self.citations)}):",
        ]
        for c in self.citations:
            lines.append(f"    • {c.source_file}  p.{c.page_start}  [{c.heading or 'no section'}]")
        lines.append(f"  Latency: {self.latency_ms:.0f}ms  |  Model: {self.model}")
        lines.append("═"*60)
        return "\n".join(lines)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class VectorlessRAGPipeline:
    """
    End-to-end Vectorless RAG pipeline using OpenAI.

    Lifecycle:
        pipeline = VectorlessRAGPipeline()
        pipeline.build()                              # parse + chunk + index
        answer = pipeline.ask("What is the revenue?")
    """

    def __init__(
        self,
        pdf_dir        : str | Path = PDF_INPUT_DIR,
        top_k          : int        = TOP_K,
        use_multi_query: bool       = False,
    ):
        self.pdf_dir         = Path(pdf_dir)
        self.top_k           = top_k
        self.use_multi_query = use_multi_query

        self._parser    = PDFParser()
        self._chunker   = StructuredChunker()
        self._retriever = BM25Retriever(top_k=top_k)
        self._client    = OpenAI(api_key=OPENAI_API_KEY)

        self._chunks  : list[Chunk] = []
        self._history : list[dict]  = []   # multi-turn message history
        self._is_built = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, pdf_paths: Optional[list[Path]] = None) -> None:
        """
        Parse, chunk, and index PDFs.

        Args:
            pdf_paths: Explicit list of PDF paths. If None, scans self.pdf_dir.
        """
        logger.info("🔨 Building RAG pipeline …")

        if pdf_paths:
            docs = [self._parser.parse(p) for p in pdf_paths]
        else:
            docs = self._parser.parse_directory(self.pdf_dir)

        if not docs:
            raise ValueError(f"No PDFs found in '{self.pdf_dir}'. Add PDFs to data/ first.")

        self._chunks = self._chunker.chunk_documents(docs)
        self._retriever.index(self._chunks)

        self._is_built = True
        logger.success(
            f"✅ Pipeline ready: {len(docs)} PDF(s), "
            f"{len(self._chunks)} chunks indexed."
        )

    def rebuild(self) -> None:
        """Re-parse and re-index all PDFs (use after adding new files)."""
        self._is_built = False
        self._history  = []
        self.build()

    # ── Ask ───────────────────────────────────────────────────────────────────

    def ask(
        self,
        query           : str,
        top_k           : Optional[int] = None,
        maintain_history: bool          = True,
        show_retrieval  : bool          = False,
    ) -> RAGResponse:
        """
        Ask a question against the indexed documents.

        Args:
            query           : Natural language question.
            top_k           : Override default top-K for this query.
            maintain_history: If True, includes prior Q&A in context.
            show_retrieval  : If True, prints retrieved chunks to console.

        Returns:
            RAGResponse with answer, citations, and metadata.
        """
        if not self._is_built:
            raise RuntimeError("Call .build() before .ask().")

        start = time.time()
        k = top_k or self.top_k

        # ── Retrieve ──────────────────────────────────────────────────────────
        if self.use_multi_query:
            paraphrases = self._generate_query_variants(query)
            results = self._retriever.retrieve_multi_query(paraphrases, top_k=k)
        else:
            results = self._retriever.retrieve(query, top_k=k)

        if show_retrieval:
            print_retrieval_results(results, query)

        # ── Build Prompt ──────────────────────────────────────────────────────
        context_block = self._build_context(results)
        user_message  = context_block + "\n" + QUERY_TEMPLATE.format(query=query)

        # ── Call OpenAI ───────────────────────────────────────────────────────
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + self._history
            + [{"role": "user", "content": user_message}]
        )

        response = self._client.chat.completions.create(
            model      = MODEL,
            max_tokens = MAX_TOKENS,
            messages   = messages,
        )

        answer = response.choices[0].message.content.strip()

        # ── Update History ────────────────────────────────────────────────────
        if maintain_history:
            self._history.append({"role": "user",      "content": user_message})
            self._history.append({"role": "assistant",  "content": answer})

        citations  = self._extract_citations(results)
        latency_ms = (time.time() - start) * 1000

        return RAGResponse(
            query       = query,
            answer      = answer,
            citations   = citations,
            chunks_used = results,
            latency_ms  = latency_ms,
            model       = MODEL,
        )

    def clear_history(self) -> None:
        """Reset conversation history for a fresh session."""
        self._history = []
        logger.info("Conversation history cleared.")

    # ── Context Builder ───────────────────────────────────────────────────────

    def _build_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieved chunks into a structured context block."""
        blocks = []
        for res in results:
            c = res.chunk
            pages = (f"{c.page_start}-{c.page_end}"
                     if c.page_start != c.page_end
                     else str(c.page_start))
            block = CONTEXT_TEMPLATE.format(
                rank       = res.rank,
                source     = c.source_file,
                pages      = pages,
                section    = c.heading_context or "(no section)",
                chunk_type = c.chunk_type.value,
                text       = c.text,
            )
            blocks.append(block)
        return "\n".join(blocks)

    # ── Citations ─────────────────────────────────────────────────────────────

    def _extract_citations(self, results: list[RetrievalResult]) -> list[Citation]:
        seen = set()
        citations = []
        for res in results:
            c = res.chunk
            key = (c.source_file, c.page_start)
            if key not in seen:
                seen.add(key)
                citations.append(Citation(
                    source_file = c.source_file,
                    page_start  = c.page_start,
                    page_end    = c.page_end,
                    chunk_id    = c.chunk_id,
                    heading     = c.heading_context,
                ))
        return citations

    # ── Multi-Query Expansion ─────────────────────────────────────────────────

    def _generate_query_variants(self, query: str) -> list[str]:
        """Ask GPT to rephrase the query 2 ways for broader retrieval."""
        prompt = (
            f"Rephrase the following question in 2 alternative ways "
            f"that preserve the same meaning. Return only the 2 rephrased "
            f"questions, one per line, no numbering:\n\n{query}"
        )
        try:
            resp = self._client.chat.completions.create(
                model      = MODEL,
                max_tokens = 150,
                messages   = [{"role": "user", "content": prompt}],
            )
            lines    = resp.choices[0].message.content.strip().split("\n")
            variants = [query] + [l.strip() for l in lines if l.strip()]
            return variants[:3]
        except Exception:
            return [query]

    # ── Inspection ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._is_built

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def get_index_stats(self) -> dict:
        if not self._is_built:
            return {"status": "not built"}
        sources = list({c.source_file for c in self._chunks})
        types   = {}
        for c in self._chunks:
            types[c.chunk_type.value] = types.get(c.chunk_type.value, 0) + 1
        return {
            "status"        : "ready",
            "total_chunks"  : len(self._chunks),
            "sources"       : sources,
            "chunk_types"   : types,
            "history_turns" : len(self._history) // 2,
        }


# ─── Interactive CLI ──────────────────────────────────────────────────────────

def run_interactive(pipeline: VectorlessRAGPipeline) -> None:
    """Run an interactive Q&A session in the terminal."""
    print("\n" + "═"*60)
    print("  🤖  Vectorless RAG — Interactive Mode (OpenAI)")
    print("  Type your question. Commands: :quit :clear :stats :help")
    print("═"*60 + "\n")

    while True:
        try:
            query = input("  ❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!\n")
            break

        if not query:
            continue

        if query.lower() in (":quit", ":q", "exit", "quit"):
            print("\n  Goodbye!\n")
            break
        elif query.lower() == ":clear":
            pipeline.clear_history()
            print("  ✅ History cleared.\n")
            continue
        elif query.lower() == ":stats":
            stats = pipeline.get_index_stats()
            print(f"\n  📊 Index stats:")
            for k, v in stats.items():
                print(f"     {k}: {v}")
            print()
            continue
        elif query.lower() == ":help":
            print("\n  Commands:")
            print("    :clear  — clear conversation history")
            print("    :stats  — show index stats")
            print("    :quit   — exit\n")
            continue

        try:
            response = pipeline.ask(query, show_retrieval=False)
            print(response)
        except Exception as e:
            logger.error(f"Error during query: {e}")
            print(f"\n  ⚠ Error: {e}\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    pipeline = VectorlessRAGPipeline(
        pdf_dir        = PDF_INPUT_DIR,
        top_k          = TOP_K,
        use_multi_query= False,
    )

    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        pipeline.build(pdf_paths=[pdf_path])
    else:
        pipeline.build()

    if len(sys.argv) > 2:
        query    = sys.argv[2]
        response = pipeline.ask(query)
        print(response)
    else:
        run_interactive(pipeline)
