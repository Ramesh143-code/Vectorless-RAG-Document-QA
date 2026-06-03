"""
retriever.py
─────────────────────────────────────────────────────────────────────────────
Vectorless RAG — BM25 + TF-IDF Retriever

This is the core "vectorless" part. Instead of embedding chunks into a
vector space, we use two classic IR algorithms:

  1. BM25   (rank-bm25)   — Best Match 25, the gold standard for keyword
                            search. Scores chunks by term frequency,
                            inverse document frequency, and document length.

  2. TF-IDF (scikit-learn) — Fallback / re-ranking layer. Catches cases
                            where BM25 misses due to exact-match reliance.

  3. Structural Boost     — Chunks from headings matching the query get
                            a score boost. Tables also get a type boost.

  4. Hybrid Fusion        — Final score = BM25_score * w1 + TFIDF_score * w2
                            + structural_boost

Why this works on structured PDFs:
  Structured PDFs (reports, manuals, forms) use consistent vocabulary.
  The terms in a user's question almost always appear verbatim in the
  relevant section. BM25 handles this better than dense retrieval which
  needs large training data to generalize.
─────────────────────────────────────────────────────────────────────────────
"""

import math
import os
import re
import string
from dataclasses import dataclass
from typing import Optional

import nltk
from dotenv import load_dotenv
from loguru import logger
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chunker import Chunk, ChunkType

load_dotenv()

# ─── NLTK Setup ───────────────────────────────────────────────────────────────
# Download quietly on first run
for _resource in ("punkt", "stopwords", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))

# ─── Config ───────────────────────────────────────────────────────────────────

TOP_K          = int(os.getenv("TOP_K_RESULTS", 5))
BM25_WEIGHT    = 0.65    # BM25 contribution to hybrid score
TFIDF_WEIGHT   = 0.35    # TF-IDF contribution to hybrid score
TABLE_BOOST    = 0.10    # Extra score for table chunks (structured data)
HEADING_BOOST  = 0.15    # Extra score when query term appears in heading trail


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    A single retrieved chunk with its scores.

    Attributes:
        chunk          : The matched Chunk object.
        bm25_score     : Raw BM25 score (not normalised).
        tfidf_score    : Cosine similarity from TF-IDF (0-1).
        structural_boost: Extra score from heading/table matching.
        final_score    : Weighted hybrid score used for ranking.
        rank           : Position in final result list (1-based).
    """
    chunk           : Chunk
    bm25_score      : float
    tfidf_score     : float
    structural_boost: float
    final_score     : float
    rank            : int = 0

    def __repr__(self) -> str:
        return (
            f"[{self.rank}] {self.chunk.chunk_type.value} "
            f"p.{self.chunk.page_start} | score={self.final_score:.4f} | "
            f"{self.chunk.text[:60]}…"
        )


# ─── Retriever ────────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    Hybrid BM25 + TF-IDF retriever over a corpus of Chunks.

    Usage:
        retriever = BM25Retriever()
        retriever.index(chunks)                  # build index once
        results = retriever.retrieve("query")    # search anytime
    """

    def __init__(
        self,
        top_k        : int   = TOP_K,
        bm25_weight  : float = BM25_WEIGHT,
        tfidf_weight : float = TFIDF_WEIGHT,
    ):
        self.top_k       = top_k
        self.bm25_weight = bm25_weight
        self.tfidf_weight= tfidf_weight

        self._chunks     : list[Chunk]  = []
        self._bm25       : Optional[BM25Okapi] = None
        self._tfidf_vec  : Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None          # sparse matrix (n_chunks × vocab)
        self._tokenized_corpus: list[list[str]] = []
        self._is_indexed  = False

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index(self, chunks: list[Chunk]) -> None:
        """
        Build BM25 and TF-IDF indexes from a list of Chunks.

        Call this once after chunking. Re-call to rebuild with new chunks.

        Args:
            chunks: Output from StructuredChunker.chunk_documents()
        """
        if not chunks:
            raise ValueError("Cannot index an empty chunk list.")

        self._chunks = chunks
        texts = [c.text for c in chunks]

        logger.info(f"  📇 Indexing {len(chunks)} chunks …")

        # Tokenize for BM25
        self._tokenized_corpus = [self._tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        # TF-IDF matrix
        self._tfidf_vec = TfidfVectorizer(
            tokenizer       = self._tokenize,
            token_pattern   = None,
            ngram_range     = (1, 2),    # unigrams + bigrams
            max_df          = 0.95,      # ignore terms in >95% of chunks
            min_df          = 1,
            sublinear_tf    = True,      # log-scale TF
        )
        self._tfidf_matrix = self._tfidf_vec.fit_transform(texts)

        self._is_indexed = True
        logger.success(
            f"✅ Index ready: {len(chunks)} chunks | "
            f"vocab size: {len(self._tfidf_vec.vocabulary_)}"
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query  : str,
        top_k  : Optional[int] = None,
        filter_source : Optional[str] = None,
        filter_type   : Optional[ChunkType] = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query         : Natural language question or keyword string.
            top_k         : Override default TOP_K for this query.
            filter_source : If set, only return chunks from this filename.
            filter_type   : If set, only return chunks of this ChunkType.

        Returns:
            List of RetrievalResult sorted by final_score descending.
        """
        if not self._is_indexed:
            raise RuntimeError("Call .index(chunks) before .retrieve().")

        k = top_k or self.top_k
        query_tokens = self._tokenize(query)

        if not query_tokens:
            logger.warning("Query tokenized to empty list — returning nothing.")
            return []

        # ── BM25 scores ───────────────────────────────────────────────────────
        bm25_raw = self._bm25.get_scores(query_tokens)
        bm25_norm = self._normalize(bm25_raw)

        # ── TF-IDF scores ─────────────────────────────────────────────────────
        query_vec    = self._tfidf_vec.transform([" ".join(query_tokens)])
        tfidf_scores = cosine_similarity(query_vec, self._tfidf_matrix)[0]

        # ── Structural boosts ─────────────────────────────────────────────────
        structural = self._structural_boosts(query_tokens)

        # ── Hybrid fusion ─────────────────────────────────────────────────────
        results: list[RetrievalResult] = []

        for i, chunk in enumerate(self._chunks):
            # Apply source/type filters
            if filter_source and chunk.source_file != filter_source:
                continue
            if filter_type and chunk.chunk_type != filter_type:
                continue

            final = (
                self.bm25_weight  * bm25_norm[i]    +
                self.tfidf_weight * float(tfidf_scores[i]) +
                structural[i]
            )

            results.append(RetrievalResult(
                chunk            = chunk,
                bm25_score       = float(bm25_raw[i]),
                tfidf_score      = float(tfidf_scores[i]),
                structural_boost = structural[i],
                final_score      = final,
            ))

        # Sort and assign ranks
        results.sort(key=lambda r: r.final_score, reverse=True)
        top_results = results[:k]
        for rank, res in enumerate(top_results, start=1):
            res.rank = rank

        return top_results

    def retrieve_multi_query(
        self,
        queries: list[str],
        top_k  : Optional[int] = None,
    ) -> list[RetrievalResult]:
        """
        Run multiple queries and fuse results by max score (score fusion).

        Useful for rephrased or decomposed questions.

        Args:
            queries: List of query strings.
            top_k  : Number of final results to return.

        Returns:
            Deduplicated, re-ranked list of RetrievalResult.
        """
        k = top_k or self.top_k
        seen: dict[str, RetrievalResult] = {}   # chunk_id → best result

        for q in queries:
            for result in self.retrieve(q, top_k=k * 2):
                cid = result.chunk.chunk_id
                if cid not in seen or result.final_score > seen[cid].final_score:
                    seen[cid] = result

        merged = sorted(seen.values(), key=lambda r: r.final_score, reverse=True)
        for rank, res in enumerate(merged[:k], start=1):
            res.rank = rank

        return merged[:k]

    # ── Structural Boosts ─────────────────────────────────────────────────────

    def _structural_boosts(self, query_tokens: list[str]) -> list[float]:
        """
        Compute a boost for each chunk based on:
          - Whether query terms appear in the chunk's heading trail
          - Whether the chunk is a TABLE (structured data preference)
        """
        query_set = set(query_tokens)
        boosts = []

        for chunk in self._chunks:
            boost = 0.0

            # Heading trail match
            trail_tokens = set(
                self._tokenize(" ".join(chunk.heading_trail))
            )
            overlap = len(query_set & trail_tokens)
            if overlap > 0:
                boost += HEADING_BOOST * min(overlap, 3) / 3   # cap at 3 matches

            # Table type boost
            if chunk.chunk_type == ChunkType.TABLE:
                boost += TABLE_BOOST

            boosts.append(boost)

        return boosts

    # ── Text Processing ───────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize and normalise text for BM25 / TF-IDF.

        Steps:
          1. Lowercase
          2. Remove punctuation
          3. NLTK word tokenize
          4. Remove stopwords and single characters
          5. Return token list
        """
        text = text.lower()
        text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
        tokens = word_tokenize(text)
        tokens = [
            t for t in tokens
            if t not in STOPWORDS and len(t) > 1 and not t.isnumeric()
        ]
        return tokens

    @staticmethod
    def _normalize(scores) -> list[float]:
        """Min-max normalise a score array to [0, 1]."""
        min_s = min(scores)
        max_s = max(scores)
        rng   = max_s - min_s
        if rng == 0:
            return [0.0] * len(scores)
        return [(s - min_s) / rng for s in scores]

    # ── Index Info ────────────────────────────────────────────────────────────

    @property
    def is_indexed(self) -> bool:
        return self._is_indexed

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Look up a chunk by its ID."""
        for c in self._chunks:
            if c.chunk_id == chunk_id:
                return c
        return None


# ─── Utility ─────────────────────────────────────────────────────────────────

def print_retrieval_results(results: list[RetrievalResult], query: str) -> None:
    """Pretty-print retrieval results to the console."""
    print("\n" + "═" * 65)
    print(f"  🔍 Query: {query}")
    print(f"  📦 Top {len(results)} results")
    print("═" * 65)

    for res in results:
        c = res.chunk
        print(f"\n  [{res.rank}] Score: {res.final_score:.4f}  "
              f"(bm25={res.bm25_score:.3f} | tfidf={res.tfidf_score:.3f} | "
              f"boost={res.structural_boost:.3f})")
        print(f"      Type   : {c.chunk_type.value}")
        print(f"      Source : {c.source_file}  p.{c.page_start}–{c.page_end}")
        print(f"      Context: {c.heading_context or '(none)'}")
        print(f"      Text   : {c.text[:200].replace(chr(10), ' ')}…")

    print("\n" + "═" * 65 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from pdf_parser import PDFParser
    from chunker import StructuredChunker

    if len(sys.argv) < 3:
        print("Usage: python retriever.py <pdf_path> '<query>'")
        print("Example: python retriever.py data/report.pdf 'What is the revenue?'")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    query    = sys.argv[2]

    # Parse → Chunk → Index → Retrieve
    parser    = PDFParser()
    chunker   = StructuredChunker()
    retriever = BM25Retriever()

    doc     = parser.parse(pdf_path)
    chunks  = chunker.chunk_document(doc)
    retriever.index(chunks)

    results = retriever.retrieve(query)
    print_retrieval_results(results, query)
