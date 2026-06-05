"""
evaluator.py
─────────────────────────────────────────────────────────────────────────────
Vectorless RAG — Evaluation Engine

Computes 6 metrics for every query automatically:

  1. Faithfulness       — Is the answer grounded in the retrieved chunks?
                          (answer tokens that appear in context / total tokens)

  2. Answer Relevance   — Does the answer actually address the query?
                          (BM25-style overlap between query and answer)

  3. Context Precision  — Are the retrieved chunks relevant to the query?
                          (query-chunk overlap, weighted by rank)

  4. Context Recall     — Do the chunks cover the answer content?
                          (answer tokens found in chunks / total answer tokens)

  5. Chunk Diversity    — Are we pulling from varied sources/sections?
                          (unique pages + sources / total chunks)

  6. Latency Score      — How fast was the response?
                          (1.0 = <500ms, scaled down to 0.0 at >5000ms)

Overall Score = weighted average of all 6 metrics (0–100)
─────────────────────────────────────────────────────────────────────────────
"""

import re
import string
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class MetricScore:
    """A single evaluation metric with score and explanation."""
    name       : str
    score      : float          # 0.0 – 1.0
    explanation: str
    icon       : str = "📊"

    @property
    def percent(self) -> int:
        return int(self.score * 100)

    @property
    def grade(self) -> str:
        if self.score >= 0.85: return "Excellent"
        if self.score >= 0.70: return "Good"
        if self.score >= 0.50: return "Fair"
        return "Poor"

    @property
    def color(self) -> str:
        if self.score >= 0.85: return "#22c55e"   # green
        if self.score >= 0.70: return "#84cc16"   # lime
        if self.score >= 0.50: return "#f59e0b"   # amber
        return "#ef4444"                           # red


@dataclass
class EvaluationResult:
    """
    Full evaluation result for one query-answer pair.
    """
    query              : str
    answer             : str
    metrics            : list[MetricScore]
    overall_score      : float          # 0.0 – 1.0
    latency_ms         : float
    chunk_count        : int
    timestamp          : float = field(default_factory=time.time)

    @property
    def overall_percent(self) -> int:
        return int(self.overall_score * 100)

    @property
    def overall_grade(self) -> str:
        if self.overall_score >= 0.85: return "Excellent"
        if self.overall_score >= 0.70: return "Good"
        if self.overall_score >= 0.50: return "Fair"
        return "Poor"

    @property
    def overall_color(self) -> str:
        if self.overall_score >= 0.85: return "#22c55e"
        if self.overall_score >= 0.70: return "#84cc16"
        if self.overall_score >= 0.50: return "#f59e0b"
        return "#ef4444"

    def to_dict(self) -> dict:
        return {
            "query"          : self.query,
            "answer_preview" : self.answer[:120] + ("…" if len(self.answer) > 120 else ""),
            "overall_score"  : round(self.overall_score, 4),
            "overall_percent": self.overall_percent,
            "overall_grade"  : self.overall_grade,
            "overall_color"  : self.overall_color,
            "latency_ms"     : round(self.latency_ms, 1),
            "chunk_count"    : self.chunk_count,
            "timestamp"      : self.timestamp,
            "metrics"        : [
                {
                    "name"       : m.name,
                    "score"      : round(m.score, 4),
                    "percent"    : m.percent,
                    "grade"      : m.grade,
                    "color"      : m.color,
                    "explanation": m.explanation,
                    "icon"       : m.icon,
                }
                for m in self.metrics
            ],
        }


# ─── Evaluator ────────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Computes evaluation metrics for every RAG query automatically.
    No ground truth needed — all metrics are reference-free.

    Usage:
        evaluator = RAGEvaluator()
        result = evaluator.evaluate(
            query    = "What is the revenue?",
            answer   = "The revenue is $1.2M ...",
            chunks   = retrieval_results,
            latency_ms = 843.0,
        )
    """

    # Metric weights for overall score
    WEIGHTS = {
        "Faithfulness"      : 0.25,
        "Answer Relevance"  : 0.25,
        "Context Precision" : 0.20,
        "Context Recall"    : 0.15,
        "Chunk Diversity"   : 0.10,
        "Latency Score"     : 0.05,
    }

    def evaluate(
        self,
        query     : str,
        answer    : str,
        chunks    : list,           # list[RetrievalResult] or list[dict]
        latency_ms: float = 0.0,
    ) -> EvaluationResult:
        """
        Run all 6 metrics and return a full EvaluationResult.

        Args:
            query     : The original user question.
            answer    : The LLM-generated answer.
            chunks    : Retrieved chunks (RetrievalResult objects or dicts).
            latency_ms: End-to-end latency in milliseconds.

        Returns:
            EvaluationResult with all metrics and overall score.
        """
        # Normalise chunks to text list + metadata
        chunk_texts, chunk_meta = self._normalise_chunks(chunks)

        # Tokenise all inputs
        q_tokens = self._tokenise(query)
        a_tokens = self._tokenise(answer)
        c_tokens = [self._tokenise(t) for t in chunk_texts]
        all_context_tokens = [t for tokens in c_tokens for t in tokens]

        # Compute all 6 metrics
        faithfulness    = self._faithfulness(a_tokens, all_context_tokens)
        answer_rel      = self._answer_relevance(q_tokens, a_tokens)
        ctx_precision   = self._context_precision(q_tokens, c_tokens)
        ctx_recall      = self._context_recall(a_tokens, c_tokens)
        diversity       = self._chunk_diversity(chunk_meta)
        latency_score   = self._latency_score(latency_ms)

        metrics = [faithfulness, answer_rel, ctx_precision, ctx_recall, diversity, latency_score]

        # Weighted overall score
        overall = sum(
            self.WEIGHTS.get(m.name, 0) * m.score
            for m in metrics
        )

        return EvaluationResult(
            query         = query,
            answer        = answer,
            metrics       = metrics,
            overall_score = min(overall, 1.0),
            latency_ms    = latency_ms,
            chunk_count   = len(chunks),
        )

    # ── Metric 1: Faithfulness ────────────────────────────────────────────────

    def _faithfulness(
        self, answer_tokens: list[str], context_tokens: list[str]
    ) -> MetricScore:
        """
        How grounded is the answer in the retrieved context?
        = unique answer tokens that appear in context / unique answer tokens
        """
        if not answer_tokens:
            return MetricScore("Faithfulness", 0.0, "No answer generated.", "🎯")

        context_set = set(context_tokens)
        answer_set  = set(answer_tokens)

        if not answer_set:
            return MetricScore("Faithfulness", 0.0, "Empty answer.", "🎯")

        grounded = answer_set & context_set
        score    = len(grounded) / len(answer_set)

        if score >= 0.85:
            explanation = "Answer is strongly grounded in retrieved context."
        elif score >= 0.60:
            explanation = "Most answer content found in context."
        elif score >= 0.40:
            explanation = "Partial grounding — some content may be hallucinated."
        else:
            explanation = "Low grounding — answer may not reflect the documents."

        return MetricScore("Faithfulness", score, explanation, "🎯")

    # ── Metric 2: Answer Relevance ────────────────────────────────────────────

    def _answer_relevance(
        self, query_tokens: list[str], answer_tokens: list[str]
    ) -> MetricScore:
        """
        Does the answer actually address the question?
        = Jaccard similarity between query and answer token sets,
          boosted by query term coverage.
        """
        if not query_tokens or not answer_tokens:
            return MetricScore("Answer Relevance", 0.0, "Empty query or answer.", "💬")

        q_set = set(query_tokens)
        a_set = set(answer_tokens)

        # Jaccard
        intersection = q_set & a_set
        union        = q_set | a_set
        jaccard      = len(intersection) / len(union) if union else 0.0

        # Query coverage: how many query terms appear in answer
        coverage = len(q_set & a_set) / len(q_set) if q_set else 0.0

        score = 0.4 * jaccard + 0.6 * coverage

        if score >= 0.5:
            explanation = "Answer directly addresses the query."
        elif score >= 0.3:
            explanation = "Answer partially addresses the query."
        else:
            explanation = "Answer may not be directly relevant to the query."

        return MetricScore("Answer Relevance", min(score * 1.5, 1.0), explanation, "💬")

    # ── Metric 3: Context Precision ───────────────────────────────────────────

    def _context_precision(
        self, query_tokens: list[str], chunk_token_lists: list[list[str]]
    ) -> MetricScore:
        """
        Are the retrieved chunks actually relevant to the query?
        Rank-weighted: top chunks matter more.
        = sum(rank_weight × chunk_relevance) / sum(rank_weights)
        """
        if not query_tokens or not chunk_token_lists:
            return MetricScore("Context Precision", 0.0, "No chunks retrieved.", "🔍")

        q_set = set(query_tokens)
        weighted_sum = 0.0
        weight_total = 0.0

        for rank, c_tokens in enumerate(chunk_token_lists, start=1):
            weight    = 1.0 / rank    # higher rank = higher weight
            c_set     = set(c_tokens)
            overlap   = len(q_set & c_set)
            relevance = overlap / len(q_set) if q_set else 0.0
            weighted_sum += weight * relevance
            weight_total += weight

        score = weighted_sum / weight_total if weight_total else 0.0

        if score >= 0.6:
            explanation = "Retrieved chunks are highly relevant to the query."
        elif score >= 0.35:
            explanation = "Most chunks are relevant; some noise present."
        else:
            explanation = "Chunks have low overlap with the query terms."

        return MetricScore("Context Precision", min(score * 1.8, 1.0), explanation, "🔍")

    # ── Metric 4: Context Recall ──────────────────────────────────────────────

    def _context_recall(
        self, answer_tokens: list[str], chunk_token_lists: list[list[str]]
    ) -> MetricScore:
        """
        Do the retrieved chunks cover the answer content?
        = answer tokens found in any chunk / total answer tokens
        """
        if not answer_tokens or not chunk_token_lists:
            return MetricScore("Context Recall", 0.0, "No answer or chunks.", "📚")

        all_ctx = set(t for tokens in chunk_token_lists for t in tokens)
        a_set   = set(answer_tokens)
        covered = a_set & all_ctx

        score = len(covered) / len(a_set) if a_set else 0.0

        if score >= 0.80:
            explanation = "Retrieved context covers most of the answer content."
        elif score >= 0.55:
            explanation = "Context covers the core answer; some gaps present."
        else:
            explanation = "Context may be missing key information for this answer."

        return MetricScore("Context Recall", score, explanation, "📚")

    # ── Metric 5: Chunk Diversity ─────────────────────────────────────────────

    def _chunk_diversity(self, chunk_meta: list[dict]) -> MetricScore:
        """
        Are we pulling from varied sections and pages?
        Prevents over-reliance on one part of the document.
        = (unique pages + unique sources) / (2 × total chunks)
        """
        if not chunk_meta:
            return MetricScore("Chunk Diversity", 0.0, "No chunks.", "🌐")

        unique_pages   = len({m.get("page", 0) for m in chunk_meta})
        unique_sources = len({m.get("source", "") for m in chunk_meta})
        total          = len(chunk_meta)

        # Normalise: max diversity = all chunks from different pages & sources
        page_diversity   = unique_pages / total if total else 0
        source_diversity = unique_sources / max(total, 1)

        score = 0.7 * page_diversity + 0.3 * source_diversity

        if score >= 0.75:
            explanation = f"Good diversity across {unique_pages} pages, {unique_sources} source(s)."
        elif score >= 0.4:
            explanation = f"Moderate diversity — {unique_pages} pages retrieved."
        else:
            explanation = "Chunks concentrated in one area of the document."

        return MetricScore("Chunk Diversity", min(score, 1.0), explanation, "🌐")

    # ── Metric 6: Latency Score ───────────────────────────────────────────────

    def _latency_score(self, latency_ms: float) -> MetricScore:
        """
        How fast was the response?
        1.0 = ≤500ms  |  0.0 = ≥5000ms  |  linear between
        """
        if latency_ms <= 500:
            score = 1.0
            explanation = f"Excellent speed ({latency_ms:.0f}ms)."
        elif latency_ms >= 5000:
            score = 0.0
            explanation = f"Slow response ({latency_ms:.0f}ms)."
        else:
            score = 1.0 - (latency_ms - 500) / 4500
            explanation = f"Response time: {latency_ms:.0f}ms."

        return MetricScore("Latency Score", score, explanation, "⚡")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _tokenise(self, text: str) -> list[str]:
        """Lowercase, remove punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
        tokens = text.split()
        # Remove very common stopwords inline (no NLTK dependency here)
        stops = {
            "the","a","an","is","are","was","were","be","been","being",
            "have","has","had","do","does","did","will","would","could",
            "should","may","might","shall","can","to","of","in","for",
            "on","with","at","by","from","as","it","its","this","that",
            "and","or","but","not","no","so","if","i","you","he","she",
            "we","they","what","which","who","when","where","how","why",
        }
        return [t for t in tokens if t not in stops and len(t) > 1]

    def _normalise_chunks(self, chunks: list) -> tuple[list[str], list[dict]]:
        """Accept either RetrievalResult objects or dicts."""
        texts = []
        meta  = []
        for c in chunks:
            if isinstance(c, dict):
                texts.append(c.get("text_preview", "") + " " + c.get("heading", ""))
                meta.append({"page": c.get("page", 0), "source": c.get("source", "")})
            else:
                # RetrievalResult object
                texts.append(c.chunk.text)
                meta.append({
                    "page"  : c.chunk.page_start,
                    "source": c.chunk.source_file,
                })
        return texts, meta


# ── Session History ────────────────────────────────────────────────────────────

class EvalSession:
    """
    Maintains a running list of EvaluationResults for a session.
    Used by the server to serve /eval/history and /eval/summary.
    """

    def __init__(self):
        self._history: list[EvaluationResult] = []

    def add(self, result: EvaluationResult) -> None:
        self._history.append(result)

    def clear(self) -> None:
        self._history.clear()

    @property
    def history(self) -> list[EvaluationResult]:
        return list(self._history)

    def summary(self) -> dict:
        """Aggregate stats across all queries in session."""
        if not self._history:
            return {"total_queries": 0}

        all_metrics: dict[str, list[float]] = {}
        for result in self._history:
            for m in result.metrics:
                all_metrics.setdefault(m.name, []).append(m.score)

        avg_overall = sum(r.overall_score for r in self._history) / len(self._history)
        avg_latency = sum(r.latency_ms for r in self._history) / len(self._history)

        return {
            "total_queries"  : len(self._history),
            "avg_overall"    : round(avg_overall, 4),
            "avg_overall_pct": int(avg_overall * 100),
            "avg_latency_ms" : round(avg_latency, 1),
            "metric_averages": {
                name: round(sum(scores) / len(scores), 4)
                for name, scores in all_metrics.items()
            },
        }
