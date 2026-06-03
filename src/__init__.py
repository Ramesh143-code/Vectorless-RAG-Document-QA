"""
Vectorless RAG on Structured PDFs
──────────────────────────────────
Public package exports for clean imports when using as a library.

Usage:
    from src import VectorlessRAGPipeline
    from src import PDFParser, StructuredChunker, BM25Retriever
"""

from pdf_parser import (
    PDFParser,
    ParsedDocument,
    ParsedPage,
    TextBlock,
    Heading,
    TableData,
    DocumentMetadata,
)

from chunker import (
    StructuredChunker,
    Chunk,
    ChunkType,
)

from retriever import (
    BM25Retriever,
    RetrievalResult,
)

from rag_pipeline import (
    VectorlessRAGPipeline,
    RAGResponse,
    Citation,
)

__all__ = [
    # Parser
    "PDFParser",
    "ParsedDocument",
    "ParsedPage",
    "TextBlock",
    "Heading",
    "TableData",
    "DocumentMetadata",
    # Chunker
    "StructuredChunker",
    "Chunk",
    "ChunkType",
    # Retriever
    "BM25Retriever",
    "RetrievalResult",
    # Pipeline
    "VectorlessRAGPipeline",
    "RAGResponse",
    "Citation",
]
