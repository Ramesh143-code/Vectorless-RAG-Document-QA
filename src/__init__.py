"""
Vectorless RAG on Structured PDFs
──────────────────────────────────
Public package exports for clean imports when using as a library.

Supports both text-based and scanned PDFs with OCR.

Usage:
    from src import PDFParser, ParsedDocument
    from src import StructuredChunker, BM25Retriever
    from src import VectorlessRAGPipeline
"""

# Core PDF Parser (with OCR support)
from .pdf_parser import (
    PDFParser,
    ParsedDocument,
    ParsedPage,
    TextBlock,
    Heading,
    TableData,
    DocumentMetadata,
    # OCR Presets and Utilities
    OCR_PRESETS,
    OCR_LANGUAGES,
    TextCleaner,
)

# Chunking Module
try:
    from .chunker import (
        StructuredChunker,
        Chunk,
        ChunkType,
    )
except ImportError:
    # Fallback if chunker doesn't exist yet
    import warnings
    warnings.warn("chunker module not found. Install or create chunker.py")
    StructuredChunker = None
    Chunk = None
    ChunkType = None

# BM25 Retriever Module
try:
    from .retriever import (
        BM25Retriever,
        RetrievalResult,
    )
except ImportError:
    # Fallback if retriever doesn't exist yet
    import warnings
    warnings.warn("retriever module not found. Install or create retriever.py")
    BM25Retriever = None
    RetrievalResult = None

# RAG Pipeline Module
try:
    from .rag_pipeline import (
        VectorlessRAGPipeline,
        RAGResponse,
        Citation,
    )
except ImportError:
    # Fallback if rag_pipeline doesn't exist yet
    import warnings
    warnings.warn("rag_pipeline module not found. Install or create rag_pipeline.py")
    VectorlessRAGPipeline = None
    RAGResponse = None
    Citation = None

# Version information
__version__ = "2.0.0"
__author__ = "Vectorless RAG Team"
__description__ = "PDF parsing with OCR support for scanned documents"

# Convenience function to create a configured parser
def create_parser(ocr_quality: str = "BALANCED", 
                  ocr_language: str = "eng",
                  parallel_processing: bool = True,
                  max_workers: int = 4) -> PDFParser:
    """
    Create a configured PDF parser with OCR settings.
    
    Args:
        ocr_quality: "FAST", "BALANCED", "HIGH_QUALITY", "VERY_HIGH", "MAXIMUM"
        ocr_language: OCR language (e.g., "eng", "eng+fra", "hin+eng")
        parallel_processing: Enable parallel OCR processing
        max_workers: Number of parallel workers
    
    Returns:
        Configured PDFParser instance
    """
    return PDFParser(
        ocr_quality=ocr_quality,
        ocr_language=ocr_language,
        parallel_processing=parallel_processing,
        max_workers=max_workers
    )


# List available OCR languages
def list_ocr_languages():
    """
    Print available OCR languages.
    """
    print("\n🌐 Available OCR Languages:")
    print("-" * 40)
    for code, name in OCR_LANGUAGES.items():
        print(f"  {code:10} - {name}")
    print("\n💡 Use '+' for multiple languages: eng+fra+deu")


# List OCR quality presets
def list_quality_presets():
    """
    Print available OCR quality presets.
    """
    print("\n📷 OCR Quality Presets:")
    print("-" * 50)
    for preset, config in OCR_PRESETS.items():
        print(f"  {preset:12} - {config['description']}")
        print(f"                  DPI: {config['dpi']}, Timeout: {config['timeout']}s")
    print("\n💡 Higher quality = slower processing")


# Module info
def info():
    """
    Print module information and capabilities.
    """
    print("=" * 60)
    print("Vectorless RAG - PDF Processing Module")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Description: {__description__}")
    print("\n✨ Features:")
    print("  • Text-based PDF extraction (fast)")
    print("  • Scanned PDF OCR (automatic fallback)")
    print("  • Multi-language OCR support")
    print("  • Configurable quality presets")
    print("  • Parallel processing for speed")
    print("  • Heading and table detection")
    print("\n🔧 OCR Configuration:")
    print(f"  • Tesseract path: Configured")
    print(f"  • Available languages: {len(OCR_LANGUAGES)}")
    print(f"  • Quality presets: {len(OCR_PRESETS)}")
    print("=" * 60)


# Define what gets imported with "from src import *"
__all__ = [
    # Core Parser
    "PDFParser",
    "ParsedDocument",
    "ParsedPage",
    "TextBlock",
    "Heading",
    "TableData",
    "DocumentMetadata",
    
    # OCR Utilities
    "OCR_PRESETS",
    "OCR_LANGUAGES",
    "TextCleaner",
    
    # Chunker (if available)
    "StructuredChunker",
    "Chunk",
    "ChunkType",
    
    # Retriever (if available)
    "BM25Retriever",
    "RetrievalResult",
    
    # Pipeline (if available)
    "VectorlessRAGPipeline",
    "RAGResponse",
    "Citation",
    
    # Convenience functions
    "create_parser",
    "list_ocr_languages",
    "list_quality_presets",
    "info",
    
    # Version
    "__version__",
    "__author__",
    "__description__",
]

# Check OCR availability on import
try:
    import pytesseract
    import fitz
    OCR_READY = True
except ImportError as e:
    OCR_READY = False
    import warnings
    warnings.warn(f"OCR dependencies not fully installed: {e}. Scanned PDFs will not work.")

# Print status message (optional, comment out if not needed)
if OCR_READY:
    print("✅ Vectorless RAG module loaded (OCR ready)")
else:
    print("⚠️ Vectorless RAG module loaded (OCR not available)")