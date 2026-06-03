"""
main.py
─────────────────────────────────────────────────────────────────────────────
Vectorless RAG on Structured PDFs — Entry Point

Run modes:
    python main.py                          → interactive Q&A on all PDFs in data/
    python main.py data/report.pdf          → interactive on a single PDF
    python main.py data/report.pdf "query"  → one-shot answer and exit
    python main.py --stats                  → show pipeline stats only
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from rag_pipeline import VectorlessRAGPipeline, run_interactive

load_dotenv()

# ─── Logging Setup ────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
    colorize=True,
)
logger.add(
    "outputs/rag.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    # --stats flag
    if args and args[0] == "--stats":
        pipeline = VectorlessRAGPipeline()
        pipeline.build()
        stats = pipeline.get_index_stats()
        print("\n📊 Pipeline Index Stats")
        print("─" * 40)
        for k, v in stats.items():
            print(f"  {k:<20}: {v}")
        print()
        return

    # Determine PDF source
    pdf_paths = None
    query     = None

    if args:
        first = Path(args[0])
        if first.suffix.lower() == ".pdf":
            if not first.exists():
                logger.error(f"File not found: {first}")
                sys.exit(1)
            pdf_paths = [first]
            if len(args) > 1:
                query = args[1]
        else:
            # Treat as query against all PDFs
            query = args[0]

    # Build pipeline
    pipeline = VectorlessRAGPipeline(use_multi_query=False)

    if pdf_paths:
        pipeline.build(pdf_paths=pdf_paths)
    else:
        pipeline.build()

    # One-shot query or interactive
    if query:
        response = pipeline.ask(query, show_retrieval=False)
        print(response)
    else:
        run_interactive(pipeline)


if __name__ == "__main__":
    main()

# uvicorn server:app --reload --port 8000
