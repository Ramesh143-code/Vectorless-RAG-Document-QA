"""
pdf_parser.py
─────────────────────────────────────────────────────────────────────────────
Vectorless RAG — Structured PDF Parser
Extracts: text blocks, tables, headings, page metadata, and document outline
from structured PDFs using pdfplumber (layout-aware) + PyMuPDF (metadata).

Output schema (per document):
    ParsedDocument
    ├── metadata        → title, author, page_count, file_name
    ├── outline         → table of contents / bookmarks (if present)
    └── pages[]
        ├── page_number
        ├── headings[]  → detected section headings with font size
        ├── blocks[]    → ordered text blocks with positional info
        └── tables[]    → extracted tables as list-of-rows
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ─── Constants ───────────────────────────────────────────────────────────────

PDF_INPUT_DIR = os.getenv("PDF_INPUT_DIR", "data/")

# Font size thresholds for heading detection
HEADING_MIN_FONT_SIZE = 11.0   # anything >= this is a candidate heading
BODY_MAX_FONT_SIZE    = 10.5   # normal body text is typically <= this


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class TextBlock:
    """A single block of continuous text on a page."""
    text: str
    page_number: int
    block_index: int
    bbox: tuple[float, float, float, float]   # (x0, y0, x1, y1)
    font_size: float = 0.0
    font_name: str = ""
    is_bold: bool = False


@dataclass
class Heading:
    """A detected section heading."""
    text: str
    page_number: int
    level: int          # 1 = largest, 2 = medium, 3 = smallest heading
    font_size: float
    bbox: tuple[float, float, float, float]


@dataclass
class TableData:
    """A table extracted from a PDF page."""
    page_number: int
    table_index: int
    rows: list[list[str]]           # list of rows; each row is list of cell strings
    bbox: tuple[float, float, float, float]

    @property
    def headers(self) -> list[str]:
        """First row treated as header."""
        return self.rows[0] if self.rows else []

    @property
    def data_rows(self) -> list[list[str]]:
        """All rows except the header."""
        return self.rows[1:] if len(self.rows) > 1 else []


@dataclass
class ParsedPage:
    """All extracted content for a single PDF page."""
    page_number: int        # 1-based
    width: float
    height: float
    raw_text: str           # full page text (flat)
    headings: list[Heading] = field(default_factory=list)
    blocks: list[TextBlock] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)


@dataclass
class DocumentMetadata:
    """Top-level PDF metadata."""
    file_name: str
    file_path: str
    page_count: int
    title: str = ""
    author: str = ""
    subject: str = ""
    creator: str = ""
    has_toc: bool = False


@dataclass
class ParsedDocument:
    """Complete parsed representation of a PDF."""
    metadata: DocumentMetadata
    outline: list[dict]         # table of contents entries [{title, page, level}]
    pages: list[ParsedPage]

    def get_all_text(self) -> str:
        """Concatenate all page text in order."""
        return "\n\n".join(p.raw_text for p in self.pages if p.raw_text.strip())

    def get_all_headings(self) -> list[Heading]:
        """Return all headings across all pages."""
        return [h for p in self.pages for h in p.headings]

    def get_all_tables(self) -> list[TableData]:
        """Return all tables across all pages."""
        return [t for p in self.pages for t in p.tables]

    def get_page(self, page_number: int) -> Optional[ParsedPage]:
        """Fetch a page by 1-based page number."""
        for p in self.pages:
            if p.page_number == page_number:
                return p
        return None


# ─── Core Parser ─────────────────────────────────────────────────────────────

class PDFParser:
    """
    Parses structured PDFs into a rich ParsedDocument object.

    Uses:
    - pdfplumber  → text blocks, tables, layout bounding boxes
    - PyMuPDF     → document metadata, TOC / outline, font details
    """

    def __init__(self, heading_min_size: float = HEADING_MIN_FONT_SIZE):
        self.heading_min_size = heading_min_size

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """
        Parse a single PDF file.

        Args:
            pdf_path: Path to the .pdf file.

        Returns:
            ParsedDocument with metadata, outline, and per-page content.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Parsing: {pdf_path.name}")

        metadata = self._extract_metadata(pdf_path)
        outline  = self._extract_outline(pdf_path)
        pages    = self._extract_pages(pdf_path)

        metadata.has_toc = len(outline) > 0

        doc = ParsedDocument(metadata=metadata, outline=outline, pages=pages)

        logger.success(
            f"✅ Parsed '{pdf_path.name}' — "
            f"{metadata.page_count} pages | "
            f"{len(doc.get_all_headings())} headings | "
            f"{len(doc.get_all_tables())} tables"
        )
        return doc

    def parse_directory(self, dir_path: str | Path = PDF_INPUT_DIR) -> list[ParsedDocument]:
        """
        Parse all .pdf files in a directory.

        Args:
            dir_path: Folder containing PDF files.

        Returns:
            List of ParsedDocument objects.
        """
        dir_path = Path(dir_path)
        pdf_files = sorted(dir_path.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in: {dir_path}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF(s) in '{dir_path}'")
        documents = []

        for pdf_file in pdf_files:
            try:
                doc = self.parse(pdf_file)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to parse '{pdf_file.name}': {e}")

        return documents

    # ── Metadata ──────────────────────────────────────────────────────────────

    def _extract_metadata(self, pdf_path: Path) -> DocumentMetadata:
        """Extract document-level metadata using PyMuPDF."""
        doc = fitz.open(str(pdf_path))
        meta = doc.metadata or {}
        doc.close()

        return DocumentMetadata(
            file_name  = pdf_path.name,
            file_path  = str(pdf_path.resolve()),
            page_count = doc.page_count,
            title      = meta.get("title", "").strip(),
            author     = meta.get("author", "").strip(),
            subject    = meta.get("subject", "").strip(),
            creator    = meta.get("creator", "").strip(),
        )

    # ── Outline / TOC ─────────────────────────────────────────────────────────

    def _extract_outline(self, pdf_path: Path) -> list[dict]:
        """
        Extract the PDF's table of contents / bookmarks.

        Returns list of dicts: [{title, page, level}]
        PyMuPDF returns TOC as [[level, title, page], ...]
        """
        doc = fitz.open(str(pdf_path))
        toc = doc.get_toc()
        doc.close()

        outline = []
        for entry in toc:
            level, title, page = entry
            outline.append({
                "level": level,
                "title": title.strip(),
                "page":  page,   # 1-based
            })

        return outline

    # ── Pages ─────────────────────────────────────────────────────────────────

    def _extract_pages(self, pdf_path: Path) -> list[ParsedPage]:
        """Extract content from every page using pdfplumber."""
        pages = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_number = i + 1   # convert to 1-based

                try:
                    parsed_page = self._parse_single_page(page, page_number)
                    pages.append(parsed_page)
                except Exception as e:
                    logger.warning(f"  ⚠ Page {page_number} failed: {e}")
                    # Append an empty page so page numbering stays consistent
                    pages.append(ParsedPage(
                        page_number = page_number,
                        width       = page.width,
                        height      = page.height,
                        raw_text    = "",
                    ))

        return pages

    def _parse_single_page(self, page, page_number: int) -> ParsedPage:
        """Parse one pdfplumber page into a ParsedPage."""

        # ── Raw text ──────────────────────────────────────────────────────────
        raw_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

        # ── Tables ────────────────────────────────────────────────────────────
        tables = self._extract_tables(page, page_number)

        # ── Text blocks with font info ────────────────────────────────────────
        blocks, headings = self._extract_blocks_and_headings(page, page_number)

        return ParsedPage(
            page_number = page_number,
            width       = page.width,
            height      = page.height,
            raw_text    = raw_text,
            headings    = headings,
            blocks      = blocks,
            tables      = tables,
        )

    # ── Tables ────────────────────────────────────────────────────────────────

    def _extract_tables(self, page, page_number: int) -> list[TableData]:
        """Extract all tables from a pdfplumber page."""
        tables = []
        raw_tables = page.extract_tables()

        for idx, raw_table in enumerate(raw_tables):
            if not raw_table:
                continue

            # Sanitize: replace None cells with empty strings, strip whitespace
            clean_rows = []
            for row in raw_table:
                clean_row = [
                    (cell.strip() if isinstance(cell, str) else "") if cell is not None else ""
                    for cell in row
                ]
                # Skip rows that are completely empty
                if any(cell for cell in clean_row):
                    clean_rows.append(clean_row)

            if not clean_rows:
                continue

            # Get bounding box of the table (pdfplumber table objects)
            table_objects = page.find_tables()
            bbox = table_objects[idx].bbox if idx < len(table_objects) else (0, 0, 0, 0)

            tables.append(TableData(
                page_number = page_number,
                table_index = idx,
                rows        = clean_rows,
                bbox        = bbox,
            ))

        return tables

    # ── Text Blocks & Headings ────────────────────────────────────────────────

    def _extract_blocks_and_headings(
        self, page, page_number: int
    ) -> tuple[list[TextBlock], list[Heading]]:
        """
        Extract word-level data from pdfplumber to reconstruct text blocks
        with font info, then detect headings by font size.
        """
        words = page.extract_words(
            x_tolerance        = 3,
            y_tolerance        = 3,
            extra_attrs        = ["fontname", "size"],   # request font metadata
            keep_blank_chars   = False,
        )

        if not words:
            return [], []

        # Group words into lines, then lines into blocks
        line_groups = self._group_words_into_lines(words)
        blocks, headings = self._build_blocks(line_groups, page_number)

        return blocks, headings

    def _group_words_into_lines(self, words: list[dict]) -> list[list[dict]]:
        """
        Group word dicts into lines based on vertical (y) proximity.
        Words on the same baseline (within 2pt) = same line.
        """
        if not words:
            return []

        lines: list[list[dict]] = []
        current_line = [words[0]]
        current_y = words[0]["top"]

        for word in words[1:]:
            if abs(word["top"] - current_y) <= 2.0:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
                current_y = word["top"]

        if current_line:
            lines.append(current_line)

        # Sort each line left→right
        for line in lines:
            line.sort(key=lambda w: w["x0"])

        return lines

    def _build_blocks(
        self, line_groups: list[list[dict]], page_number: int
    ) -> tuple[list[TextBlock], list[Heading]]:
        """
        Merge lines into paragraph blocks (gap > 8pt = new block).
        Detect headings by comparing font sizes.
        """
        if not line_groups:
            return [], []

        # First pass: collect all font sizes to determine thresholds
        all_sizes = []
        for line in line_groups:
            for w in line:
                sz = w.get("size", 0)
                if sz:
                    all_sizes.append(sz)

        # Determine heading font size threshold dynamically
        if all_sizes:
            body_size = sorted(all_sizes)[int(len(all_sizes) * 0.5)]  # median
            heading_threshold = max(body_size + 1.0, self.heading_min_size)
        else:
            heading_threshold = self.heading_min_size

        blocks: list[TextBlock]  = []
        headings: list[Heading]  = []
        block_index = 0

        # Group lines into blocks (vertical gap heuristic)
        current_block_lines: list[list[dict]] = [line_groups[0]]
        prev_bottom = max(w["bottom"] for w in line_groups[0])

        def _flush_block(block_lines: list[list[dict]]):
            nonlocal block_index
            all_words = [w for line in block_lines for w in line]
            text = " ".join(w["text"] for w in all_words).strip()
            text = re.sub(r"\s{2,}", " ", text)

            if not text:
                return

            # Dominant font size in this block
            sizes = [w.get("size", 0) for w in all_words if w.get("size")]
            avg_size = sum(sizes) / len(sizes) if sizes else 0.0

            # Font name (most common in block)
            font_names = [w.get("fontname", "") for w in all_words if w.get("fontname")]
            font_name  = max(set(font_names), key=font_names.count) if font_names else ""
            is_bold    = "bold" in font_name.lower() or "Bold" in font_name

            # Bounding box
            x0 = min(w["x0"]     for w in all_words)
            y0 = min(w["top"]    for w in all_words)
            x1 = max(w["x1"]     for w in all_words)
            y1 = max(w["bottom"] for w in all_words)

            tb = TextBlock(
                text        = text,
                page_number = page_number,
                block_index = block_index,
                bbox        = (x0, y0, x1, y1),
                font_size   = round(avg_size, 2),
                font_name   = font_name,
                is_bold     = is_bold,
            )
            blocks.append(tb)
            block_index += 1

            # Heading detection: large font OR bold short line
            is_large   = avg_size >= heading_threshold
            is_short   = len(text.split()) <= 15
            looks_like_heading = (is_large or is_bold) and is_short

            if looks_like_heading:
                # Assign heading level based on font size buckets
                if avg_size >= heading_threshold + 4:
                    level = 1
                elif avg_size >= heading_threshold + 1:
                    level = 2
                else:
                    level = 3

                headings.append(Heading(
                    text        = text,
                    page_number = page_number,
                    level       = level,
                    font_size   = round(avg_size, 2),
                    bbox        = (x0, y0, x1, y1),
                ))

        for line in line_groups[1:]:
            line_top = min(w["top"] for w in line)
            gap = line_top - prev_bottom

            if gap > 8.0:
                # Large vertical gap → new block
                _flush_block(current_block_lines)
                current_block_lines = [line]
            else:
                current_block_lines.append(line)

            prev_bottom = max(w["bottom"] for w in line)

        # Flush last block
        if current_block_lines:
            _flush_block(current_block_lines)

        return blocks, headings


# ─── Utility Functions ────────────────────────────────────────────────────────

def print_document_summary(doc: ParsedDocument) -> None:
    """Pretty-print a summary of a ParsedDocument to the console."""
    print("\n" + "═" * 60)
    print(f"  📄 {doc.metadata.file_name}")
    print("═" * 60)
    print(f"  Pages   : {doc.metadata.page_count}")
    print(f"  Title   : {doc.metadata.title or '(none)'}")
    print(f"  Author  : {doc.metadata.author or '(none)'}")
    print(f"  Has TOC : {'yes' if doc.metadata.has_toc else 'no'}")

    if doc.outline:
        print(f"\n  📑 Outline ({len(doc.outline)} entries):")
        for entry in doc.outline[:10]:
            indent = "  " * entry["level"]
            print(f"    {indent}[L{entry['level']}] {entry['title']}  (p.{entry['page']})")
        if len(doc.outline) > 10:
            print(f"    ... and {len(doc.outline) - 10} more entries")

    print(f"\n  🔖 Headings detected: {len(doc.get_all_headings())}")
    for h in doc.get_all_headings()[:8]:
        indent = "  " * h.level
        print(f"    {indent}[H{h.level}] {h.text[:70]}  (p.{h.page_number})")

    print(f"\n  📊 Tables detected: {len(doc.get_all_tables())}")
    for t in doc.get_all_tables():
        print(f"    → Page {t.page_number}, Table {t.table_index}: "
              f"{len(t.rows)} rows × {len(t.headers)} cols")

    print("═" * 60 + "\n")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    parser = PDFParser()

    # If a path is passed as argument, parse that file
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        doc = parser.parse(pdf_path)
        print_document_summary(doc)

        # Show first 500 chars of text from page 1
        if doc.pages:
            print("── Page 1 text preview ──────────────────────────────")
            print(doc.pages[0].raw_text[:500])
            print("─" * 52)

    # Otherwise parse everything in data/
    else:
        docs = parser.parse_directory(PDF_INPUT_DIR)
        for doc in docs:
            print_document_summary(doc)
