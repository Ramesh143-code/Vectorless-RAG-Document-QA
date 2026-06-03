"""
chunker.py
─────────────────────────────────────────────────────────────────────────────
Vectorless RAG — Structure-Aware Chunker

Strategy: instead of blind fixed-size windows, we chunk along the document's
OWN structure — headings, sections, tables — so every chunk carries context.

Chunking modes (applied in order of priority):
  1. Heading-bounded   → each H1/H2 section becomes a chunk
  2. Table-isolated    → each table gets its own chunk (never split)
  3. Paragraph-merged  → small paragraphs merged up to CHUNK_SIZE chars
  4. Overflow-split    → any chunk still too large is split with overlap

Output: list[Chunk]
  Each chunk carries: text, source metadata, heading trail, page range,
  chunk type, and a unique ID — everything the retriever needs.
─────────────────────────────────────────────────────────────────────────────
"""

import hashlib
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from pdf_parser import ParsedDocument, ParsedPage, Heading, TableData, TextBlock

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 500))       # max chars per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))     # overlap on overflow splits


# ─── Data Model ───────────────────────────────────────────────────────────────

class ChunkType(str, Enum):
    HEADING_SECTION = "heading_section"   # text under a heading
    TABLE           = "table"             # a full table
    PARAGRAPH       = "paragraph"         # merged paragraphs
    OVERFLOW        = "overflow"          # large section split with overlap
    FULL_PAGE       = "full_page"         # fallback: whole page as chunk


@dataclass
class Chunk:
    """
    A single retrieval unit.

    Attributes:
        chunk_id        : Stable SHA-1 hash of (source + text).
        text            : The actual content to index and retrieve.
        chunk_type      : What kind of chunk this is (see ChunkType).
        source_file     : File name of the originating PDF.
        page_start      : First page this chunk spans (1-based).
        page_end        : Last page this chunk spans (1-based).
        heading_trail   : Breadcrumb of parent headings, e.g. ["Intro", "1.2 Methods"]
        table_index     : Set if chunk_type == TABLE.
        chunk_index     : Position of this chunk within the document.
        char_count      : Length of text in characters.
    """
    text          : str
    chunk_type    : ChunkType
    source_file   : str
    page_start    : int
    page_end      : int
    heading_trail : list[str]   = field(default_factory=list)
    table_index   : Optional[int] = None
    chunk_index   : int           = 0
    chunk_id      : str           = field(init=False)

    def __post_init__(self):
        raw = f"{self.source_file}::{self.chunk_index}::{self.text[:80]}"
        self.chunk_id = hashlib.sha1(raw.encode()).hexdigest()[:12]

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def heading_context(self) -> str:
        """Human-readable heading breadcrumb, e.g. 'Section 1 > Methods'"""
        return " > ".join(self.heading_trail) if self.heading_trail else ""

    def to_dict(self) -> dict:
        return {
            "chunk_id"      : self.chunk_id,
            "text"          : self.text,
            "chunk_type"    : self.chunk_type.value,
            "source_file"   : self.source_file,
            "page_start"    : self.page_start,
            "page_end"      : self.page_end,
            "heading_trail" : self.heading_trail,
            "heading_context": self.heading_context,
            "table_index"   : self.table_index,
            "chunk_index"   : self.chunk_index,
            "char_count"    : self.char_count,
        }


# ─── Chunker ──────────────────────────────────────────────────────────────────

class StructuredChunker:
    """
    Converts a ParsedDocument into a flat list of Chunks.

    Processing pipeline per document:
      1. Build a page → active_heading map (heading trail tracker)
      2. For each page:
           a. Isolate table regions → TABLE chunks
           b. Collect non-table text blocks
           c. Group blocks under heading boundaries → HEADING_SECTION or PARAGRAPH
           d. Overflow-split any chunk that exceeds CHUNK_SIZE
      3. Assign sequential chunk_index values
      4. Return final list[Chunk]
    """

    def __init__(
        self,
        chunk_size   : int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public ────────────────────────────────────────────────────────────────

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        """
        Main entry point. Chunks an entire ParsedDocument.

        Args:
            doc: A ParsedDocument from pdf_parser.py

        Returns:
            Ordered list of Chunk objects ready for indexing.
        """
        source = doc.metadata.file_name
        chunks: list[Chunk] = []

        # Track current heading hierarchy across pages: {level: heading_text}
        heading_state: dict[int, str] = {}

        for page in doc.pages:
            page_chunks = self._process_page(page, source, heading_state)
            chunks.extend(page_chunks)

        # Assign sequential chunk_index
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            # Recompute chunk_id with final index
            raw = f"{chunk.source_file}::{chunk.chunk_index}::{chunk.text[:80]}"
            chunk.chunk_id = hashlib.sha1(raw.encode()).hexdigest()[:12]

        logger.info(
            f"  ✂ Chunked '{source}': {len(chunks)} chunks "
            f"(avg {int(sum(c.char_count for c in chunks)/max(len(chunks),1))} chars)"
        )
        return chunks

    def chunk_documents(self, docs: list[ParsedDocument]) -> list[Chunk]:
        """Chunk a list of ParsedDocuments. Returns all chunks combined."""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.success(f"✅ Total chunks across {len(docs)} doc(s): {len(all_chunks)}")
        return all_chunks

    # ── Page Processing ───────────────────────────────────────────────────────

    def _process_page(
        self,
        page          : ParsedPage,
        source        : str,
        heading_state : dict[int, str],
    ) -> list[Chunk]:
        """Process a single page into chunks."""

        if not page.raw_text.strip() and not page.tables:
            return []

        page_chunks: list[Chunk] = []

        # Step 1: get the set of bbox y-ranges occupied by tables (to skip those blocks)
        table_y_ranges = self._get_table_y_ranges(page)

        # Step 2: emit TABLE chunks
        for table in page.tables:
            table_text = self._table_to_text(table)
            if not table_text.strip():
                continue

            trail = self._build_trail(heading_state)
            chunk = Chunk(
                text         = table_text,
                chunk_type   = ChunkType.TABLE,
                source_file  = source,
                page_start   = page.page_number,
                page_end     = page.page_number,
                heading_trail= trail,
                table_index  = table.table_index,
            )
            page_chunks.append(chunk)

        # Step 3: collect non-table text blocks + update heading state
        text_blocks = self._filter_non_table_blocks(page.blocks, table_y_ranges)

        # Step 4: group blocks under heading boundaries
        section_chunks = self._group_into_sections(
            blocks       = text_blocks,
            headings     = page.headings,
            page         = page,
            source       = source,
            heading_state= heading_state,
        )
        page_chunks.extend(section_chunks)

        return page_chunks

    # ── Heading State ─────────────────────────────────────────────────────────

    def _update_heading_state(self, heading: Heading, state: dict[int, str]) -> None:
        """
        Update the running heading hierarchy.
        When an H2 is seen, clear all H3+ beneath it, etc.
        """
        state[heading.level] = heading.text
        # Clear any deeper levels
        for lvl in list(state.keys()):
            if lvl > heading.level:
                del state[lvl]

    def _build_trail(self, state: dict[int, str]) -> list[str]:
        """Return ordered list of current heading texts, shallowest first."""
        return [state[lvl] for lvl in sorted(state.keys())]

    # ── Sections ──────────────────────────────────────────────────────────────

    def _group_into_sections(
        self,
        blocks       : list[TextBlock],
        headings     : list[Heading],
        page         : ParsedPage,
        source       : str,
        heading_state: dict[int, str],
    ) -> list[Chunk]:
        """
        Walk blocks top-to-bottom. When a heading is encountered, flush
        the current buffer as a chunk and start a new section.
        """

        # Map heading bbox y0 → Heading object for quick lookup
        heading_y_map: dict[float, Heading] = {
            round(h.bbox[1], 1): h for h in headings
        }

        chunks     : list[Chunk] = []
        buffer     : list[str]   = []
        section_start_page       = page.page_number

        def flush(trail: list[str]) -> None:
            text = self._join_buffer(buffer)
            if not text:
                return
            # Split if too large
            sub_chunks = self._split_if_overflow(
                text       = text,
                chunk_type = ChunkType.HEADING_SECTION if trail else ChunkType.PARAGRAPH,
                source     = source,
                page_start = section_start_page,
                page_end   = page.page_number,
                trail      = trail,
            )
            chunks.extend(sub_chunks)
            buffer.clear()

        for block in blocks:
            block_y = round(block.bbox[1], 1)

            # Check if this block IS a heading
            matched_heading = heading_y_map.get(block_y)

            if matched_heading:
                # Flush current buffer before starting new section
                flush(self._build_trail(heading_state))
                # Update heading state with this new heading
                self._update_heading_state(matched_heading, heading_state)
                # The heading text itself goes into the new buffer as context
                buffer.append(matched_heading.text)
            else:
                buffer.append(block.text)

        # Final flush
        flush(self._build_trail(heading_state))

        return chunks

    # ── Overflow Splitter ─────────────────────────────────────────────────────

    def _split_if_overflow(
        self,
        text      : str,
        chunk_type: ChunkType,
        source    : str,
        page_start: int,
        page_end  : int,
        trail     : list[str],
    ) -> list[Chunk]:
        """
        If text fits within CHUNK_SIZE, return a single chunk.
        Otherwise split into overlapping sub-chunks.
        """
        if len(text) <= self.chunk_size:
            return [Chunk(
                text          = text,
                chunk_type    = chunk_type,
                source_file   = source,
                page_start    = page_start,
                page_end      = page_end,
                heading_trail = trail,
            )]

        # Split on sentence boundaries where possible
        sub_texts = self._split_with_overlap(text)
        chunks = []
        for sub in sub_texts:
            chunks.append(Chunk(
                text          = sub,
                chunk_type    = ChunkType.OVERFLOW,
                source_file   = source,
                page_start    = page_start,
                page_end      = page_end,
                heading_trail = trail,
            ))
        return chunks

    def _split_with_overlap(self, text: str) -> list[str]:
        """
        Split a long string into overlapping windows.
        Tries to break at sentence boundaries ('. ', '? ', '! ').
        Falls back to hard character split.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.?!])\s+', text)

        chunks  : list[str] = []
        current : list[str] = []
        current_len = 0

        for sentence in sentences:
            s_len = len(sentence)

            if current_len + s_len > self.chunk_size and current:
                # Emit current chunk
                chunks.append(" ".join(current))
                # Keep overlap: walk back from end of current until overlap chars
                overlap_buf  = []
                overlap_len  = 0
                for sent in reversed(current):
                    if overlap_len + len(sent) > self.chunk_overlap:
                        break
                    overlap_buf.insert(0, sent)
                    overlap_len += len(sent)
                current     = overlap_buf
                current_len = overlap_len

            current.append(sentence)
            current_len += s_len

        if current:
            chunks.append(" ".join(current))

        return chunks if chunks else [text[: self.chunk_size]]

    # ── Table Helpers ─────────────────────────────────────────────────────────

    def _table_to_text(self, table: TableData) -> str:
        """
        Render a TableData as a readable markdown-style text block.

        Example output:
            [TABLE] Headers: Name | Age | Score
            Row 1: Alice | 30 | 95
            Row 2: Bob | 25 | 87
        """
        lines = ["[TABLE]"]

        if table.headers:
            lines.append("Headers: " + " | ".join(h for h in table.headers if h))

        for i, row in enumerate(table.data_rows, start=1):
            row_text = " | ".join(cell for cell in row if cell)
            if row_text.strip():
                lines.append(f"Row {i}: {row_text}")

        return "\n".join(lines)

    def _get_table_y_ranges(self, page: ParsedPage) -> list[tuple[float, float]]:
        """Return (y0, y1) for each table bbox on this page."""
        return [(t.bbox[1], t.bbox[3]) for t in page.tables if t.bbox]

    def _filter_non_table_blocks(
        self,
        blocks      : list[TextBlock],
        table_ranges: list[tuple[float, float]],
    ) -> list[TextBlock]:
        """Remove blocks that fall inside a table's vertical range."""
        if not table_ranges:
            return blocks

        def inside_table(block: TextBlock) -> bool:
            by0, by1 = block.bbox[1], block.bbox[3]
            for ty0, ty1 in table_ranges:
                if by0 >= ty0 - 2 and by1 <= ty1 + 2:
                    return True
            return False

        return [b for b in blocks if not inside_table(b)]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _join_buffer(self, buffer: list[str]) -> str:
        """Join buffer lines, clean up whitespace."""
        text = " ".join(buffer)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()


# ─── Utility: Print Chunk Summary ────────────────────────────────────────────

def print_chunks_summary(chunks: list[Chunk]) -> None:
    """Print a readable summary of all chunks."""

    type_counts: dict[str, int] = {}
    for c in chunks:
        type_counts[c.chunk_type.value] = type_counts.get(c.chunk_type.value, 0) + 1

    print("\n" + "═" * 60)
    print(f"  ✂  CHUNKS SUMMARY — {len(chunks)} total")
    print("═" * 60)
    for ctype, count in sorted(type_counts.items()):
        print(f"  {ctype:<22} {count:>4} chunks")
    print()

    print("  First 10 chunks preview:")
    print("  " + "─" * 56)
    for chunk in chunks[:10]:
        trail = chunk.heading_context or "(no heading)"
        print(f"  [{chunk.chunk_index:03d}] {chunk.chunk_type.value:<18} "
              f"p.{chunk.page_start}  {chunk.char_count:>4}c")
        print(f"        📌 {trail[:55]}")
        print(f"        {chunk.text[:80].replace(chr(10),' ')}…")
        print()
    print("═" * 60 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from pdf_parser import PDFParser

    parser  = PDFParser()
    chunker = StructuredChunker()

    path = sys.argv[1] if len(sys.argv) > 1 else None

    if path:
        doc    = parser.parse(Path(path))
        chunks = chunker.chunk_document(doc)
    else:
        docs   = parser.parse_directory()
        chunks = chunker.chunk_documents(docs)

    print_chunks_summary(chunks)
