"""
pdf_parser.py
─────────────────────────────────────────────────────────────────────────────
Vectorless RAG — Advanced PDF Parser with Intelligent OCR
─────────────────────────────────────────────────────────────────────────────
Features:
- Automatic detection of text-based vs scanned PDFs
- Configurable OCR quality presets (FAST to MAXIMUM)
- Multi-language OCR support (English, French, Spanish, German, Hindi, etc.)
- Intelligent text cleaning and post-processing
- Performance optimizations for large documents
- Headings and table extraction for text-based PDFs
- Graceful fallback and error handling
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
import pdfplumber
from dotenv import load_dotenv
from loguru import logger

# ========== TESSERACT CONFIGURATION ==========
import pytesseract

# Set Tesseract path (update if different)
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_PATH = r"C:\Program Files\Tesseract-OCR\tessdata"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print(f"✅ Tesseract configured: {TESSERACT_PATH}")
else:
    print(f"⚠️ Tesseract not found at {TESSERACT_PATH}")

# Set TESSDATA_PREFIX environment variable
os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH

OCR_AVAILABLE = True
# =============================================

load_dotenv()

# ─── Constants ───────────────────────────────────────────────────────────────

PDF_INPUT_DIR = os.getenv("PDF_INPUT_DIR", "data/")
HEADING_MIN_FONT_SIZE = 11.0

# OCR Quality Presets
OCR_PRESETS = {
    "FAST": {
        "dpi": 150,
        "description": "Fastest (150 DPI) - Best for drafts and large documents",
        "preprocess": False,
        "timeout": 30
    },
    "BALANCED": {
        "dpi": 200,
        "description": "Balanced (200 DPI) - Good for most documents",
        "preprocess": True,
        "timeout": 60
    },
    "HIGH_QUALITY": {
        "dpi": 300,
        "description": "High Quality (300 DPI) - Best for printed text",
        "preprocess": True,
        "timeout": 120
    },
    "VERY_HIGH": {
        "dpi": 400,
        "description": "Very High (400 DPI) - For small fonts and dense text",
        "preprocess": True,
        "timeout": 180
    },
    "MAXIMUM": {
        "dpi": 600,
        "description": "Maximum (600 DPI) - Best quality, slowest",
        "preprocess": True,
        "timeout": 300
    }
}

# Language Support
OCR_LANGUAGES = {
    "eng": "English",
    "fra": "French",
    "deu": "German",
    "spa": "Spanish",
    "ita": "Italian",
    "por": "Portuguese",
    "rus": "Russian",
    "hin": "Hindi",
    "chi_sim": "Chinese (Simplified)",
    "chi_tra": "Chinese (Traditional)",
    "jpn": "Japanese",
    "kor": "Korean",
    "ara": "Arabic",
    "tur": "Turkish",
    "nld": "Dutch",
    "pol": "Polish",
    "swe": "Swedish"
}


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class TextBlock:
    text: str
    page_number: int
    block_index: int
    bbox: tuple[float, float, float, float]
    font_size: float = 0.0
    font_name: str = ""
    is_bold: bool = False


@dataclass
class Heading:
    text: str
    page_number: int
    level: int
    font_size: float
    bbox: tuple[float, float, float, float]


@dataclass
class TableData:
    page_number: int
    table_index: int
    rows: list[list[str]]
    bbox: tuple[float, float, float, float]

    @property
    def headers(self) -> list[str]:
        return self.rows[0] if self.rows else []

    @property
    def data_rows(self) -> list[list[str]]:
        return self.rows[1:] if len(self.rows) > 1 else []


@dataclass
class ParsedPage:
    page_number: int
    width: float
    height: float
    raw_text: str
    headings: list[Heading] = field(default_factory=list)
    blocks: list[TextBlock] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    ocr_confidence: float = 0.0
    processing_time: float = 0.0


@dataclass
class DocumentMetadata:
    file_name: str
    file_path: str
    page_count: int
    title: str = ""
    author: str = ""
    subject: str = ""
    creator: str = ""
    has_toc: bool = False
    is_scanned: bool = False
    ocr_quality: str = ""
    ocr_language: str = ""
    total_processing_time: float = 0.0


@dataclass
class ParsedDocument:
    metadata: DocumentMetadata
    outline: list[dict]
    pages: list[ParsedPage]

    def get_all_text(self) -> str:
        return "\n\n".join(p.raw_text for p in self.pages if p.raw_text.strip())

    def get_all_headings(self) -> list[Heading]:
        return [h for p in self.pages for h in p.headings]

    def get_all_tables(self) -> list[TableData]:
        return [t for p in self.pages for t in p.tables]

    def get_page(self, page_number: int) -> Optional[ParsedPage]:
        for p in self.pages:
            if p.page_number == page_number:
                return p
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "total_pages": len(self.pages),
            "pages_with_text": sum(1 for p in self.pages if p.raw_text),
            "total_headings": len(self.get_all_headings()),
            "total_tables": len(self.get_all_tables()),
            "avg_ocr_confidence": sum(p.ocr_confidence for p in self.pages) / len(self.pages) if self.pages else 0,
            "total_processing_time": self.metadata.total_processing_time
        }


# ─── Text Cleaning Utilities ─────────────────────────────────────────────────

class TextCleaner:
    """Advanced text cleaning and post-processing for OCR results"""
    
    @staticmethod
    def clean_ocr_text(text: str) -> str:
        """Clean and enhance OCR text"""
        if not text:
            return ""
        
        # Remove excessive newlines
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        # Fix common OCR errors
        corrections = {
            r'\|': 'I',           # Pipe to I
            r'0(?=[A-Za-z])': 'O', # Zero before letter to O
            r'(?<=[a-z])0': 'o',   # Zero after letter to o
            r'1(?=[A-Za-z])': 'I', # One before letter to I
            r'©': '(c)',          # Copyright symbol
            r'®': '(R)',          # Registered symbol
            r'™': '(TM)',         # Trademark symbol
            r'ﬁ': 'fi',           # Ligature fi
            r'ﬂ': 'fl',           # Ligature fl
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1\2', text)
        
        # Remove duplicate words (common OCR artifact)
        text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
        
        # Normalize spaces
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove empty lines at start and end
        text = text.strip()
        
        return text
    
    @staticmethod
    def extract_code_blocks(text: str) -> list[str]:
        """Extract potential code blocks from text"""
        code_patterns = [
            r'```(.*?)```',
            r'def\s+\w+\(.*?\):.*?(?=\n\S|\Z)',
            r'class\s+\w+.*?:.*?(?=\n\S|\Z)',
            r'import\s+\w+',
            r'from\s+\w+\s+import',
        ]
        
        code_blocks = []
        for pattern in code_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            code_blocks.extend(matches)
        
        return code_blocks


# ─── Core Parser ─────────────────────────────────────────────────────────────

class PDFParser:
    """
    Advanced PDF Parser with intelligent OCR capabilities
    """
    
    def __init__(self, 
                 heading_min_size: float = HEADING_MIN_FONT_SIZE,
                 use_ocr: bool = True,
                 ocr_quality: str = "BALANCED",
                 ocr_language: str = "eng",
                 parallel_processing: bool = True,
                 max_workers: int = 4):
        """
        Initialize PDF Parser with advanced options.
        
        Args:
            heading_min_size: Minimum font size for heading detection
            use_ocr: Enable/disable OCR for scanned PDFs
            ocr_quality: "FAST", "BALANCED", "HIGH_QUALITY", "VERY_HIGH", "MAXIMUM"
            ocr_language: OCR language(s) - use '+' for multiple (e.g., "eng+fra")
            parallel_processing: Enable parallel page processing
            max_workers: Maximum parallel workers for OCR
        """
        self.heading_min_size = heading_min_size
        self.use_ocr = use_ocr and OCR_AVAILABLE
        self.parallel_processing = parallel_processing
        self.max_workers = max_workers
        
        # OCR Configuration
        quality = ocr_quality.upper()
        if quality not in OCR_PRESETS:
            logger.warning(f"Unknown quality '{quality}', using BALANCED")
            quality = "BALANCED"
        
        self.ocr_config = OCR_PRESETS[quality]
        self.ocr_language = ocr_language
        self.ocr_quality = quality
        
        logger.info(f"📷 OCR Quality: {quality} - {self.ocr_config['description']}")
        logger.info(f"🌐 OCR Language: {ocr_language}")
        logger.info(f"⚡ Parallel Processing: {'Enabled' if parallel_processing else 'Disabled'} (workers={max_workers})")
        
        # Initialize text cleaner
        self.text_cleaner = TextCleaner()
    
    # ── Public API ────────────────────────────────────────────────────────────
    
    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """Parse a single PDF file with advanced OCR capabilities"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        start_time = time.time()
        logger.info(f"📄 Parsing: {pdf_path.name}")
        
        # Extract metadata and outline
        metadata = self._extract_metadata(pdf_path)
        outline = self._extract_outline(pdf_path)
        
        # Check if PDF is scanned
        is_scanned = self._is_scanned_pdf(pdf_path)
        metadata.is_scanned = is_scanned
        metadata.ocr_quality = self.ocr_quality if is_scanned else ""
        metadata.ocr_language = self.ocr_language if is_scanned else ""
        
        # Extract pages based on PDF type
        if is_scanned and self.use_ocr:
            logger.info(f"📸 '{pdf_path.name}' detected as scanned PDF. Using OCR...")
            pages = self._extract_pages_with_ocr_advanced(pdf_path)
        else:
            pages = self._extract_pages(pdf_path)
        
        metadata.has_toc = len(outline) > 0
        metadata.total_processing_time = time.time() - start_time
        
        doc = ParsedDocument(metadata=metadata, outline=outline, pages=pages)
        
        # Log statistics
        stats = doc.get_statistics()
        logger.success(
            f"✅ Parsed '{pdf_path.name}' — "
            f"{stats['total_pages']} pages | "
            f"{'🔍 OCR' if metadata.is_scanned else '📝 Text'} | "
            f"{stats['total_headings']} headings | "
            f"{stats['total_tables']} tables | "
            f"Time: {stats['total_processing_time']:.2f}s"
        )
        
        return doc
    
    def parse_directory(self, dir_path: str | Path = PDF_INPUT_DIR) -> list[ParsedDocument]:
        """Parse all PDF files in a directory"""
        dir_path = Path(dir_path)
        dir_path.mkdir(exist_ok=True)
        
        pdf_files = sorted(dir_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in: {dir_path}")
            logger.info(f"Please add PDF files to: {dir_path.absolute()}")
            return []
        
        logger.info(f"📁 Found {len(pdf_files)} PDF(s) in '{dir_path}'")
        documents = []
        
        for pdf_file in pdf_files:
            try:
                doc = self.parse(pdf_file)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to parse '{pdf_file.name}': {e}")
        
        return documents
    
    # ── PDF Type Detection ────────────────────────────────────────────────────
    
    def _is_scanned_pdf(self, pdf_path: Path) -> bool:
        """Detect if PDF is scanned (image-based)"""
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                pages_to_check = min(3, len(pdf.pages))
                text_found = False
                
                for i in range(pages_to_check):
                    text = pdf.pages[i].extract_text() or ""
                    if text.strip():
                        text_found = True
                        break
                
                return not text_found
        except Exception as e:
            logger.debug(f"Error checking PDF type: {e}")
            return True
    
    # ── Metadata Extraction ───────────────────────────────────────────────────
    
    def _extract_metadata(self, pdf_path: Path) -> DocumentMetadata:
        """Extract document metadata"""
        doc = fitz.open(str(pdf_path))
        meta = doc.metadata or {}
        page_count = doc.page_count
        doc.close()
        
        return DocumentMetadata(
            file_name=pdf_path.name,
            file_path=str(pdf_path.resolve()),
            page_count=page_count,
            title=meta.get("title", "").strip(),
            author=meta.get("author", "").strip(),
            subject=meta.get("subject", "").strip(),
            creator=meta.get("creator", "").strip(),
        )
    
    def _extract_outline(self, pdf_path: Path) -> list[dict]:
        """Extract table of contents"""
        doc = fitz.open(str(pdf_path))
        toc = doc.get_toc()
        doc.close()
        return [{"level": level, "title": title.strip(), "page": page} 
                for level, title, page in toc]
    
    # ─── Text-based PDF Extraction ────────────────────────────────────────────
    
    def _extract_pages(self, pdf_path: Path) -> list[ParsedPage]:
        """Extract content from text-based PDFs"""
        pages = []
        
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_number = i + 1
                
                try:
                    start_time = time.time()
                    parsed_page = self._parse_single_page(page, page_number)
                    parsed_page.processing_time = time.time() - start_time
                    pages.append(parsed_page)
                except Exception as e:
                    logger.warning(f"  ⚠ Page {page_number} failed: {e}")
                    pages.append(ParsedPage(
                        page_number=page_number,
                        width=page.width,
                        height=page.height,
                        raw_text=""
                    ))
        
        return pages
    
    def _parse_single_page(self, page, page_number: int) -> ParsedPage:
        """Parse a single page from text-based PDF"""
        raw_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
        tables = self._extract_tables(page, page_number)
        blocks, headings = self._extract_blocks_and_headings(page, page_number)
        
        return ParsedPage(
            page_number=page_number,
            width=page.width,
            height=page.height,
            raw_text=raw_text,
            headings=headings,
            blocks=blocks,
            tables=tables,
        )
    
    # ─── Advanced OCR Extraction ──────────────────────────────────────────────
    
    def _extract_pages_with_ocr_advanced(self, pdf_path: Path) -> list[ParsedPage]:
        """Advanced OCR extraction with parallel processing and quality options"""
        
        if self.parallel_processing:
            return self._extract_pages_parallel(pdf_path)
        else:
            return self._extract_pages_sequential(pdf_path)
    
    def _extract_pages_sequential(self, pdf_path: Path) -> list[ParsedPage]:
        """Sequential OCR processing (slower but uses less memory)"""
        pages = []
        
        try:
            logger.info(f"  🔍 Running OCR with {self.ocr_quality} quality preset...")
            
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                page_start = time.time()
                page = doc[page_num]
                
                logger.debug(f"    Page {page_num+1}/{total_pages} - OCR processing...")
                
                try:
                    # Perform OCR
                    text = self._perform_ocr_on_page(page)
                    
                    # Clean text
                    text = self.text_cleaner.clean_ocr_text(text)
                    
                except Exception as ocr_err:
                    logger.warning(f"      OCR error on page {page_num+1}: {ocr_err}")
                    text = ""
                
                pages.append(ParsedPage(
                    page_number=page_num + 1,
                    width=page.rect.width,
                    height=page.rect.height,
                    raw_text=text,
                    headings=[],
                    blocks=[],
                    tables=[],
                    processing_time=time.time() - page_start
                ))
            
            doc.close()
            
            pages_with_text = sum(1 for p in pages if p.raw_text)
            logger.info(f"  ✅ OCR complete: {total_pages} pages, {pages_with_text} with text")
            
        except Exception as e:
            logger.error(f"  ❌ OCR failed: {e}")
            pages = self._create_empty_pages(pdf_path)
        
        return pages
    
    def _extract_pages_parallel(self, pdf_path: Path) -> list[ParsedPage]:
        """Parallel OCR processing (faster for multi-page documents)"""
        pages = [None] * self._get_page_count(pdf_path)
        
        try:
            logger.info(f"  🔍 Running parallel OCR with {self.max_workers} workers...")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                
                doc = fitz.open(str(pdf_path))
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    future = executor.submit(self._ocr_page_worker, page, page_num + 1)
                    futures[future] = page_num
                
                for future in as_completed(futures):
                    page_num = futures[future]
                    try:
                        page_data = future.result(timeout=self.ocr_config['timeout'])
                        pages[page_num] = page_data
                    except Exception as e:
                        logger.error(f"      Page {page_num+1} failed: {e}")
                        pages[page_num] = ParsedPage(
                            page_number=page_num+1,
                            width=0, height=0, raw_text=""
                        )
                
                doc.close()
            
            # Filter out None values
            pages = [p for p in pages if p is not None]
            
            pages_with_text = sum(1 for p in pages if p.raw_text)
            logger.info(f"  ✅ Parallel OCR complete: {len(pages)} pages, {pages_with_text} with text")
            
        except Exception as e:
            logger.error(f"  ❌ Parallel OCR failed: {e}")
            pages = self._create_empty_pages(pdf_path)
        
        return pages
    
    def _ocr_page_worker(self, page, page_num: int) -> ParsedPage:
        """Worker function for parallel OCR processing"""
        page_start = time.time()
        
        try:
            text = self._perform_ocr_on_page(page)
            text = self.text_cleaner.clean_ocr_text(text)
            
            return ParsedPage(
                page_number=page_num,
                width=page.rect.width,
                height=page.rect.height,
                raw_text=text,
                headings=[],
                blocks=[],
                tables=[],
                processing_time=time.time() - page_start
            )
        except Exception as e:
            logger.error(f"      Worker failed for page {page_num}: {e}")
            return ParsedPage(
                page_number=page_num,
                width=0, height=0, raw_text="",
                processing_time=time.time() - page_start
            )
    
    def _perform_ocr_on_page(self, page) -> str:
        """Perform OCR on a single page with current settings"""
        try:
            # Use PyMuPDF's OCR
            textpage = page.get_textpage_ocr(
                language=self.ocr_language,
                dpi=self.ocr_config['dpi'],
                flags=0,
                tessdata=True
            )
            
            if textpage:
                text = textpage.extractText()
            else:
                text = ""
            
            # Optional: Preprocessing for better quality
            if self.ocr_config.get('preprocess', False):
                text = self._enhance_ocr_text(text)
            
            return text
            
        except Exception as e:
            logger.debug(f"PyMuPDF OCR error: {e}")
            # Fallback to pytesseract directly
            try:
                # Need to convert page to image first
                pix = page.get_pixmap(dpi=self.ocr_config['dpi'])
                img_data = pix.tobytes("png")
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(img_data))
                text = pytesseract.image_to_string(img, lang=self.ocr_language)
                return text
            except:
                raise e
    
    def _enhance_ocr_text(self, text: str) -> str:
        """Enhance OCR text with additional post-processing"""
        if not text:
            return text
        
        # Remove page numbers and headers (common artifacts)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that are likely page numbers
            if re.match(r'^\s*\d+\s*$', line):
                continue
            # Skip lines that are likely headers
            if len(line.strip()) < 3:
                continue
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Fix hyphenated words
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        return text
    
    def _get_page_count(self, pdf_path: Path) -> int:
        """Get total page count of PDF"""
        doc = fitz.open(str(pdf_path))
        count = doc.page_count
        doc.close()
        return count
    
    def _create_empty_pages(self, pdf_path: Path) -> list[ParsedPage]:
        """Create empty pages as fallback"""
        pages = []
        with fitz.open(str(pdf_path)) as doc:
            for i in range(doc.page_count):
                pages.append(ParsedPage(
                    page_number=i+1,
                    width=0, height=0, raw_text=""
                ))
        return pages
    
    # ── Table Extraction ──────────────────────────────────────────────────────
    
    def _extract_tables(self, page, page_number: int) -> list[TableData]:
        """Extract tables from pdfplumber page"""
        tables = []
        raw_tables = page.extract_tables()
        
        for idx, raw_table in enumerate(raw_tables):
            if not raw_table:
                continue
            
            clean_rows = []
            for row in raw_table:
                clean_row = [
                    (cell.strip() if isinstance(cell, str) else "") if cell is not None else ""
                    for cell in row
                ]
                if any(cell for cell in clean_row):
                    clean_rows.append(clean_row)
            
            if not clean_rows:
                continue
            
            table_objects = page.find_tables()
            bbox = table_objects[idx].bbox if idx < len(table_objects) else (0, 0, 0, 0)
            
            tables.append(TableData(
                page_number=page_number,
                table_index=idx,
                rows=clean_rows,
                bbox=bbox,
            ))
        
        return tables
    
    # ── Text Blocks & Headings ────────────────────────────────────────────────
    
    def _extract_blocks_and_headings(self, page, page_number: int) -> tuple[list[TextBlock], list[Heading]]:
        """Extract text blocks and detect headings"""
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            extra_attrs=["fontname", "size"],
            keep_blank_chars=False,
        )
        
        if not words:
            return [], []
        
        line_groups = self._group_words_into_lines(words)
        return self._build_blocks(line_groups, page_number)
    
    def _group_words_into_lines(self, words: list[dict]) -> list[list[dict]]:
        """Group words into lines"""
        if not words:
            return []
        
        lines = []
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
        
        for line in lines:
            line.sort(key=lambda w: w["x0"])
        
        return lines
    
    def _build_blocks(self, line_groups: list[list[dict]], page_number: int) -> tuple[list[TextBlock], list[Heading]]:
        """Build text blocks and detect headings"""
        if not line_groups:
            return [], []
        
        all_sizes = []
        for line in line_groups:
            for w in line:
                sz = w.get("size", 0)
                if sz:
                    all_sizes.append(sz)
        
        if all_sizes:
            body_size = sorted(all_sizes)[int(len(all_sizes) * 0.5)]
            heading_threshold = max(body_size + 1.0, self.heading_min_size)
        else:
            heading_threshold = self.heading_min_size
        
        blocks = []
        headings = []
        block_index = 0
        current_block_lines = [line_groups[0]]
        prev_bottom = max(w["bottom"] for w in line_groups[0])
        
        def flush_block(block_lines):
            nonlocal block_index
            all_words = [w for line in block_lines for w in line]
            text = " ".join(w["text"] for w in all_words).strip()
            text = re.sub(r"\s{2,}", " ", text)
            
            if not text:
                return
            
            sizes = [w.get("size", 0) for w in all_words if w.get("size")]
            avg_size = sum(sizes) / len(sizes) if sizes else 0.0
            font_names = [w.get("fontname", "") for w in all_words if w.get("fontname")]
            font_name = max(set(font_names), key=font_names.count) if font_names else ""
            is_bold = "bold" in font_name.lower() or "Bold" in font_name
            
            x0 = min(w["x0"] for w in all_words)
            y0 = min(w["top"] for w in all_words)
            x1 = max(w["x1"] for w in all_words)
            y1 = max(w["bottom"] for w in all_words)
            
            tb = TextBlock(
                text=text,
                page_number=page_number,
                block_index=block_index,
                bbox=(x0, y0, x1, y1),
                font_size=round(avg_size, 2),
                font_name=font_name,
                is_bold=is_bold,
            )
            blocks.append(tb)
            block_index += 1
            
            is_large = avg_size >= heading_threshold
            is_short = len(text.split()) <= 15
            if (is_large or is_bold) and is_short:
                if avg_size >= heading_threshold + 4:
                    level = 1
                elif avg_size >= heading_threshold + 1:
                    level = 2
                else:
                    level = 3
                headings.append(Heading(
                    text=text, page_number=page_number, level=level,
                    font_size=round(avg_size, 2), bbox=(x0, y0, x1, y1)
                ))
        
        for line in line_groups[1:]:
            line_top = min(w["top"] for w in line)
            gap = line_top - prev_bottom
            if gap > 8.0:
                flush_block(current_block_lines)
                current_block_lines = [line]
            else:
                current_block_lines.append(line)
            prev_bottom = max(w["bottom"] for w in line)
        
        if current_block_lines:
            flush_block(current_block_lines)
        
        return blocks, headings


# ─── Utility Functions ────────────────────────────────────────────────────────

def print_document_summary(doc: ParsedDocument) -> None:
    """Pretty-print document summary"""
    print("\n" + "═" * 70)
    print(f"  📄 {doc.metadata.file_name}")
    print("═" * 70)
    print(f"  Pages       : {doc.metadata.page_count}")
    print(f"  Title       : {doc.metadata.title or '(none)'}")
    print(f"  Author      : {doc.metadata.author or '(none)'}")
    print(f"  Type        : {'🔍 Scanned (OCR)' if doc.metadata.is_scanned else '📝 Text-based'}")
    
    if doc.metadata.is_scanned:
        print(f"  OCR Quality : {doc.metadata.ocr_quality}")
        print(f"  OCR Language: {doc.metadata.ocr_language}")
    
    print(f"  Headings    : {len(doc.get_all_headings())}")
    print(f"  Tables      : {len(doc.get_all_tables())}")
    print(f"  Time        : {doc.metadata.total_processing_time:.2f} seconds")
    
    # Show preview
    if doc.pages and doc.pages[0].raw_text:
        preview = doc.pages[0].raw_text[:200].replace('\n', ' ')
        print(f"\n  📝 Page 1 Preview: {preview[:150]}...")
    
    print("═" * 70 + "\n")


def list_available_languages():
    """Print available OCR languages"""
    print("\n🌐 Available OCR Languages:")
    print("-" * 40)
    for code, name in OCR_LANGUAGES.items():
        print(f"  {code:10} - {name}")
    print("\n💡 Use '+' for multiple languages: eng+fra+deu")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced PDF Parser with OCR")
    parser.add_argument("pdf_path", nargs="?", help="Path to PDF file (optional)")
    parser.add_argument("--quality", default="BALANCED", 
                       choices=["FAST", "BALANCED", "HIGH_QUALITY", "VERY_HIGH", "MAXIMUM"],
                       help="OCR quality preset")
    parser.add_argument("--language", default="eng",
                       help="OCR language (e.g., 'eng', 'eng+fra')")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Enable parallel processing")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--list-languages", action="store_true",
                       help="List available OCR languages")
    
    args = parser.parse_args()
    
    if args.list_languages:
        list_available_languages()
        sys.exit(0)
    
    # Initialize parser with advanced settings
    pdf_parser = PDFParser(
        ocr_quality=args.quality,
        ocr_language=args.language,
        parallel_processing=args.parallel,
        max_workers=args.workers
    )
    
    if args.pdf_path:
        # Parse single PDF
        doc = pdf_parser.parse(args.pdf_path)
        print_document_summary(doc)
        
        # Show full page 1 text
        if doc.pages and doc.pages[0].raw_text:
            print("\n── Page 1 Full Text ──────────────────────────────")
            print(doc.pages[0].raw_text[:1000])
            print("─" * 50)
    
    else:
        # Parse all PDFs in data directory
        Path(PDF_INPUT_DIR).mkdir(exist_ok=True)
        docs = pdf_parser.parse_directory(PDF_INPUT_DIR)
        
        for doc in docs:
            print_document_summary(doc)
        
        # Print summary statistics
        if docs:
            total_pages = sum(d.metadata.page_count for d in docs)
            total_time = sum(d.metadata.total_processing_time for d in docs)
            print(f"\n📊 TOTAL: {len(docs)} documents, {total_pages} pages, {total_time:.2f} seconds")