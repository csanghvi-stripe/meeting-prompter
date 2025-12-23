"""RAG Engine - Lightweight retrieval for document Q&A"""
import re
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader


class RAGEngine:
    """Lightweight RAG using keyword matching - loads all docs from folder"""

    def __init__(self, docs_dir: Path, chunk_size: int = 500, chunk_overlap: int = 100):
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[Tuple[str, str]] = []  # (text, source_filename)
        self._load_all_documents()

    def _load_all_documents(self):
        """Load all PDFs and Markdown files from docs directory"""
        if not self.docs_dir.exists():
            print(f"Warning: Docs directory not found: {self.docs_dir}")
            self.chunks = [("No documents loaded.", "none")]
            return

        pdf_files = list(self.docs_dir.glob("*.pdf"))
        md_files = list(self.docs_dir.glob("*.md"))
        all_files = pdf_files + md_files

        if not all_files:
            print(f"Warning: No PDF or Markdown files found in {self.docs_dir}")
            self.chunks = [("No documents loaded.", "none")]
            return

        total_chunks = 0
        for file_path in all_files:
            if file_path.suffix == ".pdf":
                chunks = self._load_pdf(file_path)
            else:
                chunks = self._load_markdown(file_path)
            for chunk in chunks:
                self.chunks.append((chunk, file_path.name))
            total_chunks += len(chunks)
            print(f"  Loaded {len(chunks)} chunks from {file_path.name}")

        print(f"RAG engine ready: {total_chunks} chunks from {len(all_files)} documents")

    def _load_pdf(self, pdf_path: Path) -> List[str]:
        """Load and chunk a single PDF"""
        try:
            reader = PdfReader(pdf_path)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + " "

            # Clean the text
            full_text = re.sub(r'\s+', ' ', full_text).strip()

            # Split into overlapping chunks
            chunks = []
            step = self.chunk_size - self.chunk_overlap
            for i in range(0, len(full_text), step):
                chunk = full_text[i:i + self.chunk_size]
                if len(chunk) > 50:  # Skip very short chunks
                    chunks.append(chunk)

            return chunks

        except Exception as e:
            print(f"Warning: Failed to load {pdf_path.name}: {e}")
            return []

    def _load_markdown(self, md_path: Path) -> List[str]:
        """Load and chunk a markdown file"""
        try:
            full_text = md_path.read_text(encoding='utf-8')

            # Clean the text (remove excessive whitespace but keep structure)
            full_text = re.sub(r'\n{3,}', '\n\n', full_text)
            full_text = re.sub(r' +', ' ', full_text).strip()

            # Split into overlapping chunks
            chunks = []
            step = self.chunk_size - self.chunk_overlap
            for i in range(0, len(full_text), step):
                chunk = full_text[i:i + self.chunk_size]
                if len(chunk) > 50:  # Skip very short chunks
                    chunks.append(chunk)

            return chunks

        except Exception as e:
            print(f"Warning: Failed to load {md_path.name}: {e}")
            return []

    def query(self, text: str) -> Tuple[str, float, str]:
        """
        Find most relevant chunk using Jaccard similarity.

        Returns:
            Tuple of (best_chunk, confidence_score, source_filename)
        """
        if not text or not text.strip():
            return "", 0.0, ""

        # Tokenize query
        query_words = set(self._tokenize(text))
        if not query_words:
            return "", 0.0, ""

        best_chunk = ""
        best_score = 0.0
        best_source = ""

        for chunk_text, source in self.chunks:
            chunk_words = set(self._tokenize(chunk_text))

            # Query coverage: what % of query words appear in this chunk
            intersection = len(query_words & chunk_words)
            score = intersection / len(query_words)

            if score > best_score:
                best_score = score
                best_chunk = chunk_text
                best_source = source

        return best_chunk, best_score, best_source

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, remove punctuation, split"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [w for w in text.split() if len(w) > 2]

    def get_context_preview(self, chunk: str, max_length: int = 100) -> str:
        """Get a preview of the context chunk"""
        if len(chunk) <= max_length:
            return chunk
        return chunk[:max_length] + "..."


def format_confidence(score: float) -> str:
    """Format confidence score for display"""
    percentage = score * 100
    if percentage >= 50:
        return f"🟢 {percentage:.0f}%"
    elif percentage >= 25:
        return f"🟡 {percentage:.0f}%"
    else:
        return f"🔴 {percentage:.0f}%"
