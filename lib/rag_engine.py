"""RAG Engine - Semantic retrieval for document Q&A using ColBERT."""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from pypdf import PdfReader


class RAGEngine:
    """
    RAG engine with ColBERT semantic retrieval and Jaccard fallback.

    By default, uses LFM2-ColBERT-350M for semantic understanding.
    Falls back to keyword-based Jaccard similarity if ColBERT fails to load.
    """

    def __init__(self, docs_dir: Path, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the RAG engine.

        Args:
            docs_dir: Directory containing PDF and Markdown documents
            chunk_size: Character chunk size for Jaccard fallback (default 500)
            chunk_overlap: Character overlap for Jaccard fallback (default 100)
        """
        self.docs_dir = Path(docs_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # ColBERT retriever (primary)
        self._colbert = None
        self._use_colbert = not os.environ.get("RAG_USE_FALLBACK", "").lower() in (
            "1",
            "true",
            "yes",
        )

        # Jaccard fallback data
        self.chunks: List[Tuple[str, str]] = []  # (text, source_filename)

        # Try to initialize ColBERT
        if self._use_colbert:
            try:
                self._init_colbert()
            except Exception as e:
                print(f"ColBERT unavailable, using Jaccard fallback: {e}")
                self._use_colbert = False
                self._load_all_documents()
        else:
            print("RAG_USE_FALLBACK is set, using Jaccard retrieval")
            self._load_all_documents()

    def _init_colbert(self) -> None:
        """Initialize ColBERT retriever."""
        from lib.colbert import ColBERTRetriever

        # Default index directory
        index_dir = Path("data/colbert_index")
        index_dir.mkdir(parents=True, exist_ok=True)

        self._colbert = ColBERTRetriever(index_dir, self.docs_dir)
        self._colbert.load()

    def query(self, text: str) -> Tuple[str, float, str]:
        """
        Find most relevant chunk for the query.

        Args:
            text: Query text

        Returns:
            Tuple of (best_chunk, confidence_score, source_filename)
        """
        if not text or not text.strip():
            return "", 0.0, ""

        if self._use_colbert and self._colbert:
            return self._query_colbert(text)
        else:
            return self._query_jaccard(text)

    def _query_colbert(self, text: str) -> Tuple[str, float, str]:
        """
        Query using ColBERT semantic retrieval.

        Retrieves top-3 results and combines them for richer context.
        This helps when the answer spans multiple chunks or when the
        best semantic match isn't the most relevant answer.
        """
        results = self._colbert.query(text, k=3)

        if not results:
            return "", 0.0, ""

        # Take top result's confidence and source
        _, top_confidence, top_source = results[0]

        # Combine top 3 chunks for richer context
        # Deduplicate and join with clear separators
        seen_texts = set()
        combined_chunks = []

        for chunk_text, confidence, source in results:
            # Skip duplicates or near-duplicates
            text_key = chunk_text[:100].lower()
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            # Only include if confidence is reasonable (within 20% of top)
            if confidence >= top_confidence * 0.8:
                combined_chunks.append(chunk_text)

        # Join with clear separator
        combined_context = "\n\n---\n\n".join(combined_chunks)

        return combined_context, top_confidence, top_source

    def _query_jaccard(self, text: str) -> Tuple[str, float, str]:
        """
        Query using Jaccard similarity (fallback).

        Returns:
            Tuple of (best_chunk, confidence_score, source_filename)
        """
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

    def _load_all_documents(self) -> None:
        """Load all PDFs and Markdown files for Jaccard fallback."""
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

        print(f"RAG engine ready (Jaccard): {total_chunks} chunks from {len(all_files)} documents")

    def _load_pdf(self, pdf_path: Path) -> List[str]:
        """Load and chunk a single PDF."""
        try:
            reader = PdfReader(pdf_path)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + " "

            # Clean the text
            full_text = re.sub(r"\s+", " ", full_text).strip()

            # Split into overlapping chunks
            chunks = []
            step = self.chunk_size - self.chunk_overlap
            for i in range(0, len(full_text), step):
                chunk = full_text[i : i + self.chunk_size]
                if len(chunk) > 50:  # Skip very short chunks
                    chunks.append(chunk)

            return chunks

        except Exception as e:
            print(f"Warning: Failed to load {pdf_path.name}: {e}")
            return []

    def _load_markdown(self, md_path: Path) -> List[str]:
        """Load and chunk a markdown file."""
        try:
            full_text = md_path.read_text(encoding="utf-8")

            # Clean the text (remove excessive whitespace but keep structure)
            full_text = re.sub(r"\n{3,}", "\n\n", full_text)
            full_text = re.sub(r" +", " ", full_text).strip()

            # Split into overlapping chunks
            chunks = []
            step = self.chunk_size - self.chunk_overlap
            for i in range(0, len(full_text), step):
                chunk = full_text[i : i + self.chunk_size]
                if len(chunk) > 50:  # Skip very short chunks
                    chunks.append(chunk)

            return chunks

        except Exception as e:
            print(f"Warning: Failed to load {md_path.name}: {e}")
            return []

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, remove punctuation, split."""
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return [w for w in text.split() if len(w) > 2]

    def get_context_preview(self, chunk: str, max_length: int = 100) -> str:
        """Get a preview of the context chunk."""
        if len(chunk) <= max_length:
            return chunk
        return chunk[:max_length] + "..."

    @property
    def is_using_colbert(self) -> bool:
        """Check if ColBERT is active."""
        return self._use_colbert and self._colbert is not None

    def rebuild_index(self) -> None:
        """Force rebuild of the ColBERT index."""
        if self._colbert:
            self._colbert.rebuild_index()


def format_confidence(score: float) -> str:
    """Format confidence score for display."""
    percentage = score * 100
    if percentage >= 50:
        return f"[green]{percentage:.0f}%[/green]"
    elif percentage >= 25:
        return f"[yellow]{percentage:.0f}%[/yellow]"
    else:
        return f"[red]{percentage:.0f}%[/red]"
