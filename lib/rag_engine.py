"""RAG Engine - Lightweight retrieval for Liquid specs"""
import re
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader


class RAGEngine:
    """Lightweight RAG using keyword matching (no embeddings for 8GB constraint)"""

    def __init__(self, pdf_path: Path, chunk_size: int = 500, chunk_overlap: int = 100):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[str] = []
        self._load_document()

    def _load_document(self):
        """Load and chunk the PDF document"""
        if not self.pdf_path.exists():
            print(f"Warning: RAG document not found: {self.pdf_path}")
            self.chunks = ["Liquid Neural Networks (LNNs) use ODEs for continuous data streams."]
            return

        try:
            reader = PdfReader(self.pdf_path)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + " "

            # Clean the text
            full_text = re.sub(r'\s+', ' ', full_text).strip()

            # Split into overlapping chunks
            self.chunks = []
            step = self.chunk_size - self.chunk_overlap
            for i in range(0, len(full_text), step):
                chunk = full_text[i:i + self.chunk_size]
                if len(chunk) > 50:  # Skip very short chunks
                    self.chunks.append(chunk)

            print(f"RAG loaded {len(self.chunks)} chunks from {self.pdf_path.name}")

        except Exception as e:
            print(f"Warning: Failed to load RAG document: {e}")
            self.chunks = ["Liquid Neural Networks (LNNs) use ODEs for continuous data streams."]

    def query(self, transcript: str, top_k: int = 1) -> Tuple[str, float]:
        """
        Find most relevant chunk using Jaccard similarity.

        Returns:
            Tuple of (best_chunk, confidence_score)
        """
        if not transcript or not transcript.strip():
            return "", 0.0

        # Tokenize query
        query_words = set(self._tokenize(transcript))
        if not query_words:
            return "", 0.0

        best_chunk = ""
        best_score = 0.0

        for chunk in self.chunks:
            chunk_words = set(self._tokenize(chunk))

            # Jaccard similarity
            intersection = len(query_words & chunk_words)
            union = len(query_words | chunk_words)

            if union > 0:
                score = intersection / union
                if score > best_score:
                    best_score = score
                    best_chunk = chunk

        return best_chunk, best_score

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
