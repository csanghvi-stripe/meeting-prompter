"""Token-aware document chunking for ColBERT."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader


@dataclass
class Chunk:
    """A document chunk with metadata."""

    id: str
    text: str
    source: str
    token_count: int


class TokenAwareChunker:
    """
    Token-aware chunking optimized for ColBERT.

    ColBERT has a max document length of 512 tokens. We target 400 tokens
    per chunk to leave headroom for special tokens, with 50 token overlap
    between chunks to preserve context at boundaries.
    """

    def __init__(
        self,
        tokenizer,
        target_tokens: int = 400,
        overlap_tokens: int = 50,
        min_tokens: int = 30,
    ):
        """
        Initialize the chunker.

        Args:
            tokenizer: HuggingFace tokenizer from the ColBERT model
            target_tokens: Target number of tokens per chunk (default 400)
            overlap_tokens: Number of overlapping tokens between chunks (default 50)
            min_tokens: Minimum tokens to keep a chunk (default 30)
        """
        self.tokenizer = tokenizer
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens

    def chunk_text(self, text: str, source: str) -> List[Chunk]:
        """
        Chunk text into token-aware segments.

        Args:
            text: The text to chunk
            source: Source filename for attribution

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        # Split into sentences for semantic boundaries
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_token_count = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = len(
                self.tokenizer.encode(sentence, add_special_tokens=False)
            )

            # If single sentence exceeds target, split it
            if sentence_tokens > self.target_tokens:
                # Flush current buffer first
                if current_sentences:
                    chunk = self._create_chunk(
                        current_sentences, current_token_count, source, chunk_index
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
                    current_sentences = []
                    current_token_count = 0

                # Split long sentence by character limit
                sub_chunks = self._split_long_sentence(sentence, source, chunk_index)
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
                continue

            # Check if adding this sentence exceeds target
            if current_token_count + sentence_tokens > self.target_tokens:
                # Create chunk from buffer
                chunk = self._create_chunk(
                    current_sentences, current_token_count, source, chunk_index
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new buffer with overlap
                overlap_sentences, overlap_tokens = self._get_overlap(
                    current_sentences, current_token_count
                )
                current_sentences = overlap_sentences
                current_token_count = overlap_tokens

            current_sentences.append(sentence)
            current_token_count += sentence_tokens

        # Don't forget the last chunk
        if current_sentences:
            chunk = self._create_chunk(
                current_sentences, current_token_count, source, chunk_index
            )
            if chunk:
                chunks.append(chunk)

        return chunks

    def chunk_pdf(self, pdf_path: Path) -> List[Chunk]:
        """Load and chunk a PDF file."""
        try:
            reader = PdfReader(pdf_path)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + " "

            # Clean the text
            full_text = re.sub(r"\s+", " ", full_text).strip()
            return self.chunk_text(full_text, pdf_path.name)

        except Exception as e:
            print(f"Warning: Failed to load {pdf_path.name}: {e}")
            return []

    def chunk_markdown(self, md_path: Path) -> List[Chunk]:
        """
        Load and chunk a markdown file with section awareness.

        Splits on markdown headers (## and ###) to preserve document structure.
        Each chunk includes its section header for context.
        """
        try:
            full_text = md_path.read_text(encoding="utf-8")

            # Split by markdown headers (## and ###)
            sections = self._split_by_headers(full_text)

            all_chunks = []
            for section_header, section_text in sections:
                # Prepend header to section text for context
                if section_header:
                    section_with_header = f"{section_header}\n\n{section_text}"
                else:
                    section_with_header = section_text

                # Clean the section text
                section_with_header = re.sub(r"\n{3,}", "\n\n", section_with_header)
                section_with_header = re.sub(r" +", " ", section_with_header).strip()

                # Chunk this section
                chunks = self.chunk_text(section_with_header, md_path.name)
                all_chunks.extend(chunks)

            return all_chunks

        except Exception as e:
            print(f"Warning: Failed to load {md_path.name}: {e}")
            return []

    def _split_by_headers(self, text: str) -> List[tuple]:
        """
        Split markdown text by headers.

        Returns list of (header, content) tuples.
        Headers are preserved and prepended to content.
        """
        # Match ## or ### headers
        header_pattern = r'^(#{2,3}\s+.+)$'

        lines = text.split('\n')
        sections = []
        current_header = ""
        current_content = []

        for line in lines:
            if re.match(header_pattern, line.strip()):
                # Save previous section
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections.append((current_header, content))

                # Start new section
                current_header = line.strip()
                current_content = []
            else:
                current_content.append(line)

        # Don't forget the last section
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections.append((current_header, content))

        return sections

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - handles common cases
        # Split on . ! ? followed by space and capital letter or end of string
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$", text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_chunk(
        self, sentences: List[str], token_count: int, source: str, index: int
    ) -> Optional[Chunk]:
        """Create a Chunk object from sentences."""
        if token_count < self.min_tokens:
            return None

        text = " ".join(sentences)
        chunk_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        chunk_id = f"{source}_{index}_{chunk_hash}"

        return Chunk(id=chunk_id, text=text, source=source, token_count=token_count)

    def _get_overlap(
        self, sentences: List[str], total_tokens: int
    ) -> tuple[List[str], int]:
        """Get overlap sentences from the end of the buffer."""
        if not sentences:
            return [], 0

        overlap_sentences = []
        overlap_tokens = 0

        # Work backwards to get overlap
        for sentence in reversed(sentences):
            sentence_tokens = len(
                self.tokenizer.encode(sentence, add_special_tokens=False)
            )
            if overlap_tokens + sentence_tokens > self.overlap_tokens:
                break
            overlap_sentences.insert(0, sentence)
            overlap_tokens += sentence_tokens

        return overlap_sentences, overlap_tokens

    def _split_long_sentence(
        self, sentence: str, source: str, start_index: int
    ) -> List[Chunk]:
        """Split a sentence that exceeds target tokens."""
        chunks = []
        words = sentence.split()
        current_words = []
        current_tokens = 0
        chunk_index = start_index

        for word in words:
            word_tokens = len(self.tokenizer.encode(word, add_special_tokens=False))

            if current_tokens + word_tokens > self.target_tokens:
                if current_words:
                    text = " ".join(current_words)
                    chunk_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
                    chunks.append(
                        Chunk(
                            id=f"{source}_{chunk_index}_{chunk_hash}",
                            text=text,
                            source=source,
                            token_count=current_tokens,
                        )
                    )
                    chunk_index += 1
                current_words = []
                current_tokens = 0

            current_words.append(word)
            current_tokens += word_tokens

        # Last chunk
        if current_words and current_tokens >= self.min_tokens:
            text = " ".join(current_words)
            chunk_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=f"{source}_{chunk_index}_{chunk_hash}",
                    text=text,
                    source=source,
                    token_count=current_tokens,
                )
            )

        return chunks
