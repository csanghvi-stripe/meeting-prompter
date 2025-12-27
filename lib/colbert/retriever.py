"""ColBERT retriever using LFM2-ColBERT-350M and PLAID index."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .chunker import Chunk, TokenAwareChunker
from .index_manager import IndexManager
from .normalizer import normalize_maxsim


class ColBERTRetriever:
    """
    ColBERT-based semantic retriever using LiquidAI's LFM2-ColBERT-350M model.

    Uses PLAID indexing for efficient similarity search with late interaction
    and MaxSim scoring.
    """

    MODEL_NAME = "LiquidAI/LFM2-ColBERT-350M"

    def __init__(self, index_dir: Path, docs_dir: Path):
        """
        Initialize the ColBERT retriever.

        Args:
            index_dir: Directory for PLAID index storage
            docs_dir: Directory containing source documents
        """
        self.index_dir = Path(index_dir)
        self.docs_dir = Path(docs_dir)

        self._model = None
        self._index = None
        self._retriever = None
        self._chunker = None
        self._index_manager = IndexManager(index_dir, docs_dir)
        self._chunks_metadata: Dict[str, Dict[str, str]] = {}
        self._is_loaded = False

    def load(self) -> None:
        """
        Load the ColBERT model and index.

        This should be called at startup. The model is loaded first,
        then the index is either loaded from disk or built from documents.
        """
        if self._is_loaded:
            return

        print("Loading ColBERT model (LFM2-ColBERT-350M)...")
        self._load_model()

        # Initialize chunker with model's tokenizer
        self._chunker = TokenAwareChunker(self._model.tokenizer)

        # Check if index needs rebuild
        if self._index_manager.needs_rebuild():
            print("Building PLAID index (this may take 30-60 seconds)...")
            self._build_index()
        else:
            print("Loading existing PLAID index...")
            self._load_index()

        self._is_loaded = True
        print(f"ColBERT ready: {len(self._chunks_metadata)} chunks indexed")

    def query(self, text: str, k: int = 1) -> List[Tuple[str, float, str]]:
        """
        Query the index for relevant chunks.

        Args:
            text: Query text
            k: Number of results to return

        Returns:
            List of (chunk_text, confidence, source_file) tuples
        """
        if not self._is_loaded:
            raise RuntimeError("ColBERT not loaded. Call load() first.")

        if not text or not text.strip():
            return []

        # Encode query
        query_embedding = self._model.encode(
            [text],
            batch_size=1,
            is_query=True,
            show_progress_bar=False,
        )

        # Retrieve from index
        results = self._retriever.retrieve(
            queries_embeddings=query_embedding,
            k=k,
        )

        # Process results
        output = []
        for result in results[0]:  # First query's results
            chunk_id = result["id"]
            raw_score = result["score"]

            if chunk_id not in self._chunks_metadata:
                continue

            metadata = self._chunks_metadata[chunk_id]
            confidence = normalize_maxsim(raw_score)

            output.append((metadata["text"], confidence, metadata["source"]))

        return output

    def _load_model(self) -> None:
        """Load the ColBERT model."""
        from pylate import models

        self._model = models.ColBERT(
            model_name_or_path=self.MODEL_NAME,
        )
        # Set pad token to eos token (required for LFM2)
        self._model.tokenizer.pad_token = self._model.tokenizer.eos_token

    def _build_index(self) -> None:
        """Build the PLAID index from documents."""
        from pylate import indexes, retrieve

        # Chunk all documents
        chunks = self._chunk_all_documents()
        if not chunks:
            print("Warning: No documents to index")
            return

        print(f"Encoding {len(chunks)} chunks...")

        # Encode document chunks
        chunk_texts = [c.text for c in chunks]
        chunk_ids = [c.id for c in chunks]

        embeddings = self._model.encode(
            chunk_texts,
            batch_size=32,
            is_query=False,
            show_progress_bar=True,
        )

        # Create PLAID index
        print("Building PLAID index...")
        self._index = indexes.PLAID(
            index_folder=str(self.index_dir),
            index_name="index",
            override=True,
        )

        self._index.add_documents(
            documents_ids=chunk_ids,
            documents_embeddings=embeddings,
        )

        # Create retriever
        self._retriever = retrieve.ColBERT(index=self._index)

        # Save metadata
        self._index_manager.save_manifest(chunks)
        self._chunks_metadata = {
            c.id: {"text": c.text, "source": c.source} for c in chunks
        }

    def _load_index(self) -> None:
        """Load existing PLAID index from disk."""
        from pylate import indexes, retrieve

        self._index = indexes.PLAID(
            index_folder=str(self.index_dir),
            index_name="index",
            override=False,
        )

        self._retriever = retrieve.ColBERT(index=self._index)
        self._chunks_metadata = self._index_manager.load_chunk_metadata()

    def _chunk_all_documents(self) -> List[Chunk]:
        """Chunk all documents from the docs directory."""
        chunks = []
        doc_files = self._index_manager.get_document_files()

        for file_path in doc_files:
            if file_path.suffix == ".pdf":
                file_chunks = self._chunker.chunk_pdf(file_path)
            else:
                file_chunks = self._chunker.chunk_markdown(file_path)

            chunks.extend(file_chunks)
            print(f"  Chunked {len(file_chunks)} chunks from {file_path.name}")

        return chunks

    @property
    def is_loaded(self) -> bool:
        """Check if the retriever is loaded and ready."""
        return self._is_loaded

    def rebuild_index(self) -> None:
        """Force rebuild of the index."""
        self._index_manager.clear_index()
        self._build_index()
