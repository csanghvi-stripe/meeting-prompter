"""PLAID index lifecycle management for ColBERT."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from .chunker import Chunk


class IndexManager:
    """
    Manages PLAID index lifecycle: creation, loading, and cache invalidation.

    The index is persisted to disk and only rebuilt when documents change.
    Document changes are detected via SHA256 hashes stored in a manifest.
    """

    def __init__(self, index_dir: Path, docs_dir: Path):
        """
        Initialize the index manager.

        Args:
            index_dir: Directory to store the PLAID index
            docs_dir: Directory containing source documents
        """
        self.index_dir = Path(index_dir)
        self.docs_dir = Path(docs_dir)
        self.manifest_path = self.index_dir / "docs_manifest.json"
        self.metadata_path = self.index_dir / "chunks_metadata.json"

        # Ensure directories exist
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def needs_rebuild(self) -> bool:
        """
        Check if the index needs to be rebuilt.

        Returns:
            True if documents have changed or index doesn't exist
        """
        if not self._index_exists():
            return True

        current_hashes = self._compute_doc_hashes()
        stored_hashes = self._load_manifest()

        return current_hashes != stored_hashes

    def save_manifest(self, chunks: List[Chunk]) -> None:
        """
        Save document manifest and chunk metadata after index build.

        Args:
            chunks: List of chunks that were indexed
        """
        # Save document hashes
        doc_hashes = self._compute_doc_hashes()
        with open(self.manifest_path, "w") as f:
            json.dump(doc_hashes, f, indent=2)

        # Save chunk metadata (id -> text, source mapping)
        metadata = {chunk.id: {"text": chunk.text, "source": chunk.source} for chunk in chunks}
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_chunk_metadata(self) -> Dict[str, Dict[str, str]]:
        """
        Load chunk metadata from disk.

        Returns:
            Dictionary mapping chunk_id to {"text": ..., "source": ...}
        """
        if not self.metadata_path.exists():
            return {}

        with open(self.metadata_path, "r") as f:
            return json.load(f)

    def get_document_files(self) -> List[Path]:
        """
        Get all document files from the docs directory.

        Returns:
            List of PDF and Markdown file paths
        """
        if not self.docs_dir.exists():
            return []

        pdf_files = list(self.docs_dir.glob("*.pdf"))
        md_files = list(self.docs_dir.glob("*.md"))
        return pdf_files + md_files

    def _index_exists(self) -> bool:
        """Check if a valid index exists on disk."""
        # PLAID creates multiple files in the index folder
        # Check for manifest as indicator of complete index
        return self.manifest_path.exists() and self.metadata_path.exists()

    def _compute_doc_hashes(self) -> Dict[str, str]:
        """Compute SHA256 hashes for all documents."""
        hashes = {}
        for file_path in self.get_document_files():
            file_hash = self._hash_file(file_path)
            hashes[file_path.name] = file_hash
        return hashes

    def _hash_file(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _load_manifest(self) -> Dict[str, str]:
        """Load the document manifest from disk."""
        if not self.manifest_path.exists():
            return {}

        with open(self.manifest_path, "r") as f:
            return json.load(f)

    def clear_index(self) -> None:
        """Clear all index files (for forced rebuild)."""
        import shutil

        if self.index_dir.exists():
            # Remove index subdirectory if it exists
            index_subdir = self.index_dir / "index"
            if index_subdir.exists():
                shutil.rmtree(index_subdir)

            # Remove manifest and metadata
            if self.manifest_path.exists():
                self.manifest_path.unlink()
            if self.metadata_path.exists():
                self.metadata_path.unlink()
