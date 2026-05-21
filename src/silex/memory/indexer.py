"""
Workspace Indexer — crawls and chunks the workspace for the VectorStore.

Supports smart chunking for code and markdown to maintain semantic integrity.
"""

import os
import json
import hashlib
from typing import List

from silex.memory.vector_store import VectorStore
from silex.utils.config import VYN_MANIFEST, WORKSPACE_DIR
from silex.utils.logger import setup_logger

log = setup_logger("silex.memory.indexer")

# Files to ignore during indexing
IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", "dist", ".next", "out",
    "vector_db", "backups", ".venv", "venv",
}
IGNORE_NAMES = {
    ".env", ".env.local", ".env.production", ".env.development",
    ".aria_pending_edits.json", "package-lock.json",
}
IGNORE_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".exe", ".pyc",
    ".db", ".sqlite", ".sqlite3", ".pem", ".key", ".crt", ".pfx", ".p12",
    ".log", ".lock",
}
MAX_INDEX_FILE_BYTES = 1_000_000
MANIFEST_PATH = VYN_MANIFEST

class WorkspaceIndexer:
    """
    Crawls the workspace and populates the VectorStore with semantically chunked content.
    """

    def __init__(self, vector_store: VectorStore, root_dir: str, manifest_path=None):
        self.vector_store = vector_store
        self.root_dir = os.path.abspath(root_dir)
        self.manifest_path = manifest_path or MANIFEST_PATH
        from silex.world.code_parser import CodeParser
        self.code_parser = CodeParser()

    def run(self):
        """Incrementally index changed workspace files."""
        if not getattr(self.vector_store, "is_active", False):
            log.debug(
                "Skipping workspace indexing: vector store inactive "
                "(install ChromaDB / openyfai-vyn[vector])."
            )
            return
        log.debug(f"Starting workspace indexing for: {self.root_dir}")
        previous_manifest = self._load_manifest(self.manifest_path)
        next_manifest: dict[str, dict] = {}

        indexed = 0
        skipped = 0
        for root, dirs, files in os.walk(self.root_dir):
            # Filter ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

            for file in files:
                if file.startswith(".") or file in IGNORE_NAMES:
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in IGNORE_EXTS:
                    continue

                full_path = os.path.join(root, file)
                if os.path.abspath(full_path) == os.path.abspath(str(self.manifest_path)):
                    continue
                if os.path.getsize(full_path) > MAX_INDEX_FILE_BYTES:
                    continue
                rel_path = os.path.relpath(full_path, self.root_dir)
                
                try:
                    fingerprint = self._fingerprint(full_path)
                    next_manifest[rel_path] = fingerprint
                    if previous_manifest.get(rel_path) == fingerprint:
                        skipped += 1
                        continue
                    self.vector_store.delete_by_path(rel_path)
                    self._index_file(full_path, rel_path, fingerprint)
                    indexed += 1
                except Exception as e:
                    log.error(f"Failed to index {rel_path}: {e}")

        for removed_path in set(previous_manifest) - set(next_manifest):
            self.vector_store.delete_by_path(removed_path)

        self._save_manifest(self.manifest_path, next_manifest)
        log.debug(f"Indexing complete. Indexed {indexed} changed files; skipped {skipped} unchanged files.")

    def _index_file(self, full_path: str, rel_path: str, fingerprint: dict):
        """Chunks a single file using CodeParser and adds it to the vector store."""
        chunks = self.code_parser.parse_file(full_path, root_dir=self.root_dir)
        if not chunks:
            return

        texts = [chunk.content for chunk in chunks]
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadatas.append({
                "path": rel_path,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "sha256": fingerprint["sha256"],
                "source": "workspace_index",
                "symbol_name": chunk.symbol_name,
                "symbol_type": chunk.symbol_type,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            })

        ids = [self._chunk_id(rel_path, fingerprint["sha256"], i) for i in range(len(chunks))]
        self.vector_store.add_chunks(texts, metadatas, ids=ids)

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Splits text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        return chunks

    @staticmethod
    def _fingerprint(full_path: str) -> dict:
        hasher = hashlib.sha256()
        with open(full_path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(block)
        stat = os.stat(full_path)
        return {
            "sha256": hasher.hexdigest(),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }

    @staticmethod
    def _chunk_id(rel_path: str, digest: str, index: int) -> str:
        raw = f"{rel_path}:{digest}:{index}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _load_manifest(manifest_path) -> dict[str, dict]:
        if not manifest_path.exists():
            return {}
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Workspace index manifest was unreadable; rebuilding incrementally.")
            return {}

    @staticmethod
    def _save_manifest(manifest_path, manifest: dict[str, dict]) -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

if __name__ == "__main__":
    # Test execution
    vs = VectorStore()
    indexer = WorkspaceIndexer(vs, str(WORKSPACE_DIR))
    indexer.run()
