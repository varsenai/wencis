"""
Repo Map — builds and maintains a compact, hierarchical symbol index of the codebase.
"""

from __future__ import annotations

import os
import json
import hashlib
from typing import List, Dict, Any, Optional

from silex.world.code_parser import CodeParser, CodeChunk
from silex.utils.logger import setup_logger
from silex.utils.config import WORKSPACE_DIR

log = setup_logger("silex.world.repo_map")

IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", "dist", ".next", "out",
    "vector_db", "backups", ".venv", "venv", "web_dist",
}
IGNORE_EXTS = {
    ".pyc", ".db", ".sqlite", ".sqlite3", ".log", ".lock",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf",
}
SUPPORTED_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx"}


class RepoMap:
    """
    Maintains a compact tree index of all class and function signatures in the codebase.
    """

    def __init__(self, root_dir: str, cache_path: Optional[str] = None):
        self.root_dir = os.path.abspath(root_dir)
        self.cache_path = cache_path or os.path.join(self.root_dir, ".vyn_repo_map.json")
        self.code_parser = CodeParser()
        self.index: Dict[str, Any] = {}  # rel_path -> { "sha256": str, "symbols": list[dict] }
        self._load_cache()

    def _load_cache(self):
        """Load the index and fingerprints from disk cache if available."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.index = data
                        log.info(f"Loaded repo map cache with {len(self.index)} files.")
            except Exception as e:
                log.warning(f"Failed to load repo map cache: {e}")

    def save_cache(self):
        """Save the current index to disk cache."""
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.index, f, indent=2)
            log.info("Saved repo map cache to disk.")
        except Exception as e:
            log.error(f"Failed to save repo map cache: {e}")

    def build(self, force: bool = False):
        """Walk the workspace and incrementally update the symbol index."""
        log.info(f"Building repo map for: {self.root_dir}")
        current_files = {}

        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in SUPPORTED_EXTS:
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.root_dir)
                
                try:
                    sha256 = self._get_sha256(full_path)
                    current_files[rel_path] = sha256

                    # If not force and sha256 matches, skip re-parsing
                    if not force and rel_path in self.index and self.index[rel_path].get("sha256") == sha256:
                        continue

                    # Parse signatures
                    chunks = self.code_parser.parse_file(full_path, root_dir=self.root_dir)
                    symbols = []
                    for c in chunks:
                        if c.symbol_type in ("class", "function", "method"):
                            symbols.append({
                                "name": c.symbol_name,
                                "type": c.symbol_type,
                                "signature": c.signature,
                                "start_line": c.start_line,
                                "end_line": c.end_line
                            })

                    self.index[rel_path] = {
                        "sha256": sha256,
                        "symbols": symbols
                    }
                    log.info(f"Parsed {rel_path}: {len(symbols)} symbols extracted.")

                except Exception as e:
                    log.error(f"Failed to index {rel_path} in repo map: {e}")

        # Remove files that no longer exist
        removed = set(self.index.keys()) - set(current_files.keys())
        for r in removed:
            del self.index[r]
            log.info(f"Removed {r} from repo map.")

        self.save_cache()

    def get_relevant_map(self, query: str = "", max_chars: int = 8000) -> str:
        """
        Generate a compact, hierarchical text representation of the repo map.
        Prioritizes files most relevant to the query if size budget is exceeded.
        """
        if not self.index:
            self.build()

        ranked_files = self._rank_files(query)
        lines = []
        total_chars = 0

        # We represent files with hierarchical trees:
        # path/to/file.py
        #   class MyClass
        #     def my_method(...)
        #   def global_func(...)
        
        for rel_path in ranked_files:
            file_data = self.index.get(rel_path)
            if not file_data or not file_data.get("symbols"):
                # If no symbols, just list path if space allows
                path_line = f"📄 {rel_path}\n"
                if total_chars + len(path_line) < max_chars:
                    lines.append(path_line)
                    total_chars += len(path_line)
                continue

            file_lines = [f"📄 {rel_path}"]
            
            # Format symbols hierarchically
            symbols = file_data["symbols"]
            
            # First format classes and their methods
            classes = [s for s in symbols if s["type"] == "class"]
            methods = [s for s in symbols if s["type"] == "method"]
            functions = [s for s in symbols if s["type"] == "function"]

            for cls in classes:
                file_lines.append(f"  class {cls['name']}")
                cls_methods = [m for m in methods if m["name"].startswith(f"{cls['name']}.")]
                for m in cls_methods:
                    short_name = m["name"].split(".", 1)[1]
                    sig = m["signature"]
                    # If signature is just def short_name(...), use it, otherwise clean it
                    if sig:
                        file_lines.append(f"    {sig}")
                    else:
                        file_lines.append(f"    def {short_name}(...)")

            for f in functions:
                sig = f["signature"]
                if sig:
                    file_lines.append(f"  {sig}")
                else:
                    file_lines.append(f"  def {f['name']}(...)")

            file_block = "\n".join(file_lines) + "\n\n"
            if total_chars + len(file_block) < max_chars:
                lines.append(file_block)
                total_chars += len(file_block)
            else:
                # If we exceed max chars, check if we can add just the class declarations without methods
                compact_lines = [f"📄 {rel_path} (signatures truncated)"]
                for cls in classes:
                    compact_lines.append(f"  class {cls['name']}")
                for f in functions:
                    compact_lines.append(f"  def {f['name']}(...)")
                compact_block = "\n".join(compact_lines) + "\n\n"
                
                if total_chars + len(compact_block) < max_chars:
                    lines.append(compact_block)
                    total_chars += len(compact_block)
                else:
                    # Just add the path name
                    path_line = f"📄 {rel_path} (truncated)\n\n"
                    if total_chars + len(path_line) < max_chars:
                        lines.append(path_line)
                        total_chars += len(path_line)
                    break

        return "".join(lines).strip()

    def _rank_files(self, query: str) -> List[str]:
        """Rank files based on simple overlap relevance with the query."""
        if not query:
            return sorted(self.index.keys())

        query_words = set(query.lower().split())
        rankings = []

        for rel_path, file_data in self.index.items():
            score = 0
            path_lower = rel_path.lower()
            
            # 1. Path overlap score
            for word in query_words:
                if len(word) > 2 and word in path_lower:
                    score += 10
            
            # 2. Symbol name overlap score
            for sym in file_data.get("symbols", []):
                sym_name_lower = sym["name"].lower()
                for word in query_words:
                    if len(word) > 2 and word in sym_name_lower:
                        score += 5
            
            rankings.append((rel_path, score))

        # Sort by score (descending), then by path alphabetically
        rankings.sort(key=lambda x: (-x[1], x[0]))
        return [x[0] for x in rankings]

    @staticmethod
    def _get_sha256(full_path: str) -> str:
        hasher = hashlib.sha256()
        with open(full_path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()
