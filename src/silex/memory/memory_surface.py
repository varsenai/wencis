"""
Memory Surface — generates and syncs ~/.vyn/MEMORY.md

Phase 19: This is VYN's equivalent of Hermes' MEMORY.md — a human-readable,
user-editable projection of the agent's most important memories.

Sync strategy (Option A — startup-only):
- VYN reads MEMORY.md at startup → imports any new bullet points the user added
- VYN regenerates MEMORY.md during session reflection (every 10 turns) and shutdown
- No live file watching — simplicity eliminates race conditions
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from silex.memory.memory_store import MemoryStore
from silex.models.schemas import Memory, MemorySource, MemoryType
from silex.utils.config import VYN_HOME
from silex.utils.logger import setup_logger

log = setup_logger("silex.memory_surface")

MEMORY_MD_PATH = VYN_HOME / "MEMORY.md"
MEMORY_MD_HASH_PATH = VYN_HOME / ".memory_md_hash"


class MemorySurface:
    """Generates and syncs ~/.vyn/MEMORY.md."""

    def __init__(self, memory_store: MemoryStore):
        self.memory = memory_store

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate(self) -> None:
        """Regenerate MEMORY.md from the top memories in the database."""
        all_memories = await self.memory.all_memories()
        active = [
            m for m in all_memories
            if not m.archived_at and not m.superseded_by_id
        ]

        # Group by memory_type
        groups: dict[str, list[Memory]] = {}
        for m in active:
            type_key = m.memory_type.value if isinstance(m.memory_type, MemoryType) else str(m.memory_type)
            groups.setdefault(type_key, []).append(m)

        # Build markdown
        lines = [
            "# VYN Memory — Auto-generated, user-editable",
            "",
            f"> Last synced: {datetime.now(timezone.utc).isoformat()} | {len(active)} memories",
            "> Edit this file between sessions. VYN will import changes on next startup.",
            "",
        ]

        # Type labels and emoji
        type_labels = {
            "preference": ("⚙️ Preferences", "Things the user prefers"),
            "semantic": ("🧠 Learned Facts", "Knowledge acquired through conversation"),
            "procedural": ("📋 Procedures", "How-to knowledge and workflows"),
            "project": ("📁 Projects", "Active projects and their context"),
            "character": ("🧑 About the User", "User identity and personality"),
            "normative": ("📏 Rules & Norms", "Behavioral guidelines"),
            "episodic": ("📅 Events", "Notable events and experiences"),
        }

        for type_key, (heading, description) in type_labels.items():
            memories = groups.get(type_key, [])
            if not memories:
                continue
            # Sort by importance descending, cap at 15 per category
            memories.sort(key=lambda m: m.importance, reverse=True)
            memories = memories[:15]

            lines.append(f"## {heading}")
            lines.append(f"<!-- {description} -->")
            lines.append("")
            for m in memories:
                imp = f"{m.importance:.1f}"
                lines.append(f"- {m.content} <!-- id:{m.id[:8]} imp:{imp} -->")
            lines.append("")

        content = "\n".join(lines)
        MEMORY_MD_PATH.write_text(content, encoding="utf-8")

        # Store hash for change detection
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        MEMORY_MD_HASH_PATH.write_text(content_hash, encoding="utf-8")

        log.info(f"MEMORY.md generated with {len(active)} memories at {MEMORY_MD_PATH}")

    # ------------------------------------------------------------------
    # Change Detection & Sync
    # ------------------------------------------------------------------

    async def detect_user_edits(self) -> list[str]:
        """Check if the user has edited MEMORY.md since last generation.

        Returns a list of change descriptions (one per imported memory).
        Returns an empty list if no changes detected or file doesn't exist.
        """
        if not MEMORY_MD_PATH.exists():
            return []

        current_content = MEMORY_MD_PATH.read_text(encoding="utf-8")
        current_hash = hashlib.sha256(current_content.encode()).hexdigest()

        stored_hash = ""
        if MEMORY_MD_HASH_PATH.exists():
            stored_hash = MEMORY_MD_HASH_PATH.read_text(encoding="utf-8").strip()

        if current_hash == stored_hash:
            return []  # No changes

        log.info("MEMORY.md has been modified by user. Parsing changes...")
        return await self._parse_and_sync(current_content)

    async def _parse_and_sync(self, content: str) -> list[str]:
        """Parse user-edited MEMORY.md and sync new memories back to SQLite.

        Only imports lines that look like new bullet points (no ``<!-- id:`` comment).
        Existing entries (with id comments) are left untouched.
        """
        changes: list[str] = []

        for line in content.split("\n"):
            line = line.strip()

            # Skip non-bullet lines
            if not line.startswith("- ") or line == "- ":
                continue

            # Skip lines that already have an id tag (existing memories)
            if "<!-- id:" in line:
                continue

            # Extract the memory content (strip any trailing HTML comments)
            memory_content = line[2:].strip()
            if not memory_content or memory_content.startswith("<!--"):
                continue

            # Import as a new user memory
            new_mem = Memory(
                content=memory_content,
                source=MemorySource.USER,
                importance=0.7,
                tags=["from_memory_md"],
            )
            result = await self.memory.add(new_mem)
            if result:
                changes.append(f"Added: {memory_content[:50]}")

        if changes:
            log.info(f"Synced {len(changes)} changes from MEMORY.md")

        return changes
