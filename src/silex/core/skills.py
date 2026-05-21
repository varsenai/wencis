"""
Skill Loader — Phase 19 (Compounding Intelligence).

Dynamically loads Markdown (.md) skill files from ~/.vyn/skills.
Supports YAML frontmatter for machine-parseable metadata, semantic
deduplication via ChromaDB, usage tracking, and progressive refinement.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from silex.memory.vector_store import VectorStore

from silex.core.skill_schemas import SkillFrontmatter
from silex.utils.logger import setup_logger
from silex.utils.config import VYN_SKILLS

log = setup_logger("silex.skills")

# Filenames excluded from loading as executable skills (contributor docs, etc.).
_SKIP_SKILL_NAMES = frozenset({"readme"})

# Regex to match YAML frontmatter block delimited by ---
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class SkillLoader:
    """
    Dynamically loads Markdown (.md) files from ~/.vyn/skills.
    This allows users to extend VYN's capabilities without writing Python code.
    Supports YAML frontmatter, semantic search, usage tracking, and deduplication.
    """

    def __init__(self, vector_store: "VectorStore | None" = None):
        self.skills_dir = VYN_SKILLS
        self.skills: dict[str, str] = {}
        self.skill_metadata: dict[str, SkillFrontmatter] = {}
        self.vector_store = vector_store
        self.collection = None

        if self.vector_store and self.vector_store.is_active:
            try:
                self.collection = self.vector_store.client.get_or_create_collection(
                    name="aria_skills",
                    embedding_function=self.vector_store.embedding_function,
                )
            except Exception as e:
                log.warning(f"Could not initialize vector collection for skills: {e}")

    # ------------------------------------------------------------------
    # Frontmatter Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_frontmatter(content: str) -> tuple[dict, str]:
        """Parse YAML frontmatter from ``---`` delimited block.

        Returns (metadata_dict, body) where body is the content after
        the frontmatter.  If no frontmatter is found, returns an empty
        dict and the full content.
        """
        match = _FRONTMATTER_RE.match(content)
        if not match:
            return {}, content

        try:
            metadata = yaml.safe_load(match.group(1)) or {}
            if not isinstance(metadata, dict):
                return {}, content
            body = content[match.end():]
            return metadata, body
        except yaml.YAMLError:
            return {}, content

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self) -> int:
        """Scan the skills directory and load all markdown files."""
        self.skills.clear()
        self.skill_metadata.clear()

        if not self.skills_dir.exists():
            log.warning(f"Skills directory not found at {self.skills_dir}. Creating it.")
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            return 0

        count = 0
        for file_path in self.skills_dir.glob("*.md"):
            if file_path.stem.lower() in _SKIP_SKILL_NAMES:
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    skill_name = file_path.stem
                    self.skills[skill_name] = content

                    # Parse frontmatter metadata
                    meta_dict, _body = self.parse_frontmatter(content)
                    self.skill_metadata[skill_name] = SkillFrontmatter(**{
                        k: v for k, v in meta_dict.items()
                        if k in SkillFrontmatter.model_fields
                    })

                    count += 1

                    if self.collection:
                        self.collection.upsert(
                            documents=[content],
                            metadatas=[{"name": skill_name}],
                            ids=[f"skill_{skill_name}"],
                        )
            except Exception as e:
                log.error(f"Failed to load skill {file_path.name}: {e}")

        log.info(f"Loaded {count} Markdown skills from {self.skills_dir}")
        return count

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_relevant_skills(self, query: str, limit: int = 3) -> dict[str, str]:
        """Retrieve only the top matches relevant to the user query."""
        if not self.skills:
            return {}

        if not self.collection:
            # Fallback to returning all/first few skills if vector store is not active
            return dict(list(self.skills.items())[:limit])

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(limit, len(self.skills)),
            )

            relevant = {}
            if results and results.get("metadatas") and results["metadatas"][0]:
                for meta in results["metadatas"][0]:
                    name = meta.get("name")
                    if name and name in self.skills:
                        relevant[name] = self.skills[name]
            return relevant
        except Exception as e:
            log.error(f"Error querying semantic skills: {e}")
            # Fallback
            return dict(list(self.skills.items())[:limit])

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def find_similar(self, query: str, threshold: float = 0.85) -> str | None:
        """Search existing skills via ChromaDB.

        Returns the skill name if any existing skill exceeds the
        similarity *threshold* (measured as ``1 - distance``).
        Returns ``None`` if no sufficiently similar skill exists.
        """
        if not self.collection or not self.skills:
            return None

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=1,
            )

            if (
                results
                and results.get("distances")
                and results["distances"][0]
            ):
                distance = results["distances"][0][0]
                similarity = 1.0 - distance
                if similarity >= threshold:
                    meta = results["metadatas"][0][0]
                    name = meta.get("name", "")
                    log.info(
                        f"Skill dedup: found existing skill '{name}' "
                        f"with similarity {similarity:.2f} (threshold {threshold})"
                    )
                    return name
        except Exception as e:
            log.warning(f"Skill dedup search failed (non-fatal): {e}")

        return None

    # ------------------------------------------------------------------
    # Usage Tracking & Refinement
    # ------------------------------------------------------------------

    def record_usage(self, skill_name: str) -> None:
        """Increment the ``uses`` counter and update ``last_used`` in YAML frontmatter."""
        file_path = self.skills_dir / f"{skill_name}.md"
        if not file_path.exists():
            return

        try:
            content = file_path.read_text(encoding="utf-8")
            meta_dict, body = self.parse_frontmatter(content)

            meta_dict["uses"] = meta_dict.get("uses", 0) + 1
            meta_dict["last_used"] = datetime.now(timezone.utc).isoformat()

            new_content = self._rebuild_with_frontmatter(meta_dict, body)
            file_path.write_text(new_content, encoding="utf-8")

            # Update in-memory cache
            self.skills[skill_name] = new_content
            self.skill_metadata[skill_name] = SkillFrontmatter(**{
                k: v for k, v in meta_dict.items()
                if k in SkillFrontmatter.model_fields
            })

            log.debug(f"Skill usage recorded: {skill_name} (uses={meta_dict['uses']})")
        except Exception as e:
            log.warning(f"Failed to record skill usage for '{skill_name}': {e}")

    def update_skill(self, name: str, new_body: str, extra_meta: dict | None = None) -> bool:
        """Overwrite an existing skill file body. Increments ``version``."""
        file_path = self.skills_dir / f"{name}.md"
        if not file_path.exists():
            log.warning(f"Cannot update skill '{name}': file not found")
            return False

        try:
            content = file_path.read_text(encoding="utf-8")
            meta_dict, _old_body = self.parse_frontmatter(content)

            meta_dict["version"] = meta_dict.get("version", 1) + 1
            if extra_meta:
                meta_dict.update(extra_meta)

            new_content = self._rebuild_with_frontmatter(meta_dict, new_body)
            file_path.write_text(new_content, encoding="utf-8")

            # Update in-memory cache
            self.skills[name] = new_content
            self.skill_metadata[name] = SkillFrontmatter(**{
                k: v for k, v in meta_dict.items()
                if k in SkillFrontmatter.model_fields
            })

            log.info(f"Skill updated: {name} → version {meta_dict['version']}")
            return True
        except Exception as e:
            log.error(f"Failed to update skill '{name}': {e}")
            return False

    def get_refinement_candidates(self) -> list[str]:
        """Return skill names where ``uses >= 2`` and ``refined is False``."""
        candidates = []
        for name, meta in self.skill_metadata.items():
            if meta.uses >= 2 and not meta.refined:
                candidates.append(name)
        return candidates

    def skill_stats(self) -> dict:
        """Return aggregate skill statistics."""
        total = len(self.skills)
        with_frontmatter = sum(
            1 for m in self.skill_metadata.values() if m.name
        )
        total_uses = sum(m.uses for m in self.skill_metadata.values())
        return {
            "total_skills": total,
            "with_frontmatter": with_frontmatter,
            "total_uses": total_uses,
            "refinement_candidates": len(self.get_refinement_candidates()),
        }

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_for_prompt(self, query: str | None = None) -> str:
        """Format skill metadata (name + description) for the system prompt.

        Progressive Skill Disclosure (Phase 21): only metadata is injected.
        The agent must call the ``read_skill`` tool to load the full content
        when it decides a skill is relevant to the user's request.
        """
        skills_to_format = self.skills
        if query:
            skills_to_format = self.get_relevant_skills(query)

        if not skills_to_format:
            return ""

        lines = [
            "═══════════════════════════════════════════════════════════",
            "AVAILABLE SKILLS (Metadata Only — use `read_skill` tool to load full instructions)",
            "═══════════════════════════════════════════════════════════",
            "",
        ]

        for name in skills_to_format:
            meta = self.skill_metadata.get(name)
            if meta and meta.description:
                desc = meta.description
            else:
                # Fallback: extract first non-empty, non-heading line from body
                _, body = self.parse_frontmatter(skills_to_format[name])
                first_line = next(
                    (l.strip() for l in body.split("\n") if l.strip() and not l.startswith("#")),
                    "No description",
                )
                desc = first_line[:120]

            uses_str = f" (used {meta.uses}x)" if meta and meta.uses > 0 else ""
            lines.append(f"  • {name}: {desc}{uses_str}")

        lines.append("")
        lines.append("To follow a skill, call: read_skill(name=\"<skill_name>\")")
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rebuild_with_frontmatter(meta: dict, body: str) -> str:
        """Rebuild a skill file from metadata dict and markdown body."""
        fm_str = yaml.dump(meta, default_flow_style=False, sort_keys=False).strip()
        return f"---\n{fm_str}\n---\n{body}"

    def _skill_file_path(self, name: str) -> Path | None:
        """Return the file path for a skill by name, or None if not found."""
        for suffix in ("", ".md"):
            candidate = self.skills_dir / f"{name}{suffix}"
            if candidate.exists():
                return candidate
        # Fallback: search by stem
        for f in self.skills_dir.glob("*.md"):
            if f.stem == name:
                return f
        return None

