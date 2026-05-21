"""
Skill Curator — Lifecycle management and consolidation (Phase 21).

Responsibilities:
  1. **Lifecycle transitions**: Active → Stale (30 days no use) → Archived (60 days)
  2. **Prefix cluster detection**: Groups skills by common prefix (e.g., python-*)
  3. **Umbrella consolidation**: Merges narrow skills into broad umbrella skills
  4. **Pinning**: Protects critical skills from automatic transitions
  5. **Startup guard**: Only runs if ≥7 days since last run

The Curator is non-destructive: archived skills are moved to
``~/.vyn/skills/.archive/`` and can be restored at any time.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from silex.core.skill_schemas import SkillFrontmatter
from silex.utils.logger import setup_logger

if TYPE_CHECKING:
    from silex.core.skills import SkillLoader

log = setup_logger("silex.core.curator")


class SkillCurator:
    """Manages skill lifecycle transitions and cluster consolidation."""

    STALE_DAYS = 30        # Days without use before marking stale
    ARCHIVE_DAYS = 60      # Days without use before archiving
    MIN_CLUSTER_SIZE = 3   # Minimum skills in a prefix group to consolidate
    RUN_INTERVAL_DAYS = 7  # Minimum days between automatic runs

    def __init__(
        self,
        skill_loader: "SkillLoader",
        archive_dir: Path,
        llm=None,
    ):
        self.skill_loader = skill_loader
        self.archive_dir = archive_dir
        self.llm = llm  # Optional: used for umbrella consolidation prompts

    @staticmethod
    def should_run(last_run_file: Path) -> bool:
        """Return True if ≥7 days since the last curation run."""
        if not last_run_file.exists():
            return True
        try:
            last_run_str = last_run_file.read_text(encoding="utf-8").strip()
            last_run = datetime.fromisoformat(last_run_str)
            return datetime.now(timezone.utc) - last_run >= timedelta(
                days=SkillCurator.RUN_INTERVAL_DAYS
            )
        except (ValueError, OSError):
            return True

    @staticmethod
    def record_run(last_run_file: Path) -> None:
        """Write the current timestamp to the last-run file."""
        last_run_file.write_text(
            datetime.now(timezone.utc).isoformat(), encoding="utf-8"
        )

    async def run(self) -> dict:
        """Full curation cycle. Returns a summary of actions taken."""
        log.info("Skill Curator: starting curation cycle")
        summary = {"staled": 0, "archived": 0, "consolidated": 0}

        # 1. Lifecycle transitions
        lifecycle_result = self._transition_lifecycle()
        summary["staled"] = lifecycle_result.get("staled", 0)
        summary["archived"] = lifecycle_result.get("archived", 0)

        # 2. Prefix cluster consolidation (requires LLM)
        if self.llm:
            cluster_result = await self._consolidate_clusters()
            summary["consolidated"] = cluster_result.get("consolidated", 0)

        log.info(
            f"Skill Curator: done — staled={summary['staled']}, "
            f"archived={summary['archived']}, consolidated={summary['consolidated']}"
        )
        return summary

    def _transition_lifecycle(self) -> dict:
        """Move active skills to stale, stale skills to archived.

        - Active → Stale: unused for STALE_DAYS
        - Stale → Archived: unused for ARCHIVE_DAYS (file moved to .archive/)
        - Pinned skills: always exempt

        Returns ``{staled: int, archived: int}``.
        """
        now = datetime.now(timezone.utc)
        staled = 0
        archived = 0

        for name, content in list(self.skill_loader.skills.items()):
            meta_dict, body = self.skill_loader.parse_frontmatter(content)
            meta = self.skill_loader.skill_metadata.get(name, SkillFrontmatter())

            # Skip pinned skills
            if meta.pinned:
                continue

            # Determine last used date
            last_used = self._parse_date(meta.last_used) or self._parse_date(
                meta.created_at
            )
            if not last_used:
                continue

            days_unused = (now - last_used).days

            if meta.status == "active" and days_unused >= self.STALE_DAYS:
                # Transition to stale
                self._update_skill_status(name, "stale")
                staled += 1
                log.info(f"Skill '{name}' marked stale ({days_unused} days unused)")

            elif meta.status == "stale" and days_unused >= self.ARCHIVE_DAYS:
                # Archive the skill
                self._archive_skill(name)
                archived += 1
                log.info(f"Skill '{name}' archived ({days_unused} days unused)")

        return {"staled": staled, "archived": archived}

    def _update_skill_status(self, name: str, status: str) -> None:
        """Update the status field in a skill's frontmatter and re-save."""
        if name not in self.skill_loader.skills:
            return

        content = self.skill_loader.skills[name]
        _meta_dict, body = self.skill_loader.parse_frontmatter(content)

        extra_meta = {"status": status}
        if status == "archived":
            extra_meta["archived_at"] = datetime.now(timezone.utc).isoformat()

        # Use update_skill with extra_meta so metadata is properly merged
        self.skill_loader.update_skill(name, body, extra_meta=extra_meta)

    def _archive_skill(self, name: str) -> None:
        """Move a skill file to the archive directory."""
        self._update_skill_status(name, "archived")

        # Move file to archive
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        skill_file = self.skill_loader._skill_file_path(name)
        if skill_file and skill_file.exists():
            dest = self.archive_dir / skill_file.name
            shutil.move(str(skill_file), str(dest))
            log.info(f"Moved {skill_file.name} to {self.archive_dir}")

            # Remove from in-memory loader
            self.skill_loader.skills.pop(name, None)
            self.skill_loader.skill_metadata.pop(name, None)

    async def _consolidate_clusters(self) -> dict:
        """Detect prefix clusters and merge into umbrella skills.

        A cluster is a group of ≥MIN_CLUSTER_SIZE skills sharing a common
        hyphenated prefix (e.g., ``python-deploy``, ``python-testing``).
        """
        clusters = self._detect_prefix_clusters()
        consolidated = 0

        for prefix, skill_names in clusters.items():
            if len(skill_names) < self.MIN_CLUSTER_SIZE:
                continue

            umbrella_name = f"{prefix}-umbrella"
            if umbrella_name in self.skill_loader.skills:
                continue  # already consolidated

            log.info(
                f"Consolidating {len(skill_names)} skills under prefix '{prefix}'"
            )

            try:
                await self._create_umbrella_skill(prefix, skill_names)
                consolidated += 1
            except Exception as e:
                log.warning(f"Consolidation failed for prefix '{prefix}': {e}")

        return {"consolidated": consolidated}

    def _detect_prefix_clusters(self) -> dict[str, list[str]]:
        """Group skills by common hyphenated prefix.

        Example::

            python-deploy, python-testing, python-packaging
            → prefix "python" → cluster of 3 skills
        """
        prefix_groups: dict[str, list[str]] = defaultdict(list)

        for name in self.skill_loader.skills:
            if "-" not in name:
                continue
            prefix = name.split("-")[0]
            if len(prefix) > 2:  # skip very short prefixes
                prefix_groups[prefix].append(name)

        return dict(prefix_groups)

    async def _create_umbrella_skill(
        self, prefix: str, skill_names: list[str]
    ) -> None:
        """Use the LLM to synthesize an umbrella skill from a cluster.

        The umbrella skill summarizes all sub-skills. Each sub-skill gets
        ``absorbed_into`` set to the umbrella name, and transitions to
        archived status.
        """
        # Collect all sub-skill bodies
        sub_skills = []
        for name in skill_names:
            content = self.skill_loader.skills.get(name, "")
            _, body = self.skill_loader.parse_frontmatter(content)
            sub_skills.append(f"### {name}\n{body[:500]}")

        combined = "\n\n".join(sub_skills)
        umbrella_name = f"{prefix}-umbrella"

        prompt = (
            f"You are combining {len(skill_names)} related skills with the prefix '{prefix}' "
            f"into one comprehensive umbrella skill named '{umbrella_name}'.\n\n"
            f"Sub-skills:\n{combined}\n\n"
            f"Create a single, well-organized Markdown skill that covers all the workflows. "
            f"Be concise but complete."
        )

        try:
            response = await self.llm.think(prompt, request_kind="curator")
            umbrella_body = response.text if hasattr(response, "text") else str(response)
        except Exception:
            # Fallback: just concatenate
            umbrella_body = f"# {prefix.title()} Umbrella Skill\n\n{combined}"

        # Build the umbrella skill file
        from datetime import datetime, timezone

        meta = {
            "name": umbrella_name,
            "description": f"Comprehensive {prefix} workflows (consolidated from {len(skill_names)} skills)",
            "version": 1,
            "uses": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
            "pinned": False,
        }
        umbrella_content = self._rebuild_skill_file(meta, umbrella_body)

        # Save the umbrella skill
        from silex.utils.config import VYN_SKILLS

        umbrella_path = VYN_SKILLS / f"{umbrella_name}.md"
        umbrella_path.write_text(umbrella_content, encoding="utf-8")

        # Mark sub-skills as absorbed and archive them
        for name in skill_names:
            self._update_absorbed_into(name, umbrella_name)
            self._archive_skill(name)

        # Reload skills
        self.skill_loader.load_all()

    def _update_absorbed_into(self, name: str, umbrella_name: str) -> None:
        """Set the absorbed_into field on a skill before archiving."""
        if name not in self.skill_loader.skills:
            return
        content = self.skill_loader.skills[name]
        _meta_dict, body = self.skill_loader.parse_frontmatter(content)
        self.skill_loader.update_skill(
            name, body, extra_meta={"absorbed_into": umbrella_name}
        )

    @staticmethod
    def _rebuild_skill_file(meta: dict, body: str) -> str:
        """Rebuild a skill file from metadata dict and body string."""
        import yaml

        yaml_str = yaml.dump(meta, default_flow_style=False, sort_keys=False)
        return f"---\n{yaml_str}---\n{body}"

    @staticmethod
    def _parse_date(date_str: str) -> datetime | None:
        """Parse an ISO date string, returning None on failure."""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str)
        except (ValueError, TypeError):
            return None
