"""
Skill Tool — Progressive Skill Disclosure (Phase 21).

Lets VYN read full skill instructions on demand instead of having
all skill bodies injected into the system prompt every turn.

Only skill metadata (name + description) is in the system prompt.
The agent calls ``read_skill`` when it decides a skill is relevant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from silex.tools.base import BaseTool
from silex.utils.logger import setup_logger

if TYPE_CHECKING:
    from silex.core.skills import SkillLoader

log = setup_logger("silex.tools.skill")


class ReadSkillTool(BaseTool):
    """Reads the full content of a named skill from ~/.vyn/skills/."""

    name = "read_skill"
    risk_level = "read_only"
    requires_approval = False
    description = (
        "Loads the full Markdown instructions for a skill by name. "
        "Use this when a user's request matches a skill from the AVAILABLE SKILLS list. "
        "The skill content will contain step-by-step instructions to follow."
    )
    schema = {
        "name": "string (The skill name from the AVAILABLE SKILLS list)"
    }

    def __init__(self, skill_loader: "SkillLoader"):
        self.skill_loader = skill_loader

    async def execute(self, name: str = "") -> str:
        if not name:
            return "Error: skill name is required."

        name = name.strip()
        if name not in self.skill_loader.skills:
            # Try fuzzy match (case-insensitive substring)
            matches = [s for s in self.skill_loader.skills if name.lower() in s.lower()]
            if matches:
                name = matches[0]
            else:
                available = ", ".join(list(self.skill_loader.skills.keys())[:10])
                return f"Error: Skill '{name}' not found. Available: {available}"

        content = self.skill_loader.skills[name]
        _, body = self.skill_loader.parse_frontmatter(content)

        # Track usage — only count skills the agent actually reads
        self.skill_loader.record_usage(name)

        # Bundle context: mention related skills with the same prefix
        prefix = name.split("-")[0] if "-" in name else None
        siblings = []
        if prefix and len(prefix) > 2:
            siblings = [
                s for s in self.skill_loader.skills
                if s != name and s.startswith(prefix + "-")
            ]

        bundle_banner = ""
        if siblings:
            sibling_list = ", ".join(siblings[:5])
            bundle_banner = (
                f"\n⚡ Bundle Context: Related skills in the '{prefix}' family: "
                f"{sibling_list}\n"
            )

        log.info(f"Skill loaded on demand: {name}")
        return f"--- SKILL: {name} ---{bundle_banner}\n{body.strip()}\n--- END SKILL ---"
