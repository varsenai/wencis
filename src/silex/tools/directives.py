"""
Core Directives tool for VYN.
Allows VYN to update its own core behavior rules transparently.
"""

from __future__ import annotations

from silex.tools.base import BaseTool
from silex.utils.config import VYN_DIRECTIVES_FILE
from silex.utils.logger import setup_logger

log = setup_logger("silex.tools.directives")

class UpdateDirectivesTool(BaseTool):
    """Updates the core directives file with new unbreakable rules."""

    name = "update_directives"
    risk_level = "repo_write"
    requires_approval = True

    schema = {
        "instruction": {
            "type": "string",
            "description": "What to add, remove, or change in the core directives."
        }
    }

    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def get_prompt_description(self) -> str:
        return (
            "- update_directives: Update the VYN Core Directives file to save "
            "unbreakable rules, strong user preferences, or absolute behavioral guidelines. "
            "Args: instruction (string)"
        )

    async def execute(self, instruction: str) -> str:
        """Update the directives file using the LLM to rewrite it."""
        try:
            current_content = ""
            if VYN_DIRECTIVES_FILE.exists():
                current_content = VYN_DIRECTIVES_FILE.read_text(encoding="utf-8")
                
            prompt = (
                "You are updating the VYN Core Directives markdown file. "
                "This file contains unbreakable rules and behavioral guidelines. "
                "Please rewrite the file to incorporate the following instruction.\n\n"
                f"Instruction: {instruction}\n\n"
                "Output ONLY the new markdown content for the file, nothing else."
            )
            
            response = await self.llm.think(
                system_prompt="You are a precise editor. Output only the final markdown content.",
                user_input=f"Current Content:\n{current_content}\n\n{prompt}"
            )
            
            new_content = response.response.strip()
            
            # Remove any markdown code block wrappers if the LLM added them
            if new_content.startswith("```md"):
                new_content = new_content[5:]
            elif new_content.startswith("```markdown"):
                new_content = new_content[11:]
            if new_content.startswith("```"):
                new_content = new_content[3:]
            if new_content.endswith("```"):
                new_content = new_content[:-3]
                
            new_content = new_content.strip()
            
            VYN_DIRECTIVES_FILE.write_text(new_content, encoding="utf-8")
            return f"Successfully updated Core Directives. New size: {len(new_content)} bytes."
            
        except Exception as e:
            log.error(f"Failed to update directives: {e}")
            return f"Error: Failed to update directives: {str(e)}"
