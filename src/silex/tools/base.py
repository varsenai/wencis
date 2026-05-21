"""
Base Tool Interface.
"""

from __future__ import annotations


class BaseTool:
    """Abstract base class for all ARIA tools."""
    
    name: str = "base_tool"
    description: str = "Base description."
    risk_level: str = "read_only"
    requires_approval: bool = False
    
    # We will use this schema to inform the LLM how to call the tool
    schema: dict = {}

    async def execute(self, **kwargs) -> str:
        """
        Execute the tool with the given arguments.
        Must return a string representing the outcome (success or error details).
        """
        raise NotImplementedError("Tools must implement the execute method.")

    @classmethod
    def get_prompt_description(cls) -> str:
        """Formats the tool for the system prompt."""
        approval = "requires approval" if cls.requires_approval else "auto-allowed by policy"
        return (
            f"- **{cls.name}**: {cls.description}\n"
            f"  Risk: {cls.risk_level} ({approval})\n"
            f"  Args: {cls.schema}"
        )
