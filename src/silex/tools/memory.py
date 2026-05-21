"""
Search Memory Tool.
Allows the agent to proactively query its own memory store and knowledge graph mid-turn.
"""

from __future__ import annotations

from silex.tools.base import BaseTool

class SearchMemoryTool(BaseTool):
    """Voluntarily search long-term memory and knowledge graph."""

    name = "search_memory"
    risk_level = "read_only"
    requires_approval = False

    schema = {
        "query": {
            "type": "string",
            "description": "The specific topic, fact, or past conversation to search for."
        }
    }

    def __init__(self, memory_store):
        super().__init__()
        self.memory_store = memory_store

    def get_prompt_description(self) -> str:
        return (
            "- search_memory: Voluntarily search your long-term memory and "
            "knowledge graph for specific facts, past conversations, or context you might have forgotten. "
            "Args: query (string)"
        )

    async def execute(self, query: str) -> str:
        """Search the memory store and return formatted context."""
        try:
            memories = await self.memory_store.retrieve_context(query)
            if not memories:
                return "No relevant memories found for that query."

            lines = [f"Found {len(memories)} memories related to '{query}':\n"]
            for i, mem in enumerate(memories, 1):
                tags_str = f" [{', '.join(mem.tags)}]" if mem.tags else ""
                lines.append(f"[{i}] {mem.content}")
                lines.append(f"    (Type: {mem.memory_type}, Source: {mem.source}, Confidence: {mem.confidence:.1f}){tags_str}")
                lines.append("")

            return "\n".join(lines)
        except Exception as e:
            return f"Error searching memory: {str(e)}"
