"""
Agent Tool — SpawnWorkerTool.
Allows ARIA to delegate subtasks to autonomous worker agents in parallel.
"""

from __future__ import annotations

import json
from typing import List, Optional

from silex.tools.base import BaseTool
from silex.core.agent import AgentWorker
from silex.models.schemas import AgentSpec


class SpawnWorkerTool(BaseTool):
    """Spawn a specialized worker agent to handle a subtask in parallel."""

    name = "spawn_worker"
    risk_level = "read_only"
    requires_approval = False

    schema = {
        "agent_name": {
            "type": "string",
            "description": "Short descriptive name for the worker, e.g. 'SecurityAuditor' or 'CodeReviewer'"
        },
        "persona": {
            "type": "string",
            "description": "Behavioral constraints, guidelines, and expertise instructions"
        },
        "task": {
            "type": "string",
            "description": "Specific isolated subtask the agent should complete"
        },
        "tools": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of allowed tool names (subset of active tools), e.g. ['read_file', 'list_directory']"
        },
        "read_context": {
            "type": "boolean",
            "description": "Whether to load shared/retrieved context (default: true)"
        },
        "max_turns": {
            "type": "integer",
            "description": "Max reasoning turns before returning results (default: 3)"
        }
    }

    def __init__(self, llm, tool_registry, context_builder=None):
        super().__init__()
        self.llm = llm
        self.tool_registry = tool_registry
        self.context_builder = context_builder

    def get_prompt_description(self) -> str:
        return (
            "- spawn_worker: Spawn a specialized autonomous worker agent to run in parallel on a specific task. "
            "Args: agent_name (string), persona (string), task (string), tools (array of strings, optional), "
            "read_context (boolean, optional), max_turns (integer, optional)"
        )

    async def execute(
        self,
        agent_name: str,
        persona: str,
        task: str,
        tools: Optional[List[str]] = None,
        read_context: bool = True,
        max_turns: int = 3,
        **kwargs
    ) -> str:
        """Spawn a worker agent, execute it, and return its structured results as text."""
        try:
            spec = AgentSpec(
                name=agent_name,
                persona=persona,
                task=task,
                tools=tools or [],
                read_context=read_context,
                max_turns=max_turns
            )
            worker = AgentWorker(
                spec=spec,
                llm=self.llm,
                tool_registry=self.tool_registry,
                context_builder=self.context_builder
            )
            result = await worker.execute()
            
            outcome = {
                "agent_name": result.agent_name,
                "task": result.task,
                "reasoning": result.reasoning,
                "response": result.response,
                "confidence": result.confidence,
                "proposed_edits": result.proposed_edits,
                "new_observations": result.new_observations,
                "dissent": result.dissent
            }
            return json.dumps(outcome, indent=2)
        except Exception as e:
            return f"Error executing worker agent: {e}"
