"""
Tool Registry. Manages tool registration and execution routing.

Security: All tool arguments are validated against the tool's schema
before execution to prevent injection of unexpected parameters.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from silex.core.ethics import EthicsEngine
from silex.models.schemas import EthicalAction, ToolCall, ToolResult
from silex.tools.base import BaseTool
from silex.tools.search import WebSearchTool, SemanticSearchTool
from silex.tools.file_reader import FileReaderTool
from silex.tools.code_editor import CodeEditorTool, ApplyEditTool
from silex.tools.system import ListDirectoryTool, RunTerminalCommandTool
from silex.tools.browser import BrowserTool
from silex.tools.phantom import PhantomTool
from silex.utils.logger import setup_logger
from silex.utils.config import require_tool_approvals, code_apply_enabled

log = setup_logger("silex.tools.registry")


class ToolRegistry:
    """Holds available tools and executes them based on ToolCalls."""
    
    def __init__(self, vector_store=None, db=None, session_manager=None, memory_store=None, llm=None):
        self.tools: dict[str, BaseTool] = {}
        self.vector_store = vector_store
        self.db = db
        self.session_manager = session_manager
        self.memory_store = memory_store
        self.llm = llm
        self.ethics = EthicsEngine()
        self._register_defaults()

    def _register_defaults(self):
        """Register the default tools."""
        self.register(WebSearchTool())
        self.register(FileReaderTool())
        self.register(PhantomTool())         # Phantom Simulator — dry-run before apply
        self.register(CodeEditorTool())
        self.register(ApplyEditTool())
        self.register(ListDirectoryTool())
        self.register(self._create_terminal_tool())
        self.register(BrowserTool())
        
        if self.vector_store and getattr(self.vector_store, "is_active", False):
            self.register(SemanticSearchTool(self.vector_store))

        if getattr(self, "memory_store", None):
            from silex.tools.memory import SearchMemoryTool
            self.register(SearchMemoryTool(self.memory_store))

        if getattr(self, "llm", None):
            from silex.tools.directives import UpdateDirectivesTool
            self.register(UpdateDirectivesTool(self.llm))
            from silex.tools.agent_tool import SpawnWorkerTool
            self.register(SpawnWorkerTool(self.llm, self))

    @staticmethod
    def _create_terminal_tool() -> RunTerminalCommandTool:
        """Create RunTerminalCommandTool with the best available environment.

        Selection order: Docker → Local subprocess fallback.
        """
        from silex.environments.docker_env import DockerEnvironment
        from silex.environments.local import LocalEnvironment
        from silex.utils.config import WORKSPACE_DIR, PROJECT_ROOT

        # Try Docker first
        docker_env = DockerEnvironment(
            project_dir=str(PROJECT_ROOT),
            workspace_dir=str(WORKSPACE_DIR),
        )
        if docker_env._client:
            log.info("Execution backend: Docker (container sandbox)")
            return RunTerminalCommandTool(environment=docker_env)

        # Fallback to local subprocess
        log.info("Execution backend: Local subprocess (Docker unavailable)")
        local_env = LocalEnvironment(working_dir=str(WORKSPACE_DIR))
        return RunTerminalCommandTool(environment=local_env)

    def register(self, tool: BaseTool) -> None:
        """Register a new tool."""
        self.tools[tool.name] = tool
        log.info(f"Registered tool: {tool.name}")

    def get_system_prompt_appendix(self) -> str:
        """Returns the formatted documentation of all tools for the LLM prompt."""
        if not self.tools:
            return "No tools available."
            
        docs = "AVAILABLE TOOLS:\n"
        for tool in self.tools.values():
            docs += tool.get_prompt_description() + "\n"
            
        return docs

    async def execute(self, call: ToolCall, execution_mode: str = "interactive") -> ToolResult:
        """Execute a ToolCall and return a ToolResult."""
        tool = self.tools.get(call.tool_name)
        if not tool:
            log.warning(f"Attempted to call unknown tool: {call.tool_name}")
            return ToolResult(
                tool_name=call.tool_name,
                actual_outcome="Error: Tool not found in registry.",
                success=False,
                error="Tool not found"
            )

        # ── Parse arguments ──────────────────────────────────────
        args_dict = {}
        if isinstance(call.arguments, str):
            try:
                args_dict = json.loads(call.arguments)
            except json.JSONDecodeError:
                return ToolResult(
                    tool_name=call.tool_name,
                    actual_outcome="Error: Failed to parse arguments as JSON.",
                    success=False,
                    error="JSON parse error"
                )
        elif isinstance(call.arguments, dict):
            args_dict = call.arguments

        # ── Validate arguments against schema ────────────────────
        # Only allow keys defined in the tool's schema
        allowed_keys = set(tool.schema.keys()) if tool.schema else set()
        unexpected_keys = set(args_dict.keys()) - allowed_keys
        if unexpected_keys:
            log.warning(
                f"Tool {call.tool_name}: rejected unexpected args: {unexpected_keys}"
            )
            # Strip unexpected keys rather than crash
            args_dict = {k: v for k, v in args_dict.items() if k in allowed_keys}

        log.info(f"Executing {call.tool_name} with args: {list(args_dict.keys())}")

        ethical_decision = self.ethics.evaluate_tool_call(
            call,
            tool,
            args_dict,
            execution_mode=execution_mode,
        )
        await self._log_ethical_decision(call, ethical_decision)

        if ethical_decision.action == EthicalAction.REFUSE:
            return ToolResult(
                tool_name=call.tool_name,
                actual_outcome=(
                    f"Error: Ethical policy refused {call.tool_name}. "
                    f"{ethical_decision.rationale}"
                ),
                success=False,
                error="ethical_refusal",
                ethical_decision=ethical_decision,
            )

        if ethical_decision.action == EthicalAction.ESCALATE:
            approval_id = await self._queue_approval(
                tool,
                args_dict,
                f"{call.expected_outcome} | {ethical_decision.rationale}",
            )
            return ToolResult(
                tool_name=call.tool_name,
                actual_outcome=(
                    f"Error: Approval required for {call.tool_name} "
                    f"(risk={tool.risk_level}, approval_id={approval_id}). "
                    f"Principle={ethical_decision.principle}."
                ),
                success=False,
                error="approval_required",
                ethical_decision=ethical_decision,
            )

        if self._approval_required(tool):
            approval_id = await self._queue_approval(tool, args_dict, call.expected_outcome)
            return ToolResult(
                tool_name=call.tool_name,
                actual_outcome=(
                    f"Error: Approval required for {call.tool_name} "
                    f"(risk={tool.risk_level}, approval_id={approval_id})."
                ),
                success=False,
                error="approval_required",
                ethical_decision=ethical_decision,
            )

        try:
            outcome = await tool.execute(**args_dict)
            # Tools return human-readable strings, so normalize the common error prefixes.
            success = not outcome.lower().startswith("error:")
            
            return ToolResult(
                tool_name=call.tool_name,
                actual_outcome=outcome,
                success=success,
                error=outcome if not success else None,
                ethical_decision=ethical_decision,
            )
        except Exception as e:
            log.error(f"Tool {call.tool_name} crashed: {e}")
            return ToolResult(
                tool_name=call.tool_name,
                actual_outcome="Error executing tool: internal error occurred.",
                success=False,
                error="Internal tool execution error",
                ethical_decision=ethical_decision,
            )

    def _approval_required(self, tool: BaseTool) -> bool:
        if not tool.requires_approval or not require_tool_approvals():
            return False
        # When the operator enables code_apply, code-editor tools are
        # auto-approved at this layer.  The ethics engine has ALREADY run
        # (lines 108-144 above) — this flag only skips the manual approval
        # queue, it does NOT bypass ethical evaluation.
        if tool.risk_level == "repo_write" and code_apply_enabled():
            return False
        return True

    async def _queue_approval(self, tool: BaseTool, args_dict: dict, reason: str) -> str:
        approval_id = str(uuid.uuid4())
        if self.db:
            await self.db.execute(
                """
                INSERT INTO tool_approvals (
                    id, session_id, tool_name, risk_level, arguments_json,
                    expected_outcome, reason, status, created_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    self.session_manager.current.id if self.session_manager and self.session_manager.current else None,
                    tool.name,
                    tool.risk_level,
                    json.dumps(args_dict),
                    reason or "Tool requested by model.",
                    reason or "Tool requested by model.",
                    "pending",
                    datetime.now(timezone.utc).isoformat(),
                    None,
                ),
            )
        return approval_id

    async def _log_ethical_decision(self, call: ToolCall, decision) -> None:
        if not self.db:
            return
        await self.db.execute(
            """
            INSERT INTO ethical_decisions (
                id, session_id, turn_number, tool_name, principle, action,
                rationale, risk_level, requires_consent, uncertainty, context,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                self.session_manager.current.id if self.session_manager and self.session_manager.current else None,
                (self.session_manager.current.turn_count + 1) if self.session_manager and self.session_manager.current else 0,
                call.tool_name,
                decision.principle,
                decision.action.value,
                decision.rationale,
                decision.risk_level.value,
                int(decision.requires_consent),
                decision.uncertainty,
                decision.context,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    async def get_pending_approvals(self) -> list[dict]:
        if not self.db:
            return []
        return await self.db.fetch_all(
            "SELECT * FROM tool_approvals WHERE status = 'pending' ORDER BY created_at DESC"
        )

    async def resolve_approval(self, approval_id: str, status: str) -> bool:
        if status not in {"approved", "rejected"}:
            return False
        if not self.db:
            return False
        now = datetime.now(timezone.utc).isoformat()
        approval = await self.db.fetch_one(
            "SELECT * FROM tool_approvals WHERE id = ?",
            (approval_id,),
        )
        if not approval:
            return False

        execution_result_json = None
        if status == "approved":
            tool = self.tools.get(approval["tool_name"])
            if tool:
                args_dict = json.loads(approval["arguments_json"])
                try:
                    outcome = await tool.execute(**args_dict)
                    execution_result_json = json.dumps(
                        {
                            "success": not outcome.lower().startswith("error:"),
                            "actual_outcome": outcome,
                        }
                    )
                except Exception as exc:
                    execution_result_json = json.dumps(
                        {
                            "success": False,
                            "actual_outcome": "Error executing approved tool.",
                            "error": str(exc),
                        }
                    )

        await self.db.execute(
            "UPDATE tool_approvals SET status = ?, resolved_at = ?, execution_result_json = ? WHERE id = ?",
            (status, now, execution_result_json, approval_id),
        )
        return True
