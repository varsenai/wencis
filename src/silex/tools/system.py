"""
System-level tools for directory exploration and terminal execution.

Phase 21: Terminal execution delegates to a BaseEnvironment backend
(Docker, Local, SSH). The Docker-specific logic now lives in
``silex.environments.docker_env``.
"""

from __future__ import annotations

import os
from pathlib import Path

from silex.tools.base import BaseTool
from silex.utils.config import terminal_execution_enabled, WORKSPACE_DIR
from silex.utils.logger import setup_logger

log = setup_logger("silex.tools.system")
WORKSPACE_ROOT = WORKSPACE_DIR
BLOCKED_PATH_PARTS = {".git", "node_modules", ".venv", "venv", "__pycache__"}


def _resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError:
        raise ValueError("path is outside the project directory")
    if any(part in BLOCKED_PATH_PARTS for part in resolved.parts):
        raise ValueError("path includes a restricted directory")
    return resolved

class ListDirectoryTool(BaseTool):
    """Lists contents of a directory to map the codebase."""

    name = "list_directory"
    risk_level = "read_only"
    description = (
        "Lists the files and folders inside a specified directory path. "
        "Use this to explore the project structure."
    )
    schema = {
        "path": "string (The absolute or relative directory path to list)"
    }

    async def execute(self, path: str = ".") -> str:
        try:
            safe_path = _resolve_project_path(path)
        except ValueError as e:
            return f"Error: Access denied — {e}."

        if not safe_path.exists():
            return f"Error: Path '{path}' does not exist."
        if not safe_path.is_dir():
            return f"Error: Path '{path}' is not a directory."

        try:
            items = os.listdir(safe_path)
            directories = []
            files = []
            
            for item in items:
                full_path = safe_path / item
                if full_path.is_dir():
                    directories.append(f"📁 {item}/")
                else:
                    files.append(f"📄 {item}")
                    
            directories.sort()
            files.sort()
            
            output = f"Contents of {safe_path}:\n"
            output += "\n".join(directories + files)
            return output
            
        except Exception as e:
            return f"Error listing directory: {e}"


class RunTerminalCommandTool(BaseTool):
    """Executes commands via the configured execution environment.

    Phase 21: Delegates to a ``BaseEnvironment`` instance (Local, Docker,
    or SSH) instead of hardcoding Docker.  The active backend is set at
    init time and can be swapped without changing tool code.
    """

    name = "run_terminal_command"
    risk_level = "sandbox_write"
    requires_approval = True
    schema = {
        "command": "string (The command to execute)"
    }

    def __init__(self, environment=None):
        from silex.environments.base import BaseEnvironment
        self.environment: BaseEnvironment | None = environment
        # Dynamic description based on active backend
        if environment:
            self.description = (
                f"Executes a command in the '{environment.name}' execution environment. "
                "The working directory and safety restrictions depend on the backend."
            )
        else:
            self.description = (
                "Executes a command in the configured execution environment. "
                "No backend is currently configured."
            )

    async def execute(self, command: str) -> str:
        if not self.environment:
            return (
                "Error: No execution environment configured. "
                "Set VYN_EXECUTION_BACKEND in your .env file."
            )

        if not terminal_execution_enabled():
            return (
                "Error: Execution blocked. Autonomous terminal execution is currently "
                "disabled by the user for safety reasons. You must ask the user to "
                "set ARIA_ENABLE_TERMINAL_EXECUTION=true in the .env file to enable execution."
            )

        if not await self.environment.is_available():
            env_info = await self.environment.get_info()
            return (
                f"Error: Execution environment '{self.environment.name}' is not available. "
                f"Details: {env_info}"
            )

        log.info(f"Executing via {self.environment.name}: {command}")

        output, exit_code = await self.environment.execute(command)
        env_info = await self.environment.get_info()

        return (
            f"--- OUTPUT ({env_info['name']}) ---\n"
            f"{output}\n"
            f"--- END OUTPUT ---\n"
            f"Exit Code: {exit_code}"
        )
