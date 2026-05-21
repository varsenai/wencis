"""
BaseEnvironment — abstract interface for command execution backends.

All execution tools route through this interface.  Swapping the backend
(Local → Docker → SSH) requires zero changes to tool code.

Phase 21 — decouples RunTerminalCommandTool from its Docker-only impl.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEnvironment(ABC):
    """Abstract base for execution environments."""

    name: str = "base"

    @abstractmethod
    async def execute(self, command: str, timeout: int = 60) -> tuple[str, int]:
        """Execute *command* and return ``(stdout_stderr, exit_code)``."""
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Return True if this environment is ready to accept commands."""
        ...

    @abstractmethod
    async def get_info(self) -> dict:
        """Return a metadata dict describing the environment."""
        ...
