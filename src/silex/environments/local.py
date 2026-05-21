"""
Local subprocess execution backend.

Runs commands via ``asyncio.create_subprocess_shell`` in a restricted
working directory with a hard timeout.  Always available — no external
dependencies needed.

Phase 21 — fallback environment when Docker is unavailable.
"""

from __future__ import annotations

import asyncio
import platform
from pathlib import Path

from silex.environments.base import BaseEnvironment
from silex.utils.logger import setup_logger

log = setup_logger("silex.environments.local")


class LocalEnvironment(BaseEnvironment):
    """Execute commands via local subprocess with timeout and directory restriction."""

    name = "local"

    def __init__(self, working_dir: str | Path, default_timeout: int = 60):
        self.working_dir = Path(working_dir)
        self.default_timeout = default_timeout

    async def execute(self, command: str, timeout: int | None = None) -> tuple[str, int]:
        """Run *command* via the system shell in *working_dir*.

        Returns ``(combined_output, exit_code)``.  On timeout, the
        process is killed and exit_code is set to ``-1``.
        """
        timeout = timeout or self.default_timeout
        self.working_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Local exec: {command!r} (timeout={timeout}s, cwd={self.working_dir})")

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self.working_dir),
            )

            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                output = stdout.decode("utf-8", errors="replace") if stdout else ""
                return output, proc.returncode or 0

            except asyncio.TimeoutError:
                log.warning(f"Local exec timed out after {timeout}s — killing process")
                proc.kill()
                await proc.wait()
                return f"Error: Command timed out after {timeout} seconds.", -1

        except Exception as e:
            log.error(f"Local exec failed: {e}")
            return f"Error: {e}", -1

    async def is_available(self) -> bool:
        """Local environment is always available."""
        return True

    async def get_info(self) -> dict:
        """Return metadata about the local environment."""
        return {
            "name": "local",
            "os": platform.system(),
            "platform": platform.platform(),
            "cwd": str(self.working_dir),
        }
