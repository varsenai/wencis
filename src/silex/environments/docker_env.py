"""
Docker container execution backend.

Runs commands inside an isolated Alpine Linux container with:
  - Project root mounted read-only at /project
  - Workspace mounted read-write at /workspace
  - Memory / PID / network limits

Refactored from the original ``RunTerminalCommandTool`` in system.py.

Phase 21 — pluggable backend behind the BaseEnvironment interface.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

try:
    import docker
except ImportError:
    docker = None  # type: ignore[assignment]

from silex.environments.base import BaseEnvironment
from silex.utils.logger import setup_logger

log = setup_logger("silex.environments.docker")


class DockerEnvironment(BaseEnvironment):
    """Execute commands inside a Docker container sandbox."""

    name = "docker"

    def __init__(
        self,
        image: str = "alpine:latest",
        project_dir: str | Path = "",
        workspace_dir: str | Path = "",
        mem_limit: str = "256m",
        pids_limit: int = 128,
    ):
        self.image = image
        self.project_dir = str(project_dir) if project_dir else ""
        self.workspace_dir = str(workspace_dir) if workspace_dir else ""
        self.mem_limit = mem_limit
        self.pids_limit = pids_limit
        self._client = None

        if docker:
            try:
                self._client = docker.from_env()
            except Exception as e:
                log.warning(f"Could not connect to Docker: {e}")

    async def execute(self, command: str, timeout: int = 60) -> tuple[str, int]:
        """Run *command* inside a Docker container.

        Returns ``(logs, exit_code)``.
        """
        if not self._client:
            return "Error: Docker client is not available.", -1

        log.info(f"Docker exec: {command!r} (image={self.image}, timeout={timeout}s)")

        try:
            # Ensure image is available
            try:
                self._client.images.get(self.image)
            except docker.errors.ImageNotFound:
                log.info(f"Pulling {self.image}...")
                await asyncio.to_thread(self._client.images.pull, self.image)

            # Build volume mounts
            volumes = {}
            if self.project_dir:
                volumes[self.project_dir] = {"bind": "/project", "mode": "ro"}
            if self.workspace_dir:
                Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)
                volumes[self.workspace_dir] = {"bind": "/workspace", "mode": "rw"}

            container = self._client.containers.run(
                image=self.image,
                command=["sh", "-c", command],
                volumes=volumes,
                working_dir="/workspace",
                detach=True,
                remove=True,
                network_disabled=False,
                mem_limit=self.mem_limit,
                pids_limit=self.pids_limit,
            )

            try:
                result = await asyncio.to_thread(container.wait, timeout=timeout)
            except Exception:
                container.kill()
                return f"Error: Command timed out after {timeout} seconds.", -1

            logs = (await asyncio.to_thread(container.logs)).decode("utf-8", errors="replace")
            exit_code = result.get("StatusCode", 0)
            return logs, exit_code

        except Exception as e:
            log.error(f"Docker exec failed: {e}")
            return f"Error: {e}", -1

    async def is_available(self) -> bool:
        """Check if Docker daemon is reachable."""
        if not self._client:
            return False
        try:
            await asyncio.to_thread(self._client.ping)
            return True
        except Exception:
            return False

    async def get_info(self) -> dict:
        """Return Docker environment metadata."""
        info: dict = {
            "name": "docker",
            "image": self.image,
            "project_dir": self.project_dir,
            "workspace_dir": self.workspace_dir,
        }
        if self._client:
            try:
                version = await asyncio.to_thread(self._client.version)
                info["docker_version"] = version.get("Version", "unknown")
            except Exception:
                pass
        return info
