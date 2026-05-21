"""
File Reader Tool for accessing local files securely.

Security: All reads are sandboxed to the project root directory.
Dotfiles and sensitive paths are blocked to prevent credential leaks.
"""

from __future__ import annotations

from pathlib import Path
import aiofiles
from silex.tools.base import BaseTool
from silex.utils.logger import setup_logger
from silex.utils.config import WORKSPACE_DIR

log = setup_logger("silex.tools.file_reader")

# ---------------------------------------------------------------------------
# Security — define the sandbox boundary
# ---------------------------------------------------------------------------

# Sandbox root: resolved from config.py (VYN_WORKSPACE or VYN_WORKSPACE env override).
# This ensures pip-installed copies don't accidentally expose site-packages.
_PROJECT_ROOT = WORKSPACE_DIR

# Patterns that are ALWAYS blocked, even inside the project root
_BLOCKED_NAMES = {
    ".env", ".env.local", ".env.production", ".env.development",
    ".git", ".gitignore",
}
_BLOCKED_PREFIXES = {".env"}  # Catches .env.anything
_BLOCKED_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}


def _is_path_safe(path: Path) -> tuple[bool, str]:
    """
    Validate that a resolved path is safe to read.

    Returns (is_safe, reason) — reason is empty if safe.
    """
    # 1. Must be inside the project root
    try:
        path.relative_to(_PROJECT_ROOT)
    except ValueError:
        return False, f"Access denied — path is outside the project directory ({_PROJECT_ROOT.name}/)."

    # 2. No dotfiles / sensitive files
    if path.name in _BLOCKED_NAMES:
        return False, f"Access denied — '{path.name}' is a restricted file."

    for prefix in _BLOCKED_PREFIXES:
        if path.name.startswith(prefix):
            return False, f"Access denied — files starting with '{prefix}' are restricted."

    # 3. No path component should be a blocked directory
    for part in path.parts:
        if part in _BLOCKED_DIRS:
            return False, f"Access denied — the '{part}/' directory is restricted."

    return True, ""


class FileReaderTool(BaseTool):
    name = "read_file"
    risk_level = "read_only"
    description = (
        "Reads the text content of a local file. "
        "Files must be inside the project directory. "
        "Dotfiles (.env, .git) are blocked for security."
    )
    schema = {
        "file_path": "string (the path to the file, relative to the project root)",
    }

    async def execute(self, **kwargs) -> str:
        file_path = kwargs.get("file_path")
        if not file_path:
            return "Error: 'file_path' argument is required."

        candidate = Path(file_path)
        if not candidate.is_absolute():
            candidate = _PROJECT_ROOT / candidate
        path = candidate.resolve()

        # ── Security gate ──────────────────────────────────────────
        safe, reason = _is_path_safe(path)
        if not safe:
            log.warning(f"Blocked file read attempt: {path} — {reason}")
            return f"Error: {reason}"
        # ───────────────────────────────────────────────────────────

        log.info(f"Reading file: {path}")

        if not path.exists():
            return f"Error: File not found at path: {path}"

        if not path.is_file():
            return f"Error: Path is not a file: {path}"

        try:
            async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
                # Read up to first 10,000 characters to prevent context overflow
                content = await f.read(10000)

            if len(content) == 10000:
                content += "\n...[TRUNCATED due to length]"

            return f"Contents of {path.name}:\n\n{content}"

        except UnicodeDecodeError:
            return f"Error: File {path.name} appears to be binary or has an unsupported encoding."
        except Exception as e:
            log.error(f"Failed to read file {path}: {e}")
            return "Error reading file: could not read the file."
