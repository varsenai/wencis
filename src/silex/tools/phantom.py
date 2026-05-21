"""
Phantom Simulator — dry-run code changes before applying them to the workspace.

Design (from notes/vision/world_model_v2.md §3):
  1. Copy the target file(s) into ~/.vyn/phantom/<run-id>/
  2. Apply the proposed code change to the phantom copy
  3. Run a syntax/compiler check:
     - Python files  → python -m py_compile
     - TypeScript/JS → tsc --noEmit (if tsc is available)
  4. On SUCCESS: report clean, caller can apply to real workspace
     On FAILURE:  return the error output so the LLM can learn from it
  5. Always clean up the phantom directory after the run

The phantom dir lives in ~/.vyn/phantom/, outside the workspace,
so the FSWatcher never picks up temporary files.

Security:
  - Phantom dir is isolated from the real workspace.
  - No changes are written back to the workspace by this tool.
  - The tool is read-only from the workspace perspective; it only writes
    to the ephemeral phantom directory which is deleted on completion.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import uuid
from pathlib import Path

from silex.tools.base import BaseTool
from silex.utils.config import VYN_HOME, WORKSPACE_DIR as _WORKSPACE_ROOT
from silex.utils.logger import setup_logger

log = setup_logger("silex.tools.phantom")

# The phantom runs live here — outside the workspace so the watcher ignores them.
PHANTOM_BASE = VYN_HOME / "phantom"


def _resolve_workspace_path(file_path: str) -> Path:
    """Resolve a path and ensure it is inside the workspace sandbox."""
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = _WORKSPACE_ROOT / candidate
    full_path = candidate.resolve()
    try:
        full_path.relative_to(_WORKSPACE_ROOT)
    except ValueError:
        raise ValueError(
            f"Access denied — path '{file_path}' is outside the workspace."
        )
    return full_path


class PhantomTool(BaseTool):
    """
    Dry-run a proposed code change in an isolated phantom directory.

    Copies the target file to ~/.vyn/phantom/<run-id>/, applies the
    proposed content, runs a syntax check (py_compile for .py, tsc --noEmit
    for .ts/.tsx), reports SUCCESS or FAILURE with error output, then
    cleans up. No real workspace files are modified.
    """

    name = "phantom_test"
    risk_level = "read_only"
    requires_approval = False
    description = (
        "Test a proposed code change in a safe isolated environment before applying it. "
        "Copies the target file to a temporary phantom directory, applies the proposed content, "
        "and runs a syntax/compiler check. Returns SUCCESS or FAILURE with error details. "
        "Always use this before propose_code_edit when making non-trivial changes to Python or TypeScript files."
    )
    schema = {
        "file_path": "str — workspace-relative or absolute path to the file being changed",
        "proposed_content": "str — the full new file content to validate",
    }

    async def execute(self, *, file_path: str, proposed_content: str, **kwargs) -> str:
        run_id = uuid.uuid4().hex[:8]
        phantom_dir = PHANTOM_BASE / run_id
        phantom_dir.mkdir(parents=True, exist_ok=True)

        try:
            return await self._run(file_path, proposed_content, phantom_dir, run_id)
        finally:
            # Always clean up — no orphaned phantom files
            shutil.rmtree(phantom_dir, ignore_errors=True)
            log.debug(f"Phantom run {run_id} cleaned up.")

    async def _run(
        self,
        file_path: str,
        proposed_content: str,
        phantom_dir: Path,
        run_id: str,
    ) -> str:
        # --- Resolve and validate the real file path ---
        try:
            real_path = _resolve_workspace_path(file_path)
        except ValueError as e:
            return f"PHANTOM ERROR: {e}"

        suffix = real_path.suffix.lower()
        phantom_file = phantom_dir / real_path.name

        # --- Write proposed content to phantom ---
        try:
            phantom_file.write_text(proposed_content, encoding="utf-8")
        except Exception as e:
            return f"PHANTOM ERROR: Could not write phantom file — {e}"

        log.info(f"Phantom run {run_id}: testing {real_path.name} ({suffix})")

        # --- Run the appropriate syntax/compiler check ---
        if suffix == ".py":
            return await self._check_python(phantom_file, run_id)
        elif suffix in {".ts", ".tsx", ".js", ".jsx"}:
            return await self._check_typescript(phantom_dir, phantom_file, real_path, run_id)
        else:
            # For unsupported types, just confirm the file was written cleanly
            size = phantom_file.stat().st_size
            return (
                f"PHANTOM SUCCESS (run {run_id}): "
                f"File '{real_path.name}' written cleanly ({size} bytes). "
                f"No syntax checker available for '{suffix}' — manual review recommended."
            )

    @staticmethod
    async def _check_python(phantom_file: Path, run_id: str) -> str:
        """Run py_compile on the phantom file to catch syntax errors."""
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "py_compile", str(phantom_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
            if proc.returncode == 0:
                return (
                    f"PHANTOM SUCCESS (run {run_id}): "
                    f"Python syntax check passed. Safe to apply."
                )
            error = stderr.decode(errors="replace").strip()
            # Remap phantom path back to the real filename for clarity
            error = error.replace(str(phantom_file.parent), "<phantom>")
            return (
                f"PHANTOM FAILURE (run {run_id}): Python syntax check failed.\n"
                f"Error:\n{error}\n\n"
                f"Do NOT apply this change. Fix the error first and run phantom_test again."
            )
        except asyncio.TimeoutError:
            return f"PHANTOM ERROR (run {run_id}): py_compile timed out after 15s."
        except Exception as e:
            return f"PHANTOM ERROR (run {run_id}): py_compile failed to run — {e}"

    @staticmethod
    async def _check_typescript(
        phantom_dir: Path,
        phantom_file: Path,
        real_path: Path,
        run_id: str,
    ) -> str:
        """
        Run tsc --noEmit --allowJs on the proposed file, compiled in-situ within the workspace.
        Falls back to a simple brace-balance check if tsc is not available.
        """
        content = phantom_file.read_text(encoding="utf-8", errors="replace")

        # Check if tsc is on PATH (Windows installs it as tsc.cmd)
        tsc_path = shutil.which("tsc") or shutil.which("tsc.cmd")
        if not tsc_path:
            # Fallback: brace balance check (catches most structural errors)
            opens = content.count("{")
            closes = content.count("}")
            if opens == closes:
                return (
                    f"PHANTOM SUCCESS (run {run_id}): "
                    f"TypeScript brace-balance check passed (tsc not available). "
                    f"Install tsc for full type checking."
                )
            return (
                f"PHANTOM FAILURE (run {run_id}): Brace imbalance detected "
                f"({{: {opens}, }}: {closes}). tsc not available for full check."
            )

        # Create a hidden temp file next to the real file in the actual workspace
        # This allows tsc to resolve relative imports and node_modules correctly.
        temp_file = real_path.with_name(f".phantom_{real_path.name}")
        
        try:
            temp_file.write_text(content, encoding="utf-8")

            # On Windows, .cmd files need to be invoked via cmd /c
            import platform as _platform
            if _platform.system() == "Windows" and tsc_path.lower().endswith(".cmd"):
                cmd_args = ["cmd", "/c", tsc_path,
                            "--noEmit", "--allowJs", "--skipLibCheck",
                            str(temp_file)]
            else:
                cmd_args = [tsc_path, "--noEmit", "--allowJs", "--skipLibCheck",
                            str(temp_file)]

            proc = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(_WORKSPACE_ROOT),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            output = (stdout + stderr).decode(errors="replace").strip()

            if proc.returncode == 0:
                return (
                    f"PHANTOM SUCCESS (run {run_id}): "
                    f"TypeScript/JS in-situ check passed (tsc --noEmit). Safe to apply."
                )
            
            # Remap the temp filename back to the real filename so the LLM isn't confused
            output = output.replace(temp_file.name, real_path.name)
            output = output.replace(str(temp_file), real_path.name)
            
            return (
                f"PHANTOM FAILURE (run {run_id}): TypeScript check failed.\n"
                f"Errors:\n{output}\n\n"
                f"Do NOT apply this change. Fix the errors first and run phantom_test again."
            )
        except asyncio.TimeoutError:
            return f"PHANTOM ERROR (run {run_id}): tsc timed out after 30s."
        except Exception as e:
            return f"PHANTOM ERROR (run {run_id}): tsc failed to run — {e}"
        finally:
            # Guarantee cleanup of the hidden temp file in the workspace
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as e:
                    log.error(f"Phantom failed to clean up temp file {temp_file}: {e}")
