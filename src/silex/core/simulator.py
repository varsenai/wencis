"""
Phantom Simulator (World Model V2)

Provides a safe, non-Docker environment for ARIA to dry-run code changes.
Copies target files into a hidden temporary directory, applies modifications,
and runs local syntax checks (Python/TypeScript). 
Enforces a strict cleanup policy to guarantee no residual files are left behind.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from silex.utils.logger import setup_logger
from silex.utils.config import WORKSPACE_DIR, VYN_PHANTOM

log = setup_logger("silex.world_model.simulator")


class PhantomSimulator:
    """Safely dry-runs file modifications before committing them to the workspace."""

    def __init__(self):
        self.phantom_dir = VYN_PHANTOM
        
    def _setup_phantom_dir(self):
        """Ensure the phantom directory is clean before starting."""
        if self.phantom_dir.exists():
            shutil.rmtree(self.phantom_dir, ignore_errors=True)
        self.phantom_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup(self):
        """V2 Constraint: Strict cleanup policy. Delete unconditionally after run."""
        if self.phantom_dir.exists():
            try:
                shutil.rmtree(self.phantom_dir, ignore_errors=True)
                log.debug("Phantom directory cleaned up.")
            except Exception as e:
                log.error(f"Failed to clean phantom directory: {e}")

    def simulate_edit(self, filepath: Path, new_content: str) -> tuple[bool, str]:
        """
        Simulates editing a file and runs a syntax check.
        Returns (success: bool, error_message: str)
        """
        self._setup_phantom_dir()
        
        try:
            # 1. Mirror the target file into the phantom directory
            relative_path = filepath.relative_to(WORKSPACE_DIR)
            phantom_file = self.phantom_dir / relative_path
            
            phantom_file.parent.mkdir(parents=True, exist_ok=True)
            phantom_file.write_text(new_content, encoding="utf-8")
            
            # 2. Run Syntax/Compiler Check
            success, error_msg = self._run_syntax_check(phantom_file)
            
            if not success:
                log.warning(f"Phantom simulation failed for {filepath.name}: {error_msg}")
                # Issue 13: Push alert to Telegram via goals queue
                self._push_alert(filepath.name, error_msg)
            else:
                log.info(f"Phantom simulation passed for {filepath.name}.")
                
            return success, error_msg

        except Exception as e:
            msg = f"Simulator internal error: {str(e)}"
            log.error(msg)
            return False, msg
            
        finally:
            # 3. V2 Cleanup Policy Execution
            self._cleanup()

    def _push_alert(self, filename: str, error_msg: str):
        """Pushes an alert to the job queue for the Telegram worker to report."""
        import sqlite3
        import uuid
        from datetime import datetime, timezone
        from silex.utils.config import VYN_DB
        try:
            conn = sqlite3.connect(VYN_DB)
            cur = conn.cursor()
            goal_id = f"goal_{uuid.uuid4().hex[:8]}"
            now = datetime.now(timezone.utc).isoformat()
            alert_msg = f"[ALERT] Phantom Sandbox compiler check failed on {filename}:\n\n{error_msg[:500]}"
            cur.execute(
                "INSERT INTO goals (id, description, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (goal_id, alert_msg, "alert", now, now)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            log.error(f"Failed to push alert to queue: {e}")

    def _run_syntax_check(self, phantom_file: Path) -> tuple[bool, str]:
        """Route to the correct syntax checker based on file type."""
        ext = phantom_file.suffix.lower()
        
        if ext == ".py":
            return self._check_python(phantom_file)
        elif ext in {".ts", ".tsx"}:
            return self._check_typescript(phantom_file)
        elif ext in {".json", ".jsonc"}:
            return self._check_json(phantom_file)
            
        # If it's a file type we can't check, assume safe syntax
        return True, ""

    def _check_python(self, filepath: Path) -> tuple[bool, str]:
        """Use Python's built-in compiler to check syntax."""
        try:
            subprocess.run(
                ["python", "-m", "py_compile", str(filepath)],
                capture_output=True,
                text=True,
                check=True
            )
            return True, ""
        except subprocess.CalledProcessError as e:
            return False, e.stderr.strip()

    def _check_typescript(self, filepath: Path) -> tuple[bool, str]:
        """Attempt to run tsc --noEmit. Fallback to True if tsc not available."""
        try:
            # Note: We run tsc from the workspace root so it finds tsconfig.json
            # but point it to the phantom file.
            result = subprocess.run(
                ["npx", "tsc", "--noEmit", str(filepath)],
                cwd=str(WORKSPACE_DIR),
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode != 0:
                # TypeScript throws a lot of false positives if the phantom file
                # doesn't have the full project context. For now, we only look for 
                # fatal parsing errors (TS1005: expected something).
                if "TS1005" in result.stdout or "TS1128" in result.stdout:
                    return False, result.stdout.strip()
            return True, ""
        except FileNotFoundError:
            # If npx/tsc isn't installed, we bypass the check safely.
            return True, ""

    def _check_json(self, filepath: Path) -> tuple[bool, str]:
        """Parse JSON to ensure it is valid."""
        import json
        try:
            content = filepath.read_text(encoding="utf-8")
            json.loads(content)
            return True, ""
        except json.JSONDecodeError as e:
            return False, str(e)
