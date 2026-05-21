"""
Code Editor Tool — allows ARIA to propose and apply file changes.

Security (v1.0.5):
  - All paths are sandboxed to ARIA_WORKSPACE (not the repo root).
  - The autonomous apply bypass has been removed. ALL edits flow through
    the ethics engine and approval queue in the ToolRegistry.
  - The code-apply operator flag now controls whether the registry
    auto-approves code edits, NOT whether ethics is skipped.
  - Dotfiles and sensitive directories are always blocked.
"""

import json
import uuid
from pathlib import Path

from silex.tools.base import BaseTool
from silex.utils.logger import setup_logger
from silex.utils.config import VYN_PENDING_EDITS

log = setup_logger("silex.tools.code_editor")

# ---------------------------------------------------------------------------
# Security — sandbox boundary (matches file_reader.py)
# ---------------------------------------------------------------------------

# Sandbox root: resolved from config.py (VYN_WORKSPACE or env override).
# This ensures pip-installed copies don't accidentally write to site-packages.
from silex.utils.config import WORKSPACE_DIR as _WORKSPACE_ROOT  # noqa: E402

PENDING_EDITS_FILE = VYN_PENDING_EDITS
BLOCKED_FILE_PREFIXES = (".env",)
BLOCKED_PATH_PARTS = {".git", "node_modules", ".venv", "venv", "__pycache__"}


def _resolve_workspace_path(file_path: str) -> Path:
    """Resolve a user-supplied path inside the ARIA workspace sandbox.

    Security guarantees:
      1. The resolved path must be inside _WORKSPACE_ROOT.
      2. Dotfiles (.env*) are always blocked.
      3. Sensitive directories (.git, node_modules, etc.) are blocked.
    """
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = _WORKSPACE_ROOT / candidate
    full_path = candidate.resolve()

    try:
        full_path.relative_to(_WORKSPACE_ROOT)
    except ValueError:
        raise ValueError(
            f"Access denied — path is outside the workspace directory ({_WORKSPACE_ROOT.name}/)."
        )

    if full_path.name.startswith(BLOCKED_FILE_PREFIXES):
        raise ValueError("Access denied — environment files are restricted.")
    if any(part in BLOCKED_PATH_PARTS for part in full_path.parts):
        raise ValueError("Access denied — restricted directory component.")

    return full_path


class CodeEditorTool(BaseTool):
    """
    Allows ARIA to propose changes to files in the workspace.

    Security: This tool NEVER applies edits directly. It always creates
    a draft proposal. The ToolRegistry's ethics engine and approval queue
    decide whether and when the edit is actually applied.
    """

    name = "propose_code_edit"
    risk_level = "repo_write"
    requires_approval = True
    description = (
        "Propose a change to a file in the workspace. The change will NOT be "
        "applied immediately — it will be saved as a draft pending human approval. "
        "Ask the user to approve the edit in your response."
    )

    schema = {
        "file_path": "string (relative path to the file inside the workspace)",
        "target_content": "string (the exact block of code to replace. Leave empty to overwrite the entire file)",
        "replacement_content": "string (the new code to insert)",
        "explanation": "string (why you are making this change)"
    }

    async def execute(self, **kwargs) -> str:
        file_path = kwargs.get("file_path")
        target_content = kwargs.get("target_content", "")
        replacement_content = kwargs.get("replacement_content", "")
        explanation = kwargs.get("explanation", "No explanation provided.")

        if not file_path or not replacement_content:
            return "ERROR: file_path and replacement_content are required."

        try:
            full_path = _resolve_workspace_path(file_path)
        except ValueError as e:
            return f"ERROR: {e}"

        # Verify target content if provided
        if target_content and full_path.exists():
            with open(full_path, "r", encoding="utf-8") as f:
                current_code = f.read()
                if target_content not in current_code:
                    return "ERROR: The target_content was not found exactly as written in the file."
                if current_code.count(target_content) != 1:
                    return "ERROR: The target_content is ambiguous. It appears multiple times in the file. Please provide a larger, unique block of code to replace."

        # Load existing pending edits
        pending_edits = []
        if PENDING_EDITS_FILE.exists():
            try:
                with open(PENDING_EDITS_FILE, "r", encoding="utf-8") as f:
                    pending_edits = json.load(f)
                    if not isinstance(pending_edits, list):
                        pending_edits = []
            except (json.JSONDecodeError, OSError, ValueError):
                pending_edits = []

        proposal = {
            "id": str(uuid.uuid4())[:8],
            "file_path": str(full_path),
            "target_content": target_content,
            "replacement_content": replacement_content,
            "explanation": explanation
        }

        # ALL edits go to the pending queue. The ethics engine and approval
        # flow in the ToolRegistry decide when they are applied.
        # The operator's code-apply flag is handled by the registry's
        # _approval_required() method, NOT here.
        pending_edits.append(proposal)
        with open(PENDING_EDITS_FILE, "w", encoding="utf-8") as f:
            json.dump(pending_edits, f, indent=4)

        return (
            f"DRAFT CREATED (ID: {proposal['id']}) for {file_path}.\n"
            f"STATUS: PENDING HUMAN APPROVAL.\n"
            f"INSTRUCTION: Ask the user to 'approve edit {proposal['id']}' or 'approve all edits'."
        )

    def _apply_edit_logic(self, proposal: dict):
        """Apply a single edit proposal to disk.

        Called ONLY by ApplyEditTool after the edit has passed through
        the ethics engine and approval queue.
        """
        import shutil
        from datetime import datetime
        from silex.utils.config import VYN_BACKUPS

        full_path = _resolve_workspace_path(proposal["file_path"])
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if full_path.exists():
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = VYN_BACKUPS / f"{full_path.name}_{timestamp}.bak"
                shutil.copy2(full_path, backup_path)
                log.info(f"Failsafe backup created: {backup_path}")
            except Exception as e:
                log.warning(f"Failed to create failsafe backup for {full_path}: {e}")

        if proposal["target_content"] and full_path.exists():
            with open(full_path, "r", encoding="utf-8") as f:
                current_code = f.read()
            new_code = current_code.replace(proposal["target_content"], proposal["replacement_content"], 1)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_code)
        else:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(proposal["replacement_content"])


class ApplyEditTool(BaseTool):
    """
    Applies pending code edits that have been approved.
    """

    name = "apply_approved_edit"
    risk_level = "repo_write"
    requires_approval = True
    description = (
        "Applies one or all pending code edits. "
        "Usage: Provide 'edit_id' for a specific file, or leave empty to apply ALL pending edits."
    )

    schema = {
        "edit_id": "string (optional, the ID of a specific edit to apply. If omitted, all edits are applied)"
    }

    async def execute(self, **kwargs) -> str:
        edit_id = kwargs.get("edit_id")

        if not PENDING_EDITS_FILE.exists():
            return "ERROR: No pending edits found."

        try:
            with open(PENDING_EDITS_FILE, "r", encoding="utf-8") as f:
                pending = json.load(f)

            if not pending:
                return "ERROR: No pending edits in the list."

            if edit_id:
                # Apply specific edit
                to_apply = [e for e in pending if e["id"] == edit_id]
                if not to_apply:
                    return f"ERROR: No edit found with ID {edit_id}."

                edit = to_apply[0]
                CodeEditorTool()._apply_edit_logic(edit)

                # Remove from list
                remaining = [e for e in pending if e["id"] != edit_id]
                if remaining:
                    with open(PENDING_EDITS_FILE, "w", encoding="utf-8") as f:
                        json.dump(remaining, f, indent=4)
                else:
                    PENDING_EDITS_FILE.unlink()

                return f"SUCCESS: Edit {edit_id} for {Path(edit['file_path']).name} applied."

            else:
                # Apply ALL
                for edit in pending:
                    CodeEditorTool()._apply_edit_logic(edit)

                PENDING_EDITS_FILE.unlink()
                return f"SUCCESS: All {len(pending)} pending edits have been applied."

        except Exception as e:
            return f"ERROR applying edit: {str(e)}"
