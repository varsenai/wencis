"""
Lightweight ethical policy checks for tool execution.

The goal is not to solve morality in code. The goal is to make high-impact
actions legible, auditable, and easier to escalate before ARIA acts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from silex.models.schemas import EthicalAction, EthicalDecision, ToolCall, ToolRisk
from silex.tools.base import BaseTool


PRINCIPLES = {
    "beneficence": "Preserve life, reduce suffering, and support flourishing.",
    "non_domination": "Do not coerce, trap, or impose control over people.",
    "truthfulness": "Prefer calibrated truth over manipulation or false certainty.",
    "consent_and_privacy": "Treat access to data, devices, and systems as morally significant.",
    "corrigibility": "Respect approval flows, operator oversight, tests, and explicit policy.",
    "moral_uncertainty": "Escalate when uncertainty or irreversible impact is high.",
    "identity_continuity": "Grow through stable commitments, not opportunistic rule-breaking.",
}


class EthicsEngine:
    """Applies a small constitutional check before tool execution."""

    HIGH_RISK = {
        ToolRisk.SANDBOX_WRITE,
        ToolRisk.REPO_WRITE,
        ToolRisk.DESTRUCTIVE,
        ToolRisk.EXTERNAL_SIDE_EFFECT,
    }

    def evaluate_tool_call(
        self,
        call: ToolCall,
        tool: BaseTool,
        args_dict: dict[str, Any],
        *,
        execution_mode: str = "interactive",
    ) -> EthicalDecision:
        """Return a traceable proceed/escalate/refuse decision."""
        risk = self._normalize_risk(tool.risk_level)
        execution_mode = execution_mode or "interactive"

        if risk == ToolRisk.DESTRUCTIVE:
            return EthicalDecision(
                action=EthicalAction.REFUSE,
                principle="beneficence",
                rationale=(
                    "Destructive actions are not allowed to proceed automatically. "
                    "They create disproportionate irreversible risk."
                ),
                risk_level=risk,
                requires_consent=True,
                uncertainty=0.1,
                context=execution_mode,
            )

        if execution_mode == "background" and risk != ToolRisk.READ_ONLY:
            return EthicalDecision(
                action=EthicalAction.ESCALATE,
                principle="corrigibility",
                rationale=(
                    "Background autonomy may not perform non-read-only or externally "
                    "visible actions without explicit approval."
                ),
                risk_level=risk,
                requires_consent=True,
                uncertainty=0.45,
                context=execution_mode,
            )

        if risk in self.HIGH_RISK:
            return EthicalDecision(
                action=EthicalAction.ESCALATE,
                principle="consent_and_privacy",
                rationale=(
                    "This action can change code, execute commands, or create durable "
                    "side effects. It requires explicit human oversight."
                ),
                risk_level=risk,
                requires_consent=True,
                uncertainty=0.25,
                context=execution_mode,
            )

        if risk == ToolRisk.NETWORK:
            if self._looks_sensitive_network_request(args_dict):
                return EthicalDecision(
                    action=EthicalAction.ESCALATE,
                    principle="consent_and_privacy",
                    rationale=(
                        "This network action appears to involve credentials, tokens, "
                        "accounts, or private data and should be escalated."
                    ),
                    risk_level=risk,
                    requires_consent=True,
                    uncertainty=0.4,
                    context=execution_mode,
                )

            return EthicalDecision(
                action=EthicalAction.PROCEED,
                principle="truthfulness",
                rationale=(
                    "Network access is permitted for scoped information gathering, "
                    "but it should stay task-relevant and privacy-aware."
                ),
                risk_level=risk,
                requires_consent=False,
                uncertainty=0.2,
                context=execution_mode,
            )

        if self._looks_like_broad_file_access(args_dict):
            return EthicalDecision(
                action=EthicalAction.ESCALATE,
                principle="consent_and_privacy",
                rationale=(
                    "The requested path looks broad enough to expose more data than "
                    "the current task likely needs."
                ),
                risk_level=risk,
                requires_consent=True,
                uncertainty=0.35,
                context=execution_mode,
            )

        return EthicalDecision(
            action=EthicalAction.PROCEED,
            principle="truthfulness",
            rationale=(
                "This action is low risk and proportionate to the current task."
            ),
            risk_level=risk,
            requires_consent=False,
            uncertainty=0.1,
            context=execution_mode,
        )

    @staticmethod
    def _normalize_risk(risk_level: str) -> ToolRisk:
        try:
            return ToolRisk(risk_level)
        except ValueError:
            return ToolRisk.READ_ONLY

    @staticmethod
    def _looks_sensitive_network_request(args_dict: dict[str, Any]) -> bool:
        text = " ".join(str(v).lower() for v in args_dict.values())
        sensitive_markers = (
            "token",
            "password",
            "secret",
            "cookie",
            "session",
            "auth",
            "credential",
            "login",
            "private",
        )
        return any(marker in text for marker in sensitive_markers)

    @staticmethod
    def _looks_like_broad_file_access(args_dict: dict[str, Any]) -> bool:
        raw_path = args_dict.get("path") or args_dict.get("directory")
        if not raw_path:
            return False
        path = Path(str(raw_path))
        broad_markers = {"", ".", "..", "/", "\\", "c:\\", "d:\\", "e:\\"}
        normalized = str(path).strip().lower()
        if normalized in broad_markers:
            return True
        return path.is_absolute() and len(path.parts) <= 1
