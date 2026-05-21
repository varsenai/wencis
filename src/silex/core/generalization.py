"""
Generalization Engine — Phase 6.

Extracts universal principles from domain-specific causal observations
and provides them as context for cross-domain reasoning.
"""

from __future__ import annotations

import json

from silex.llm.base import SupportsLLM
from silex.models.schemas import (
    CausalObservation,
    PrincipleExtractionResponse,
    UniversalPrinciple,
)
from silex.storage.database import Database
from silex.utils.logger import setup_logger

log = setup_logger("silex.generalization")

EXTRACTION_PROMPT = """You are an Abstract Principle Extractor.
You receive specific causal observations from a particular domain.
Your job is to determine whether a UNIVERSAL, DOMAIN-AGNOSTIC principle
can be extracted from them.

Rules:
1. A universal principle must be TRUE across at least 3 different domains.
2. It must be stated WITHOUT domain-specific jargon.
3. Give it a memorable name (e.g., "The Friction Law", "The Compounding Principle").
4. If the observations are too narrow or domain-specific for abstraction, set has_principle to false.

Examples of good principles:
- "The Bottleneck Principle": Performance of any system is limited by its weakest component.
  (Applies to: manufacturing, software, biology, economics, military strategy)
- "The Network Effect Law": The value of a network grows non-linearly with each new participant.
  (Applies to: social media, telephone systems, languages, marketplaces, ecosystems)

Do NOT force a principle where none exists. Quality over quantity."""


class GeneralizationEngine:
    """Manages extraction and retrieval of universal principles."""

    def __init__(self, llm_client: SupportsLLM, db: Database):
        self.llm = llm_client
        self.db = db

    async def abstract_principles(
        self, observations: list[CausalObservation]
    ) -> UniversalPrinciple | None:
        """
        Given new causal observations, attempt to extract a universal principle.
        Returns the principle if one was found, None otherwise.
        """
        if not observations:
            return None

        obs_text = "\n".join(
            f"- {o.from_concept} --[{o.relationship}]--> {o.to_concept} "
            f"(strength: {o.strength}, evidence: {o.evidence})"
            for o in observations
        )

        content = f"CAUSAL OBSERVATIONS:\n{obs_text}\n\nExtract a universal principle if one exists."

        try:
            extraction = await self.llm.complete_json(
                schema=PrincipleExtractionResponse,
                system_prompt=EXTRACTION_PROMPT,
                user_input=content,
                temperature=0.3,
                request_kind="generalization",
            )

            if not extraction.has_principle or not extraction.name:
                log.debug("No universal principle extracted from this batch.")
                return None

            # Check for duplicates
            existing = await self.db.fetch_all(
                "SELECT name FROM principles WHERE name = ?",
                (extraction.name,)
            )
            if existing:
                log.debug(f"Principle '{extraction.name}' already exists. Skipping.")
                return None

            # Persist
            principle = UniversalPrinciple(
                name=extraction.name,
                statement=extraction.statement,
                original_domain=extraction.original_domain,
                applicable_domains=extraction.applicable_domains,
                source_observations=[
                    f"{o.from_concept} -> {o.to_concept}" for o in observations
                ],
            )

            await self.db.execute(
                """
                INSERT INTO principles (
                    id, name, statement, original_domain,
                    applicable_domains_json, source_observations_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    principle.id,
                    principle.name,
                    principle.statement,
                    principle.original_domain,
                    json.dumps(principle.applicable_domains),
                    json.dumps(principle.source_observations),
                    principle.created_at,
                ),
            )

            log.info(f"New principle extracted: '{principle.name}'")
            return principle

        except Exception as e:
            log.warning(f"Principle extraction failed (non-fatal): {e}")
            return None

    async def get_all_principles(self) -> list[UniversalPrinciple]:
        """Fetch all stored universal principles."""
        rows = await self.db.fetch_all(
            "SELECT * FROM principles ORDER BY created_at DESC"
        )
        return [
            UniversalPrinciple(
                id=r["id"],
                name=r["name"],
                statement=r["statement"],
                original_domain=r["original_domain"],
                applicable_domains=json.loads(r["applicable_domains_json"]),
                source_observations=json.loads(r["source_observations_json"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def format_for_prompt(self, principles: list[UniversalPrinciple]) -> str:
        """Format principles for injection into the system prompt."""
        if not principles:
            return ""

        lines = [
            "═══════════════════════════════════════════════════════════",
            f"UNIVERSAL PRINCIPLES ({len(principles)} discovered)",
            "═══════════════════════════════════════════════════════════",
            "Use these to draw structural analogies across domains:",
            "",
        ]

        for p in principles:
            domains = ", ".join(p.applicable_domains[:5])
            lines.append(f"  📐 [{p.name}]: {p.statement}")
            lines.append(f"     Learned from: {p.original_domain}")
            lines.append(f"     Also applies to: {domains}")
            lines.append("")

        return "\n".join(lines)
