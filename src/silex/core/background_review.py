"""
Background Review Daemon — Post-Turn Learning Agent (Phase 21).

After every N turns (default: 3), this lightweight daemon analyzes
recent conversation for learning opportunities:

  1. **User corrections**: "no, I meant…", "that's wrong", "actually…"
  2. **Failed tool calls**: errors, retries, self-healing loops
  3. **Workarounds**: user-demonstrated techniques the agent didn't know
  4. **Skill candidates**: multi-step workflows worth saving

It then writes memories or creates/updates skills autonomously.
Restricted to memory + skill mutations only — no execution tools.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from silex.utils.logger import setup_logger

if TYPE_CHECKING:
    from silex.core.skills import SkillLoader
    from silex.memory.memory_store import MemoryStore

log = setup_logger("silex.core.background_review")

# Patterns that indicate the user corrected the agent
_CORRECTION_MARKERS = [
    "no, i meant",
    "that's wrong",
    "that's not right",
    "that is wrong",
    "actually,",
    "not what i asked",
    "i said",
    "you misunderstood",
    "wrong.",
    "incorrect",
    "no no",
    "stop.",
    "let me clarify",
]


class BackgroundReviewer:
    """Post-turn learning agent that runs as a background task.

    Analyzes recent conversation history for learning signals and
    writes memories/skills from the findings.
    """

    def __init__(
        self,
        llm,
        skill_loader: "SkillLoader",
        memory_store: "MemoryStore",
        db,
        review_interval: int = 3,
    ):
        self.llm = llm
        self.skill_loader = skill_loader
        self.memory = memory_store
        self.db = db
        self.review_interval = review_interval
        self._turn_counter = 0

    async def maybe_review(self, recent_turns: list, session_id: str) -> None:
        """Called after each turn.  Triggers a review every N turns."""
        self._turn_counter += 1
        if self._turn_counter < self.review_interval:
            log.debug(
                f"Background review: skip ({self._turn_counter}/{self.review_interval})"
            )
            return

        self._turn_counter = 0
        log.info("Background review daemon triggered — analyzing recent conversation")

        try:
            await self._review(recent_turns, session_id)
        except Exception as e:
            log.warning(f"Background review failed (non-fatal): {e}")

    async def _review(self, recent_turns: list, session_id: str) -> None:
        """Full review cycle: detect corrections, failures, and skill candidates."""
        if not recent_turns:
            return

        # ── 1. Detect user corrections ────────────────────────────
        corrections = self._detect_corrections(recent_turns)
        for correction in corrections:
            await self._store_correction_memory(correction, session_id)

        # ── 2. Detect failed tool approaches ──────────────────────
        failures = await self._detect_failed_approaches(session_id)
        for failure in failures:
            await self._store_failure_memory(failure, session_id)

        # ── 3. Check skill refinement candidates ──────────────────
        try:
            candidates = self.skill_loader.get_refinement_candidates()
            for c in candidates:
                log.info(f"Skill '{c}' qualifies for refinement — will refine in background")
                # Refinement is handled by cognitive_loop._refine_skill()
                # We just log the candidates here; the loop picks them up
        except Exception as e:
            log.warning(f"Refinement candidate check failed: {e}")

        # ── 4. LLM-powered analysis (if enough turns) ────────────
        if len(recent_turns) >= 4:
            await self._llm_review(recent_turns, session_id)

    def _detect_corrections(self, turns: list) -> list[dict]:
        """Scan recent user messages for correction patterns.

        Returns a list of dicts with keys: ``user_said``, ``turn_number``,
        ``correction_marker``.
        """
        corrections = []
        for turn in turns:
            user_lower = turn.user_input.lower().strip()
            for marker in _CORRECTION_MARKERS:
                if marker in user_lower:
                    corrections.append({
                        "user_said": turn.user_input[:300],
                        "turn_number": turn.turn_number,
                        "correction_marker": marker,
                    })
                    break  # one correction per turn
        return corrections

    async def _detect_failed_approaches(self, session_id: str) -> list[dict]:
        """Query the database for recent tool failures in this session."""
        if not self.db:
            return []
        try:
            rows = await self.db.fetch_all(
                """
                SELECT failure_type, description, created_at
                FROM session_failures
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT 5
                """,
                (session_id,),
            )
            return [dict(r) for r in rows] if rows else []
        except Exception as e:
            log.debug(f"Failed approach detection query failed: {e}")
            return []

    async def _store_correction_memory(
        self, correction: dict, session_id: str
    ) -> None:
        """Store a user correction as a memory."""
        from silex.models.schemas import Memory, MemorySource

        content = (
            f"User correction (turn {correction['turn_number']}): "
            f"{correction['user_said']}"
        )
        mem = Memory(
            content=content,
            source=MemorySource.REFLECTION,
            importance=0.6,
            level=1,
            tags=["user_correction", "background_review"],
            provenance={"session_id": session_id, "marker": correction["correction_marker"]},
        )
        await self.memory.add(mem)
        log.info(f"📝 Background review: stored correction memory (turn {correction['turn_number']})")

    async def _store_failure_memory(self, failure: dict, session_id: str) -> None:
        """Store a failed approach as a negative memory."""
        from silex.models.schemas import Memory, MemorySource

        content = (
            f"Failed approach ({failure.get('failure_type', 'unknown')}): "
            f"{failure.get('description', 'No description')}"
        )
        mem = Memory(
            content=content,
            source=MemorySource.REFLECTION,
            importance=0.5,
            level=1,
            tags=["failed_approach", "background_review"],
            provenance={"session_id": session_id},
        )
        await self.memory.add(mem)
        log.info(f"📝 Background review: stored failure memory")

    async def _llm_review(self, recent_turns: list, session_id: str) -> None:
        """Use the fast LLM to analyze conversation for deeper insights.

        Only called when there are ≥4 turns to analyze.  Uses structured
        output to extract learnings.
        """
        from silex.core.skill_schemas import BackgroundReviewResponse

        # Build transcript
        transcript_lines = []
        for t in recent_turns[-6:]:  # last 6 turns max
            user = t.user_input[:300].replace("\n", " ")
            resp = t.response[:400].replace("\n", " ")
            transcript_lines.append(f"Turn {t.turn_number} — User: {user} | VYN: {resp}")
        transcript = "\n".join(transcript_lines)

        prompt = (
            "You are a background learning agent. Analyze this conversation transcript "
            "and extract learning signals. Focus on:\n"
            "1. Things the user corrected or clarified\n"
            "2. Approaches that failed and why\n"
            "3. New facts or preferences to remember\n"
            "4. Multi-step workflows that could be saved as reusable skills\n\n"
            "Be concise. Only include genuinely useful learnings, not noise."
        )

        try:
            from silex.runtime.settings import get_provider_settings
            provider_settings = get_provider_settings(None)

            review = await self.llm.complete_json(
                schema=BackgroundReviewResponse,
                system_prompt=prompt,
                user_input=f"TRANSCRIPT:\n{transcript}",
                model_override=provider_settings.get("fast_model"),
                temperature=0.2,
                request_kind="background_review",
            )

            # Process new memories
            for mem_content in review.new_memories[:3]:  # cap at 3
                from silex.models.schemas import Memory, MemorySource
                mem = Memory(
                    content=mem_content,
                    source=MemorySource.REFLECTION,
                    importance=0.5,
                    level=1,
                    tags=["background_review", "llm_extracted"],
                    provenance={"session_id": session_id},
                )
                await self.memory.add(mem)

            if review.new_memories:
                log.info(f"📝 Background review: stored {len(review.new_memories)} LLM-extracted memories")

            # Log skill candidates (synthesis is handled by _synthesize_skill)
            for sc in review.skill_candidates:
                log.info(f"💡 Background review: skill candidate detected: {sc.get('name', 'unnamed')}")

        except Exception as e:
            log.warning(f"LLM background review failed (non-fatal): {e}")
